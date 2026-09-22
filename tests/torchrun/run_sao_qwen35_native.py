"""Native Qwen3.5-4B FSDP SAO/PPO microbatch preflight.

This script is a bounded torchrun harness for real production engines. It uses
actual rollout fixture rows, records identity/readback evidence, and avoids
full CPU parameter snapshots by comparing sampled local FSDP shards only.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.cli_args import (
    FSDPEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    PPOCriticConfig,
)
from areal.engine.fsdp_engine import FSDPPPOActor, FSDPPPOCritic
from areal.trainer.ppo.update import update_critic_before_actor
from areal.trainer.ppo.value_checkpoint import validate_value_artifact

REQUIRED_SAMPLE_KEYS = {
    "input_tokens",
    "output_tokens",
    "behavior_logprobs",
    "behavior_versions",
    "reward",
    "stop_reason",
    "truncated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--value", type=Path)
    parser.add_argument("--samples", type=Path, nargs="*")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=["sao", "ppo"], default="sao")
    parser.add_argument("--allow-base-critic", action="store_true")
    parser.add_argument("--allow-synthetic-value", action="store_true")
    parser.add_argument("--samples-per-rank", type=int, default=2)
    parser.add_argument("--truncate-tokens", type=int, default=0)
    parser.add_argument("--engine-microbatches", type=int, default=2)
    parser.add_argument("--max-tokens-per-mb", type=int)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--require-world-size", type=int, default=4)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=5e-6)
    parser.add_argument("--sample-param-elements", type=int, default=8)
    parser.add_argument("--check-recovery", action="store_true")
    args = parser.parse_args()
    if (args.samples is None) == (args.batch is None):
        raise ValueError("Provide exactly one of --samples or --batch")
    if args.value is None and not args.allow_base_critic:
        raise ValueError("--value is required unless --allow-base-critic is set")
    if args.samples_per_rank != 2:
        raise ValueError("This qualification requires exactly 2 trajectories per rank")
    return args


def optimizer_config(lr: float) -> OptimizerConfig:
    return OptimizerConfig(
        lr=lr,
        weight_decay=0.0,
        gradient_clipping=1.0,
        lr_scheduler_type="constant",
        warmup_steps=0,
    )


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a real finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return value


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_existing_files(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): file_digest(path) for path in paths if path.is_file()}


def _row_from_sample(sample: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    missing = REQUIRED_SAMPLE_KEYS - sample.keys()
    if missing:
        raise ValueError(f"sample {sample.get('request_id')} missing keys: {missing}")
    prompt = [int(x) for x in sample["input_tokens"]]
    output = [int(x) for x in sample["output_tokens"]]
    if not output:
        raise ValueError(f"sample {sample.get('request_id')} has no output_tokens")
    if len(sample["behavior_logprobs"]) != len(output):
        raise ValueError(
            f"sample {sample.get('request_id')} behavior_logprobs length "
            f"{len(sample['behavior_logprobs'])} != output length {len(output)}"
        )
    if len(sample["behavior_versions"]) != len(output):
        raise ValueError(
            f"sample {sample.get('request_id')} behavior_versions length "
            f"{len(sample['behavior_versions'])} != output length {len(output)}"
        )
    full = prompt + output
    original_len = len(full)
    if max_tokens > 0 and len(full) > max_tokens:
        raise ValueError(
            f"sample {sample.get('request_id')} length {len(full)} exceeds cap "
            f"{max_tokens}; loader should select fitting rows instead of truncating"
        )
    prompt_len = len(prompt)
    output_len = len(output)
    behavior = list(sample.get("behavior_logprobs", []))[-output_len:]
    versions = [int(x) for x in sample["behavior_versions"]]
    loss_mask = [0.0] * prompt_len + [1.0] * output_len
    logprobs = [0.0] * prompt_len + [
        _finite_float(x, "behavior_logprobs") for x in behavior
    ]
    behavior_versions = [-1] * prompt_len + versions
    return {
        "input_ids": torch.tensor([full], dtype=torch.long),
        "attention_mask": torch.ones(1, len(full), dtype=torch.long),
        "loss_mask": torch.tensor([loss_mask], dtype=torch.float32),
        "logprobs": torch.tensor([logprobs], dtype=torch.float32),
        "versions": torch.tensor([behavior_versions], dtype=torch.long),
        "behavior_versions": torch.tensor([behavior_versions], dtype=torch.long),
        "rewards": torch.tensor([_finite_float(sample.get("reward"), "reward")]),
        "terminated": torch.tensor(
            [bool(sample.get("stop_reason") == "stop" and not sample["truncated"])]
        ),
        "truncated": torch.tensor([bool(sample.get("truncated", False))]),
        "_identity": {
            "request_id": sample.get("request_id"),
            "source_id": sample.get("source_id"),
            "benchmark": sample.get("benchmark"),
            "task_id": sample.get("task_id"),
            "sample_idx": sample.get("sample_idx"),
            "answer": sample.get("answer"),
            "parse_status": sample.get("parse_status"),
            "stop_reason": sample.get("stop_reason"),
            "truncated": sample.get("truncated"),
            "original_len": original_len,
            "used_len": len(full),
            "prompt_len": prompt_len,
            "output_len": output_len,
            "raw_token_fixture": True,
        },
    }


def _tensor_row(row: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    converted = {}
    for key, value in row.items():
        if key.startswith("_"):
            converted[key] = value
        elif isinstance(value, torch.Tensor):
            converted[key] = value.detach().cpu()
        else:
            converted[key] = torch.tensor(value)
    if max_tokens > 0 and converted["input_ids"].shape[-1] > max_tokens:
        raise ValueError("serialized row exceeds --truncate-tokens cap")
    return converted


def expanded_sample_paths(args: argparse.Namespace) -> list[Path]:
    if args.samples is None:
        return [args.batch]
    sample_paths = []
    for path in args.samples:
        sample_paths.extend(
            sorted(path.glob("train-*.jsonl")) if path.is_dir() else [path]
        )
    return sorted(sample_paths)


def load_all_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if args.samples is not None:
        for path in expanded_sample_paths(args):
            with path.open() as stream:
                for line in stream:
                    if line.strip():
                        sample = json.loads(line)
                        total_tokens = len(sample["input_tokens"]) + len(
                            sample["output_tokens"]
                        )
                        if (
                            args.truncate_tokens > 0
                            and total_tokens > args.truncate_tokens
                        ):
                            continue
                        rows.append(_row_from_sample(sample, args.truncate_tokens))
    else:
        payload = torch.load(args.batch, map_location="cpu")
        if isinstance(payload, dict):
            payload = payload.get("rows", payload.get("samples"))
        if not isinstance(payload, list):
            raise ValueError("--batch must contain a list or a dict with rows/samples")
        for row in payload:
            try:
                rows.append(_tensor_row(row, args.truncate_tokens))
            except ValueError:
                if args.truncate_tokens <= 0:
                    raise
    return rows


def load_rank_rows(
    args: argparse.Namespace, rank: int, world: int
) -> list[dict[str, Any]]:
    rows = load_all_rows(args)
    required = world * args.samples_per_rank
    if len(rows) < required:
        raise ValueError(f"fixture has {len(rows)} rows, need {required}")
    start = rank * args.samples_per_rank
    return rows[start : start + args.samples_per_rank]


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for row in rows:
        clean.append(
            {key: value for key, value in row.items() if not key.startswith("_")}
        )
    return clean


def apply_termination_contract(
    rows: list[dict[str, Any]], algorithm: str
) -> list[dict[str, Any]]:
    adjusted = copy.deepcopy(rows)
    for row in adjusted:
        identity = row.get("_identity")
        if not isinstance(identity, dict):
            continue
        if algorithm == "sao" and identity.get("stop_reason") == "length":
            row["terminated"] = torch.tensor([True])
            row["truncated"] = torch.tensor([False])
            identity["termination_transform"] = (
                "sao_finite_budget_length_stop_treated_terminal"
            )
        else:
            identity["termination_transform"] = "input_contract_preserved"
    return adjusted


def move_rows(rows: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    moved = []
    for row in rows:
        moved.append(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in row.items()
            }
        )
    return moved


def jsonable_stats(stats: Any) -> Any:
    if isinstance(stats, dict):
        return {key: jsonable_stats(value) for key, value in stats.items()}
    if isinstance(stats, list):
        return [jsonable_stats(value) for value in stats]
    if isinstance(stats, torch.Tensor):
        return stats.detach().cpu().tolist()
    if isinstance(stats, int | float | str | bool) or stats is None:
        return stats
    return repr(stats)


def sample_local_shards(
    engine: Any, elements_per_param: int
) -> dict[str, dict[str, Any]]:
    sampled = {}
    with torch.no_grad():
        for name, param in engine.model.named_parameters():
            local = param.detach()
            if hasattr(local, "to_local"):
                local = local.to_local()
            flat = local.reshape(-1)
            take = min(elements_per_param, flat.numel())
            if take == 0:
                continue
            indices = torch.linspace(
                0, flat.numel() - 1, steps=take, device=flat.device
            ).long()
            cpu = flat.index_select(0, indices).float().cpu().contiguous()
            sampled[name] = {
                "shape": list(local.shape),
                "dtype": str(local.dtype),
                "numel": int(local.numel()),
                "sampled": int(take),
                "sample_indices": indices.cpu().tolist(),
                "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                "sum": float(cpu.sum().item()),
                "abs_sum": float(cpu.abs().sum().item()),
            }
    return sampled


def count_changed(before: dict[str, Any], after: dict[str, Any]) -> int:
    changed = 0
    for name, old in before.items():
        new = after.get(name)
        if new is not None and old["sha256"] != new["sha256"]:
            changed += 1
    return changed


def snapshots_equal(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return before == after


def optimizer_steps_since_init(engine: Any) -> int:
    steps = []
    for state in engine.optimizer.state.values():
        step = state.get("step") if isinstance(state, dict) else None
        if isinstance(step, torch.Tensor):
            steps.append(int(step.detach().cpu().max().item()))
        elif isinstance(step, int | float):
            steps.append(int(step))
    return max(steps) if steps else 0


def sample_optimizer_state(engine: Any, elements_per_tensor: int) -> dict[str, Any]:
    sampled = {
        "param_group_lrs": [group["lr"] for group in engine.optimizer.param_groups],
        "optimizer_steps_since_init": optimizer_steps_since_init(engine),
        "state": {},
    }
    for param_idx, state in enumerate(engine.optimizer.state.values()):
        state_sample = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                local = value.detach()
                if hasattr(local, "to_local"):
                    local = local.to_local()
                flat = local.reshape(-1)
                take = min(elements_per_tensor, flat.numel())
                if take == 0:
                    state_sample[key] = {"shape": list(value.shape), "sampled": 0}
                    continue
                indices = torch.linspace(
                    0, flat.numel() - 1, steps=take, device=flat.device
                ).long()
                cpu = flat.index_select(0, indices).float().cpu().contiguous()
                state_sample[key] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sample_indices": indices.cpu().tolist(),
                    "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                    "sum": float(cpu.sum().item()),
                }
            elif isinstance(value, int | float | str | bool) or value is None:
                state_sample[key] = value
            else:
                state_sample[key] = repr(value)
        sampled["state"][str(param_idx)] = state_sample
    return sampled


def state_snapshot(engine: Any, elements_per_tensor: int) -> dict[str, Any]:
    return {
        "params": sample_local_shards(engine, elements_per_tensor),
        "optimizer": sample_optimizer_state(engine, elements_per_tensor),
        "scheduler": jsonable_stats(engine.lr_scheduler.state_dict()),
        "engine_optimizer_steps": engine.optimizer_steps_since_init,
    }


def perturb_recoverable_state(engine: Any) -> None:
    with torch.no_grad():
        for param in engine.model.parameters():
            if param.numel() > 0 and param.is_floating_point():
                param.add_(0.125)
                break
        for state in engine.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    flat = value.reshape(-1)
                    if value.is_floating_point():
                        flat[0].add_(0.25)
                    else:
                        flat[0].add_(1)
                    break
            else:
                continue
            break
    engine.lr_scheduler.step()
    engine.optimizer_steps_since_init += 7


def recovery_roundtrip(
    actor: Any,
    critic: Any,
    recovery_path: Path,
    elements_per_tensor: int,
) -> dict[str, Any]:
    actor_path = recovery_path / "actor"
    critic_path = recovery_path / "critic"
    before = {
        "actor": state_snapshot(actor, elements_per_tensor),
        "critic": state_snapshot(critic, elements_per_tensor),
    }
    actor.save(SaveLoadMeta(str(actor_path), "dcp", True))
    critic.save(SaveLoadMeta(str(critic_path), "dcp", True))
    perturb_recoverable_state(actor)
    perturb_recoverable_state(critic)
    perturbed = {
        "actor": state_snapshot(actor, elements_per_tensor),
        "critic": state_snapshot(critic, elements_per_tensor),
    }
    if snapshots_equal(before["actor"], perturbed["actor"]):
        raise RuntimeError("actor negative recovery check did not perturb state")
    if snapshots_equal(before["critic"], perturbed["critic"]):
        raise RuntimeError("critic negative recovery check did not perturb state")
    if before["actor"]["optimizer"] == perturbed["actor"]["optimizer"]:
        raise RuntimeError("actor optimizer negative recovery check did not change")
    if before["critic"]["optimizer"] == perturbed["critic"]["optimizer"]:
        raise RuntimeError("critic optimizer negative recovery check did not change")
    actor.load(SaveLoadMeta(str(actor_path), "dcp", True))
    critic.load(SaveLoadMeta(str(critic_path), "dcp", True))
    restored = {
        "actor": state_snapshot(actor, elements_per_tensor),
        "critic": state_snapshot(critic, elements_per_tensor),
    }
    if not snapshots_equal(before["actor"], restored["actor"]):
        raise RuntimeError("actor recovery snapshot mismatch after load")
    if not snapshots_equal(before["critic"], restored["critic"]):
        raise RuntimeError("critic recovery snapshot mismatch after load")
    return {
        "checked": True,
        "format": "dcp",
        "path": str(recovery_path),
        "negative_check": "perturbed sampled params optimizer state and scheduler",
        "restored": True,
        "actor_optimizer_steps_since_init": optimizer_steps_since_init(actor),
        "critic_optimizer_steps_since_init": optimizer_steps_since_init(critic),
    }


def prepared_microbatch_count(
    engine: Any, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    batch = engine._normalize_batch_input(copy.deepcopy(rows))[0]
    mb_list = engine._prepare_mb_list(batch)
    return {
        "count": len(mb_list.mbs),
        "group_lens": [int(x) for x in getattr(mb_list, "group_lens", [])],
        "padded_to_lengths": [
            int(x) if x is not None else None
            for x in getattr(mb_list, "padded_to_lengths", [])
        ],
    }


def observe_losses(engine: Any, role: str, losses: list[dict[str, Any]]) -> None:
    compute = engine._compute_logprobs_and_loss

    def record(*args: Any, **kwargs: Any) -> torch.Tensor:
        loss = compute(*args, **kwargs)
        losses.append({"role": role, "loss": float(loss.detach().float().cpu().item())})
        return loss

    engine._compute_logprobs_and_loss = record


def wrap_train_batch(engine: Any, role: str, calls: list[dict[str, Any]]) -> None:
    train_batch = engine.train_batch

    def record(*args: Any, **kwargs: Any) -> dict[str, float]:
        stats = train_batch(*args, **kwargs)
        calls.append({"role": role, "stats": jsonable_stats(stats)})
        return stats

    engine.train_batch = record


def assert_train_call_evidence(
    calls: list[dict[str, Any]],
    *,
    expected_microbatches: int,
    actor_lr: float,
    critic_lr: float,
) -> None:
    expected_lr = {"actor": actor_lr, "critic": critic_lr}
    for call in calls:
        role = call["role"]
        stats = call["stats"]
        if stats.get("num_micro_batches") != expected_microbatches:
            raise RuntimeError(f"{role} train call did not use n_mbs=2: {stats}")
        if stats.get("update_successful") != 1.0:
            raise RuntimeError(f"{role} train call was not successful: {stats}")
        grad_norm = stats.get("grad_norm")
        if (
            not isinstance(grad_norm, int | float)
            or not math.isfinite(float(grad_norm))
            or grad_norm <= 0
        ):
            raise RuntimeError(f"{role} grad_norm is not finite positive: {stats}")
        if stats.get("lr") != expected_lr[role]:
            raise RuntimeError(
                f"{role} lr {stats.get('lr')} != expected {expected_lr[role]}"
            )


def check_isolation(actor: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    batched = actor.compute_logp(copy.deepcopy(rows))
    separate = [actor.compute_logp([copy.deepcopy(row)])[0] for row in rows]
    for actual, expected in zip(batched, separate):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    perturbed = copy.deepcopy(rows)
    perturbed[0]["input_ids"].fill_(actor.tokenizer.eos_token_id or 0)
    changed = actor.compute_logp(perturbed)
    for actual, expected in zip(changed[1:], batched[1:]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {
        "batched_equals_separate": True,
        "neighbor_perturbation_preserved_other_rows": True,
    }


def validate_value_contract(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any] | None, str]:
    if args.value is None:
        return args.model, None, "base_critic_unsealed_interim"
    manifest = validate_value_artifact(args.value, require_pretrained=False)
    kind = manifest["qualification"]["kind"]
    if kind == "synthetic" and not args.allow_synthetic_value:
        raise ValueError("Synthetic value artifact requires --allow-synthetic-value")
    contract = {
        "identity": manifest["identity"],
        "protocol": manifest["protocol"],
        "require_pretrained": False,
    }
    return args.value, contract, f"sealed_value_{kind}"


def main() -> None:
    args = parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != args.require_world_size:
        raise ValueError(f"Expected {args.require_world_size} ranks, got {world}")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(17)
    actor = critic = None
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        value_path, value_contract, value_mode = validate_value_contract(args)
        all_rows = load_all_rows(args)
        required_rows = args.samples_per_rank * world
        if len(all_rows) < required_rows:
            raise ValueError(f"fixture has {len(all_rows)} rows, need {required_rows}")
        selected_rows = all_rows[:required_rows]
        start = rank * args.samples_per_rank
        rank_rows = apply_termination_contract(
            selected_rows[start : start + args.samples_per_rank], args.algorithm
        )
        raw = move_rows(public_rows(rank_rows), torch.device("cuda"))
        identities = [row.get("_identity") for row in rank_rows]
        selected_lengths = [int(row["input_ids"].shape[-1]) for row in selected_rows]
        source_hashes = {
            "model": hash_existing_files(
                [
                    args.model / "config.json",
                    args.model / "tokenizer.json",
                    args.model / "tokenizer_config.json",
                ]
            ),
            "fixtures": hash_existing_files(expanded_sample_paths(args)),
        }
        if args.batch is not None:
            source_hashes["batch"] = hash_existing_files([args.batch])
        mb_spec = MicroBatchSpec(
            n_mbs=args.engine_microbatches,
            max_tokens_per_mb=args.max_tokens_per_mb,
        )
        common = dict(
            experiment_name="sao-qwen35-native-preflight",
            trial_name=f"{args.algorithm}-dp{world}-mb{args.engine_microbatches}",
            backend=f"fsdp:d{world}",
            attn_impl=args.attention,
            dtype=args.dtype,
            optimizer_dtype="float32",
            disable_dropout=True,
            gradient_checkpointing=True,
            fsdp=FSDPEngineConfig(memory_efficient_load=True),
            mb_spec=mb_spec,
            ppo_n_minibatches=1,
        )
        actor = FSDPPPOActor(
            PPOActorConfig(
                **common,
                path=str(args.model),
                optimizer=optimizer_config(args.actor_lr),
                kl_ctl=0.0,
                discount=1.0,
                gae_lambda=(
                    "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"
                    if args.algorithm == "sao"
                    else 1.0
                ),
                gae_lambda_kwargs={"alpha": 1.5} if args.algorithm == "sao" else {},
                critic_gae_lambda=1.0,
                reward_norm=None,
                adv_norm=None,
                recompute_logprob=False,
                use_decoupled_loss=False,
                use_direct_dis_loss=args.algorithm == "sao",
            )
        )
        critic = FSDPPPOCritic(
            PPOCriticConfig(
                **common,
                path=str(value_path),
                optimizer=optimizer_config(args.critic_lr),
                is_critic=True,
                eps_clip=None if args.algorithm == "sao" else 0.5,
                value_contract=value_contract,
            )
        )
        if actor.model_config.model_type != "qwen3_5":
            raise RuntimeError(
                f"Expected Qwen3.5 model_type qwen3_5, got {actor.model_config.model_type}"
            )
        ft = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=args.samples_per_rank * world,
            train_batch_size=args.samples_per_rank * world,
        )
        for engine in (actor, critic):
            engine.create_process_group()
            engine.initialize(None, ft)
        events = []
        actor.optimizer.register_step_post_hook(lambda *unused: events.append("actor"))
        critic.optimizer.register_step_post_hook(
            lambda *unused: events.append("critic")
        )
        train_calls: list[dict[str, Any]] = []
        loss_events: list[dict[str, Any]] = []
        observe_losses(actor, "actor", loss_events)
        observe_losses(critic, "critic", loss_events)
        wrap_train_batch(actor, "actor", train_calls)
        wrap_train_batch(critic, "critic", train_calls)
        actor_mb = prepared_microbatch_count(actor, raw)
        critic_mb = prepared_microbatch_count(critic, raw)
        if args.max_tokens_per_mb is None:
            if actor_mb["count"] != args.engine_microbatches:
                raise RuntimeError(f"actor actual microbatches: {actor_mb}")
            if critic_mb["count"] != args.engine_microbatches:
                raise RuntimeError(f"critic actual microbatches: {critic_mb}")
        values = critic.compute_values(copy.deepcopy(raw))
        for row, value in zip(raw, values):
            row["values"] = value
        fixed = actor.compute_advantages(copy.deepcopy(raw))
        before = {
            "actor": sample_local_shards(actor, args.sample_param_elements),
            "critic": sample_local_shards(critic, args.sample_param_elements),
        }
        targets = [row["returns"].detach().clone() for row in fixed]
        if args.algorithm == "sao":
            report = update_critic_before_actor(actor, critic, raw, fixed, 2)
            expected_events = ["critic", "critic", "actor"]
        else:
            actor_report = actor.ppo_update(copy.deepcopy(fixed))
            critic_report = critic.ppo_update(copy.deepcopy(fixed))
            actor.step_lr_scheduler()
            critic.step_lr_scheduler()
            report = {"critic": [critic_report], "actor": actor_report}
            expected_events = ["actor", "critic"]
        after = {
            "actor": sample_local_shards(actor, args.sample_param_elements),
            "critic": sample_local_shards(critic, args.sample_param_elements),
        }
        assert_train_call_evidence(
            train_calls,
            expected_microbatches=2,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
        changed = {
            "actor": count_changed(before["actor"], after["actor"]),
            "critic": count_changed(before["critic"], after["critic"]),
        }
        if changed["actor"] <= 0:
            raise RuntimeError("actor sampled local shards did not change")
        if changed["critic"] <= 0:
            raise RuntimeError("critic sampled local shards did not change")
        for row, target in zip(fixed, targets):
            torch.testing.assert_close(row["returns"], target, rtol=0, atol=0)
        if events != expected_events:
            raise RuntimeError(f"optimizer order {events} != {expected_events}")
        isolation = check_isolation(actor, raw)
        recovery = {"checked": False}
        if args.check_recovery:
            recovery = recovery_roundtrip(
                actor, critic, args.output / "recovery-dcp", args.sample_param_elements
            )
        result = {
            "rank": rank,
            "world_size": world,
            "algorithm": args.algorithm,
            "value_mode": value_mode,
            "value_contract_require_pretrained": False if value_contract else None,
            "allow_base_critic": args.allow_base_critic,
            "allow_synthetic_value": args.allow_synthetic_value,
            "sample_identities": identities,
            "sequence_lengths": [int(row["input_ids"].shape[-1]) for row in raw],
            "selected_first8_sequence_lengths": selected_lengths[:8],
            "selected_sequence_min": min(selected_lengths),
            "selected_sequence_max": max(selected_lengths),
            "source_hashes": source_hashes,
            "truncation": {
                "max_tokens": args.truncate_tokens,
                "selection_filter_only": args.truncate_tokens > 0,
                "mutates_token_prefix": False,
                "disclosed": any(
                    item is not None and item["original_len"] != item["used_len"]
                    for item in identities
                ),
            },
            "requested_engine_microbatches": args.engine_microbatches,
            "actor_prepared_microbatches": actor_mb,
            "critic_prepared_microbatches": critic_mb,
            "train_batch_calls": train_calls,
            "loss_events": loss_events,
            "optimizer_events": events,
            "expected_optimizer_events": expected_events,
            "actor_lr": actor.optimizer.param_groups[0]["lr"],
            "critic_lr": critic.optimizer.param_groups[0]["lr"],
            "report": jsonable_stats(report),
            "sampled_shard_changes": changed,
            "sampled_shards_before": before,
            "sampled_shards_after": after,
            "isolation": isolation,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "recovery": recovery,
            "scope": (
                "native Qwen3.5-4B FSDP integration evidence only; "
                "not pretrained critic quality qualification"
            ),
            "config_readback": {
                "discount": actor.config.discount,
                "gae_lambda": actor.config.gae_lambda,
                "gae_lambda_kwargs": actor.config.gae_lambda_kwargs,
                "critic_gae_lambda": actor.config.critic_gae_lambda,
                "reward_norm": actor.config.reward_norm,
                "adv_norm": actor.config.adv_norm,
                "use_direct_dis_loss": actor.config.use_direct_dis_loss,
                "gradient_checkpointing": actor.config.gradient_checkpointing,
                "critic_eps_clip": critic.config.eps_clip,
                "actor_ppo_n_minibatches": actor.config.ppo_n_minibatches,
                "critic_ppo_n_minibatches": critic.config.ppo_n_minibatches,
                "mb_spec": {
                    "n_mbs": actor.config.mb_spec.n_mbs,
                    "max_tokens_per_mb": actor.config.mb_spec.max_tokens_per_mb,
                },
            },
        }
        out = args.output / f"rank{rank}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, result)
        if rank == 0:
            (args.output / "args.json").write_text(
                json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n"
            )
            (args.output / "summary.json").write_text(
                json.dumps(gathered, indent=2, sort_keys=True) + "\n"
            )
    finally:
        for engine in (critic, actor):
            if engine is not None:
                engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
