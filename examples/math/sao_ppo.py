# SPDX-License-Identifier: Apache-2.0
"""Run the pinned async PPO contract through AReaL's native trainer."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import requests
import torch
from datasets import load_from_disk
from omegaconf import OmegaConf

from scripts.sao.async_eval import AsyncEvalPPOTrainer, SaoPPOConfig

from areal.api.cli_args import (
    PPOConfig,
    load_expr_config,
    parse_cli_args,
    to_structured_cfg,
)
from areal.infra.rpc.rtensor import RTensor
from areal.trainer.ppo.validation import verify_gamma_one_episodic_returns
from areal.utils import logging
from areal.utils.network import format_hostport

logger = logging.getLogger("SaoPpo")


def _safetensors_weight_keys(path: Path, role: str) -> set[str]:
    from safetensors import safe_open

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"{role} safetensors index has no weight_map: {index_path}"
            )
        missing = sorted(
            name
            for name in set(weight_map.values())
            if not (path / str(name)).is_file()
        )
        if missing:
            raise ValueError(
                f"{role} safetensors index references missing shards: {missing}"
            )
        keys = set()
        for shard_name in sorted(set(weight_map.values())):
            with safe_open(
                path / str(shard_name), framework="pt", device="cpu"
            ) as handle:
                keys.update(handle.keys())
        return keys

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        return set()

    keys = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def validate_hf_checkpoint_paths(config: PPOConfig) -> None:
    """Lightweight CPU-only validation for the selected actor and critic HF dirs."""
    if config.critic is None:
        raise ValueError("A critic config is required before checkpoint validation")

    actor_path = Path(config.actor.path).expanduser()
    critic_path = Path(config.critic.path).expanduser()
    for role, path in (("actor", actor_path), ("critic", critic_path)):
        if not path.is_dir():
            raise ValueError(
                f"{role} checkpoint path must be a local directory: {path}"
            )
        if not (path / "config.json").is_file():
            raise ValueError(f"{role} checkpoint is missing config.json: {path}")
        has_model_file = any(
            (path / name).is_file()
            for name in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        ) or any(path.glob("*.safetensors"))
        if not has_model_file:
            raise ValueError(f"{role} checkpoint has no HF model shard: {path}")

    if actor_path.resolve() == critic_path.resolve():
        raise ValueError("Actor Base and pretrained critic checkpoints must differ")

    critic_keys = _safetensors_weight_keys(critic_path, "critic")
    if "score.weight" not in critic_keys:
        raise ValueError(
            "Pretrained critic checkpoint must expose score.weight in safetensors headers"
        )


def validate_contract(config: SaoPPOConfig, *, preflight: bool = False) -> None:
    AsyncEvalPPOTrainer._validate_save_eval_sync(config)
    expected = {
        "actor.backend": (config.actor.backend, "fsdp:d4p1t1"),
        "critic.backend": (
            config.critic.backend if config.critic else None,
            "fsdp:d4p1t1",
        ),
        "rollout.backend": (config.rollout.backend, "sglang:d3p1t1"),
        "evaluation_rollout.backend": (
            config.evaluation_rollout.backend,
            "sglang:d1p1t1",
        ),
        "scheduler.type": (config.scheduler.type, "local"),
        "cluster.n_nodes": (config.cluster.n_nodes, 1),
        "cluster.n_gpus_per_node": (config.cluster.n_gpus_per_node, 8),
        "actor.discount": (config.actor.discount, 1.0),
        "actor.loss_reduction": (config.actor.loss_reduction, "token_mean"),
        "actor.gae_lambda": (config.actor.gae_lambda, 0.95),
        "actor.critic_gae_lambda": (config.actor.critic_gae_lambda, 1.0),
        "actor.gae_timestep_unit": (config.actor.gae_timestep_unit, "token"),
        "actor.reward_scaling": (config.actor.reward_scaling, 1.0),
        "actor.reward_bias": (config.actor.reward_bias, 0.0),
        "actor.use_decoupled_loss": (config.actor.use_decoupled_loss, True),
        "actor.recompute_logprob": (config.actor.recompute_logprob, True),
        "actor.prox_logp_method": (config.actor.prox_logp_method, "recompute"),
        "actor.ppo_n_minibatches": (config.actor.ppo_n_minibatches, 1),
        "actor.reward_norm": (config.actor.reward_norm, None),
        "actor.adv_norm": (config.actor.adv_norm, None),
        "actor.eps_clip": (config.actor.eps_clip, 0.2),
        "actor.eps_clip_higher": (config.actor.eps_clip_higher, 0.28),
        "actor.kl_ctl": (config.actor.kl_ctl, 0.0),
        "actor.use_sapo_loss": (config.actor.use_sapo_loss, False),
        "actor.use_cispo_loss": (config.actor.use_cispo_loss, False),
        "actor.overlong_reward_penalty": (config.actor.overlong_reward_penalty, False),
        "actor.importance_sampling_level": (
            config.actor.importance_sampling_level,
            "token",
        ),
        "dynamic_bs": (config.dynamic_bs, False),
        "num_critic_only_steps": (config.num_critic_only_steps, 0),
        "gconfig.reward_normalization": (config.gconfig.reward_normalization, False),
        "train_dataset.drop_last": (config.train_dataset.drop_last, False),
        "actor.init_from_scratch": (config.actor.init_from_scratch, False),
    }
    if not preflight:
        expected.update(
            {
                "total_train_epochs": (config.total_train_epochs, 1),
                "total_train_steps": (config.total_train_steps, None),
                "gconfig.max_new_tokens": (config.gconfig.max_new_tokens, 8192),
                "gconfig.n_samples": (config.gconfig.n_samples, 8),
                "eval_gconfig.n_samples": (config.eval_gconfig.n_samples, 2),
                "train_dataset.batch_size": (config.train_dataset.batch_size, 16),
                "saver.freq_steps": (config.saver.freq_steps, 20),
                "recover.freq_steps": (config.recover.freq_steps, 20),
                "evaluator.eval_before_train": (
                    config.evaluator.eval_before_train,
                    True,
                ),
                "evaluator.freq_steps": (config.evaluator.freq_steps, 20),
            }
        )
    mismatches = {
        key: values for key, values in expected.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"PPO contract mismatch (observed, expected): {mismatches}")
    for role in ("actor", "rollout", "evaluation_rollout"):
        role_config = getattr(config, role)
        if role_config.scheduling_strategy.type != "separation":
            raise ValueError(f"{role} must use separate GPU workers")
        if not role_config.scheduling_spec or any(
            spec.gpu != 1 for spec in role_config.scheduling_spec
        ):
            raise ValueError(f"{role} workers must reserve one GPU each")
    if config.critic is None or not config.critic.is_critic:
        raise ValueError("A trainable scalar critic is required")
    if (
        config.critic.scheduling_strategy.type != "colocation"
        or config.critic.scheduling_strategy.target != "actor"
    ):
        raise ValueError("critic must colocate with actor on the four training GPUs")
    if config.critic.loss_reduction != "token_mean":
        raise ValueError("critic.loss_reduction must be token_mean")
    if config.critic.path == config.actor.path:
        raise ValueError("Actor Base and pretrained critic checkpoints must differ")
    if config.critic.init_from_scratch:
        raise ValueError("Pretrained critic must load from SAO_CRITIC_PATH")
    if config.critic.eps_clip != 0.2:
        raise ValueError("critic.eps_clip must be 0.2")
    if config.critic.ppo_n_minibatches != 1:
        raise ValueError("critic.ppo_n_minibatches must be 1")
    rejection = config.actor.rejection_sampling
    if rejection is None or (
        rejection.level,
        rejection.action,
        rejection.metric,
        rejection.upper,
        rejection.lower,
    ) != ("token", "mask", "ratio", 5.0, None):
        raise ValueError("Expected official GSM8K token ratio rejection above 5")
    optimizers = {
        "actor": (config.actor.optimizer, 1.0e-6),
        "critic": (config.critic.optimizer, 5.0e-6),
    }
    for role, (optimizer, lr) in optimizers.items():
        if optimizer is None or (
            optimizer.type,
            optimizer.lr,
            optimizer.weight_decay,
            optimizer.beta1,
            optimizer.beta2,
            optimizer.eps,
            optimizer.warmup_steps_proportion,
            optimizer.lr_scheduler_type,
            optimizer.warmup_steps,
            optimizer.gradient_clipping,
        ) != ("adam", lr, 0.01, 0.9, 0.98, 1.0e-8, 0.0, "constant", 0, 1.0):
            raise ValueError(f"{role} optimizer must match the SAO PPO recipe")
    if config.ref is not None or config.teacher is not None:
        raise ValueError("No reference/privileged teacher belongs in this PPO baseline")
    if config.sglang.attention_backend == "fa3":
        raise ValueError("FA3 is not the A100 backend")


def check_step_metrics(
    data: dict, completed_step: int, updates_since_init: int | None = None
) -> dict:
    """Reject skipped or non-finite updates using native engine metrics."""
    roles = {}
    for role in ("actor", "critic"):
        metrics = {
            key: value
            for key, value in data.items()
            if ("critic/" in key if role == "critic" else "critic/" not in key)
        }
        grads = {k: v for k, v in metrics.items() if k.endswith("grad_norm")}
        successes = {
            k: v for k, v in metrics.items() if k.endswith("update_successful")
        }
        counts = {
            k: v for k, v in metrics.items() if k.endswith("optimizer_steps_since_init")
        }
        if not grads or not successes or not counts:
            raise RuntimeError(
                f"Missing {role} optimizer evidence at step {completed_step}"
            )
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in grads.values()):
            raise RuntimeError(
                f"{role} has a non-finite/zero gradient at step {completed_step}"
            )
        if any(float(v) != 1 for v in successes.values()):
            raise RuntimeError(
                f"{role} skipped an optimizer update at step {completed_step}"
            )
        expected_count = (
            completed_step if updates_since_init is None else updates_since_init
        )
        if any(float(v) != expected_count for v in counts.values()):
            raise RuntimeError(
                f"{role} optimizer count mismatch: {counts}, expected {expected_count}"
            )
        roles[role] = {
            "grad_norm": grads,
            "update_successful": successes,
            "optimizer_steps": counts,
        }
    for key, value in data.items():
        if (
            "loss" in key
            and isinstance(value, (float, int))
            and not math.isfinite(value)
        ):
            raise RuntimeError(f"Non-finite metric {key} at step {completed_step}")
    return roles


def source_id_digest(source_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(source_ids) + "\n").encode()).hexdigest()


def source_key(source_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(str(source_id).encode()).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def source_key_digest(source_ids: list[str]) -> str:
    return hashlib.sha256(
        (
            "\n".join(str(source_key(source_id)) for source_id in source_ids) + "\n"
        ).encode()
    ).hexdigest()


def weight_only_start_step_from_env() -> int:
    raw = os.environ.get("SAO_WEIGHT_ONLY_START_STEP")
    if raw in (None, ""):
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("SAO_WEIGHT_ONLY_START_STEP must be an integer") from exc
    if value < 0:
        raise ValueError("SAO_WEIGHT_ONLY_START_STEP must be non-negative")
    return value


def _global_epoch_source_ids(
    trainer,
    train_dataset,
    *,
    config: PPOConfig,
    count: int,
) -> list[str]:
    sampler = trainer.train_dataloader.sampler
    dataset_len = len(train_dataset)
    generator = torch.Generator()
    generator.manual_seed(int(getattr(sampler, "seed", config.seed)))
    if getattr(sampler, "shuffle", False):
        indices = torch.randperm(dataset_len, generator=generator).tolist()
    else:
        indices = list(range(dataset_len))
    total_size = int(getattr(sampler, "total_size", len(indices)))
    if not getattr(sampler, "drop_last", False) and total_size > len(indices):
        padding_size = total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
    else:
        indices = indices[:total_size]
    if count > len(indices):
        raise ValueError(
            f"Weight-only start needs {count} skipped prompts, but the seeded epoch "
            f"contains only {len(indices)} prompts"
        )
    source_ids = list(train_dataset["source_id"])
    return [str(source_ids[index]) for index in indices[:count]]


def _load_expected_weight_only_digest(
    source: Path,
    *,
    start_step: int,
    batch_size: int,
) -> dict[str, str]:
    if source.is_dir():
        continuation = source / "weight-only-continuation.json"
        if continuation.is_file():
            return _load_expected_weight_only_digest(
                continuation, start_step=start_step, batch_size=batch_size
            )
        consumed = None
        if (source / "consumed").is_dir():
            consumed = source / "consumed"
        elif (source / "1.json").is_file():
            consumed = source
        if consumed is not None:
            keys = []
            seen = set()
            for step in range(1, start_step + 1):
                path = consumed / f"{step}.json"
                if not path.is_file():
                    raise ValueError(f"Missing consumed evidence file: {path}")
                for row in json.loads(path.read_text(encoding="utf-8")):
                    key = int(row["audit_source_key"])
                    if key not in seen:
                        seen.add(key)
                        keys.append(str(key))
            expected = start_step * batch_size
            if len(keys) != expected:
                raise ValueError(
                    f"Consumed evidence has {len(keys)} unique prompts, expected {expected}"
                )
            return {
                "source_key_sha256": hashlib.sha256(
                    ("\n".join(keys) + "\n").encode()
                ).hexdigest()
            }
        epoch_order = source / "epoch-order.json"
        if epoch_order.is_file():
            return _load_expected_weight_only_digest(
                epoch_order, start_step=start_step, batch_size=batch_size
            )
        raise ValueError(f"No supported source evidence found in {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    if "skipped_source_id_sha256" in payload or "skipped_source_key_sha256" in payload:
        return {
            "source_id_sha256": payload.get("skipped_source_id_sha256"),
            "source_key_sha256": payload.get("skipped_source_key_sha256"),
        }
    if "source_id_order_sha256" in payload and "source_ids" in payload:
        ids = [str(item) for item in payload["source_ids"][: start_step * batch_size]]
        return {
            "source_id_sha256": source_id_digest(ids),
            "source_key_sha256": source_key_digest(ids),
        }
    for key in (
        "source_id_sha256",
        "consumed_source_id_sha256",
        "source_key_sha256",
        "consumed_source_key_sha256",
    ):
        if key in payload:
            return {key.replace("consumed_", ""): payload[key]}
    raise ValueError(f"Unsupported source evidence payload: {source}")


def prepare_weight_only_continuation(
    trainer,
    train_dataset,
    evidence: Path,
    *,
    config: PPOConfig,
    initial_step: int,
) -> None:
    if initial_step == 0:
        return
    if trainer.recover_info is not None:
        raise ValueError(
            "SAO_WEIGHT_ONLY_START_STEP is only valid when native recovery did not load"
        )

    skipped_prompt_count = initial_step * config.train_dataset.batch_size
    skipped_source_ids = _global_epoch_source_ids(
        trainer,
        train_dataset,
        config=config,
        count=skipped_prompt_count,
    )
    payload = {
        "mode": "weight_only",
        "initial_step": initial_step,
        "next_cumulative_step": initial_step + 1,
        "skipped_prompt_count": skipped_prompt_count,
        "train_batch_size": config.train_dataset.batch_size,
        "seed": config.seed,
        "skipped_source_id_sha256": source_id_digest(skipped_source_ids),
        "skipped_source_key_sha256": source_key_digest(skipped_source_ids),
        "source_evidence": os.environ.get("SAO_WEIGHT_ONLY_SOURCE_EVIDENCE"),
        "optimizer_state": "reset",
        "async_rollout_state": "reset",
    }

    expected_id_digest = os.environ.get("SAO_WEIGHT_ONLY_SOURCE_IDS_SHA256")
    expected_key_digest = os.environ.get("SAO_WEIGHT_ONLY_SOURCE_KEYS_SHA256")
    source_evidence = os.environ.get("SAO_WEIGHT_ONLY_SOURCE_EVIDENCE")
    if not source_evidence and not expected_id_digest and not expected_key_digest:
        raise ValueError(
            "Weight-only continuation requires prior-source evidence. Set "
            "SAO_WEIGHT_ONLY_SOURCE_EVIDENCE, SAO_WEIGHT_ONLY_SOURCE_IDS_SHA256, "
            "or SAO_WEIGHT_ONLY_SOURCE_KEYS_SHA256."
        )
    if source_evidence:
        expected = _load_expected_weight_only_digest(
            Path(source_evidence).expanduser(),
            start_step=initial_step,
            batch_size=config.train_dataset.batch_size,
        )
        expected_id_digest = expected.get("source_id_sha256") or expected_id_digest
        expected_key_digest = expected.get("source_key_sha256") or expected_key_digest
    if expected_id_digest and expected_id_digest != payload["skipped_source_id_sha256"]:
        raise ValueError(
            "Skipped source_id digest does not match prior evidence: "
            f"observed={payload['skipped_source_id_sha256']} expected={expected_id_digest}"
        )
    if (
        expected_key_digest
        and expected_key_digest != payload["skipped_source_key_sha256"]
    ):
        raise ValueError(
            "Skipped source_key digest does not match prior evidence: "
            f"observed={payload['skipped_source_key_sha256']} expected={expected_key_digest}"
        )

    iterator = iter(trainer.train_dataloader)
    for _ in range(initial_step):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError(
                f"Cannot skip {initial_step} dataloader batches for weight-only start"
            ) from exc
    state = trainer.train_dataloader.state_dict()
    trainer.train_dataloader.load_state_dict(state)
    (evidence / "weight-only-continuation.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def verify_published_policy(trainer, evidence: Path, step: int) -> None:
    """Read the updated actor and every rollout replica on the same fixed tokens."""
    ids = trainer.tokenizer.encode(
        "Question: What is 2+3? Answer: 5", add_special_tokens=False
    )
    target_len = 4
    if len(ids) <= target_len + 1:
        raise ValueError("Publication probe needs a nonempty scoring prefix")
    mask = torch.zeros((1, len(ids)), dtype=torch.bool)
    mask[:, -target_len:] = True
    probes = [
        {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.bool),
            "loss_mask": mask.clone(),
        }
        for _ in range(4)
    ]
    actor = RTensor.localize(trainer.actor.compute_logp(probes))
    rollout = RTensor.localize(trainer.rollout.compute_logp(probes))
    rows = []
    for index, (a, b) in enumerate(zip(actor, rollout, strict=True)):
        # FSDP returns next-token predictions; scoring writes at token positions.
        a = a.reshape(-1).float()[-target_len - 1 : -1].cpu()
        b = b.reshape(-1).float()[-target_len:].cpu()
        diff = (a - b).abs()
        rows.append(
            {
                "replica_index": index,
                "actor": a.tolist(),
                "rollout": b.tolist(),
                "max_abs_error": diff.max().item(),
            }
        )
    passed = all(row["max_abs_error"] <= 0.15 for row in rows)
    (evidence / f"published-policy-{step}.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "atol": 0.15,
                "rtol": 0.0,
                "version": trainer.rollout.get_version(),
                "token_ids": ids,
                "scored_token_ids": ids[-target_len:],
                "replicas": rows,
            },
            indent=2,
        )
        + "\n"
    )
    if not passed:
        raise RuntimeError(
            f"Published policy logprob parity failed; evidence={evidence / f'published-policy-{step}.json'}"
        )


def split_trajectory_groups(batch: list[dict]) -> list[dict]:
    """Dispatch a tail such as five prompts as twenty individual PPO trajectories."""
    individual = []
    for group in batch:
        count = group["input_ids"].shape[0]
        for i in range(count):
            individual.append(
                {
                    key: value[i : i + 1]
                    if torch.is_tensor(value) and value.ndim and value.shape[0] == count
                    else value
                    for key, value in group.items()
                }
            )
    return individual


def install_audit_hooks(
    trainer,
    evidence: Path,
    *,
    preflight: bool = False,
    initial_step: int = 0,
) -> None:
    """Keep native PPO execution; record consumed IDs and completed RPC spans."""
    consumed_dir = evidence / "consumed"
    consumed_dir.mkdir(exist_ok=True)
    start_step = (
        trainer.recover_info.last_step_info.next().global_step
        if trainer.recover_info is not None
        else initial_step
    )
    optimizer_steps_base = (
        trainer.recover_info.trainer_state.get("optimizer_steps_base", 0)
        if trainer.recover_info is not None
        else initial_step
    )
    state = {"step": start_step, "spans": []}
    original_prepare = trainer.actor.prepare_batch

    def prepare(*args, **kwargs):
        batch = original_prepare(*args, **kwargs)
        state["step"] += 1
        state["spans"] = []
        keys = (
            "audit_source_key",
            "audit_task_id",
            "audit_sample_idx",
            "terminated",
            "truncated",
        )
        metadata = RTensor.localize([{k: item[k] for k in keys} for item in batch])
        records = []
        for group in metadata:
            columns = {k: v.reshape(-1).tolist() for k, v in group.items()}
            records.extend(
                dict(zip(keys, row))
                for row in zip(*(columns[k] for k in keys), strict=True)
            )
        if not records:
            raise RuntimeError("Empty consumed PPO batch")
        (consumed_dir / f"{state['step']}.json").write_text(json.dumps(records) + "\n")
        if len(batch) % trainer.actor.parallel_strategy.dp_size:
            local_batch = RTensor.localize(batch)
            individual = split_trajectory_groups(local_batch)
            trainer.actor.clear_batches(batch)
            batch = individual
        return batch

    trainer.actor.prepare_batch = prepare
    original_save_training_state = trainer._save_training_state

    def save_training_state(*, epoch, epoch_step, global_step, force=False):
        force = force or (
            initial_step > 0
            and trainer.recover_info is None
            and global_step == initial_step
        )
        return original_save_training_state(
            epoch=epoch,
            epoch_step=epoch_step,
            global_step=global_step,
            force=force,
        )

    trainer._save_training_state = save_training_state
    original_advantages = trainer.actor.compute_advantages

    def advantages(*args, **kwargs):
        result = original_advantages(*args, **kwargs)
        if preflight or state["step"] <= 5:
            keys = (
                "values",
                "returns",
                "rewards",
                "loss_mask",
                "attention_mask",
                "terminated",
                "truncated",
                "bootstrap_mask",
            )
            local = RTensor.localize(
                [{key: group[key] for key in keys} for group in result]
            )
            report = verify_gamma_one_episodic_returns(
                local,
                reward_scaling=trainer.config.actor.reward_scaling,
                reward_bias=trainer.config.actor.reward_bias,
                reward_clip=trainer.config.actor.reward_clip,
            )
            if report["bootstrapped"] != 0:
                raise RuntimeError(
                    "SAO PPO critic-return audit unexpectedly bootstrapped"
                )
            report["step"] = state["step"]
            report["consumed_sha256"] = hashlib.sha256(
                (consumed_dir / f"{state['step']}.json").read_bytes()
            ).hexdigest()
            path = evidence / "returns"
            path.mkdir(exist_ok=True)
            (path / f"{state['step']}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        return result

    trainer.actor.compute_advantages = advantages
    for role in ("actor", "critic"):
        engine = getattr(trainer, role)
        original_update = engine.ppo_update

        def update(*args, _fn=original_update, _role=role, **kwargs):
            # Profiler RPCs can wait for decode activity; they are diagnostics,
            # not a prerequisite for a valid optimizer update.
            if (
                os.environ.get("SAO_PROFILE_ROLLOUT", "0") == "1"
                and not preflight
                and _role == "actor"
                and state["step"] <= 5
            ):
                for index, server in enumerate(trainer.rollout.server_infos):
                    address = format_hostport(server.host, server.port)
                    response = requests.post(
                        f"http://{address}/start_profile",
                        json={
                            "output_dir": str(
                                evidence
                                / "profiler"
                                / f"rollout-{index}-step-{state['step']}"
                            ),
                            "num_steps": 128,
                            "activities": ["CPU", "GPU"],
                            "profile_prefix": f"ppo-step-{state['step']}",
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
            start_ns = time.time_ns()
            result = _fn(*args, **kwargs)
            state["spans"].append(
                {"role": _role, "started_ns": start_ns, "completed_ns": time.time_ns()}
            )
            return result

        engine.ppo_update = update

    original_commit = trainer.stats_logger.commit

    def commit(epoch, step, global_step, data):
        trainer.check_evaluation()
        completed_step = global_step + 1
        roles = check_step_metrics(
            data, completed_step, completed_step - optimizer_steps_base
        )
        if data.get("ppo_actor/explicit_termination") != 1:
            raise RuntimeError(
                "PPO update did not consume explicit episode termination metadata"
            )
        step_dir = evidence / "steps"
        step_dir.mkdir(exist_ok=True)
        payload = {
            "completed_step": completed_step,
            "raw_global_step": global_step,
            "epoch": epoch,
            "epoch_step": step,
            "recorded_ns": time.time_ns(),
            "role_updates": roles,
            "update_spans": state["spans"],
            "published_version": trainer.rollout.get_version(),
            "metrics": data,
        }
        if preflight or completed_step <= 5:
            return_path = evidence / "returns" / f"{completed_step}.json"
            payload["returns_audit_sha256"] = hashlib.sha256(
                return_path.read_bytes()
            ).hexdigest()
        if payload["published_version"] != completed_step:
            raise RuntimeError(
                "Rollout version was not published after the joint update"
            )
        if preflight:
            verify_published_policy(trainer, evidence, completed_step)
        (step_dir / f"{completed_step}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )
        original_commit(epoch, step, global_step, data)

    trainer.stats_logger.commit = commit
    original_trace = trainer._save_perf_tracer

    def trace_and_supervise(step):
        original_trace(step)
        completed = step + 1
        if completed > 5:
            return
        trainer.rollout.pause()
        step_path = evidence / "steps" / f"{completed}.json"
        digest = hashlib.sha256(step_path.read_bytes()).hexdigest()
        gate_dir = evidence / "supervision"
        gate_dir.mkdir(exist_ok=True)
        gate_path = gate_dir / f"{completed}.json"
        logger.info(
            "Step %s awaits evidence review at %s (sha256=%s)",
            completed,
            gate_path,
            digest,
        )
        deadline = time.monotonic() + 1800
        while True:
            if gate_path.exists():
                review = json.loads(gate_path.read_text())
                if (
                    review.get("step_sha256") != digest
                    or review.get("passed") is not True
                ):
                    raise RuntimeError(f"Invalid supervision record: {gate_path}")
                break
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Step {completed} supervision was not completed within 30 minutes"
                )
            time.sleep(2)

    trainer._save_perf_tracer = trace_and_supervise


def main(args: list[str]) -> None:
    config_only = "--check-config" in args
    args = [x for x in args if x != "--check-config"]
    if config_only:
        cfg, _ = parse_cli_args(args)
        cfg = to_structured_cfg(cfg, SaoPPOConfig)
        config = OmegaConf.to_object(cfg)
        assert isinstance(config, SaoPPOConfig)
    else:
        config, _ = load_expr_config(args, SaoPPOConfig)
    preflight = os.environ.get("SAO_PREFLIGHT", "0") == "1"
    validate_contract(config, preflight=preflight)
    validate_hf_checkpoint_paths(config)
    if config_only:
        sys.stdout.write(
            json.dumps(dataclasses.asdict(config), indent=2, default=str) + "\n"
        )
        return

    run_root = Path(config.cluster.fileroot)
    evidence = run_root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "resolved-config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    dataset = load_from_disk(config.train_dataset.path)
    train_dataset, valid_dataset = dataset["train"], dataset["test"]
    if len(valid_dataset) != 700 and not preflight:
        raise ValueError(
            f"Expected all 700 evaluation problems, got {len(valid_dataset)}"
        )

    workflow = "areal.workflow.sao_math.AuditedMathWorkflow"
    workflow_kwargs = {
        "reward_fn": "areal.reward.math_prd.math_prd_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": False,
        "audit_dir": str(evidence / "samples"),
    }
    eval_kwargs = {**workflow_kwargs, "gconfig": config.eval_gconfig}

    with AsyncEvalPPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        weight_only_initial_step = weight_only_start_step_from_env()
        trainer.rollout.prepare_batch = functools.partial(
            trainer.rollout.prepare_batch, finite_epoch=True, fail_on_rejection=True
        )
        # DistributedSampler otherwise uses its own default seed=0.
        trainer.train_dataloader.sampler.seed = config.seed
        source_ids = list(train_dataset["source_id"])
        order = [source_ids[i] for i in trainer.train_dataloader.sampler]
        (evidence / "epoch-order.json").write_text(
            json.dumps(
                {
                    "seed": config.seed,
                    "source_ids": order,
                    "source_id_order_sha256": hashlib.sha256(
                        ("\n".join(order) + "\n").encode()
                    ).hexdigest(),
                    "dataloader_steps": len(trainer.train_dataloader),
                    "preflight": preflight,
                },
                indent=2,
            )
            + "\n"
        )
        prepare_weight_only_continuation(
            trainer,
            train_dataset,
            evidence,
            config=config,
            initial_step=weight_only_initial_step,
        )
        trainer._apply_initial_step_policy_version(weight_only_initial_step)
        install_audit_hooks(
            trainer,
            evidence,
            preflight=preflight,
            initial_step=weight_only_initial_step,
        )
        if preflight and trainer.recover_info is None:
            verify_published_policy(trainer, evidence, 0)
        if config.evaluator.eval_before_train and trainer.recover_info is None:
            # Consume only the initial trigger; do not advance the periodic evaluation cadence.
            trainer.evaluator.freq_ctl.check(epochs=0, steps=0)
            trainer._evaluate_fn(workflow, eval_kwargs)
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_kwargs,
            initial_step=weight_only_initial_step,
        )
        (evidence / "epoch-finished.json").write_text(
            json.dumps(
                {
                    "train_dataset_rows": len(train_dataset),
                    "expected_steps": len(trainer.train_dataloader),
                    "preflight": preflight,
                    "finished_ns": time.time_ns(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
