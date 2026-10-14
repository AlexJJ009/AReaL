# SPDX-License-Identifier: Apache-2.0
"""Fit the existing tau2 head critic on stored rollouts, without inference services.

Run with torchrun and a training-only critic backend matching its world size.
``--epochs`` counts passes over saved rows, not epochs of the original task pack.
``--check-config`` reads config/data only; it never creates an engine or process group.
"""

from __future__ import annotations

# CLI streams must be routed before importing modules that construct log handlers.
# ruff: noqa: E402
import argparse
import glob
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Keep the CLI readback machine-readable; imported AReaL log handlers use stdout.
# Bind those handlers to stderr before importing the engine and restore only the
# result stream at closeout. Library imports do not alter the caller's streams.
_CLI_STDOUT = sys.stdout if __name__ == "__main__" else None
if _CLI_STDOUT is not None:
    sys.stdout = sys.stderr

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from examples.tau2.critic_head import Tau2HeadCriticConfig, Tau2HeadFSDPPPOCritic
from examples.tau2.utils import Tau2PPOConfig
from scripts.qualification.critic_pretrain import aggregate_by_key, aggregate_validation
from scripts.tau2.train_critic import (
    _pad_for_dp,
    _value_target_row,
    evaluate_tau2_critic,
)
from scripts.tau2.train_critic_head import validate_head_config

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import to_structured_cfg
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.trainer.ppo.update import require_single_update
from areal.utils import stats_tracker


def first_token_parity(row: dict) -> float:
    """Binary planted label visible to every action prefix: first input token parity.

    This is a deterministic smoke signal, not a claim about tau2 task success.
    """
    return float(int(torch.as_tensor(row["input_ids"]).reshape(-1)[0]) % 2)


def import_label_override(path: str | None) -> Callable[[dict], float] | None:
    """Import a row -> scalar training-label function; no tokenizer/API required."""
    if path is None:
        return None
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError("label override must be module:function")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError("label override must name a callable")
    return function


def target_row(row: dict, override: Callable[[dict], float] | None = None) -> dict:
    """Preserve the online loss alignment, optionally replace the training return."""
    if override is not None:
        label = float(override(row))
        if not math.isfinite(label):
            raise ValueError("label override returned a non-finite value")
        row = dict(row, reward=label)
    return _value_target_row(row, zero_loss=bool(row.get("_zero_loss_padding", False)))


def replay_batches(rows: list[dict], batch_size: int, *, seed: int | None = None):
    """One complete saved-data pass, retaining a short final batch."""
    indices = list(range(len(rows)))
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(rows), generator=generator).tolist()
    for start in range(0, len(rows), batch_size):
        yield [rows[index] for index in indices[start : start + batch_size]]


def parse_config(argv: list[str]):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-rollouts", action="append", required=True)
    parser.add_argument("--validation-rollouts", action="append", required=True)
    parser.add_argument("--validation-label-override")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--label-override")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--head-init", help="Exported linear probe head .pt")
    parser.add_argument("--check-config", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    raw = OmegaConf.load(args.config)
    if "config" in raw:
        raw = raw.config
    raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(overrides))
    config = OmegaConf.to_object(to_structured_cfg(raw, Tau2PPOConfig))
    if args.head_lr is not None:
        config.critic_head_lr = args.head_lr
    if args.backbone_lr is not None:
        config.critic.optimizer.lr = args.backbone_lr
    if args.warmup_steps is not None:
        config.critic.optimizer.warmup_steps = args.warmup_steps
        config.critic.optimizer.warmup_steps_proportion = 0.0
    validate_head_config(config)
    batch_size = (
        args.batch_size if args.batch_size is not None else config.train_batch_episodes
    )
    if args.epochs <= 0 or batch_size <= 0 or args.eval_every <= 0:
        raise ValueError("epochs, batch-size, and eval-every must be positive")
    if not math.isfinite(config.critic.optimizer.lr) or config.critic.optimizer.lr < 0:
        raise ValueError("backbone learning rate must be finite and nonnegative")
    if config.critic_head_lr is not None and not math.isfinite(config.critic_head_lr):
        raise ValueError("head learning rate must be finite")
    if (config.critic.optimizer.warmup_steps or 0) < 0:
        raise ValueError("warmup steps must be nonnegative")
    allocation = ModelAllocation.from_str(config.critic.backend)
    if allocation.backend != "fsdp":
        raise ValueError("offline head critic requires fsdp backend")
    if any(
        getattr(allocation.parallel, field) != 1
        for field in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
            "expert_parallel_size",
            "expert_tensor_parallel_size",
        )
    ):
        raise ValueError(
            "offline replay currently supports pure data parallel FSDP only"
        )
    if config.critic.ppo_n_minibatches != 1:
        raise ValueError("offline replay requires critic.ppo_n_minibatches=1")
    if config.critic.eps_clip is not None:
        raise ValueError("offline terminal-return replay requires critic.eps_clip=null")
    # Only engine settings are consumed. Keep every output namespace owned by
    # this replay even when the source config came from a completed online run.
    for component in (
        config.cluster,
        config.saver,
        config.stats_logger,
        config.rollout,
    ):
        if hasattr(component, "fileroot"):
            component.fileroot = str(Path(args.output_dir).resolve())
        if hasattr(component, "experiment_name"):
            component.experiment_name = "tau2-critic-offline"
        if hasattr(component, "trial_name"):
            component.trial_name = Path(args.output_dir).name
    config.critic.experiment_name = "tau2-critic-offline"
    config.critic.trial_name = Path(args.output_dir).name
    config.perf_tracer = None
    critic_config = Tau2HeadCriticConfig.from_critic(
        config.critic,
        head_bias_init=config.critic_head_bias_init,
        head_lr=config.critic_head_lr,
    )
    return args, config, critic_config, allocation, batch_size


class LocalEvaluationCritic:
    """Adapt direct workers to the existing fixed-rollout validation evaluator.

    Partition padded rows across pure DP workers and gather predictions in row
    order. The engine reduces validation gradients using its own global token
    denominator, matching the controller evaluator without RPC or simulators.
    """

    def __init__(self, engine):
        self.engine = engine
        self.parallel_strategy = SimpleNamespace(dp_size=engine.parallel_helper.dp_size)

    def _local_rows(self, data):
        dp_size = self.parallel_strategy.dp_size
        if len(data) % dp_size:
            raise ValueError("validation batch must be padded to DP size")
        rank = dist.get_rank(self.engine.dp_group)
        count = len(data) // dp_size
        return data[rank * count : (rank + 1) * count]

    def compute_values(self, data):
        values = [
            value.detach().cpu()
            for value in self.engine.compute_values(self._local_rows(data))
        ]
        gathered = [None] * self.parallel_strategy.dp_size
        dist.all_gather_object(gathered, values, group=self.engine.cpu_group)
        return [value for shard in gathered for value in shard]

    def grad_norm(self, data):
        return self.engine.grad_norm(self._local_rows(data))

    def clear_batches(self, *_args):
        self.engine.clear_batches()


def validate_probe_artifact(path: str, checkpoint: str) -> dict:
    """Check linear head provenance and tensor contract without model resources."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("requires_sigmoid", False):
        raise ValueError(
            "Logistic probe heads require sigmoid and cannot initialize linear critic"
        )
    source = artifact.get("base_checkpoint", artifact.get("checkpoint"))
    if source != checkpoint:
        raise ValueError("probe head base checkpoint does not match critic.path")
    manifest_path = artifact.get("feature_manifest")
    expected_hash = artifact.get("feature_manifest_sha256")
    if manifest_path or expected_hash:
        if not manifest_path or not expected_hash:
            raise ValueError(
                "probe feature manifest path and SHA256 must both be supplied"
            )
        manifest_bytes = Path(manifest_path).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
            raise ValueError("probe feature manifest SHA256 mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest.get("checkpoint", manifest.get("base_checkpoint")) != checkpoint:
            raise ValueError("probe feature manifest checkpoint mismatch")
        if (
            manifest.get("feature_semantics")
            != "score_input_final_norm_shifted_action_prefix"
        ):
            raise ValueError("probe feature semantics do not match scalar critic")
    else:
        # Tiny smoke artifacts may lack a feature cache; disclose this in readback.
        manifest = None
    state = artifact.get("state_dict", artifact)
    weight = state.get("score.weight", state.get("weight"))
    bias = state.get("score.bias", state.get("bias"))
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] != 1:
        raise ValueError("probe head weight must have shape [1, hidden_size]")
    if not isinstance(bias, torch.Tensor) or bias.shape != (1,):
        raise ValueError("probe head bias must have shape [1]")
    if manifest is not None and manifest.get("hidden_size") != weight.shape[1]:
        raise ValueError("probe feature manifest hidden_size mismatch")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("probe head must be finite")
    return state


def load_probe_head(engine, path: str) -> None:
    """Load a CPU-exported head after normal engine initialization."""
    state = validate_probe_artifact(path, engine.config.path)
    for name in ("weight", "bias"):
        value = state.get(f"score.{name}", state.get(name))
        parameter = getattr(engine.model.score, name)
        if (
            value is None
            or parameter is None
            or tuple(value.shape) != tuple(parameter.shape)
        ):
            raise ValueError(f"probe head {name} shape does not match critic score")
        # DTensor copy needs the full value distributed with the score placement.
        if hasattr(parameter, "device_mesh"):
            from torch.distributed.tensor import distribute_tensor

            value = distribute_tensor(
                value.to(parameter.device, dtype=parameter.dtype),
                parameter.device_mesh,
                parameter.placements,
            )
        with torch.no_grad():
            parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))


def evaluate_banks(holder, banks, validation, evidence, step):
    """Evaluate each bank once; pool official metric rows without extra forwards."""
    reports = {}
    for name, bank in banks.items():
        directory = evidence if len(banks) == 1 else evidence / name
        directory.mkdir(parents=True, exist_ok=True)
        reports[name] = evaluate_tau2_critic(
            holder, bank, evidence_dir=directory, completed_step=step
        )
    if len(banks) > 1:
        rows = [row for report in reports.values() for row in report["rows"]]
        pooled = {
            "completed_step": step,
            "input_records": len(validation),
            "rows": rows,
            "summary": aggregate_validation(rows),
            "by_domain": aggregate_by_key(rows, "domain"),
            "by_reward": aggregate_by_key(rows, "target"),
            "by_truncation": aggregate_by_key(rows, "truncated"),
            "grad_norm_scope": "per_bank_validation_loss_backward_no_optimizer_step",
            "grad_norm_by_bank": {
                name: report["grad_norm"] for name, report in reports.items()
            },
        }
        (evidence / f"validation-step-{step:06d}.json").write_text(
            json.dumps(pooled, indent=2), encoding="utf-8"
        )
    stats_tracker.export_all(reduce_group=holder.critic.engine.dp_group)


def main(argv: list[str]) -> dict[str, Any]:
    args, config, critic_config, allocation, batch_size = parse_config(argv)
    # Shared stored-data reader enforces real tensor/mask/reward contracts.
    from scripts.tau2.critic_features import (
        action_positions,
        iter_rollouts,
        load_stored_rollouts,
        validate_splits,
    )

    paths = []
    for pattern in args.train_rollouts:
        matches = glob.glob(pattern)
        if not matches:
            raise ValueError(f"No training rollout files match {pattern}")
        paths.extend(matches)
    if len(paths) != len(set(paths)):
        raise ValueError("Training inputs contain duplicate files")
    train = iter_rollouts(paths, split="train", run="replay", batch_size=batch_size)
    banks = {}
    validation = []
    validation_override = import_label_override(args.validation_label_override)
    for index, path in enumerate(args.validation_rollouts):
        bank = load_stored_rollouts(path)
        validate_splits(train, bank)
        copied = []
        for row in bank:
            row = dict(row)
            row["episode_id"] = f"bank-{index}:{row.get('episode_id', len(copied))}"
            if validation_override:
                label = float(validation_override(row))
                if not math.isfinite(label):
                    raise ValueError(
                        "validation label override returned a non-finite value"
                    )
                row["reward"] = label
            copied.append(row)
        banks[f"bank-{index}"] = copied
        validation.extend(copied)
    if not train or not validation:
        raise ValueError("training and validation rollouts must both be nonempty")
    override = import_label_override(args.label_override)
    for row in train + validation:
        positions = action_positions(row)
        if not positions.numel():
            raise ValueError("Stored rollout has no action tokens")
        target_row(row)
    for row in train:
        target_row(row, override)
    steps_per_epoch = math.ceil(len(train) / batch_size)
    total_steps = steps_per_epoch * args.epochs
    if (critic_config.optimizer.warmup_steps or 0) >= total_steps:
        raise ValueError("warmup steps must be smaller than total optimizer steps")
    summary = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "replay_epochs": args.epochs,
        "batch_size": batch_size,
        "total_train_steps": total_steps,
        "label_override": args.label_override,
        "validation_label_override": args.validation_label_override,
        "validation_targets": "synthetic_diagnostic"
        if validation_override
        else "stored_official_reward_unchanged",
        "validation_banks": {name: len(bank) for name, bank in banks.items()},
        "critic_config": asdict(critic_config),
        "engine": "examples.tau2.critic_head.Tau2HeadFSDPPPOCritic",
        "shuffle": args.shuffle,
    }
    if args.head_init:
        validate_probe_artifact(args.head_init, critic_config.path)
        probe = torch.load(args.head_init, map_location="cpu", weights_only=True)
        summary["head_init_provenance"] = (
            "feature_manifest_sha256_and_checkpoint_verified"
            if probe.get("feature_manifest_sha256")
            else "partial_checkpoint_path_and_tensor_shape_only"
        )
    if args.check_config:
        return summary
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full FSDP replay requires CUDA; use CPU unit smoke for loss/optimizer verification"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != allocation.parallel.world_size:
        raise ValueError("torchrun world size must match critic.backend")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty replay output directory")
    os.environ["AREAL_RUNTIME_AUDIT_DIR"] = str(output / "runtime-audit")
    engine = Tau2HeadFSDPPPOCritic(critic_config)
    engine.create_process_group(allocation.parallel)
    rank = dist.get_rank()
    evidence = output if rank == 0 else output / f"rank-{rank}"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "resolved-config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    try:
        # FinetuneSpec assumes drop_last: use virtual padded size solely for exact
        # scheduler step count; replay retains all actual saved rows.
        engine.initialize(
            None, FinetuneSpec(args.epochs, steps_per_epoch * batch_size, batch_size)
        )
        if args.head_init:
            load_probe_head(engine, args.head_init)
        holder = SimpleNamespace(critic=LocalEvaluationCritic(engine))
        evaluate_banks(holder, banks, validation, evidence, 0)
        dp_size = engine.parallel_helper.dp_size
        dp_rank = dist.get_rank(engine.dp_group)
        completed = 0
        for epoch in range(args.epochs):
            for rows in replay_batches(
                train, batch_size, seed=config.seed + epoch if args.shuffle else None
            ):
                rows, _ = _pad_for_dp(rows, dp_size)
                per_rank = len(rows) // dp_size
                local = rows[dp_rank * per_rank : (dp_rank + 1) * per_rank]
                batch = [target_row(row, override) for row in local]
                lr_used = [group["lr"] for group in engine.optimizer.param_groups]
                report = engine.ppo_update(batch)
                require_single_update(report, "offline critic")
                engine.lr_scheduler_step()
                engine.clear_batches()
                completed += 1
                metrics = stats_tracker.export_all(reduce_group=engine.dp_group)
                metrics.update(report)
                metrics.update(completed_step=completed, epoch=epoch, lr_used=lr_used)
                (evidence / f"train-step-{completed:06d}.json").write_text(
                    json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
                )
                if completed % args.eval_every == 0 or completed == total_steps:
                    evaluate_banks(holder, banks, validation, evidence, completed)
        engine.save(
            SaveLoadMeta(
                path=str(output / "checkpoint-final"),
                weight_format="hf",
                with_optim=True,
                tokenizer=engine.tokenizer,
            )
        )
    finally:
        engine.destroy()
    return summary


if __name__ == "__main__":
    result = main(sys.argv[1:])
    if int(os.environ.get("RANK", "0")) == 0:
        _CLI_STDOUT.write(json.dumps(result, indent=2) + "\n")
