# SPDX-License-Identifier: Apache-2.0

"""Train a scalar critic from online τ² train-task episodes."""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from examples.tau2.contracts import (
    MAX_ASSISTANT_TOKENS,
    OFFICIAL_TAU2_REPOSITORY,
    OFFICIAL_TAU2_REVISION,
    SUPPORTED_DOMAINS,
    resolve_pinned_hf_snapshot,
)
from examples.tau2.evaluation import repeat_groups_for_dispatch
from examples.tau2.train import collection_provenance, get_tau2_dataset
from examples.tau2.utils import Tau2PPOConfig
from scripts.qualification.critic_pretrain import (
    actor_update_metrics,
    aggregate_by_key,
    aggregate_validation,
    save_initial_critic,
    validate_config,
    write_json,
)

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.infra.rpc.rtensor import RTensor
from areal.utils.recover import RecoverInfo


def _source_file_hashes() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    files = (
        "scripts/tau2/train_critic.py",
        "areal/experimental/openai/client.py",
        "areal/experimental/openai/proxy/workflow.py",
        "areal/engine/sglang_remote.py",
        "examples/tau2/agent.py",
        "examples/tau2/contracts.py",
        "examples/tau2/evaluation.py",
        "examples/tau2/train.py",
        "examples/tau2/user_simulator.py",
        "examples/tau2/utils.py",
    )
    return {
        path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest()
        for path in files
    }


def _checkpoint_root(*, fileroot: str, experiment_name: str, trial_name: str) -> Path:
    return (
        Path(fileroot)
        / "checkpoints"
        / getpass.getuser()
        / experiment_name
        / trial_name
    )


def _model_checkpoint_path(
    *,
    fileroot: str,
    experiment_name: str,
    trial_name: str,
    name: str,
    epoch: int,
    epoch_step: int,
    global_step: int,
) -> Path:
    return (
        _checkpoint_root(
            fileroot=fileroot,
            experiment_name=experiment_name,
            trial_name=trial_name,
        )
        / name
        / f"epoch{epoch}epochstep{epoch_step}globalstep{global_step}"
    )


def _recover_checkpoint_path(
    *, fileroot: str, experiment_name: str, trial_name: str, name: str
) -> Path:
    return (
        _checkpoint_root(
            fileroot=fileroot,
            experiment_name=experiment_name,
            trial_name=trial_name,
        )
        / name
        / "recover_checkpoint"
    )


def _final_step_paths(config: Tau2PPOConfig, *, train_rows: int) -> dict[str, Any]:
    steps_per_epoch = math.ceil(train_rows / config.train_dataset.batch_size)
    final_completed_step = int(config.total_train_steps)
    final_global_step = final_completed_step - 1
    final_epoch = final_global_step // steps_per_epoch
    final_epoch_step = final_global_step % steps_per_epoch
    final_checkpoint_path = _model_checkpoint_path(
        fileroot=config.saver.fileroot,
        experiment_name=config.saver.experiment_name,
        trial_name=config.saver.trial_name,
        epoch=final_epoch,
        epoch_step=final_epoch_step,
        global_step=final_global_step,
        name="critic",
    )
    recover_root = _checkpoint_root(
        fileroot=config.recover.fileroot,
        experiment_name=config.recover.experiment_name,
        trial_name=config.recover.trial_name,
    )
    return {
        "final_completed_step": final_completed_step,
        "final_global_step": final_global_step,
        "final_epoch": final_epoch,
        "final_epoch_step": final_epoch_step,
        "final_critic_checkpoint": str(final_checkpoint_path),
        "recover_info": str(recover_root / "recover_info"),
        "recover_actor_checkpoint": str(
            _recover_checkpoint_path(
                fileroot=config.recover.fileroot,
                experiment_name=config.recover.experiment_name,
                trial_name=config.recover.trial_name,
                name="default",
            )
        ),
        "recover_critic_checkpoint": str(
            _recover_checkpoint_path(
                fileroot=config.recover.fileroot,
                experiment_name=config.recover.experiment_name,
                trial_name=config.recover.trial_name,
                name="critic",
            )
        ),
    }


def _read_required_json(path: Path, *, description: str) -> Any:
    if not path.exists():
        raise RuntimeError(f"Missing {description}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _final_summary_payload(
    config: Tau2PPOConfig,
    *,
    evidence_dir: Path,
    data_summary: dict[str, Any],
    select_best: bool,
) -> dict[str, Any]:
    final_completed_step = int(config.total_train_steps)
    final_validation_path = (
        evidence_dir / f"validation-step-{final_completed_step:06d}.json"
    )
    final_metrics_path = (
        evidence_dir / "train-metrics" / f"step-{final_completed_step:06d}.json"
    )
    final_validation = _read_required_json(
        final_validation_path, description="final validation report"
    )
    final_metrics = _read_required_json(final_metrics_path, description="final metrics")
    if final_validation.get("completed_step") != final_completed_step:
        raise RuntimeError(
            "Final validation report completed_step mismatch: "
            f"expected={final_completed_step} observed={final_validation.get('completed_step')}"
        )
    if final_metrics.get("completed_step") != final_completed_step:
        raise RuntimeError(
            "Final train metrics completed_step mismatch: "
            f"expected={final_completed_step} observed={final_metrics.get('completed_step')}"
        )
    full_planned_steps = _expected_train_steps(
        config, train_rows=int(data_summary["train_rows"])
    )
    paths = _final_step_paths(config, train_rows=int(data_summary["train_rows"]))
    final_checkpoint = Path(paths["final_critic_checkpoint"])
    if not final_checkpoint.exists():
        raise RuntimeError(f"Missing final critic checkpoint: {final_checkpoint}")
    recover_info = RecoverInfo.load(paths["recover_info"])
    if recover_info.last_step_info.global_step != paths["final_global_step"]:
        raise RuntimeError(
            "Final recover info global_step mismatch: "
            f"expected={paths['final_global_step']} "
            f"observed={recover_info.last_step_info.global_step}"
        )
    if (
        recover_info.trainer_state.get("num_critic_only_steps")
        != config.num_critic_only_steps
    ):
        raise RuntimeError(
            "Final recover info critic-only state mismatch: "
            f"expected={config.num_critic_only_steps} "
            f"observed={recover_info.trainer_state.get('num_critic_only_steps')}"
        )
    if recover_info.trainer_state.get("policy_version") != 0:
        raise RuntimeError(
            "Tau2 critic-only final recover info must keep actor policy_version=0, "
            f"observed={recover_info.trainer_state.get('policy_version')}"
        )
    for key in ("recover_actor_checkpoint", "recover_critic_checkpoint"):
        checkpoint_path = Path(paths[key])
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Missing final recovery checkpoint {key}: {checkpoint_path}"
            )
    return {
        "status": "complete",
        "completed_steps": final_completed_step,
        "full_planned_steps": full_planned_steps,
        "run_scope": (
            "full_formal"
            if config.experiment_mode == "formal"
            else (
                "full_tune"
                if final_completed_step == full_planned_steps
                else "bounded_tune_probe"
            )
        ),
        "data_summary": data_summary,
        "final_validation_report": str(final_validation_path),
        "final_metrics": str(final_metrics_path),
        "final_critic_checkpoint": paths["final_critic_checkpoint"],
        "recovery_paths": {
            "recover_info": paths["recover_info"],
            "recover_actor_checkpoint": paths["recover_actor_checkpoint"],
            "recover_critic_checkpoint": paths["recover_critic_checkpoint"],
        },
        "selection_policy": (
            "dev_best_checkpoint_selection"
            if select_best
            else "formal_test_metrics_only_no_selection"
        ),
    }


def _validation_steps(config: Tau2PPOConfig, profile: str) -> tuple[int, ...]:
    _ = profile
    final_step = int(config.total_train_steps or 0)
    if final_step <= 0:
        return ()
    freq = config.evaluator.freq_steps
    if freq is None or freq <= 0:
        return (0, final_step)
    return tuple(sorted({0, final_step, *range(freq, final_step + 1, freq)}))


def _cadence_steps(total_steps: int | None, freq_steps: int | None) -> tuple[int, ...]:
    if total_steps is None or total_steps <= 0:
        return ()
    regular = (
        range(freq_steps, total_steps + 1, freq_steps)
        if freq_steps and freq_steps > 0
        else ()
    )
    return tuple(sorted({total_steps, *regular}))


def _expected_train_steps(config: Tau2PPOConfig, *, train_rows: int) -> int:
    return math.ceil(train_rows / config.train_dataset.batch_size) * int(
        config.total_train_epochs
    )


def validate_tau2_critic_config(
    config: Tau2PPOConfig,
    *,
    train_rows: int | None = None,
    validation_rows: int | None = None,
) -> None:
    validate_config(config)
    if config.algorithm != "critic" or config.experiment_mode not in ("tune", "formal"):
        raise ValueError(
            "Tau2 critic training uses algorithm=critic, experiment_mode=tune|formal"
        )
    if tuple(config.domains) != SUPPORTED_DOMAINS:
        raise ValueError("Tau2 critic training must use all three official domains")
    if config.train_batch_episodes != config.train_dataset.batch_size:
        raise ValueError("train_batch_episodes must equal train_dataset.batch_size")
    if config.train_dataset.drop_last:
        raise ValueError("Tau2 critic online fit must keep the tail batch")
    if config.task_limit is not None:
        raise ValueError(
            "Tau2 critic training ignores task_limit; use tune step caps instead"
        )
    if config.valid_dataset is not None:
        raise ValueError("Tau2 critic uses a fixed dev rollout hook, not valid_dataset")
    if config.rollout.queue_size < 2 * config.train_dataset.batch_size:
        raise ValueError(
            "Online critic rollout.queue_size must cover the active train batch "
            "plus one reserved consumer batch"
        )
    if config.gconfig.n_samples != 1:
        raise ValueError("Tau2 critic training requires gconfig.n_samples=1")
    if config.actor.use_direct_dis_loss:
        raise ValueError("Tau2 critic fitting must not apply Direct DIS actor loss")
    if not config.actor.offload:
        raise ValueError("Tau2 critic production fit requires actor.offload=true")
    if not config.rollout.deterministic_sampling:
        raise ValueError(
            "Tau2 critic production fit requires rollout.deterministic_sampling=true"
        )
    if config.critic is None:
        raise ValueError("Tau2 critic fit requires a critic config")
    if (
        config.critic.loss_reduction != "token_mean"
        or config.critic.eps_clip is not None
    ):
        raise ValueError("Tau2 critic fit requires token_mean plain MSE")
    if (
        not config.actor.disable_dropout
        or not config.critic.disable_dropout
        or config.actor.ppo_n_minibatches != 1
        or config.critic.ppo_n_minibatches != 1
    ):
        raise ValueError(
            "Tau2 critic uniform tail replication requires disabled dropout and "
            "one PPO minibatch for actor and critic"
        )
    if train_rows is not None:
        expected_steps = _expected_train_steps(config, train_rows=train_rows)
        if (
            config.experiment_mode == "formal"
            and config.total_train_steps != expected_steps
        ):
            raise ValueError(
                "Formal tau2 critic fit requires total_train_steps to match "
                f"ceil(train_rows / batch_size) * epochs: expected={expected_steps} "
                f"observed={config.total_train_steps}"
            )
        if config.experiment_mode == "tune" and not (
            0 < int(config.total_train_steps) <= expected_steps
        ):
            raise ValueError(
                "Tune tau2 critic fit requires total_train_steps in "
                f"[1, {expected_steps}], observed={config.total_train_steps}"
            )
    if config.total_train_steps != config.num_critic_only_steps:
        raise ValueError("Every tau2 critic step must stay inside critic-only warmup")
    if config.critic_updates_before_actor != 0:
        raise ValueError("Tau2 critic pretraining must not run SAO actor updates")
    if config.recover.mode not in ("auto", "disabled"):
        raise ValueError("Tau2 critic recovery must be auto or explicitly disabled")
    if config.gconfig.max_tokens != 32768 or config.gconfig.max_new_tokens != 4096:
        raise ValueError("Tau2 critic requires 32K context and 4K response cap")
    if config.econfig.enable_thinking or config.econfig.add_thinking_tool:
        raise ValueError("Tau2 critic uses the non-thinking policy contract")
    if (
        config.critic.optimizer.lr_scheduler_type != "constant"
        or config.critic.optimizer.warmup_steps != 0
        or config.critic.optimizer.warmup_steps_proportion != 0
    ):
        raise ValueError("Tau2 critic requires constant LR with zero warmup")
    if not config.critic.freeze_critic_attention:
        raise ValueError("Tau2 critic production fit requires frozen attention")
    if validation_rows is not None:
        expected_validation_rows = 100 if config.experiment_mode == "formal" else 36
        if validation_rows != expected_validation_rows:
            raise ValueError(
                "Tau2 critic fixed validation set has unexpected task count: "
                f"expected={expected_validation_rows} observed={validation_rows}"
            )


def fill_derived_train_steps(config: Tau2PPOConfig, *, train_rows: int) -> int:
    expected_steps = _expected_train_steps(config, train_rows=train_rows)
    if config.total_train_steps is None:
        config.total_train_steps = expected_steps
    if config.num_critic_only_steps in (None, 0):
        config.num_critic_only_steps = expected_steps
    return expected_steps


def resolve_training_snapshots(config: Tau2PPOConfig, *, snapshot_resolver=None) -> str:
    """Resolve the pinned actor/backbone source before constructing GPU workers."""

    actor_source = config.actor.path
    actor_snapshot = resolve_pinned_hf_snapshot(
        actor_source,
        snapshot_resolver=snapshot_resolver,
    )
    config.actor.path = actor_snapshot
    config.tokenizer_path = actor_snapshot
    config.rollout.tokenizer_path = actor_snapshot
    config.sglang.model_path = actor_snapshot
    config.vllm.model = actor_snapshot
    if config.critic is not None and config.critic.path == actor_source:
        config.critic.path = actor_snapshot
    return actor_snapshot


def build_tau2_critic_datasets(
    config: Tau2PPOConfig,
) -> tuple[Any, Any, dict[str, Any]]:
    provenance = collection_provenance(config, actor_source=config.actor.path)
    validation_split = "test" if config.experiment_mode == "formal" else "dev"
    dataset_kwargs = dict(
        domain=SUPPORTED_DOMAINS,
        provenance=provenance,
        seed=config.seed,
        experiment_mode=config.experiment_mode,
        critic_dev_fraction=config.critic_dev_fraction,
        critic_tasks_per_domain=config.critic_tasks_per_domain,
        critic_rollouts_per_task=1,
        algorithm=config.algorithm,
    )
    train_dataset = get_tau2_dataset(**dataset_kwargs, split="train")
    validation_dataset = get_tau2_dataset(**dataset_kwargs, split=validation_split)
    train_ids = {(str(row["domain"]), str(row["task_id"])) for row in train_dataset}
    validation_ids = {
        (str(row["domain"]), str(row["task_id"])) for row in validation_dataset
    }
    if train_ids & validation_ids:
        raise ValueError("Tau2 critic train/validation task split is not disjoint")
    summary = {
        "status": "checked",
        "source": "official_tau2_train_tasks",
        "experiment_mode": config.experiment_mode,
        "validation_split": validation_split,
        "split_seed": config.seed,
        "dev_fraction": config.critic_dev_fraction,
        "policy_provenance": provenance,
        "reward_protocol": "official-binary-v1",
        "train_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "unique_train_tasks": len(train_ids),
        "unique_validation_tasks": len(validation_ids),
        "episodes_by_split_domain": {
            f"train:{domain}": sum(
                1 for row in train_dataset if row["domain"] == domain
            )
            for domain in SUPPORTED_DOMAINS
        }
        | {
            f"{validation_split}:{domain}": sum(
                1 for row in validation_dataset if row["domain"] == domain
            )
            for domain in SUPPORTED_DOMAINS
        },
    }
    if validation_split == "dev":
        summary["dev_rows"] = len(validation_dataset)
    else:
        summary["test_rows"] = len(validation_dataset)
    return train_dataset, validation_dataset, summary


def _to_2d_tensor(row: dict[str, Any], key: str, dtype: torch.dtype) -> torch.Tensor:
    value = row[key]
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=dtype)
    value = value.detach().cpu().to(dtype=dtype)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError(f"{key} must be 1D or 2D, got shape {tuple(value.shape)}")
    return value


def _row_reward(row: dict[str, Any]) -> float:
    value = row.get("reward", row.get("official_score", row.get("rewards")))
    if value is None:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            value = metadata.get("official_score")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1)[-1].item()
    return float(value)


def _critic_loss_mask(row: dict[str, Any], *, zero_loss: bool = False) -> torch.Tensor:
    mask = _to_2d_tensor(row, "loss_mask", torch.bool)
    mask = torch.roll(mask, shifts=-1, dims=-1)
    mask[:, -1] = False
    if zero_loss:
        mask.zero_()
    return mask


def _value_target_row(
    row: dict[str, Any], *, zero_loss: bool = False
) -> dict[str, torch.Tensor]:
    input_ids = _to_2d_tensor(row, "input_ids", torch.long)
    width = input_ids.shape[-1]
    reward = _row_reward(row)
    return {
        "input_ids": input_ids,
        "attention_mask": _to_2d_tensor(row, "attention_mask", torch.bool),
        "loss_mask": _critic_loss_mask(row, zero_loss=zero_loss),
        "values": torch.zeros((1, width), dtype=torch.float32),
        "returns": torch.full((1, width), reward, dtype=torch.float32),
    }


def _critic_dp_size(trainer: PPOTrainer) -> int:
    strategy = trainer.critic.parallel_strategy
    for attr in ("dp_size", "data_parallel_size"):
        value = getattr(strategy, attr, None)
        if value is not None:
            return int(value)
    return 1


def _pad_for_dp(
    rows: list[dict[str, Any]], dp_size: int
) -> tuple[list[dict[str, Any]], int]:
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    padded = list(rows)
    real_count = len(rows)
    while len(padded) % dp_size:
        padded.append(dict(rows[len(padded) % real_count], _zero_loss_padding=True))
    return padded, real_count


def _metric_row(row: dict[str, Any], values: torch.Tensor) -> dict[str, Any]:
    mask = _critic_loss_mask(row).reshape(-1)
    predicted = values.reshape(-1).float().cpu()
    if predicted.numel() != mask.numel():
        raise ValueError(
            "Critic validation prediction width does not match row: "
            f"predicted={predicted.numel()} observed={mask.numel()}"
        )
    predicted = predicted[mask].double()
    if predicted.numel() == 0:
        raise ValueError(f"Critic validation row has no action tokens: {row}")
    if not torch.isfinite(predicted).all():
        raise ValueError("Non-finite critic validation predictions")
    reward = _row_reward(row)
    target = torch.full_like(predicted, reward)
    error = predicted - target
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    task_id = str(row.get("task_id", metadata.get("task_id", "unknown")))
    domain = str(row.get("domain", metadata.get("domain", "unknown")))
    source_id = str(row.get("source_id", f"tau2:{domain}:{task_id}"))
    return {
        "source_id": source_id,
        "episode_id": str(row.get("episode_id", source_id)),
        "domain": domain,
        "task_id": task_id,
        "target": reward,
        "truncated": bool(row.get("truncated", False)),
        "n_tokens": int(predicted.numel()),
        "prediction_mean": float(predicted.mean().item()),
        "prediction_min": float(predicted.min().item()),
        "prediction_max": float(predicted.max().item()),
        "error_sum": float(error.sum().item()),
        "sse": float(error.square().sum().item()),
        "target_sum": float(target.sum().item()),
        "target2_sum": float(target.square().sum().item()),
    }


def _localize_remote_values(remote_values: Any) -> list[torch.Tensor]:
    return [value.detach().cpu() for value in RTensor.localize(remote_values)]


def _grad_norm_value(report: Any) -> float:
    reports = report if isinstance(report, list) else [report]
    values = [float(item["grad_norm"]) for item in reports]
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"Non-finite critic grad norm report: {report}")
    return max(values)


def _clear_critic_batches(
    trainer: PPOTrainer, tensor_batch: list[dict[str, torch.Tensor]]
) -> None:
    clear_batches = getattr(trainer.critic, "clear_batches", None)
    if clear_batches is not None:
        clear_batches(tensor_batch)


def _group_by_domain(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        grouped[str(row.get("domain", metadata.get("domain")))].append(row)
    return dict(grouped)


def evaluate_tau2_critic(
    trainer: PPOTrainer,
    rows: list[dict[str, Any]],
    *,
    evidence_dir: Path,
    completed_step: int,
) -> dict[str, Any]:
    if getattr(trainer, "_should_offload_actor", False):
        trainer._offload_model(trainer.actor, role="actor")
    if getattr(trainer, "_should_offload_critic", False):
        trainer._onload_model(trainer.critic, role="critic")
    try:
        dp_size = _critic_dp_size(trainer)
        chunk_size = max(
            dp_size, int(os.environ.get("TAU2_CRITIC_VALIDATION_CHUNK", "4"))
        )
        metric_rows: list[dict[str, Any]] = []
        padded_total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            padded, real_count = _pad_for_dp(chunk, dp_size)
            padded_total += len(padded)
            tensor_batch = [
                {
                    "input_ids": _to_2d_tensor(row, "input_ids", torch.long),
                    "attention_mask": _to_2d_tensor(row, "attention_mask", torch.bool),
                    "loss_mask": _to_2d_tensor(row, "loss_mask", torch.bool),
                }
                for row in padded
            ]
            remote_values = None
            try:
                remote_values = trainer.critic.compute_values(tensor_batch)
                values = _localize_remote_values(remote_values)
            finally:
                if remote_values is None:
                    trainer.critic.clear_batches(tensor_batch)
                else:
                    trainer.critic.clear_batches(tensor_batch, remote_values)
            for row, value in zip(
                padded[:real_count], values[:real_count], strict=True
            ):
                metric_rows.append(_metric_row(row, value))

        grad_norms = {}
        for name, group_rows in {"overall": rows, **_group_by_domain(rows)}.items():
            padded, _ = _pad_for_dp(group_rows, dp_size)
            tensor_batch = [
                _value_target_row(
                    row,
                    zero_loss=bool(row.get("_zero_loss_padding", False)),
                )
                for row in padded
            ]
            try:
                report = trainer.critic.grad_norm(tensor_batch)
            finally:
                _clear_critic_batches(trainer, tensor_batch)
            grad_norms[name] = _grad_norm_value(report)
    finally:
        if getattr(trainer, "_should_offload_critic", False):
            trainer._offload_model(trainer.critic, role="critic")

    summary = aggregate_validation(metric_rows)
    payload = {
        "completed_step": completed_step,
        "input_records": len(rows),
        "padded_records": padded_total,
        "dp_size": dp_size,
        "rows": metric_rows,
        "summary": summary,
        "by_domain": aggregate_by_key(metric_rows, "domain"),
        "by_reward": aggregate_by_key(metric_rows, "target"),
        "by_truncation": aggregate_by_key(metric_rows, "truncated"),
        "grad_norm": grad_norms,
        "grad_norm_scope": "validation_loss_backward_no_optimizer_step",
    }
    write_json(evidence_dir / f"validation-step-{completed_step:06d}.json", payload)
    return payload


def tau2_validation_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "tau2_critic_validation/completed_step": float(report["completed_step"]),
        "tau2_critic_validation/overall_mse": report["summary"]["mse"],
        "tau2_critic_validation/overall_macro_mse": report["summary"]["macro_mse"],
        "tau2_critic_validation/overall_explained_variance_defined": float(
            report["summary"]["explained_variance_defined"]
        ),
        "tau2_critic_validation/overall_grad_norm": report["grad_norm"]["overall"],
        "tau2_critic_validation/overall_n_tokens": float(report["summary"]["n_tokens"]),
    }
    if report["summary"]["explained_variance"] is not None:
        metrics["tau2_critic_validation/overall_explained_variance"] = report[
            "summary"
        ]["explained_variance"]
    for domain, summary in report["by_domain"].items():
        metrics[f"tau2_critic_validation/{domain}_mse"] = summary["mse"]
        metrics[f"tau2_critic_validation/{domain}_grad_norm"] = report["grad_norm"][
            domain
        ]
        metrics[f"tau2_critic_validation/{domain}_explained_variance_defined"] = float(
            summary["explained_variance_defined"]
        )
        if summary["explained_variance"] is not None:
            metrics[f"tau2_critic_validation/{domain}_explained_variance"] = summary[
                "explained_variance"
            ]
        metrics[f"tau2_critic_validation/{domain}_n_tokens"] = float(
            summary["n_tokens"]
        )
    return metrics


def _selection_value(report: dict[str, Any]) -> float:
    value = report["summary"]["mse"]
    if not math.isfinite(float(value)):
        raise ValueError(f"Non-finite validation MSE: {value}")
    return float(value)


def _validation_rows(dataset: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixed_validation_manifest(
    config: Tau2PPOConfig,
    validation_rows: list[dict[str, Any]],
    *,
    workflow: str,
    workflow_kwargs: dict[str, Any],
) -> dict[str, Any]:
    validation_split = "test" if config.experiment_mode == "formal" else "dev"
    payload = {
        "actor_path": config.actor.path,
        "critic_path": config.critic.path if config.critic is not None else None,
        "experiment_mode": config.experiment_mode,
        "validation_split": validation_split,
        "official_tau2": {
            "repository": OFFICIAL_TAU2_REPOSITORY,
            "revision": OFFICIAL_TAU2_REVISION,
        },
        "adapter_source_hashes": _source_file_hashes(),
        "task_ids": [
            {
                "domain": str(row["domain"]),
                "task_id": str(row["task_id"]),
                "source_id": str(row.get("source_id", "")),
            }
            for row in validation_rows
        ],
        "generation": {
            "max_tokens": config.gconfig.max_tokens,
            "max_new_tokens": config.gconfig.max_new_tokens,
            "temperature": config.gconfig.temperature,
            "top_p": config.gconfig.top_p,
            "top_k": config.gconfig.top_k,
            "frequency_penalty": config.gconfig.frequency_penalty,
            "seed": config.gconfig.seed,
        },
        "environment": dataclasses.asdict(config.econfig),
        "workflow": workflow,
        "workflow_kwargs": workflow_kwargs,
    }
    return {
        "schema": "tau2-fixed-validation-rollouts-v1",
        "digest": _json_digest(payload),
        "payload": payload,
    }


def _detach_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _detach_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_detach_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_cpu(value) for value in obj)
    return obj


def _rollout_task_ids(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "domain": str(row["domain"]),
            "task_id": str(row["task_id"]),
            "source_id": str(row.get("source_id", "")),
        }
        for row in rows
    ]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_fixed_validation_rollouts(
    rows: Any, expected_task_ids: list[dict[str, str]]
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("Fixed validation rollout cache must contain a list of rows")
    if _rollout_task_ids(rows) != expected_task_ids:
        raise RuntimeError(
            "Fixed validation rollout cache task identities do not match"
        )
    for index, row in enumerate(rows):
        try:
            input_ids = _to_2d_tensor(row, "input_ids", torch.long)
            attention_mask = _to_2d_tensor(row, "attention_mask", torch.bool)
            loss_mask = _to_2d_tensor(row, "loss_mask", torch.bool)
            _row_reward(row)
        except Exception as exc:
            raise RuntimeError(
                f"Fixed validation rollout cache row {index} is malformed"
            ) from exc
        if (
            attention_mask.shape != input_ids.shape
            or loss_mask.shape != input_ids.shape
        ):
            raise RuntimeError(
                "Fixed validation rollout cache tensor shape mismatch at row "
                f"{index}: input={tuple(input_ids.shape)} "
                f"attention={tuple(attention_mask.shape)} loss={tuple(loss_mask.shape)}"
            )
    return rows


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _attach_task_identity(rollout: dict[str, Any], row: dict[str, Any]) -> None:
    metadata = (
        rollout.get("metadata") if isinstance(rollout.get("metadata"), dict) else {}
    )
    expected_domain = str(row["domain"])
    expected_task_id = str(row["task_id"])
    observed_domain = rollout.get("domain", metadata.get("domain"))
    observed_task_id = rollout.get("task_id", metadata.get("task_id"))
    if observed_domain is not None and str(observed_domain) != expected_domain:
        raise RuntimeError(
            "Tau2 rollout result order drifted within the current batch: "
            f"expected domain={expected_domain} observed={observed_domain}"
        )
    if observed_task_id is not None and str(observed_task_id) != expected_task_id:
        raise RuntimeError(
            "Tau2 rollout result order drifted within the current batch: "
            f"expected task_id={expected_task_id} observed={observed_task_id}"
        )
    rollout.update(
        domain=expected_domain,
        task_id=expected_task_id,
        source_id=str(
            row.get("source_id", f"tau2:{expected_domain}:{expected_task_id}")
        ),
    )


def collect_tau2_validation_rollouts(
    trainer: PPOTrainer,
    validation_rows: list[dict[str, Any]],
    *,
    workflow: str,
    workflow_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    rollout = getattr(trainer, "eval_rollout", None)
    if rollout is None:
        raise RuntimeError(
            "Tau2 fixed validation evaluation requires trainer.eval_rollout"
        )
    start_proxy = getattr(rollout, "start_proxy", None)
    if callable(start_proxy):
        start_proxy()

    results: list[dict[str, Any]] = []
    batch_size = trainer.config.train_dataset.batch_size
    for start in range(0, len(validation_rows), batch_size):
        rows = validation_rows[start : start + batch_size]
        remote_batch = rollout.rollout_batch(
            rows,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=1,
            reward_normalization=False,
            drop_incomplete_group=False,
        )
        try:
            if len(remote_batch) != len(rows):
                raise RuntimeError(
                    "Tau2 fixed validation evaluation received incomplete rollout batch: "
                    f"{len(remote_batch)} results for {len(rows)} inputs"
                )
            localized = _detach_cpu(RTensor.localize(remote_batch))
            for row, rollout_row in zip(rows, localized, strict=True):
                _attach_task_identity(rollout_row, row)
            results.extend(localized)
        finally:
            if remote_batch:
                trainer.actor.clear_batches(*remote_batch)
    return results


def load_or_collect_fixed_validation_rollouts(
    trainer: PPOTrainer,
    validation_rows: list[dict[str, Any]],
    *,
    evidence_dir: Path,
    workflow: str,
    workflow_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = _fixed_validation_manifest(
        trainer.config,
        validation_rows,
        workflow=workflow,
        workflow_kwargs=workflow_kwargs,
    )
    manifest_path = evidence_dir / "fixed-validation-manifest.json"
    rollouts_path = evidence_dir / "fixed-validation-rollouts.pt"
    if manifest_path.exists() or rollouts_path.exists():
        if not manifest_path.exists() or not rollouts_path.exists():
            raise RuntimeError("Incomplete fixed validation rollout cache")
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed.get("digest") != manifest["digest"]:
            raise RuntimeError(
                "Fixed validation rollout cache does not match current task/config digest"
            )
        expected_sha256 = observed.get("rollouts_sha256")
        if not expected_sha256:
            raise RuntimeError(
                "Fixed validation rollout cache manifest lacks rollouts_sha256"
            )
        observed_sha256 = _file_sha256(rollouts_path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                "Fixed validation rollout cache tensor file hash mismatch: "
                f"expected={expected_sha256} observed={observed_sha256}"
            )
        rows = torch.load(rollouts_path, map_location="cpu", weights_only=True)
        return _validate_fixed_validation_rollouts(
            rows, manifest["payload"]["task_ids"]
        )

    rollouts = collect_tau2_validation_rollouts(
        trainer,
        validation_rows,
        workflow=workflow,
        workflow_kwargs=workflow_kwargs,
    )
    rollouts = _validate_fixed_validation_rollouts(
        rollouts, manifest["payload"]["task_ids"]
    )
    tmp_rollouts = rollouts_path.with_suffix(rollouts_path.suffix + ".tmp")
    torch.save(rollouts, tmp_rollouts)
    os.replace(tmp_rollouts, rollouts_path)
    manifest = dict(manifest, rollouts_sha256=_file_sha256(rollouts_path))
    _write_json_atomic(manifest_path, manifest)
    return rollouts


def install_tau2_validation_hook(
    trainer: PPOTrainer,
    fixed_validation_rollouts: list[dict[str, Any]],
    *,
    evidence_dir: Path,
    validation_steps: tuple[int, ...],
    best: dict[str, Any] | None,
    select_best: bool,
) -> None:
    original_commit = trainer.stats_logger.commit

    def commit(epoch: int, step: int, global_step: int, data: dict[str, Any]) -> None:
        nonlocal best
        completed = global_step + 1
        if completed == trainer.config.total_train_steps and (
            not trainer.config.saver.freq_steps
            or completed % trainer.config.saver.freq_steps
        ):
            trainer._save_training_state(
                epoch=epoch, epoch_step=step, global_step=global_step, force=True
            )
        actor_metrics = actor_update_metrics(data)
        if actor_metrics:
            raise RuntimeError(
                "Actor optimizer emitted update metrics during tau2 critic-only "
                f"step {completed}: {actor_metrics}"
            )
        if completed in validation_steps:
            report = evaluate_tau2_critic(
                trainer,
                fixed_validation_rollouts,
                evidence_dir=evidence_dir,
                completed_step=completed,
            )
            data.update(tau2_validation_metrics(report))
            current = _selection_value(report)
            if select_best and (best is None or current < float(best["metric_value"])):
                trainer.saver.save(
                    trainer.critic,
                    epoch,
                    step,
                    global_step,
                    tokenizer=trainer.tokenizer,
                    processor=trainer.processor,
                    name="critic-best",
                    force=True,
                )
                best = {
                    "selection_metric": "mse",
                    "metric_value": current,
                    "completed_step": completed,
                    "checkpoint_name": "critic-best",
                    "validation_report": str(
                        evidence_dir / f"validation-step-{completed:06d}.json"
                    ),
                }
                write_json(evidence_dir / "best-validation.json", best)
        write_json(
            evidence_dir / "train-metrics" / f"step-{completed:06d}.json",
            {
                "completed_step": completed,
                "epoch": epoch,
                "epoch_step": step,
                "global_step": global_step,
                "metrics": data,
            },
        )
        data["tau2_critic_train/completed_step"] = completed
        original_commit(epoch, step, global_step, data)

    trainer.stats_logger.commit = commit


class Tau2CriticOnlineTrainer(PPOTrainer):
    """PPOTrainer wrapper that preserves finite τ² task-tail coverage."""

    def _init_impl(self, config, *args, **kwargs):
        super()._init_impl(config, *args, **kwargs)
        self._tau2_train_epoch = (
            self.recover_info.last_step_info.epoch if self.recover_info else 0
        )
        self._tau2_train_iterator = None
        self._tau2_next_train_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info
            else 0
        )
        if self.recover_info is None:
            self.train_dataloader.sampler.seed = config.seed
        self.rollout.prepare_batch = self._strict_prepare_training_batch

    def _next_strict_rows(self, dataloader) -> list[dict[str, Any]]:
        if self._tau2_train_iterator is None:
            if hasattr(dataloader, "sampler") and hasattr(
                dataloader.sampler, "set_epoch"
            ):
                dataloader.sampler.set_epoch(self._tau2_train_epoch)
            self._tau2_train_iterator = iter(dataloader)
        try:
            batch = next(self._tau2_train_iterator)
        except StopIteration:
            self._tau2_train_epoch += 1
            if hasattr(dataloader, "sampler") and hasattr(
                dataloader.sampler, "set_epoch"
            ):
                dataloader.sampler.set_epoch(self._tau2_train_epoch)
            self._tau2_train_iterator = iter(dataloader)
            batch = next(self._tau2_train_iterator)
        rows = [dict(row) for row in batch]
        if not rows:
            raise RuntimeError("Tau2 strict train iterator produced an empty batch")
        return rows

    def _strict_prepare_training_batch(
        self,
        dataloader,
        workflow,
        workflow_kwargs=None,
        should_accept_fn=None,
        group_size=1,
        dynamic_bs=False,
        reward_normalization=False,
        drop_incomplete_group=False,
        **kwargs,
    ):
        if dynamic_bs:
            raise RuntimeError(
                "Tau2 critic strict batching does not support dynamic_bs"
            )
        _ = kwargs
        rows = self._next_strict_rows(dataloader)
        logical_step = self._tau2_next_train_step
        task_base = logical_step * self.config.train_dataset.batch_size
        for offset, row in enumerate(rows):
            self.rollout.submit(
                data=row,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs or {},
                should_accept_fn=should_accept_fn,
                task_id=task_base + offset,
                group_size=group_size,
                reward_normalization=reward_normalization,
                drop_incomplete_group=drop_incomplete_group,
            )
        groups = [
            result
            for result in self.rollout.wait(count=len(rows))
            if result is not None
        ]
        if len(groups) != len(rows):
            raise RuntimeError(
                "Tau2 critic train rollout received incomplete current batch: "
                f"{len(groups)} results for {len(rows)} inputs"
            )
        for row, rollout_row in zip(rows, groups, strict=True):
            _attach_task_identity(rollout_row, row)
        physical, replicas = repeat_groups_for_dispatch(
            groups, self.actor.parallel_strategy.dp_size
        )
        self._batch_counts = {
            "tau2_batch/real_prompts": len(groups),
            "tau2_batch/real_episodes": len(groups) * self.config.gconfig.n_samples,
            "tau2_batch/physical_prompt_groups": len(physical),
            "tau2_batch/dispatch_replication_factor": replicas,
            "tau2_batch/logical_step": logical_step,
            "tau2_batch/task_id_start": task_base,
        }
        self._tau2_next_train_step += 1
        return physical

    def _export_and_commit_stats(self, epoch, epoch_step, global_step):
        stats = self.actor.export_stats()
        if self.critic is not None:
            stats.update(
                {
                    f"critic/{key}": value
                    for key, value in self.critic.export_stats().items()
                }
            )
        stats.update(self.rollout.export_stats())
        if self.eval_rollout is not None:
            stats.update(self.eval_rollout.export_stats())
        stats.update(getattr(self, "_batch_counts", {}))
        stats["ppo/critic_only"] = 1
        stats["ppo/policy_version"] = 0
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)


def _workflow_kwargs(config: Tau2PPOConfig) -> dict[str, Any]:
    return dict(
        econfig=dataclasses.asdict(config.econfig),
        gen_args=dict(
            temperature=config.gconfig.temperature,
            top_p=config.gconfig.top_p,
            frequency_penalty=config.gconfig.frequency_penalty,
            seed=config.gconfig.seed,
            max_completion_tokens=MAX_ASSISTANT_TOKENS,
            max_total_tokens=config.econfig.context_window_tokens,
        ),
        timeout=config.episode_timeout_seconds,
        infra_retries=config.infra_retries,
    )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-config", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    config, _ = load_expr_config(
        ["--config", args.config, *overrides],
        Tau2PPOConfig,
    )
    train_dataset, validation_dataset, data_summary = build_tau2_critic_datasets(config)
    fill_derived_train_steps(config, train_rows=len(train_dataset))
    validate_tau2_critic_config(
        config,
        train_rows=len(train_dataset),
        validation_rows=len(validation_dataset),
    )
    validation_steps = _validation_steps(config, "default")
    save_steps = _cadence_steps(config.total_train_steps, config.saver.freq_steps)
    workflow_kwargs = _workflow_kwargs(config)
    if args.check_config:
        print(
            OmegaConf.to_yaml(
                {
                    "config": OmegaConf.structured(config),
                    "data_summary": data_summary,
                    "actor_source": config.actor.path,
                    "train_rows": len(train_dataset),
                    "validation_rows": len(validation_dataset),
                    "validation_split": data_summary["validation_split"],
                    "validation_steps": list(validation_steps),
                    "save_steps": list(save_steps),
                    "side_effects": {
                        "api_calls": 0,
                        "gpu_processes": 0,
                        "queue_tasks": 0,
                        "model_snapshot_resolution": 0,
                    },
                },
                resolve=True,
            )
        )
        return

    actor_snapshot = resolve_training_snapshots(config)
    evidence_dir = Path(config.cluster.fileroot) / "tau2-critic-fit-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    workflow = "examples.tau2.agent.Tau2AgentWorkflow"
    write_json(
        evidence_dir / "resolved-config.json",
        {
            "config": dataclasses.asdict(config),
            "data_summary": data_summary,
            "actor_snapshot": actor_snapshot,
            "validation_steps": list(validation_steps),
            "save_steps": list(save_steps),
            "grad_norm_scope": "validation_loss_backward_no_optimizer_step",
            "trace_task_id_scheme": {
                "train": "global_step * train_batch_size + row_offset",
                "resume": "starts at recover_info.last_step_info.next().global_step",
                "uncommitted_retry": "may reuse the failed step task ids; earlier committed steps are preserved",
            },
        },
    )
    with Tau2CriticOnlineTrainer(config, train_dataset=train_dataset) as trainer:
        if trainer.recover_info is None:
            save_initial_critic(trainer)
        fixed_validation_rollouts = load_or_collect_fixed_validation_rollouts(
            trainer,
            _validation_rows(validation_dataset),
            evidence_dir=evidence_dir,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
        )
        select_best = config.experiment_mode != "formal"
        best_path = evidence_dir / "best-validation.json"
        best = (
            json.loads(best_path.read_text(encoding="utf-8"))
            if select_best and best_path.exists()
            else None
        )
        recovered_completed = (
            trainer.recover_info.last_step_info.next().global_step
            if trainer.recover_info is not None
            else 0
        )
        step0 = evaluate_tau2_critic(
            trainer,
            fixed_validation_rollouts,
            evidence_dir=evidence_dir,
            completed_step=recovered_completed,
        )
        if select_best and best is None:
            best = {
                "selection_metric": "mse",
                "metric_value": _selection_value(step0),
                "completed_step": recovered_completed,
                "checkpoint_name": "critic-initial",
                "validation_report": str(
                    evidence_dir / f"validation-step-{recovered_completed:06d}.json"
                ),
            }
            write_json(best_path, best)
        trainer.stats_logger.commit(0, -1, -1, tau2_validation_metrics(step0))
        install_tau2_validation_hook(
            trainer,
            fixed_validation_rollouts,
            evidence_dir=evidence_dir,
            validation_steps=validation_steps,
            best=best,
            select_best=select_best,
        )
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )
        summary = _final_summary_payload(
            config,
            evidence_dir=evidence_dir,
            data_summary=data_summary,
            select_best=select_best,
        )
        if select_best and best_path.exists():
            summary["best_validation"] = json.loads(
                best_path.read_text(encoding="utf-8")
            )
        write_json(evidence_dir / "summary.json", summary)


if __name__ == "__main__":
    main(sys.argv[1:])
