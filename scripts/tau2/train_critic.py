# SPDX-License-Identifier: Apache-2.0

"""Train a scalar critic from sealed τ² fixed-policy episodes."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from omegaconf import OmegaConf

from examples.tau2.contracts import (
    SUPPORTED_DOMAINS,
    resolve_pinned_hf_snapshot,
    split_critic_tasks,
)
from scripts.qualification.critic_pretrain import (
    actor_update_metrics,
    aggregate_by_key,
    aggregate_validation,
    save_initial_critic,
    validate_config,
    write_json,
)
from scripts.tau2.critic_data import (
    PROVENANCE_FIELDS,
    _load_official_train_ids,
    iter_critic_rows,
)

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.infra.rpc.rtensor import RTensor


def _validation_steps(config: PPOConfig, profile: str) -> tuple[int, ...]:
    _ = profile
    final_step = int(config.total_train_steps or 0)
    if final_step <= 0:
        return (0,)
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


def _expected_split_counter(
    official_train_ids: Iterable[tuple[str, str]],
    *,
    seed: int,
    dev_fraction: float,
) -> Counter[tuple[str, str]]:
    task_rows = [
        {"domain": domain, "task_id": task_id, "split": "train"}
        for domain, task_id in sorted(official_train_ids)
    ]
    expected_rows = split_critic_tasks(
        task_rows,
        seed=seed,
        dev_fraction=dev_fraction,
    )
    return Counter(
        (str(row["critic_split"]), str(row["domain"])) for row in expected_rows
    )


def _assert_full_coverage_split(
    rows: dict[str, list[dict]],
    identities: list[tuple[str, str]],
    official_train_ids: set[tuple[str, str]],
    *,
    seed: int,
    dev_fraction: float,
) -> None:
    observed = Counter(
        (str(row["critic_split"]), str(row["domain"]))
        for split_rows in rows.values()
        for row in split_rows
    )
    expected = _expected_split_counter(
        official_train_ids,
        seed=seed,
        dev_fraction=dev_fraction,
    )
    if observed != expected:
        raise ValueError(
            "Critic full-coverage fit requires the configured task-stratified "
            f"split over all three domains: expected={dict(expected)} "
            f"observed={dict(observed)}"
        )
    if set(identities) != official_train_ids:
        missing = sorted(official_train_ids - set(identities))
        extra = sorted(set(identities) - official_train_ids)
        raise ValueError(
            "Critic full-coverage rows must cover the official train tasks exactly: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def _assert_generic_split(rows: dict[str, list[dict]]) -> None:
    observed = Counter(
        (str(row["critic_split"]), str(row["domain"]))
        for split_rows in rows.values()
        for row in split_rows
    )
    missing = [
        f"{split}:{domain}"
        for split in ("train", "dev")
        for domain in SUPPORTED_DOMAINS
        if observed[(split, domain)] < 1
    ]
    if missing:
        raise ValueError(
            "Critic fit requires at least one train/dev episode per domain: "
            f"missing={missing}"
        )


def _load_episode_rows(
    *data_paths: Path,
    require_full_coverage: bool = False,
    split_seed: int = 42,
    dev_fraction: float = 0.2,
) -> tuple[dict, list[dict], list[dict]]:
    official_train_ids = _load_official_train_ids()
    rows: dict[str, list[dict]] = {"train": [], "dev": []}
    for row in iter_critic_rows(*data_paths, official_train_ids=official_train_ids):
        rows[str(row["critic_split"])].append(row)
    train_rows = rows["train"]
    if not train_rows:
        raise ValueError("Critic fit found no train rows")
    expected_provenance = {key: str(train_rows[0][key]) for key in PROVENANCE_FIELDS}
    all_rows = rows["train"] + rows["dev"]
    identities = [(str(row["domain"]), str(row["task_id"])) for row in all_rows]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Critic fit requires at most one episode per official task identity"
        )
    if require_full_coverage:
        _assert_full_coverage_split(
            rows,
            identities,
            official_train_ids,
            seed=split_seed,
            dev_fraction=dev_fraction,
        )
    else:
        _assert_generic_split(rows)
    observed = Counter(
        (str(row["critic_split"]), str(row["domain"]))
        for split_rows in rows.values()
        for row in split_rows
    )
    summary = {
        "status": "checked",
        "require_full_coverage": require_full_coverage,
        "split_seed": split_seed,
        "dev_fraction": dev_fraction,
        "data_paths": [str(path.expanduser().resolve()) for path in data_paths],
        "policy_provenance": expected_provenance,
        "reward_protocol": "official-binary-v1",
        "train_rows": len(rows["train"]),
        "dev_rows": len(rows["dev"]),
        "unique_tasks": len(set(identities)),
        "episodes_by_split_domain": {
            f"{split}:{domain}": int(count)
            for (split, domain), count in sorted(observed.items())
        },
    }
    return summary, rows["train"], rows["dev"]


def validate_tau2_critic_config(
    config: PPOConfig,
    *,
    train_rows: int | None = None,
) -> None:
    validate_config(config)
    if config.train_dataset.drop_last:
        raise ValueError("Tau2 critic production fit must keep the tail batch")
    if config.valid_dataset is None:
        raise ValueError("Tau2 critic validation requires a valid_dataset")
    largest_prompt_batch = max(
        config.train_dataset.batch_size,
        config.valid_dataset.batch_size,
    )
    if config.rollout.queue_size < 2 * largest_prompt_batch:
        raise ValueError(
            "Offline critic rollout.queue_size must cover the active replay batch "
            "plus one reserved consumer batch"
        )
    if config.gconfig.n_samples != 1:
        raise ValueError("Offline critic replay requires gconfig.n_samples=1")
    if config.actor.use_direct_dis_loss:
        raise ValueError("Offline critic fitting must not apply Direct DIS actor loss")
    if config.critic is None:
        raise ValueError("Tau2 critic fit requires a critic config")
    if (
        config.critic.loss_reduction != "token_mean"
        or config.critic.eps_clip is not None
    ):
        raise ValueError("Tau2 critic fit requires token_mean plain MSE")
    if train_rows is not None:
        expected_steps = math.ceil(train_rows / config.train_dataset.batch_size) * int(
            config.total_train_epochs
        )
        if config.total_train_steps != expected_steps:
            raise ValueError(
                "Tau2 critic fit requires total_train_steps to match "
                f"ceil(train_rows / batch_size) * epochs: expected={expected_steps} "
                f"observed={config.total_train_steps}"
            )
    if config.total_train_steps != config.num_critic_only_steps:
        raise ValueError(
            "Every offline critic step must stay inside critic-only warmup"
        )
    if config.recover.mode not in ("auto", "disabled"):
        raise ValueError("Offline critic recovery must be auto or explicitly disabled")


def fill_derived_train_steps(config: PPOConfig, *, train_rows: int) -> int:
    expected_steps = math.ceil(train_rows / config.train_dataset.batch_size) * int(
        config.total_train_epochs
    )
    if config.total_train_steps is None:
        config.total_train_steps = expected_steps
    if config.num_critic_only_steps in (None, 0):
        config.num_critic_only_steps = expected_steps
    return expected_steps


def resolve_training_snapshots(config: PPOConfig, *, snapshot_resolver=None) -> str:
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


def _as_batch_tensor(row: dict[str, Any], key: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([row[key]], dtype=dtype)


def _critic_loss_mask(row: dict[str, Any], *, zero_loss: bool = False) -> torch.Tensor:
    mask = torch.tensor([row["loss_mask"]], dtype=torch.bool)
    mask = torch.roll(mask, shifts=-1, dims=-1)
    mask[:, -1] = False
    if zero_loss:
        mask.zero_()
    return mask


def _value_target_row(
    row: dict[str, Any], *, zero_loss: bool = False
) -> dict[str, torch.Tensor]:
    width = len(row["input_ids"])
    reward = float(row["reward"])
    return {
        "input_ids": _as_batch_tensor(row, "input_ids", torch.long),
        "attention_mask": _as_batch_tensor(row, "attention_mask", torch.bool),
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
            "Critic validation prediction width does not match sealed row: "
            f"predicted={predicted.numel()} sealed={mask.numel()}"
        )
    predicted = predicted[mask].double()
    if predicted.numel() == 0:
        raise ValueError(
            f"Critic validation row has no action tokens: {row['episode_id']}"
        )
    if not torch.isfinite(predicted).all():
        raise ValueError("Non-finite critic validation predictions")
    target = torch.full_like(predicted, float(row["reward"]))
    error = predicted - target
    return {
        "source_id": str(row.get("source_id", row["episode_id"])),
        "episode_id": str(row["episode_id"]),
        "domain": str(row["domain"]),
        "task_id": str(row["task_id"]),
        "target": float(row["reward"]),
        "truncated": bool(row["truncated"]),
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
        grouped[str(row["domain"])].append(row)
    return dict(grouped)


def evaluate_tau2_critic(
    trainer: PPOTrainer,
    rows: list[dict[str, Any]],
    *,
    evidence_dir: Path,
    completed_step: int,
) -> dict[str, Any]:
    dp_size = _critic_dp_size(trainer)
    chunk_size = max(dp_size, int(os.environ.get("TAU2_CRITIC_VALIDATION_CHUNK", "4")))
    metric_rows: list[dict[str, Any]] = []
    padded_total = 0
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        padded, real_count = _pad_for_dp(chunk, dp_size)
        padded_total += len(padded)
        tensor_batch = [
            {
                "input_ids": _as_batch_tensor(row, "input_ids", torch.long),
                "attention_mask": _as_batch_tensor(row, "attention_mask", torch.bool),
                "loss_mask": _as_batch_tensor(row, "loss_mask", torch.bool),
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
        for row, value in zip(padded[:real_count], values[:real_count], strict=True):
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
        "grad_norm_scope": "dev_loss_backward_no_optimizer_step",
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


def install_tau2_validation_hook(
    trainer: PPOTrainer,
    dev_rows: list[dict[str, Any]],
    *,
    evidence_dir: Path,
    validation_steps: tuple[int, ...],
    best: dict[str, Any] | None,
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
                dev_rows,
                evidence_dir=evidence_dir,
                completed_step=completed,
            )
            data.update(tau2_validation_metrics(report))
            current = _selection_value(report)
            if best is None or current < float(best["metric_value"]):
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


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", required=True, action="append")
    parser.add_argument("--require-full-coverage", action="store_true")
    parser.add_argument("--critic-dev-fraction", type=float, default=0.2)
    parser.add_argument("--check-config", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    config, _ = load_expr_config(
        ["--config", args.config, *overrides],
        PPOConfig,
    )
    data_paths = tuple(Path(raw) for raw in args.episodes)
    data_summary, train_rows, dev_rows = _load_episode_rows(
        *data_paths,
        require_full_coverage=args.require_full_coverage,
        split_seed=config.seed,
        dev_fraction=args.critic_dev_fraction,
    )
    fill_derived_train_steps(config, train_rows=len(train_rows))
    validate_tau2_critic_config(config, train_rows=len(train_rows))
    validation_steps = _validation_steps(config, "default")
    save_steps = _cadence_steps(config.total_train_steps, config.saver.freq_steps)
    if args.check_config:
        print(
            OmegaConf.to_yaml(
                {
                    "config": OmegaConf.structured(config),
                    "data_summary": data_summary,
                    "actor_source": config.actor.path,
                    "train_rows": len(train_rows),
                    "dev_rows": len(dev_rows),
                    "require_full_coverage": args.require_full_coverage,
                    "critic_dev_fraction": args.critic_dev_fraction,
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
    write_json(
        evidence_dir / "resolved-config.json",
        {
            "config": dataclasses.asdict(config),
            "data_summary": data_summary,
            "actor_snapshot": actor_snapshot,
            "validation_steps": list(validation_steps),
            "save_steps": list(save_steps),
            "grad_norm_scope": "dev_loss_backward_no_optimizer_step",
        },
    )
    with PPOTrainer(
        config,
        train_dataset=Dataset.from_list(train_rows),
        valid_dataset=Dataset.from_list(dev_rows),
    ) as trainer:
        trainer.rollout.prepare_batch = functools.partial(
            trainer.rollout.prepare_batch, finite_epoch=True, fail_on_rejection=True
        )
        recovered_completed = (
            trainer.recover_info.last_step_info.next().global_step
            if trainer.recover_info is not None
            else 0
        )
        if trainer.recover_info is None:
            save_initial_critic(trainer)
        step0 = evaluate_tau2_critic(
            trainer,
            dev_rows,
            evidence_dir=evidence_dir,
            completed_step=recovered_completed,
        )
        best_path = evidence_dir / "best-validation.json"
        if best_path.exists():
            best = json.loads(best_path.read_text(encoding="utf-8"))
        else:
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
            dev_rows,
            evidence_dir=evidence_dir,
            validation_steps=validation_steps,
            best=best,
        )
        trainer.train(
            workflow="examples.tau2.critic_replay.Tau2CriticReplayWorkflow",
            workflow_kwargs={},
            eval_workflow="examples.tau2.critic_replay.Tau2CriticReplayWorkflow",
            eval_workflow_kwargs={},
        )
        write_json(
            evidence_dir / "summary.json",
            {
                "status": "complete",
                "data_summary": data_summary,
                "best_validation": json.loads(best_path.read_text(encoding="utf-8")),
            },
        )


if __name__ == "__main__":
    main(sys.argv[1:])
