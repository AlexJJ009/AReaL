# SPDX-License-Identifier: Apache-2.0
"""Native critic-only PPO driver with fixed-trajectory validation."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.infra.rpc.rtensor import RTensor

REQUIRED_JSONL_KEYS = {
    "source_id",
    "benchmark",
    "input_tokens",
    "output_tokens",
    "reward",
    "truncated",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_validation_records(
    path: Path, *, max_records: int | None = None
) -> list[dict]:
    records = []
    seen_by_source: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if max_records is not None and len(records) >= max_records:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_JSONL_KEYS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_no} missing keys: {sorted(missing)}")
            input_tokens = row["input_tokens"]
            output_tokens = row["output_tokens"]
            reward = float(row["reward"])
            if not input_tokens or not output_tokens:
                raise ValueError(f"{path}:{line_no} empty input/output tokens")
            if not math.isfinite(reward):
                raise ValueError(f"{path}:{line_no} non-finite reward")
            if reward not in (0.0, 1.0):
                raise ValueError(f"{path}:{line_no} expected binary reward")
            source_id = str(row["source_id"])
            sample_idx = int(row.get("sample_idx", seen_by_source.get(source_id, 0)))
            seen_by_source[source_id] = seen_by_source.get(source_id, 0) + 1
            records.append(
                {
                    "source_id": source_id,
                    "sample_idx": sample_idx,
                    "benchmark": str(row["benchmark"]),
                    "input_tokens": [int(x) for x in input_tokens],
                    "output_tokens": [int(x) for x in output_tokens],
                    "reward": reward,
                    "truncated": bool(row["truncated"]),
                }
            )
    if not records:
        raise ValueError(f"No validation records loaded from {path}")
    return records


def make_validation_item(record: dict) -> dict[str, Any]:
    ids = record["input_tokens"] + record["output_tokens"]
    prompt_len = len(record["input_tokens"])
    loss_mask = torch.zeros((1, len(ids)), dtype=torch.bool)
    loss_mask[:, prompt_len:] = True
    return {
        "source_id": record["source_id"],
        "sample_idx": int(record.get("sample_idx", 0)),
        "benchmark": record["benchmark"],
        "prompt_len": prompt_len,
        "response_len": len(record["output_tokens"]),
        "target": float(record["reward"]),
        "truncated": bool(record["truncated"]),
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.bool),
        "loss_mask": loss_mask,
    }


def pad_for_dp(
    items: list[dict[str, Any]], dp_size: int
) -> tuple[list[dict[str, Any]], int]:
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    padded = list(items)
    real_count = len(items)
    while len(padded) % dp_size:
        duplicate = dict(items[len(padded) % real_count])
        duplicate["_padding_duplicate"] = True
        padded.append(duplicate)
    return padded, real_count


def response_values(
    values: torch.Tensor, *, prompt_len: int, token_count: int
) -> torch.Tensor:
    start = prompt_len - 1
    end = token_count - 1
    if start < 0 or end <= start:
        raise ValueError(
            f"Invalid response value slice prompt_len={prompt_len} token_count={token_count}"
        )
    flat = values.reshape(-1).float()
    if flat.numel() < token_count:
        raise ValueError("Critic output is shorter than the input token sequence")
    selected = flat[start:end]
    if not torch.isfinite(selected).all():
        raise ValueError("Non-finite critic validation predictions")
    return selected


def aggregate_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_tokens = sum(int(row["n_tokens"]) for row in rows)
    if total_tokens <= 0:
        raise ValueError("Validation aggregate has no response tokens")
    sse = sum(float(row["sse"]) for row in rows)
    error_sum = sum(float(row["error_sum"]) for row in rows)
    sum_target = sum(float(row["target_sum"]) for row in rows)
    sum_target2 = sum(float(row["target2_sum"]) for row in rows)
    mean_target = sum_target / total_tokens
    var_target = sum_target2 / total_tokens - mean_target * mean_target
    mse = sse / total_tokens
    error_mean = error_sum / total_tokens
    residual_var = max(0.0, mse - error_mean * error_mean)
    explained_variance = None
    ev_defined = var_target > 0.0
    if ev_defined:
        explained_variance = 1.0 - residual_var / var_target
    rewards = [float(row["target"]) for row in rows]
    truncated = [bool(row["truncated"]) for row in rows]
    per_traj_mse = [float(row["sse"]) / int(row["n_tokens"]) for row in rows]
    nontruncated_rows = [row for row in rows if not row["truncated"]]
    nontruncated_tokens = sum(int(row["n_tokens"]) for row in nontruncated_rows)
    nontruncated_mse = None
    if nontruncated_tokens:
        nontruncated_mse = (
            sum(float(row["sse"]) for row in nontruncated_rows) / nontruncated_tokens
        )
    return {
        "n_trajectories": len(rows),
        "n_tokens": total_tokens,
        "mse": mse,
        "macro_mse": sum(per_traj_mse) / len(per_traj_mse),
        "bias": error_mean,
        "residual_var": residual_var,
        "target_mean": mean_target,
        "target_var": var_target,
        "explained_variance": explained_variance,
        "explained_variance_defined": ev_defined,
        "reward_mean": sum(rewards) / len(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "truncated_count": sum(int(x) for x in truncated),
        "truncated_rate": sum(int(x) for x in truncated) / len(truncated),
        "nontruncated_count": sum(int(not x) for x in truncated),
        "nontruncated_rate": sum(int(not x) for x in truncated) / len(truncated),
        "nontruncated_mse": nontruncated_mse,
    }


def aggregate_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    names = sorted({str(row[key]) for row in rows})
    return {
        name: aggregate_validation([row for row in rows if str(row[key]) == name])
        for name in names
    }


def _critic_dp_size(trainer: PPOTrainer) -> int:
    strategy = trainer.critic.parallel_strategy
    for attr in ("dp_size", "data_parallel_size"):
        value = getattr(strategy, attr, None)
        if value is not None:
            return int(value)
    return 1


def _tensor_only(item: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": item["input_ids"],
        "attention_mask": item["attention_mask"],
        "loss_mask": item["loss_mask"],
    }


def actor_probe(trainer: PPOTrainer, evidence_dir: Path, completed_step: int) -> dict:
    item = make_validation_item(
        {
            "source_id": "fixed-actor-probe",
            "benchmark": "fixed",
            "input_tokens": trainer.tokenizer.encode(
                "Question: What is 2+3? Answer:", add_special_tokens=False
            ),
            "output_tokens": trainer.tokenizer.encode(" 5", add_special_tokens=False),
            "reward": 1.0,
            "truncated": False,
        }
    )
    padded, real_count = pad_for_dp([item], _critic_dp_size(trainer))
    tensor_batch = [_tensor_only(entry) for entry in padded]
    remote_values = None
    try:
        remote_values = trainer.actor.compute_logp(tensor_batch)
        tensors = RTensor.localize(remote_values)
        values = [
            tensor.detach().float().cpu().reshape(-1).tolist()
            for tensor in tensors[:real_count]
        ]
    finally:
        if remote_values is None:
            trainer.actor.clear_batches(tensor_batch)
        else:
            trainer.actor.clear_batches(tensor_batch, remote_values)
    report = {
        "completed_step": completed_step,
        "sha256": sha_json(values),
        "values": values,
    }
    write_json(evidence_dir / f"actor-probe-step-{completed_step:06d}.json", report)
    return report


def evaluate_critic(
    trainer: PPOTrainer,
    records: list[dict],
    *,
    evidence_dir: Path,
    completed_step: int,
) -> dict[str, Any]:
    items = [make_validation_item(record) for record in records]
    dp_size = _critic_dp_size(trainer)
    chunk_size = int(os.environ.get("CRITIC_PRETRAIN_VALIDATION_CHUNK", "4"))
    chunk_size = max(int(dp_size), chunk_size)
    rows = []
    padded_total = 0
    for start in range(0, len(items), chunk_size):
        chunk = items[start : start + chunk_size]
        padded, real_count = pad_for_dp(chunk, dp_size)
        padded_total += len(padded)
        tensor_batch = [_tensor_only(item) for item in padded]
        remote_values = None
        try:
            remote_values = trainer.critic.compute_values(tensor_batch)
            tensors = RTensor.localize(remote_values)
        finally:
            if remote_values is None:
                trainer.critic.clear_batches(tensor_batch)
            else:
                trainer.critic.clear_batches(tensor_batch, remote_values)
        for item, values in zip(padded[:real_count], tensors[:real_count], strict=True):
            token_count = int(item["input_ids"].shape[1])
            predicted = response_values(
                values, prompt_len=int(item["prompt_len"]), token_count=token_count
            ).double()
            target = torch.full_like(predicted, float(item["target"]))
            error = predicted - target
            n_tokens = int(predicted.numel())
            rows.append(
                {
                    "source_id": item["source_id"],
                    "sample_idx": item["sample_idx"],
                    "benchmark": item["benchmark"],
                    "target": float(item["target"]),
                    "truncated": bool(item["truncated"]),
                    "prompt_len": int(item["prompt_len"]),
                    "response_len": int(item["response_len"]),
                    "n_tokens": n_tokens,
                    "prediction_mean": float(predicted.mean().item()),
                    "prediction_min": float(predicted.min().item()),
                    "prediction_max": float(predicted.max().item()),
                    "error_sum": float(error.sum().item()),
                    "sse": float(error.square().sum().item()),
                    "target_sum": float(target.sum().item()),
                    "target2_sum": float(target.square().sum().item()),
                }
            )
    summary = aggregate_validation(rows)
    payload = {
        "completed_step": completed_step,
        "recorded_ns": time.time_ns(),
        "dp_size": int(dp_size),
        "input_records": len(records),
        "padded_records": padded_total,
        "rows": rows,
        "summary": summary,
        "by_benchmark": aggregate_by_key(rows, "benchmark"),
        "by_reward": aggregate_by_key(rows, "target"),
        "by_truncation": aggregate_by_key(rows, "truncated"),
    }
    write_json(evidence_dir / f"validation-step-{completed_step:06d}.json", payload)
    return payload


def validation_metrics(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "critic_validation/completed_step": report["completed_step"],
        "critic_validation/mse": summary["mse"],
        "critic_validation/macro_mse": summary["macro_mse"],
        "critic_validation/bias": summary["bias"],
        "critic_validation/nontruncated_mse": summary["nontruncated_mse"],
        "critic_validation/nontruncated_rate": summary["nontruncated_rate"],
        "critic_validation/explained_variance": summary["explained_variance"],
        "critic_validation/explained_variance_defined": int(
            summary["explained_variance_defined"]
        ),
        "critic_validation/n_tokens": summary["n_tokens"],
        "critic_validation/truncated_rate": summary["truncated_rate"],
    }


def actor_update_metrics(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if "critic" not in key.lower()
        and any(
            marker in key.lower()
            for marker in (
                "update_successful",
                "optimizer_steps_since_init",
                "grad_norm",
            )
        )
    }


def install_validation_hook(
    trainer: PPOTrainer,
    records: list[dict],
    *,
    evidence_dir: Path,
    baseline_actor_sha: str,
) -> None:
    original_commit = trainer.stats_logger.commit

    def commit(epoch: int, step: int, global_step: int, data: dict[str, Any]) -> None:
        completed = global_step + 1
        actor_metrics = actor_update_metrics(data)
        if actor_metrics:
            raise RuntimeError(
                f"Actor optimizer emitted update metrics during critic-only step {completed}: "
                f"{actor_metrics}"
            )
        if completed % 5 == 0:
            probe = actor_probe(trainer, evidence_dir, completed)
            if probe["sha256"] != baseline_actor_sha:
                raise RuntimeError(
                    f"Actor fixed probe changed at step {completed}: "
                    f"{probe['sha256']} != {baseline_actor_sha}"
                )
            report = evaluate_critic(
                trainer, records, evidence_dir=evidence_dir, completed_step=completed
            )
            data.update(validation_metrics(report))
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
        data["critic_train/completed_step"] = completed
        original_commit(epoch, step, global_step, data)

    trainer.stats_logger.commit = commit


def validate_config(config: PPOConfig) -> None:
    if config.critic is None or not config.critic.is_critic:
        raise ValueError("critic_pretrain requires critic.is_critic=True")
    if config.num_critic_only_steps < 1:
        raise ValueError("critic_pretrain requires num_critic_only_steps > 0")
    if config.total_train_steps is None:
        raise ValueError("critic_pretrain requires explicit total_train_steps")
    if config.total_train_steps > config.num_critic_only_steps:
        raise ValueError("actor would update after critic-only window")
    if (
        config.actor.discount != 1.0
        or config.actor.critic_gae_lambda != 1.0
        or config.actor.reward_scaling != 1.0
        or config.actor.reward_bias != 0.0
        or config.actor.reward_norm is not None
        or config.actor.kl_ctl != 0.0
        or config.actor.overlong_reward_penalty
    ):
        raise ValueError(
            "Fixed binary MC validation requires matching raw binary MC training targets"
        )


def save_initial_critic(trainer: PPOTrainer) -> None:
    trainer.saver.save(
        trainer.critic,
        epoch=0,
        step=-1,
        global_step=-1,
        tokenizer=trainer.tokenizer,
        processor=trainer.processor,
        name="critic-initial",
        force=True,
    )


def write_wandb_binding(trainer: PPOTrainer, evidence_dir: Path) -> None:
    try:
        import wandb

        run = wandb.run
        if run is not None:
            run.define_metric("critic/*", step_metric="critic_train/completed_step")
            run.define_metric(
                "critic_validation/*",
                step_metric="critic_validation/completed_step",
            )
        payload = {
            "enabled": run is not None,
            "url": run.url if run is not None else None,
            "id": run.id if run is not None else None,
            "name": run.name if run is not None else None,
            "project": run.project
            if run is not None
            else trainer.config.stats_logger.wandb.project,
            "entity": run.entity
            if run is not None
            else trainer.config.stats_logger.wandb.entity,
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "enabled": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    write_json(evidence_dir / "wandb-binding.json", payload)


def run(config: PPOConfig, *, validation_jsonl: Path) -> None:
    validate_config(config)
    evidence_dir = Path(config.cluster.fileroot) / "critic-pretrain-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    records = load_validation_records(
        validation_jsonl,
        max_records=int(os.environ.get("CRITIC_PRETRAIN_MAX_VALIDATION_RECORDS", "0"))
        or None,
    )
    write_json(
        evidence_dir / "resolved-config.json",
        {
            "config": dataclasses.asdict(config),
            "validation_jsonl": str(validation_jsonl),
            "validation_records": len(records),
        },
    )
    dataset = load_from_disk(config.train_dataset.path)
    workflow = "areal.workflow.sao_math.AuditedMathWorkflow"
    workflow_kwargs = {
        "reward_fn": "areal.reward.math_prd.math_prd_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": False,
        "audit_dir": str(evidence_dir / "samples"),
    }
    with PPOTrainer(
        config,
        train_dataset=dataset[config.train_dataset.split],
        valid_dataset=dataset[config.valid_dataset.split],
    ) as trainer:
        write_wandb_binding(trainer, evidence_dir)
        trainer.rollout.prepare_batch = functools.partial(
            trainer.rollout.prepare_batch, finite_epoch=True, fail_on_rejection=True
        )
        trainer.train_dataloader.sampler.seed = config.seed
        recovered_completed = (
            trainer.recover_info.last_step_info.next().global_step
            if trainer.recover_info is not None
            else 0
        )
        if trainer.recover_info is None:
            save_initial_critic(trainer)
        step0 = evaluate_critic(
            trainer,
            records,
            evidence_dir=evidence_dir,
            completed_step=recovered_completed,
        )
        baseline_path = evidence_dir / "actor-probe-baseline.json"
        current_actor = actor_probe(trainer, evidence_dir, recovered_completed)
        if trainer.recover_info is None:
            baseline_actor = current_actor
            write_json(baseline_path, baseline_actor)
        else:
            if not baseline_path.exists():
                raise RuntimeError(f"Missing saved actor baseline: {baseline_path}")
            baseline_actor = json.loads(baseline_path.read_text())
            if current_actor["sha256"] != baseline_actor["sha256"]:
                raise RuntimeError(
                    f"Recovered actor probe differs from saved baseline: "
                    f"{current_actor['sha256']} != {baseline_actor['sha256']}"
                )
        trainer.stats_logger.commit(
            0,
            -1,
            -1,
            validation_metrics(step0),
        )
        install_validation_hook(
            trainer,
            records,
            evidence_dir=evidence_dir,
            baseline_actor_sha=baseline_actor["sha256"],
        )
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs={**workflow_kwargs, "gconfig": config.eval_gconfig},
        )
        write_json(
            evidence_dir / "summary.json",
            {
                "status": "complete",
                "initial_validation": step0["summary"],
                "completed_ns": time.time_ns(),
            },
        )


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-validation-records", type=int)
    args, overrides = parser.parse_known_args(argv)
    config_args = ["--config", args.config, *overrides]
    if args.max_steps is not None:
        config_args.append(f"total_train_steps={args.max_steps}")
    if args.max_validation_records is not None:
        os.environ["CRITIC_PRETRAIN_MAX_VALIDATION_RECORDS"] = str(
            args.max_validation_records
        )
    config, _ = load_expr_config(config_args, PPOConfig)
    run(config, validation_jsonl=Path(args.validation_jsonl))


if __name__ == "__main__":
    main(sys.argv[1:])
