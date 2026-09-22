# SPDX-License-Identifier: Apache-2.0
"""Native GPU qualification for PPO critic-only warmup and recovery.

This is intentionally an experiment driver rather than a second trainer.  The
native ``PPOTrainer`` owns rollout collection, optimizer steps, publication, and
checkpoint recovery; this file only records bounded evidence around that loop.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk

from examples.math.sao_ppo import verify_published_policy

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.infra.rpc.rtensor import RTensor
from areal.trainer.ppo.validation import verify_gamma_one_episodic_returns


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(_json_value(payload), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _probe_batch(tokenizer: Any) -> list[dict[str, torch.Tensor]]:
    ids = tokenizer.encode("Question: What is 2+3? Answer: 5", add_special_tokens=False)
    if len(ids) < 5:
        raise RuntimeError("Fixed qualification probe has too few tokens")
    loss_mask = torch.zeros((1, len(ids)), dtype=torch.bool)
    loss_mask[:, -4:] = True
    item = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.bool),
        "loss_mask": loss_mask,
    }
    # Keep one request per local FSDP replica.  This avoids an empty local
    # result when the probe is broadcast through a four-way data-parallel role.
    return [dict(item) for _ in range(4)]


def _local_tensors(result: Any) -> list[torch.Tensor]:
    localized = RTensor.localize(result)
    if torch.is_tensor(localized):
        return [localized.detach().float().cpu()]
    if not isinstance(localized, (list, tuple)):
        raise TypeError(
            f"Expected tensor list from native probe, got {type(localized)!r}"
        )
    tensors = []
    for item in localized:
        if not torch.is_tensor(item):
            raise TypeError(f"Expected tensor probe item, got {type(item)!r}")
        tensors.append(item.detach().float().cpu())
    return tensors


def _actor_probe(
    trainer: PPOTrainer, batch: list[dict[str, torch.Tensor]]
) -> dict[str, Any]:
    tensors = _local_tensors(trainer.actor.compute_logp(batch))
    values = [tensor.reshape(-1).tolist() for tensor in tensors]
    return {"values": values, "sha256": _sha256_json(values)}


def _critic_probe(
    trainer: PPOTrainer, batch: list[dict[str, torch.Tensor]]
) -> dict[str, Any]:
    tensors = _local_tensors(trainer.critic.compute_values(batch))
    values = [tensor.reshape(-1).tolist() for tensor in tensors]
    return {"values": values, "sha256": _sha256_json(values)}


def _max_abs_delta(before: dict[str, Any], after: dict[str, Any]) -> float:
    lhs = torch.tensor(before["values"], dtype=torch.float32)
    rhs = torch.tensor(after["values"], dtype=torch.float32)
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"Probe shape changed: {lhs.shape} -> {rhs.shape}")
    return float((lhs - rhs).abs().max().item())


def _metric_evidence(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "update_successful",
        "optimizer_steps_since_init",
        "grad_norm",
        "lr",
        "learning_rate",
    )
    return {
        key: _json_value(value)
        for key, value in data.items()
        if any(key.endswith(suffix) or suffix in key for suffix in keys)
    }


def _role_metrics(data: dict[str, Any], role: str) -> dict[str, Any]:
    selected = {}
    for key, value in data.items():
        if key.endswith("__count"):
            continue
        lowered = key.lower()
        is_critic = "critic" in lowered
        if (role == "critic") != is_critic:
            continue
        if any(
            marker in lowered
            for marker in (
                "update_successful",
                "optimizer_steps_since_init",
                "grad_norm",
                "/lr",
                "learning_rate",
            )
        ):
            selected[key] = value
    return selected


def _retain_step10_critic_checkpoint(trainer: PPOTrainer, evidence: Path) -> None:
    run_root = Path(trainer.config.cluster.fileroot)
    candidates = sorted(run_root.rglob("critic/recover_checkpoint"))
    if not candidates:
        raise RuntimeError("Native step-10 critic recover checkpoint was not found")
    source = candidates[-1]
    retained = run_root / "retained-critic-only-step10" / "critic-recover_checkpoint"
    if retained.exists():
        raise FileExistsError(f"Refusing to overwrite retained checkpoint: {retained}")
    shutil.copytree(source, retained)
    shutil.copy2(
        evidence / "resolved-config.json", retained.parent / "resolved-config.json"
    )
    recovery_info = source.parent.parent / "recover_info"
    shutil.copytree(recovery_info, retained.parent / "recover_info")
    _write_json(
        run_root / "retained-critic-only-step10" / "manifest.json",
        {
            "status": "functional_only",
            "completed_step": 10,
            "source": str(source),
            "retained": str(retained),
            "source_sha": os.environ.get("EXPECTED_SOURCE_COMMIT"),
            "config_sha256": hashlib.sha256(
                (evidence / "resolved-config.json").read_bytes()
            ).hexdigest(),
        },
    )


def _stage_name(args: argparse.Namespace) -> str:
    value = (
        args.stage
        or os.environ.get("SAO_WARMUP_STAGE")
        or os.environ.get("SAO_WARMUP_STOP_AFTER_STAGE")
        or "initial"
    ).lower()
    if value in {"initial", "stage1", "stop"}:
        return "initial"
    if value in {"resume", "stage2", "final"}:
        return "resume"
    raise ValueError(f"Unsupported warmup stage {value!r}")


def _install_evidence_hooks(
    trainer: PPOTrainer,
    evidence: Path,
    *,
    stage: str,
    probe_batch: list[dict[str, torch.Tensor]],
) -> None:
    start_step = (
        trainer.recover_info.last_step_info.next().global_step
        if trainer.recover_info is not None
        else 0
    )
    state: dict[str, Any] = {"step": start_step - 1, "probes": {}}
    steps_dir = evidence / "steps"
    probes_dir = evidence / "probes"
    returns_dir = evidence / "returns"
    for directory in (steps_dir, probes_dir, returns_dir):
        directory.mkdir(parents=True, exist_ok=True)

    original_advantages = trainer.actor.compute_advantages

    def compute_advantages(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = original_advantages(batch)
        step = state["step"] + 1
        state["step"] = step
        report = verify_gamma_one_episodic_returns(
            RTensor.localize(
                [
                    {
                        key: item[key]
                        for key in (
                            "values",
                            "returns",
                            "rewards",
                            "loss_mask",
                            "attention_mask",
                            "terminated",
                            "truncated",
                            "bootstrap_mask",
                        )
                    }
                    for item in result
                ]
            ),
            reward_scaling=trainer.config.actor.reward_scaling,
            reward_bias=trainer.config.actor.reward_bias,
            reward_clip=trainer.config.actor.reward_clip,
        )
        source_keys = RTensor.localize(
            [{"audit_source_key": item["audit_source_key"]} for item in result]
        )
        report["consumed_source_keys"] = [
            int(key)
            for item in source_keys
            for key in item["audit_source_key"].flatten().tolist()
        ]
        report.update({"stage": stage, "global_step": step})
        _write_json(returns_dir / f"{step + 1:06d}.json", report)
        return result

    trainer.actor.compute_advantages = compute_advantages
    original_commit = trainer.stats_logger.commit

    def commit(epoch: int, step: int, global_step: int, data: dict[str, Any]) -> None:
        completed = global_step + 1
        _write_json(evidence / "raw-metrics" / f"{completed:06d}.json", data)
        actor_metrics = _role_metrics(data, "actor")
        critic_metrics = _role_metrics(data, "critic")
        if not any("update_successful" in key for key in critic_metrics):
            raise RuntimeError(f"Missing critic optimizer evidence at step {completed}")
        if not any(
            "lr" in key.lower() or "learning_rate" in key.lower()
            for key in critic_metrics
        ):
            raise RuntimeError(f"Missing critic LR evidence at step {completed}")
        critic_successes = [
            float(value)
            for key, value in critic_metrics.items()
            if "update_successful" in key
        ]
        if any(value != 1.0 for value in critic_successes):
            raise RuntimeError(f"Critic optimizer update skipped at step {completed}")
        critic_only = completed <= trainer.config.num_critic_only_steps
        if critic_only and any("update_successful" in key for key in actor_metrics):
            raise RuntimeError(
                f"Actor optimizer updated during warmup at step {completed}"
            )
        if not critic_only and not any(
            "update_successful" in key for key in actor_metrics
        ):
            raise RuntimeError(f"Missing actor optimizer evidence at step {completed}")
        original_commit(epoch, step, global_step, data)
        if completed == 10:
            _retain_step10_critic_checkpoint(trainer, evidence)
        actor_probe = _actor_probe(trainer, probe_batch)
        critic_probe = _critic_probe(trainer, probe_batch)
        probe = {
            "stage": stage,
            "completed_step": completed,
            "recorded_ns": time.time_ns(),
            "actor": actor_probe,
            "critic": critic_probe,
        }
        _write_json(probes_dir / f"{completed:06d}.json", probe)
        _write_json(
            steps_dir / f"{completed:06d}.json",
            {
                "stage": stage,
                "completed_step": completed,
                "raw_global_step": global_step,
                "epoch": epoch,
                "epoch_step": step,
                "policy_version": data.get("ppo/policy_version"),
                "critic_only": data.get("ppo/critic_only"),
                "actor_optimizer": _json_value(actor_metrics),
                "critic_optimizer": _json_value(critic_metrics),
                "optimizer": _metric_evidence(data),
                "metrics": data,
                "probe_sha256": _sha256_json(probe),
            },
        )
        state["probes"][completed] = probe

    trainer.stats_logger.commit = commit


def _validate_summary(evidence: Path, *, stage: str) -> dict[str, Any]:
    probes = {
        int(path.stem): json.loads(path.read_text())
        for path in sorted((evidence / "probes").glob("*.json"))
        if path.stem.isdigit()
    }
    steps = {
        int(path.stem): json.loads(path.read_text())
        for path in sorted((evidence / "steps").glob("*.json"))
    }
    if not steps:
        raise RuntimeError("No native PPO step evidence was written")
    ordered_steps = sorted(steps)
    versions = [steps[step]["policy_version"] for step in ordered_steps]
    expected_versions = [0 if step <= 10 else step - 10 for step in ordered_steps]
    if stage == "resume" and versions != expected_versions:
        raise RuntimeError(
            f"Unexpected policy versions: observed={versions}, expected={expected_versions}"
        )
    if stage == "initial" and any(version != 0 for version in versions):
        raise RuntimeError(f"Policy version changed during initial warmup: {versions}")
    warmup_steps = [step for step in ordered_steps if step <= 10]
    start_probe_path = evidence / "probes" / "stage-start.json"
    if not start_probe_path.exists():
        raise RuntimeError(f"Missing stage-start probe: {start_probe_path}")
    start_probe = json.loads(start_probe_path.read_text())
    if warmup_steps:
        baseline = start_probe["actor"]
        warmup_deltas = {
            step: _max_abs_delta(baseline, probes[step]["actor"])
            for step in warmup_steps
        }
        if any(delta != 0.0 for delta in warmup_deltas.values()):
            raise RuntimeError(
                f"Actor changed during critic-only warmup: {warmup_deltas}"
            )
    joint_steps = [step for step in ordered_steps if step > 10]
    critic_delta = None
    if joint_steps and 10 in probes:
        critic_delta = _max_abs_delta(
            probes[10]["critic"], probes[joint_steps[0]]["critic"]
        )
        if critic_delta <= 1e-7:
            raise RuntimeError(
                f"Critic fixed probe did not change after joint update: {critic_delta}"
            )
    restore_probe_delta = None
    previous_stage = evidence.parent / "initial" / "probes" / "000005.json"
    if stage == "resume" and previous_stage.exists():
        previous = json.loads(previous_stage.read_text())
        restore_probe_delta = {
            "actor": _max_abs_delta(previous["actor"], start_probe["actor"]),
            "critic": _max_abs_delta(previous["critic"], start_probe["critic"]),
        }
        if restore_probe_delta["actor"] != 0.0 or restore_probe_delta["critic"] != 0.0:
            raise RuntimeError(
                f"Actor probe changed across native recovery: {restore_probe_delta}"
            )
    summary = {
        "stage": stage,
        "steps": ordered_steps,
        "policy_versions": versions,
        "warmup_actor_exact_deltas": (warmup_deltas if warmup_steps else {}),
        "first_joint_critic_max_abs_delta": critic_delta,
        "restore_probe_delta": restore_probe_delta,
        "return_reports": len(list((evidence / "returns").glob("*.json"))),
        "native_recovery_state_present": (
            any((evidence.parent.parent / "checkpoints").rglob("trainer_state.json"))
        ),
    }
    _write_json(evidence / f"summary-{stage}.json", summary)
    return summary


def _run_stage(config: PPOConfig, *, stage: str, evidence: Path) -> dict[str, Any]:
    if stage == "initial":
        config.total_train_steps = int(os.environ.get("SAO_WARMUP_INITIAL_STEPS", "5"))
    else:
        config.total_train_steps = int(os.environ.get("SAO_WARMUP_FINAL_STEPS", "12"))
    if config.total_train_steps < 1:
        raise ValueError("total_train_steps must be positive")

    dataset = load_from_disk(config.train_dataset.path)
    train_dataset = dataset[config.train_dataset.split]
    valid_dataset = dataset[config.valid_dataset.split]
    stage_evidence = evidence / stage
    stage_evidence.mkdir(parents=True, exist_ok=True)
    _write_json(
        stage_evidence / "resolved-config.json",
        {
            "stage": stage,
            "source_sha": os.environ.get("EXPECTED_SOURCE_COMMIT"),
            "config": dataclasses.asdict(config),
        },
    )
    workflow = "areal.workflow.sao_math.AuditedMathWorkflow"
    workflow_kwargs = {
        "reward_fn": "areal.reward.math_prd.math_prd_reward_fn",
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": False,
        "audit_dir": str(stage_evidence / "samples"),
    }
    eval_kwargs = {**workflow_kwargs, "gconfig": config.eval_gconfig}
    with PPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        if stage == "resume" and trainer.recover_info is None:
            raise RuntimeError("Resume stage did not load the native step-5 checkpoint")
        trainer.rollout.prepare_batch = functools.partial(
            trainer.rollout.prepare_batch, finite_epoch=True, fail_on_rejection=True
        )
        trainer.train_dataloader.sampler.seed = config.seed
        probe_batch = _probe_batch(trainer.tokenizer)
        _install_evidence_hooks(
            trainer, stage_evidence, stage=stage, probe_batch=probe_batch
        )
        stage_start = {
            "stage": stage,
            "completed_step": (
                trainer.recover_info.last_step_info.next().global_step
                if trainer.recover_info is not None
                else 0
            ),
            "actor": _actor_probe(trainer, probe_batch),
            "critic": _critic_probe(trainer, probe_batch),
        }
        _write_json(stage_evidence / "probes" / "stage-start.json", stage_start)
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_kwargs,
        )
        if stage == "resume":
            verify_published_policy(trainer, stage_evidence, 12)
    return _validate_summary(stage_evidence, stage=stage)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("initial", "resume"))
    args, hydra_args = parser.parse_known_args(argv)
    config, _ = load_expr_config(["--config", args.config, *hydra_args], PPOConfig)
    run_root = Path(config.cluster.fileroot)
    evidence = run_root / "warmup-evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    stage = _stage_name(args)
    summary = _run_stage(config, stage=stage, evidence=evidence)
    _write_json(
        evidence / "latest.json",
        {"stage": stage, "completed_ns": time.time_ns(), "summary": summary},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
