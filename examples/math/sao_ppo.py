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

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.infra.rpc.rtensor import RTensor
from areal.utils import logging
from areal.utils.network import format_hostport

logger = logging.getLogger("SaoPpo")


def validate_contract(config: PPOConfig, *, preflight: bool = False) -> None:
    expected = {
        "actor.use_decoupled_loss": (config.actor.use_decoupled_loss, False),
        "actor.recompute_logprob": (config.actor.recompute_logprob, False),
        "actor.rejection_sampling": (config.actor.rejection_sampling, None),
        "actor.reward_norm": (config.actor.reward_norm, None),
        "actor.eps_clip": (config.actor.eps_clip, 0.2),
        "actor.eps_clip_higher": (config.actor.eps_clip_higher, None),
        "actor.kl_ctl": (config.actor.kl_ctl, 0.0),
        "actor.use_sapo_loss": (config.actor.use_sapo_loss, False),
        "actor.use_cispo_loss": (config.actor.use_cispo_loss, False),
        "actor.overlong_reward_penalty": (config.actor.overlong_reward_penalty, False),
        "actor.importance_sampling_level": (
            config.actor.importance_sampling_level,
            "token",
        ),
        "dynamic_bs": (config.dynamic_bs, False),
        "gconfig.reward_normalization": (config.gconfig.reward_normalization, False),
        "train_dataset.drop_last": (config.train_dataset.drop_last, False),
    }
    if not preflight:
        expected.update(
            {
                "total_train_epochs": (config.total_train_epochs, 1),
                "total_train_steps": (config.total_train_steps, None),
                "gconfig.max_new_tokens": (config.gconfig.max_new_tokens, 8192),
                "gconfig.n_samples": (config.gconfig.n_samples, 4),
                "eval_gconfig.n_samples": (config.eval_gconfig.n_samples, 4),
                "saver.freq_steps": (config.saver.freq_steps, 20),
                "evaluator.freq_steps": (config.evaluator.freq_steps, 20),
            }
        )
    mismatches = {
        key: values for key, values in expected.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"PPO contract mismatch (observed, expected): {mismatches}")
    if config.critic is None or not config.critic.is_critic:
        raise ValueError("A trainable scalar critic is required")
    if config.critic.path != config.actor.path:
        raise ValueError("Actor and critic must share the pinned Base checkpoint")
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
        torch.testing.assert_close(a, b, rtol=0, atol=0.15)
    (evidence / f"published-policy-{step}.json").write_text(
        json.dumps(
            {
                "passed": True,
                "version": trainer.rollout.get_version(),
                "token_ids": ids,
                "scored_token_ids": ids[-target_len:],
                "replicas": rows,
            },
            indent=2,
        )
        + "\n"
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


def install_audit_hooks(trainer, evidence: Path, *, preflight: bool = False) -> None:
    """Keep native PPO execution; record consumed IDs and completed RPC spans."""
    consumed_dir = evidence / "consumed"
    consumed_dir.mkdir(exist_ok=True)
    start_step = (
        trainer.recover_info.last_step_info.next().global_step
        if trainer.recover_info is not None
        else 0
    )
    state = {"step": start_step, "spans": []}
    original_prepare = trainer.actor.prepare_batch

    def prepare(*args, **kwargs):
        batch = original_prepare(*args, **kwargs)
        state["step"] += 1
        state["spans"] = []
        keys = ("audit_source_key", "audit_task_id", "audit_sample_idx")
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
    for role in ("actor", "critic"):
        engine = getattr(trainer, role)
        original_update = engine.ppo_update

        def update(*args, _fn=original_update, _role=role, **kwargs):
            if not preflight and _role == "actor" and state["step"] <= 5:
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
        completed_step = global_step + 1
        roles = check_step_metrics(data, completed_step, completed_step - start_step)
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
    config, _ = load_expr_config(args, PPOConfig)
    preflight = os.environ.get("SAO_PREFLIGHT", "0") == "1"
    validate_contract(config, preflight=preflight)
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

    with PPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
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
        install_audit_hooks(trainer, evidence, preflight=preflight)
        if config.evaluator.eval_before_train and trainer.recover_info is None:
            # Consume only the initial trigger; do not advance step20 cadence.
            trainer.evaluator.freq_ctl.check(epochs=0, steps=0)
            trainer._evaluate_fn(workflow, eval_kwargs)
        trainer.train(
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow,
            eval_workflow_kwargs=eval_kwargs,
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
