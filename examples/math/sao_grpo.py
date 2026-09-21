# SPDX-License-Identifier: Apache-2.0
"""Run the SAO math dataset through AReaL's native official GRPO recipe."""

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

from datasets import load_from_disk
from omegaconf import OmegaConf

from areal import PPOTrainer
from areal.api.cli_args import (
    GRPOConfig,
    load_expr_config,
    parse_cli_args,
    to_structured_cfg,
)
from areal.infra.rpc.rtensor import RTensor
from areal.utils import logging
from areal.utils.lr_scheduler import get_num_warmup_steps

logger = logging.getLogger("SaoGrpo")


def validate_contract(config: GRPOConfig, *, preflight: bool = False) -> None:
    expected = {
        "actor.optimizer.lr": (config.actor.optimizer.lr, 6.0e-6),
        "actor.optimizer.weight_decay": (config.actor.optimizer.weight_decay, 0.017),
        "actor.optimizer.warmup_steps": (config.actor.optimizer.warmup_steps, 5),
        "actor.optimizer.warmup_steps_proportion": (
            config.actor.optimizer.warmup_steps_proportion,
            0.001,
        ),
        "actor.optimizer.lr_scheduler_type": (
            config.actor.optimizer.lr_scheduler_type,
            "constant",
        ),
        "actor.eps_clip": (config.actor.eps_clip, 0.4),
        "actor.reward_scaling": (config.actor.reward_scaling, 10.0),
        "actor.reward_bias": (config.actor.reward_bias, -0.5),
        "actor.use_decoupled_loss": (config.actor.use_decoupled_loss, True),
        "actor.recompute_logprob": (config.actor.recompute_logprob, True),
        "actor.kl_ctl": (config.actor.kl_ctl, 0.0),
        "actor.ppo_n_minibatches": (config.actor.ppo_n_minibatches, 1),
        "actor.use_sapo_loss": (config.actor.use_sapo_loss, False),
        "actor.use_cispo_loss": (config.actor.use_cispo_loss, False),
        "actor.overlong_reward_penalty": (
            config.actor.overlong_reward_penalty,
            False,
        ),
        "dynamic_bs": (config.dynamic_bs, False),
        "gconfig.reward_normalization": (config.gconfig.reward_normalization, False),
        "train_dataset.drop_last": (config.train_dataset.drop_last, True),
        "sglang.attention_backend": (config.sglang.attention_backend, "flashinfer"),
    }
    if not preflight:
        expected.update(
            {
                "total_train_epochs": (config.total_train_epochs, 1),
                "total_train_steps": (config.total_train_steps, None),
                "gconfig.max_new_tokens": (config.gconfig.max_new_tokens, 8192),
                "gconfig.max_tokens": (config.gconfig.max_tokens, 9216),
                "gconfig.n_samples": (config.gconfig.n_samples, 4),
                "eval_gconfig.n_samples": (config.eval_gconfig.n_samples, 4),
                "saver.freq_steps": (config.saver.freq_steps, 20),
                "evaluator.freq_steps": (config.evaluator.freq_steps, 20),
                "evaluator.eval_before_train": (
                    config.evaluator.eval_before_train,
                    True,
                ),
            }
        )
    mismatches = {
        key: values for key, values in expected.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"GRPO contract mismatch (observed, expected): {mismatches}")
    if config.critic is not None:
        raise ValueError("SAO GRPO review candidate must not configure a critic")
    if config.ref is not None:
        raise ValueError("SAO GRPO review candidate keeps KL at 0 and no ref model")
    if config.teacher is not None:
        raise ValueError("SAO GRPO review candidate must not configure a teacher")
    if config.actor.path != os.environ.get("SAO_MODEL_PATH"):
        raise ValueError("actor.path must resolve from SAO_MODEL_PATH")
    if config.train_dataset.path != os.environ.get("SAO_DATA_PATH"):
        raise ValueError("train_dataset.path must resolve from SAO_DATA_PATH")
    rejection = config.actor.rejection_sampling
    if rejection is None or (
        rejection.level,
        rejection.action,
        rejection.metric,
        rejection.upper,
        rejection.lower,
    ) != ("token", "mask", "ratio", 5.0, None):
        raise ValueError("Expected official GSM8K token ratio rejection above 5")
    reward_norm = config.actor.reward_norm
    if reward_norm is None or (
        reward_norm.mean_level,
        reward_norm.std_level,
        reward_norm.group_size,
    ) != ("group", "group", config.gconfig.n_samples):
        raise ValueError("Expected group reward_norm with group_size == n_samples")
    adv_norm = config.actor.adv_norm
    if adv_norm is None or (adv_norm.mean_level, adv_norm.std_level) != (
        "batch",
        "batch",
    ):
        raise ValueError("Expected batch-level advantage normalization")


def _check_actor_step_metrics(
    data: dict, completed_step: int, updates_since_init: int | None = None
) -> dict:
    grads = {k: v for k, v in data.items() if k.endswith("grad_norm")}
    successes = {k: v for k, v in data.items() if k.endswith("update_successful")}
    counts = {k: v for k, v in data.items() if k.endswith("optimizer_steps_since_init")}
    if not grads or not successes or not counts:
        raise RuntimeError(f"Missing actor optimizer evidence at step {completed_step}")
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in grads.values()):
        raise RuntimeError(
            f"Actor has a non-finite/zero gradient at step {completed_step}"
        )
    if any(float(v) != 1 for v in successes.values()):
        raise RuntimeError(
            f"Actor skipped an optimizer update at step {completed_step}"
        )
    expected_count = (
        completed_step if updates_since_init is None else updates_since_init
    )
    if any(float(v) != expected_count for v in counts.values()):
        raise RuntimeError(
            f"Actor optimizer count mismatch: {counts}, expected {expected_count}"
        )
    for key, value in data.items():
        if (
            "loss" in key
            and isinstance(value, (float, int))
            and not math.isfinite(value)
        ):
            raise RuntimeError(f"Non-finite metric {key} at step {completed_step}")
    return {
        "grad_norm": grads,
        "update_successful": successes,
        "optimizer_steps": counts,
    }


def write_step_count_evidence(
    config: GRPOConfig,
    train_dataset,
    valid_dataset,
    evidence: Path,
    *,
    preflight: bool = False,
) -> dict:
    evidence.mkdir(parents=True, exist_ok=True)
    train_rows = len(train_dataset)
    train_batch_size = config.train_dataset.batch_size
    if config.train_dataset.drop_last:
        expected_steps = train_rows // train_batch_size
        consumed_prompts = expected_steps * train_batch_size
    else:
        expected_steps = math.ceil(train_rows / train_batch_size)
        consumed_prompts = train_rows
    dropped_prompts = train_rows - consumed_prompts
    expected_trajectories = consumed_prompts * config.gconfig.n_samples
    actual_warmup_steps = get_num_warmup_steps(config.actor.optimizer, expected_steps)
    payload = {
        "train_dataset_rows": train_rows,
        "valid_dataset_rows": len(valid_dataset),
        "train_prompts_per_step": train_batch_size,
        "samples_per_prompt": config.gconfig.n_samples,
        "expected_optimizer_steps": expected_steps,
        "expected_consumed_prompts": consumed_prompts,
        "dropped_tail_prompts": dropped_prompts,
        "expected_train_trajectories": expected_trajectories,
        "last_step_prompt_count": train_batch_size if expected_steps else 0,
        "last_step_trajectory_count": train_batch_size * config.gconfig.n_samples
        if expected_steps
        else 0,
        "drop_last": config.train_dataset.drop_last,
        "group_semantics": "Only complete prompt groups are consumed; no tail is split into singleton GRPO trajectories.",
        "optimizer_warmup_steps_proportion": (
            config.actor.optimizer.warmup_steps_proportion
        ),
        "resolved_warmup_steps": actual_warmup_steps,
        "warmup_note": (
            "Official GSM8K GRPO keeps warmup_steps_proportion=0.001. "
            "User-approved warmup_steps=5 takes precedence over that proportion."
        ),
    }
    (evidence / "step-counts.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if not preflight and valid_dataset is not None and len(valid_dataset) != 700:
        raise ValueError(
            f"Expected all 700 evaluation problems, got {len(valid_dataset)}"
        )
    if actual_warmup_steps != 5:
        raise ValueError("Expected fixed warmup to resolve to 5 optimizer updates")
    return payload


def install_audit_hooks(trainer, evidence: Path) -> None:
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
            raise RuntimeError("Empty consumed GRPO batch")
        (consumed_dir / f"{state['step']}.json").write_text(
            json.dumps(records) + "\n", encoding="utf-8"
        )
        return batch

    trainer.actor.prepare_batch = prepare
    original_update = trainer.actor.ppo_update

    def update(*args, **kwargs):
        start_ns = time.time_ns()
        result = original_update(*args, **kwargs)
        state["spans"].append(
            {"role": "actor", "started_ns": start_ns, "completed_ns": time.time_ns()}
        )
        return result

    trainer.actor.ppo_update = update
    original_commit = trainer.stats_logger.commit

    def commit(epoch, step, global_step, data):
        completed_step = global_step + 1
        actor_update = _check_actor_step_metrics(
            data, completed_step, completed_step - start_step
        )
        if data.get("ppo_actor/explicit_termination") != 1:
            raise RuntimeError(
                "GRPO update did not consume explicit episode termination metadata"
            )
        step_dir = evidence / "steps"
        step_dir.mkdir(exist_ok=True)
        payload = {
            "completed_step": completed_step,
            "raw_global_step": global_step,
            "epoch": epoch,
            "epoch_step": step,
            "recorded_ns": time.time_ns(),
            "actor_update": actor_update,
            "update_spans": state["spans"],
            "published_version": trainer.rollout.get_version(),
            "consumed_sha256": hashlib.sha256(
                (consumed_dir / f"{completed_step}.json").read_bytes()
            ).hexdigest(),
            "metrics": data,
        }
        if payload["published_version"] != completed_step:
            raise RuntimeError("Rollout version was not published after actor update")
        (step_dir / f"{completed_step}.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )
        original_commit(epoch, step, global_step, data)

    trainer.stats_logger.commit = commit


def main(args: list[str]) -> None:
    config_only = "--check-config" in args
    args = [x for x in args if x != "--check-config"]
    if config_only:
        cfg, _ = parse_cli_args(args)
        cfg = to_structured_cfg(cfg, GRPOConfig)
        config = OmegaConf.to_object(cfg)
        assert isinstance(config, GRPOConfig)
    else:
        config, _ = load_expr_config(args, GRPOConfig)
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
    step_counts = write_step_count_evidence(
        config, train_dataset, valid_dataset, evidence, preflight=preflight
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
                    "expected_optimizer_steps": step_counts["expected_optimizer_steps"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if len(trainer.train_dataloader) != step_counts["expected_optimizer_steps"]:
            raise ValueError(
                "Dataloader step count changed after trainer construction: "
                f"{len(trainer.train_dataloader)} vs {step_counts['expected_optimizer_steps']}"
            )
        install_audit_hooks(trainer, evidence)
        if config.evaluator.eval_before_train and trainer.recover_info is None:
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
                    "finished_ns": time.time_ns(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
