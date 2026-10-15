# SPDX-License-Identifier: Apache-2.0
"""Uniform multi-pack PPO training and a fixed three-domain test set."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset

from examples.tau2 import benchmark_variant, task_pack
from examples.tau2.contracts import build_sampling_schedule
from examples.tau2.utils import Tau2PPOConfig


def build_ppo_train_dataset(
    config: Tau2PPOConfig, dataset_kwargs: dict[str, Any]
) -> Dataset:
    """Use official fixed A/R train plus packs; budgets cannot bias task coverage."""
    from examples.tau2.train import get_tau2_dataset

    selected = set(config.domains)
    if not {"airline", "retail"} <= selected:
        raise ValueError("PPO training requires Airline and Retail")
    if not config.train_task_packs:
        raise ValueError("PPO requires train_task_packs (AReaL pack v2)")
    packs = [str(Path(p).resolve(strict=True)) for p in config.train_task_packs]
    if len(set(packs)) != len(packs):
        raise ValueError("Duplicate task pack paths")
    config.train_task_packs = packs
    official_kwargs = {**dataset_kwargs, "domain": ("airline", "retail")}
    rows = [
        {**r, "task_pack": None}
        for r in get_tau2_dataset(**official_kwargs, split="train")
    ]
    for pack in packs:
        domains = tuple(task_pack.domains(pack))
        if not domains or not set(domains) <= selected:
            raise ValueError(
                f"Pack domains {domains} must be nonempty and selected in domains"
            )
        pack_rows = task_pack._task_pack_rows(
            pack, domains, provenance=dataset_kwargs["provenance"], seed=config.seed
        )
        # The source_id is also used for traces and retry identity. Include the
        # manifest-verified pack root, not only its domain-local task id.
        rows.extend(
            {**r, "source_id": f"pack:{pack}:{r['domain']}:{r['task_id']}"}
            for r in pack_rows
        )
    identities = [(r["domain"], r["task_id"], r["task_pack"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate tasks in PPO pool")
    counts = Counter(r["domain"] for r in rows)
    if set(counts) != selected:
        raise ValueError("Every selected training domain requires tasks")
    expected = {d: n * config.samples_per_task_per_epoch for d, n in counts.items()}
    if (
        config.domain_effective_episodes
        and config.domain_effective_episodes != expected
    ):
        raise ValueError(
            f"PPO budgets must repeat every task equally: expected {expected}"
        )
    config.domain_effective_episodes = expected
    scheduled = build_sampling_schedule(
        rows, domain_effective_episodes=expected, algorithm="ppo", seed=config.seed
    )
    if config.task_limit is not None and config.task_limit < len(scheduled):
        raise ValueError("task_limit must not truncate the PPO schedule")
    if config.train_dataset.drop_last:
        raise ValueError("PPO uniform coverage requires drop_last=false")
    random.Random(config.seed).shuffle(scheduled)
    return Dataset.from_list(scheduled)


def build_ppo_eval_dataset(
    config: Tau2PPOConfig, dataset_kwargs: dict[str, Any]
) -> Dataset:
    """Fixed-v3 Airline/Retail test and unmodified official Telecom test."""
    from examples.tau2.train import get_tau2_dataset

    domains = ("airline", "retail", "telecom")
    benchmark_variant.validate_task_files("fixed", domains)
    kwargs = {**dataset_kwargs, "domain": domains, "experiment_mode": "formal"}
    rows = list(get_tau2_dataset(**kwargs, split="test"))
    counts = Counter(r["domain"] for r in rows)
    if counts != {"airline": 20, "retail": 40, "telecom": 40}:
        raise ValueError(f"PPO evaluation requires 20/40/40 test tasks, got {counts}")
    if len({(r["domain"], r["task_id"]) for r in rows}) != 100:
        raise ValueError("Duplicate PPO evaluation tasks")
    return Dataset.from_list(rows)


def ppo_config_summary(config: Tau2PPOConfig, train: Dataset, valid: Dataset) -> dict:
    """Small readback of resolved settings, coverage, and global/policy clocks."""
    return {
        "model": config.actor.path,
        "critic_init": config.critic.path,
        "critic_head_init": (
            {"weight": 0, "bias": config.critic_head_bias_init}
            if config.critic.path == config.actor.path
            else "checkpoint unchanged"
        ),
        "parser": config.rollout.agent.tool_call_parser,
        "algorithm": config.algorithm,
        "loss": "clipped PPO (decoupled async correction)",
        "gamma": config.actor.discount,
        "actor_lambda": config.actor.gae_lambda,
        "critic_lambda": config.actor.critic_gae_lambda,
        "advantage_whitening": config.actor.adv_norm is not None,
        "actor_clip": config.actor.eps_clip,
        "actor_clip_upper": config.actor.eps_clip_higher,
        "value_clip": config.critic.eps_clip,
        "actor_lr": config.actor.optimizer.lr,
        "critic_lr": config.critic.optimizer.lr,
        "freeze_critic_attention": config.critic.freeze_critic_attention,
        "warmup_steps": config.num_critic_only_steps,
        "batch_episodes": config.train_batch_episodes,
        "samples_per_task_per_epoch": config.samples_per_task_per_epoch,
        "domain_budgets_per_epoch": config.domain_effective_episodes,
        "scheduled_episodes_per_epoch": len(train),
        "epochs": config.total_train_epochs,
        "global_train_steps": config.total_train_steps,
        "actor_update_steps": config.total_train_steps - config.num_critic_only_steps,
        "total_train_episodes": config.effective_episodes,
        "eval_tasks": dict(Counter(r["domain"] for r in valid)),
        "eval_repeats": config.eval_gconfig.n_samples,
        "eval_temperature": config.eval_gconfig.temperature,
        "eval_before_train": config.evaluator.eval_before_train,
        "eval_every_global_steps": config.evaluator.freq_steps,
        "saver_every_global_steps": config.saver.freq_steps,
        "eval_steps": sorted(
            {
                *([0] if config.evaluator.eval_before_train else []),
                *range(
                    config.evaluator.freq_steps,
                    config.total_train_steps + 1,
                    config.evaluator.freq_steps,
                ),
                config.total_train_steps,
            }
        ),
        "max_head_offpolicyness": config.rollout.max_head_offpolicyness,
        "recompute_logprob": config.actor.recompute_logprob,
        "use_decoupled_loss": config.actor.use_decoupled_loss,
    }
