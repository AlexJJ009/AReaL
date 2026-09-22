# SPDX-License-Identifier: Apache-2.0

"""Critic-first batch updates using production engine operations."""

import copy
import math
from dataclasses import replace
from typing import Any

import torch

from areal.api.io_struct import FinetuneSpec
from areal.utils import stats_tracker


def critic_optimizer_spec(spec: FinetuneSpec, updates: int) -> FinetuneSpec:
    """Budget scheduler steps for the critic's repeated dataset passes."""
    return replace(spec, total_train_epochs=spec.total_train_epochs * max(updates, 1))


def summarize_updates(stats: list[dict[str, float]]) -> dict[str, float]:
    """Report engine-confirmed optimizer work, never infer success from calls."""
    successful = sum(s.get("update_successful", 0) == 1 for s in stats)
    effective = sum(
        s.get("update_successful", 0) == 1
        and math.isfinite(s.get("grad_norm", float("nan")))
        and s.get("grad_norm", 0) > 0
        and s.get("lr", 0) > 0
        for s in stats
    )
    return {
        "attempted": float(len(stats)),
        "successful": float(successful),
        "skipped": float(len(stats) - successful),
        "effective": float(effective),
        "nonfinite": float(
            sum(not math.isfinite(s.get("grad_norm", float("nan"))) for s in stats)
        ),
    }


def require_single_update(report: Any, role: str) -> None:
    """Controller tensor dispatch repeats each DP report per assigned row."""
    reports = report if isinstance(report, list) else [report]
    if not reports:
        raise RuntimeError(f"Missing {role} optimizer report")
    for item in reports:
        expected = {"attempted": 1.0, "successful": 1.0, "skipped": 0.0}
        if (
            not isinstance(item, dict)
            or any(item.get(key) != value for key, value in expected.items())
            or item.get("nonfinite", 0) != 0
        ):
            raise RuntimeError(f"{role} did not complete one optimizer update: {item}")


def _effective_updates(report: Any) -> float:
    reports = report if isinstance(report, list) else [report]
    return min(item["effective"] for item in reports)


def update_critic_before_actor(
    actor: Any,
    critic: Any,
    rollout_batch: list[dict[str, Any]],
    critic_batch: list[dict[str, Any]],
    updates: int,
) -> dict[str, Any]:
    """Fit fixed pre-update targets, refresh values, then update actor once.

    Inputs are one episode per row with explicit terminated/truncated metadata.
    The caller has computed initial values/targets and waited for checkpoint
    staging. Policy publication happens only on return. Microbatch accumulation
    is inside the engine; each update call must perform one full optimizer step.
    """
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("critic updates must be a positive integer")
    if not rollout_batch or any(
        "terminated" not in row or "truncated" not in row for row in rollout_batch
    ):
        raise ValueError("Critic-first updates require explicit episode boundaries")
    raw = copy.deepcopy(rollout_batch)
    fixed = copy.deepcopy(critic_batch)
    for row in fixed:
        # Controller mode carries immutable RTensor handles. Targets are already
        # detached at the worker's GAE boundary; don't fetch them onto controller.
        for key in ("returns", "values"):
            if isinstance(row[key], torch.Tensor):
                row[key] = row[key].detach().clone()
    critic_reports = []
    for _ in range(updates):
        report = critic.ppo_update(copy.deepcopy(fixed))
        require_single_update(report, "critic")
        critic_reports.append(report)
        critic.step_lr_scheduler()
    refreshed = critic.compute_values(raw)
    if len(refreshed) != len(raw):
        raise RuntimeError("Critic returned the wrong number of value trajectories")
    for row, value in zip(raw, refreshed):
        row["values"] = value.detach() if isinstance(value, torch.Tensor) else value
    actor_batch = actor.compute_advantages(raw)
    if len(actor_batch) != len(fixed):
        raise RuntimeError("Actor returned the wrong number of advantage trajectories")
    for row, target in zip(actor_batch, fixed):
        row["returns"] = target["returns"]
    actor_report = actor.ppo_update(actor_batch)
    require_single_update(actor_report, "actor")
    actor.step_lr_scheduler()
    with stats_tracker.scope("sao_updates"):
        stats_tracker.scalar(
            critic_attempted=float(updates),
            critic_successful=float(updates),
            critic_effective=sum(_effective_updates(r) for r in critic_reports),
            actor_attempted=1.0,
            actor_successful=1.0,
            actor_effective=_effective_updates(actor_report),
        )
    return {"critic": critic_reports, "actor": actor_report}
