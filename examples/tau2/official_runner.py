# SPDX-License-Identifier: Apache-2.0
"""Official ``tau2 run`` scheduling semantics on an AReaL rollout controller.

The episode itself is the pinned official orchestrator and evaluator
(``examples/tau2/agent.py``). This module owns what ``tau2 run`` does above one
episode, so offline and in-training evaluation share one definition:

- ``tau2/runner/batch.py``: each (task, trial) pair is one episode.
- ``tau2/runner/progress.py``: a failed attempt is replayed from scratch on any
  exception, up to ``DEFAULT_RETRY_ATTEMPTS`` more times. The replay happens in
  ``OpenAIProxyWorkflow`` with a fresh proxy session per attempt
  (``Tau2AgentWorkflow.should_retry_episode``). Once retries are exhausted the
  episode becomes an ``INFRASTRUCTURE_ERROR`` placeholder; the run continues.
- ``tau2/metrics/agent_metrics.py``: infrastructure errors are excluded before
  ``avg_reward`` and ``pass^k`` are computed, and their count is reported.

AReaL only delivers results that carry policy tokens. An episode the user ended
before any policy call, or one whose retries were exhausted, comes back as a
bare rejection; the episode's own attempt record tells the two apart.

The controller contract is ``submit(row, workflow, workflow_kwargs, ...)`` and
``dispatcher.wait_results(1, timeout=None, with_task_ids=True)``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from tau2.data_model.simulation import TerminationReason
from tau2.metrics.agent_metrics import is_successful, pass_hat_k

from examples.tau2.utils import trajectory_record_dir

from areal.infra.rpc.rtensor import RTensor

WORKFLOW = "examples.tau2.agent.Tau2AgentWorkflow"
COMPLETED = "completed"
INFRASTRUCTURE_ERROR = TerminationReason.INFRASTRUCTURE_ERROR.value


def expand_trials(tasks: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    """One row per (task, trial), with a run-unique episode identity."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = []
    for task in tasks:
        for trial in range(repeats):
            row = dict(task)
            row["trial"] = trial
            row["source_id"] = f"{row['domain']}:{row['task_id']}:trial{trial}"
            row["attempt_id"] = row["source_id"]
            rows.append(row)
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluation source IDs")
    return rows


def episode_workflow_kwargs(
    workflow_kwargs: dict[str, Any], trajectory_root: Path
) -> dict[str, Any]:
    """Bind evaluation outcome rules and this run's attempt-record directory."""

    return {
        **workflow_kwargs,
        "evaluation": True,
        "trajectory_dir": str(trajectory_root),
    }


def iter_episode_results(
    controller: Any,
    rows: list[dict[str, Any]],
    *,
    workflow_kwargs: dict[str, Any],
    max_inflight: int,
    workflow: str = WORKFLOW,
) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Keep ``max_inflight`` episodes running; yield each as soon as it ends.

    A rejected episode is yielded with ``None``; it stays attributable because
    results are returned with their task IDs.
    """

    if max_inflight <= 0:
        raise ValueError("Evaluation max_concurrent_rollouts must be positive")
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluation source IDs")
    pending: dict[int, dict[str, Any]] = {}
    completed: set[int] = set()
    next_row = 0
    while pending or next_row < len(rows):
        while next_row < len(rows) and len(pending) < max_inflight:
            row = rows[next_row]
            task_id = controller.submit(
                row,
                workflow=workflow,
                workflow_kwargs=workflow_kwargs,
                group_size=1,
                is_eval=True,
                reward_normalization=False,
                drop_incomplete_group=False,
            )
            if task_id in pending or task_id in completed:
                raise RuntimeError(
                    "Duplicate rollout task IDs returned by controller.submit"
                )
            pending[task_id] = row
            next_row += 1
        results = controller.dispatcher.wait_results(
            1, timeout=None, with_task_ids=True
        )
        if len(results) != 1:
            raise RuntimeError("Evaluation coverage mismatch: expected one result")
        task_id, result = results[0]
        if task_id in completed:
            raise RuntimeError(f"Duplicate rollout task result: {task_id}")
        if task_id not in pending:
            raise RuntimeError(f"Unexpected rollout task ID: {task_id}")
        completed.add(task_id)
        yield pending.pop(task_id), None if result is None else result.trajectory
    if len(completed) != len(rows):
        raise RuntimeError("Evaluation coverage mismatch: missing rollout task results")


def _episode_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": row["domain"],
        "task_id": row["task_id"],
        "split": row.get("split", "test"),
        "trial": int(row["trial"]),
        "source_id": row["source_id"],
        "attempt_id": row["attempt_id"],
    }


def _binary_score(row: dict[str, Any], score: float) -> float:
    if not math.isfinite(score) or score not in (0.0, 1.0):
        raise RuntimeError(
            f"Official τ² score must be binary for {row['source_id']}: {score}"
        )
    return score


def record_from_trajectory(
    row: dict[str, Any], trajectory: dict[str, Any]
) -> dict[str, Any]:
    """Score an exported episode from its tensors."""

    localized = RTensor.localize(trajectory)
    rewards = localized.get("rewards")
    if rewards is None:
        raise RuntimeError(f"Missing rewards for {row['source_id']}")
    score = float(rewards.reshape(-1)[-1].item())
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite reward for {row['source_id']}: {score}")
    official_scores = localized.get("official_scores")
    official_score = (
        float(official_scores.reshape(-1)[-1].item())
        if official_scores is not None
        else score
    )
    task_budget_failure = localized.get("task_budget_failure")
    return {
        "status": COMPLETED,
        **_episode_identity(row),
        "reward": score,
        "official_score": _binary_score(row, official_score),
        "task_budget_failure": bool(task_budget_failure.reshape(-1)[-1].item())
        if task_budget_failure is not None
        else False,
    }


def attempt_records(trajectory_root: Path, source_id: str) -> list[dict[str, Any]]:
    """All attempt records of one episode, oldest first."""

    directory = trajectory_record_dir(trajectory_root, source_id)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in directory.glob("*.json")
    ]
    return sorted(records, key=lambda r: r["recorded_at"])


def record_from_unexported(
    row: dict[str, Any], trajectory_root: Path
) -> dict[str, Any]:
    """Resolve an episode that returned no policy tokens.

    Its last attempt either finished without any policy call (scored, as
    official τ² scores a first-turn user stop) or failed after the last retry
    (an infrastructure error, excluded from metrics).
    """

    attempts = attempt_records(trajectory_root, row["source_id"])
    last = attempts[-1] if attempts else None
    if (
        last is not None
        and last.get("status") == COMPLETED
        and last.get("policy_turns") == 0
    ):
        score = _binary_score(row, float(last["official_score"]))
        return {
            "status": COMPLETED,
            **_episode_identity(row),
            "reward": score,
            "official_score": score,
            "task_budget_failure": bool(last.get("task_budget_failure", False)),
            "policy_turns": 0,
            "stop_reason": last.get("stop_reason"),
            "attempts": len(attempts),
        }
    if last is None:
        error_type, error = "MissingAttemptRecord", "No attempt record was written"
    elif last.get("status") == COMPLETED:
        # The simulation ended but the proxy export or reward write failed.
        error_type = "ExportFailure"
        error = "Episode finished with policy turns but exported no trajectory"
    else:
        error_type, error = last.get("error_type"), last.get("error")
    return {
        "status": INFRASTRUCTURE_ERROR,
        **_episode_identity(row),
        "attempts": len(attempts),
        "error_type": error_type,
        "error": error,
    }


def collect_episodes(
    controller: Any,
    rows: list[dict[str, Any]],
    *,
    workflow_kwargs: dict[str, Any],
    max_inflight: int,
    trajectory_root: Path,
    on_record: Callable[[dict[str, Any]], None] | None = None,
    workflow: str = WORKFLOW,
) -> list[dict[str, Any]]:
    """Run every row to an official outcome; never abort on an episode failure."""

    trajectory_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for row, trajectory in iter_episode_results(
        controller,
        rows,
        workflow_kwargs=episode_workflow_kwargs(workflow_kwargs, trajectory_root),
        max_inflight=max_inflight,
        workflow=workflow,
    ):
        record = (
            record_from_unexported(row, trajectory_root)
            if trajectory is None
            else record_from_trajectory(row, trajectory)
        )
        records.append(record)
        if on_record is not None:
            on_record(record)
    return records


def _pass_hat_ks(records: list[dict[str, Any]], repeats: int) -> dict[int, float]:
    by_task: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        by_task[(record["domain"], str(record["task_id"]))].append(
            is_successful(float(record["official_score"]))
        )
    if not by_task:
        return {}
    # Official clips k to the fewest evaluated trials of any task.
    max_k = min(repeats, min(len(v) for v in by_task.values()))
    return {
        k: sum(pass_hat_k(len(v), sum(v), k) for v in by_task.values()) / len(by_task)
        for k in range(1, max_k + 1)
    }


def official_metrics(records: list[dict[str, Any]], *, repeats: int) -> dict[str, Any]:
    """``tau2.metrics.agent_metrics`` over resolved episodes, per domain and pooled."""

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        evaluated = [r for r in subset if r["status"] == COMPLETED]
        excluded = [r for r in subset if r["status"] == INFRASTRUCTURE_ERROR]
        scores = [float(r["official_score"]) for r in evaluated]
        reasons: dict[str, int] = defaultdict(int)
        for record in excluded:
            reasons[str(record.get("error_type"))] += 1
        return {
            "episodes": len(subset),
            "evaluated_episodes": len(evaluated),
            "infra_excluded_episodes": len(excluded),
            "infra_excluded_by_type": dict(reasons),
            "policy_free_episodes": sum(r.get("policy_turns") == 0 for r in evaluated),
            "avg_reward": sum(scores) / len(scores) if scores else 0.0,
            "pass_hat_k": _pass_hat_ks(evaluated, repeats),
        }

    domains = sorted({r["domain"] for r in records})
    return {
        "overall": summarize(records),
        "by_domain": {
            domain: summarize([r for r in records if r["domain"] == domain])
            for domain in domains
        },
    }
