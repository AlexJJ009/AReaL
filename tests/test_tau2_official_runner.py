# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from tau2.metrics.agent_metrics import pass_hat_k

from examples.tau2 import official_runner
from examples.tau2.utils import trajectory_record_dir


class _Controller:
    """Rolling controller whose results arrive in a scripted order."""

    def __init__(self, results=None, order=None, task_ids=None):
        self.dispatcher = self
        self.results = results or {}
        self.order = order
        self.task_ids = task_ids
        self.submitted = []
        self.calls = []
        self.max_seen_inflight = 0
        self.inflight = []

    def submit(self, row, workflow, workflow_kwargs, **kwargs):
        assert kwargs["group_size"] == 1 and kwargs["is_eval"]
        task_id = (
            self.task_ids[len(self.submitted)]
            if self.task_ids is not None
            else len(self.submitted)
        )
        self.submitted.append((task_id, row, workflow, workflow_kwargs))
        self.inflight.append(task_id)
        self.max_seen_inflight = max(self.max_seen_inflight, len(self.inflight))
        return task_id

    def wait_results(self, count, timeout=None, with_task_ids=False):
        assert count == 1 and timeout is None and with_task_ids
        self.calls.append(len(self.inflight))
        task_id = self.order.pop(0) if self.order else self.inflight[0]
        if task_id in self.inflight:
            self.inflight.remove(task_id)
        return [(task_id, self.results.get(task_id))]


def _rows(n=3):
    return official_runner.expand_trials(
        [{"domain": "retail", "task_id": str(i)} for i in range(n)], 1
    )


def _trajectory(score):
    return SimpleNamespace(trajectory={"rewards": torch.tensor([score])})


def _write_attempt(root, source_id, name, **fields):
    directory = trajectory_record_dir(root, source_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(fields))


def test_expand_trials_gives_each_task_trial_a_unique_identity():
    rows = official_runner.expand_trials(
        [{"domain": "airline", "task_id": "7", "split": "test"}], 2
    )

    assert [r["source_id"] for r in rows] == ["airline:7:trial0", "airline:7:trial1"]
    assert [r["trial"] for r in rows] == [0, 1]
    assert all(r["attempt_id"] == r["source_id"] for r in rows)
    with pytest.raises(ValueError, match="positive"):
        official_runner.expand_trials([], 0)
    with pytest.raises(ValueError, match="Duplicate"):
        official_runner.expand_trials([{"domain": "a", "task_id": "1"}] * 2, 1)


def test_iter_episode_results_refills_and_binds_rejections_by_task_id():
    rows = _rows(4)
    exported = _trajectory(1.0)
    controller = _Controller(results={0: exported}, order=[1, 0, 3, 2])

    yielded = list(
        official_runner.iter_episode_results(
            controller, rows, workflow_kwargs={"k": 1}, max_inflight=2
        )
    )

    assert controller.max_seen_inflight == 2
    assert [row["task_id"] for row, _ in yielded] == ["1", "0", "3", "2"]
    assert yielded[0][1] is None
    assert yielded[1][1] is exported.trajectory


@pytest.mark.parametrize(
    "controller, match",
    [
        (_Controller(order=[9]), "Unexpected rollout task ID"),
        (_Controller(task_ids=[0, 0]), "Duplicate rollout task IDs"),
    ],
)
def test_iter_episode_results_rejects_unattributable_results(controller, match):
    with pytest.raises(RuntimeError, match=match):
        list(
            official_runner.iter_episode_results(
                controller, _rows(2), workflow_kwargs={}, max_inflight=2
            )
        )


def test_iter_episode_results_rejects_duplicate_result():
    controller = _Controller(order=[0, 0])

    with pytest.raises(RuntimeError, match="Duplicate rollout task result"):
        list(
            official_runner.iter_episode_results(
                controller, _rows(2), workflow_kwargs={}, max_inflight=2
            )
        )


def test_unexported_policy_free_episode_is_scored_from_last_attempt(tmp_path):
    row = _rows(1)[0]
    _write_attempt(
        tmp_path,
        row["source_id"],
        "b",
        recorded_at=1.0,
        status="error",
        error_type="ValueError",
        error="empty",
        policy_turns=0,
    )
    _write_attempt(
        tmp_path,
        row["source_id"],
        "a",
        recorded_at=2.0,
        status="completed",
        policy_turns=0,
        official_score=1.0,
        stop_reason="user_stop",
    )

    record = official_runner.record_from_unexported(row, tmp_path)

    assert record["status"] == "completed"
    assert record["official_score"] == 1.0
    assert record["policy_turns"] == 0
    assert record["attempts"] == 2
    assert record["stop_reason"] == "user_stop"


def test_unexported_episode_with_exhausted_retries_is_infrastructure_error(tmp_path):
    row = _rows(1)[0]
    for i in range(4):
        _write_attempt(
            tmp_path,
            row["source_id"],
            str(i),
            recorded_at=float(i),
            status="error",
            error_type="EmptyUserResponseError",
            error=f"attempt {i}",
            policy_turns=0,
        )

    record = official_runner.record_from_unexported(row, tmp_path)

    assert record["status"] == official_runner.INFRASTRUCTURE_ERROR
    assert record["attempts"] == 4
    assert record["error_type"] == "EmptyUserResponseError"
    assert record["error"] == "attempt 3"


@pytest.mark.parametrize(
    "attempt, error_type",
    [
        (None, "MissingAttemptRecord"),
        ({"status": "completed", "policy_turns": 3}, "ExportFailure"),
    ],
)
def test_unexported_episode_without_explanation_is_infrastructure_error(
    tmp_path, attempt, error_type
):
    row = _rows(1)[0]
    if attempt is not None:
        _write_attempt(tmp_path, row["source_id"], "a", recorded_at=1.0, **attempt)

    record = official_runner.record_from_unexported(row, tmp_path)

    assert record["status"] == official_runner.INFRASTRUCTURE_ERROR
    assert record["error_type"] == error_type


def test_collect_episodes_binds_evaluation_kwargs_and_never_aborts(tmp_path):
    rows = _rows(3)
    _write_attempt(
        tmp_path,
        rows[1]["source_id"],
        "a",
        recorded_at=1.0,
        status="completed",
        policy_turns=0,
        official_score=0.0,
        stop_reason="user_stop",
    )
    controller = _Controller(results={0: _trajectory(1.0)})
    seen = []

    records = official_runner.collect_episodes(
        controller,
        rows,
        workflow_kwargs={"timeout": 1800.0},
        max_inflight=3,
        trajectory_root=tmp_path,
        on_record=seen.append,
    )

    kwargs = controller.submitted[0][3]
    assert kwargs == {
        "timeout": 1800.0,
        "evaluation": True,
        "trajectory_dir": str(tmp_path),
    }
    assert [r["status"] for r in records] == [
        "completed",
        "completed",
        official_runner.INFRASTRUCTURE_ERROR,
    ]
    assert seen == records


def test_official_metrics_excludes_infra_errors_and_matches_pass_hat_k():
    def record(task_id, score, status="completed"):
        return {
            "domain": "retail",
            "task_id": task_id,
            "status": status,
            "official_score": score,
        }

    records = [
        record("a", 1.0),
        record("a", 1.0),
        record("a", 0.0),
        record("b", 0.0),
        record("b", 1.0),
        record("b", 0.0),
        record("c", 1.0),
        record("c", 1.0),
        {
            "domain": "retail",
            "task_id": "c",
            "status": "infrastructure_error",
            "error_type": "ValueError",
        },
    ]

    metrics = official_runner.official_metrics(records, repeats=3)["overall"]

    assert metrics["episodes"] == 9
    assert metrics["evaluated_episodes"] == 8
    assert metrics["infra_excluded_episodes"] == 1
    assert metrics["infra_excluded_by_type"] == {"ValueError": 1}
    assert metrics["avg_reward"] == pytest.approx(5 / 8)
    # Only k in (1, 2, 4) is reported, and never above repeats.
    assert sorted(metrics["pass_hat_k"]) == [1, 2]
    assert metrics["pass_hat_k_tasks"] == {1: 3, 2: 3}
    for k in (1, 2):
        expected = (pass_hat_k(3, 2, k) + pass_hat_k(3, 1, k) + pass_hat_k(2, 2, k)) / 3
        assert metrics["pass_hat_k"][k] == pytest.approx(expected)


def test_official_metrics_reports_mean_and_pass_hat_1_2_4_over_eligible_tasks():
    def record(task_id, score):
        return {
            "domain": "retail",
            "task_id": task_id,
            "status": "completed",
            "official_score": score,
        }

    trials = {"a": [1.0, 0.0, 0.0, 0.0], "b": [0.0] * 4, "c": [1.0, 1.0, 0.0]}
    records = [record(t, s) for t, scores in trials.items() for s in scores]

    metrics = official_runner.official_metrics(records, repeats=4)["overall"]

    assert "pass_at_k" not in metrics
    assert sorted(metrics["pass_hat_k"]) == [1, 2, 4]
    # Task c lost a trial, so pass^4 averages over the two 4-trial tasks only.
    assert metrics["tasks"] == 3
    assert metrics["pass_hat_k_tasks"] == {1: 3, 2: 3, 4: 2}
    assert metrics["pass_hat_k"][4] == pytest.approx(
        (pass_hat_k(4, 1, 4) + pass_hat_k(4, 0, 4)) / 2
    )
    expected = (pass_hat_k(4, 1, 2) + pass_hat_k(4, 0, 2) + pass_hat_k(3, 2, 2)) / 3
    assert metrics["pass_hat_k"][2] == pytest.approx(expected)
    # Per-task mean, then mean over tasks (not the pooled episode mean 3/11).
    assert metrics["mean_at_k"] == pytest.approx((1 / 4 + 0 + 2 / 3) / 3)
