# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/tau2/evaluate.py"
SPEC = importlib.util.spec_from_file_location("tau2_evaluate", MODULE_PATH)
tau2_evaluate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tau2_evaluate)


def _official_rows():
    rows = []
    counts = {"airline": 20, "retail": 40, "telecom": 40}
    for domain, count in counts.items():
        for idx in range(count):
            rows.append(
                {"domain": domain, "task_id": f"{domain}-{idx}", "split": "test"}
            )
    return rows


def test_script_imports_without_running_main():
    """Importing the standalone entrypoint does not execute evaluation."""

    assert callable(tau2_evaluate.main)
    assert callable(tau2_evaluate.run)


def test_build_eval_rows_expands_official_test_repeats(monkeypatch):
    """Official test rows are expanded into independently identified trials."""

    monkeypatch.setattr(tau2_evaluate, "get_tau2_dataset", lambda **_: _official_rows())
    config = type(
        "Config",
        (),
        {"seed": 42, "experiment_mode": "formal"},
    )()

    rows = tau2_evaluate.build_eval_rows(config, repeats=8)

    assert len(rows) == 800
    assert rows[0]["source_id"] == "airline:airline-0:trial0"
    assert rows[1]["source_id"] == "airline:airline-0:trial1"
    assert rows[7]["trial"] == 7
    assert rows[8]["source_id"] == "airline:airline-1:trial0"


def test_aggregate_reports_pass1_not_best_of_repeats():
    """Repeated trials are averaged directly as pass@1."""

    records = [
        {
            "status": "completed",
            "domain": "airline",
            "task_id": "a",
            "trial": 0,
            "official_score": 1.0,
        },
        {
            "status": "completed",
            "domain": "airline",
            "task_id": "a",
            "trial": 1,
            "official_score": 0.0,
        },
        {
            "status": "completed",
            "domain": "retail",
            "task_id": "r",
            "trial": 0,
            "official_score": 0.0,
        },
        {
            "status": "completed",
            "domain": "retail",
            "task_id": "r",
            "trial": 1,
            "official_score": 1.0,
        },
    ]

    summary = tau2_evaluate.aggregate(records, planned=4, repeats=2)

    assert summary["status"] == "complete"
    assert summary["metrics"]["eval/pass1_mean"] == 0.5
    assert summary["by_domain"]["airline"] == {"episodes": 2, "pass1_mean": 0.5}
    assert summary["by_trial"]["0"] == {"episodes": 2, "pass1_mean": 0.5}


def test_aggregate_marks_infra_failure_partial_without_reward_zero():
    """Infrastructure failures reduce coverage and do not enter score means."""

    records = [
        {
            "status": "completed",
            "domain": "airline",
            "task_id": "a",
            "trial": 0,
            "official_score": 1.0,
        },
        {
            "status": "infra_failed",
            "domain": "airline",
            "task_id": "b",
            "trial": 0,
            "error": "RuntimeError: worker died",
        },
    ]

    summary = tau2_evaluate.aggregate(records, planned=2, repeats=1)

    assert summary["status"] == "partial"
    assert summary["completed_episodes"] == 1
    assert summary["infra_failed_episodes"] == 1
    assert summary["metrics"]["eval/pass1_mean"] == 1.0


def test_bind_eval_config_preserves_validation_sampling(tmp_path):
    """Standalone eval keeps the resolved eval config except explicit N=1 trials."""

    gconfig = SimpleNamespace(
        n_samples=8,
        seed=None,
        reward_normalization=True,
        drop_incomplete_group=True,
        temperature=0.25,
        top_p=0.9,
    )
    eval_gconfig = SimpleNamespace(
        n_samples=8,
        seed=None,
        reward_normalization=True,
        drop_incomplete_group=True,
        temperature=1.0,
        top_p=1.0,
    )
    config = SimpleNamespace(
        cluster=SimpleNamespace(
            fileroot="/old",
            name_resolve=SimpleNamespace(nfs_record_root="/old/name-resolve"),
        ),
        experiment_name="exp",
        trial_name="trial",
        actor=SimpleNamespace(path="old-model"),
        tokenizer_path="old-model",
        sglang=SimpleNamespace(model_path="old-model"),
        vllm=SimpleNamespace(model="old-model"),
        evaluation_rollout=SimpleNamespace(
            experiment_name="exp",
            trial_name="trial",
            fileroot="/old",
            tokenizer_path="old-model",
            max_head_offpolicyness=2,
            consumer_batch_size=32,
        ),
        rollout=None,
        gconfig=gconfig,
        eval_gconfig=eval_gconfig,
    )

    bound = tau2_evaluate.bind_eval_config(
        config, model_path="/models/policy", output_root=tmp_path
    )

    assert bound.eval_gconfig.temperature == 1.0
    assert bound.eval_gconfig.top_p == 1.0
    assert bound.eval_gconfig.n_samples == 1
    assert bound.eval_gconfig.seed is None
    assert bound.gconfig.temperature == 0.25


def test_assert_fresh_output_rejects_existing_results(tmp_path):
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tau2_evaluate.assert_fresh_output(tmp_path, check_config=True)


def test_record_from_trajectory_localizes_rtensor_reward(monkeypatch):
    """A completed trajectory produces a compact per-task ledger record."""

    monkeypatch.setattr(
        tau2_evaluate.RTensor,
        "localize",
        lambda obj: obj,
    )
    row = {
        "domain": "telecom",
        "task_id": "t-1",
        "split": "test",
        "trial": 3,
        "source_id": "telecom:t-1:trial3",
        "attempt_id": "telecom:t-1:trial3",
    }
    trajectory = {
        "rewards": torch.tensor([0.25]),
        "official_scores": torch.tensor([1.0]),
        "task_budget_failure": torch.tensor([False]),
    }

    record = tau2_evaluate.record_from_trajectory(row, trajectory)

    assert record["status"] == "completed"
    assert record["official_score"] == 1.0
    assert record["reward"] == 0.25
    assert record["trial"] == 3


def test_write_record_appends_jsonl(tmp_path):
    path = tmp_path / "episodes.jsonl"

    tau2_evaluate.write_record(path, {"b": 2, "a": 1})
    tau2_evaluate.write_record(path, {"a": 3})

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"a": 1, "b": 2}, {"a": 3}]


def test_wait_for_submitted_maps_reordered_dispatcher_results():
    """Concurrent trial batches preserve task identity from dispatcher results."""

    class Result:
        def __init__(self, task_id, trajectory):
            self.task_id = task_id
            self.trajectory = trajectory

    class Dispatcher:
        def wait_results(self, count, timeout=None):
            assert count == 2
            assert timeout is None
            return [
                Result(12, {"rewards": "second"}),
                Result(11, {"rewards": "first"}),
            ]

    controller = SimpleNamespace(dispatcher=Dispatcher())
    submitted = [(11, {"source_id": "first"}), (12, {"source_id": "second"})]

    mapped = tau2_evaluate.wait_for_submitted(controller, submitted)

    assert mapped == [
        ({"source_id": "first"}, {"rewards": "first"}),
        ({"source_id": "second"}, {"rewards": "second"}),
    ]


def test_wait_for_submitted_rejects_missing_identity():
    class Dispatcher:
        def wait_results(self, count, timeout=None):
            return []

    controller = SimpleNamespace(dispatcher=Dispatcher())

    with pytest.raises(RuntimeError, match="coverage mismatch"):
        tau2_evaluate.wait_for_submitted(controller, [(1, {"source_id": "x"})])


def test_record_from_trajectory_rejects_missing_reward():
    row = {"source_id": "airline:a:trial0"}

    with pytest.raises(RuntimeError, match="Missing rewards"):
        tau2_evaluate.record_from_trajectory(row, {})
