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


def _official_rows(domains=("airline", "retail", "telecom")):
    rows = []
    counts = {"airline": 20, "retail": 40, "telecom": 40}
    for domain, count in counts.items():
        if domain not in domains:
            continue
        for idx in range(count):
            rows.append(
                {"domain": domain, "task_id": f"{domain}-{idx}", "split": "test"}
            )
    return rows


def test_script_imports_without_running_main():
    """Importing the standalone entrypoint does not execute evaluation."""

    assert callable(tau2_evaluate.main)
    assert callable(tau2_evaluate.run)


@pytest.mark.parametrize(
    "temperature_args, expected", [([], 0.0), (["--temperature", "1"], 1.0)]
)
def test_load_eval_config_historical_sampling_uses_requested_temperature(
    monkeypatch, temperature_args, expected
):
    """New evaluation overrides a historical config without changing training."""

    config = SimpleNamespace(
        gconfig=SimpleNamespace(temperature=1.0),
        eval_gconfig=SimpleNamespace(temperature=0.75, seed=None),
    )
    monkeypatch.setattr(
        tau2_evaluate, "load_expr_config", lambda *_: (config, "historical.yaml")
    )
    monkeypatch.setattr(tau2_evaluate, "validate_tau2_recipe", lambda _: None)
    args = tau2_evaluate.parse_args(
        [
            "--config",
            "historical.yaml",
            "--model-path",
            "checkpoint",
            "--output-root",
            "eval",
        ]
        + temperature_args
    )

    loaded, _ = tau2_evaluate.load_eval_config(args)

    assert loaded.eval_gconfig.temperature == expected
    assert loaded.gconfig.temperature == 1.0
    assert args.repeats == 4


@pytest.mark.parametrize("temperature", ["-1", "nan", "inf"])
def test_load_eval_config_invalid_temperature_rejects(monkeypatch, temperature):
    """Invalid overrides cannot silently produce an unintended sampling policy."""

    config = SimpleNamespace(eval_gconfig=SimpleNamespace(seed=None))
    monkeypatch.setattr(
        tau2_evaluate, "load_expr_config", lambda *_: (config, "historical.yaml")
    )
    args = tau2_evaluate.parse_args(
        [
            "--config",
            "historical.yaml",
            "--model-path",
            "checkpoint",
            "--output-root",
            "eval",
            "--temperature",
            temperature,
        ]
    )

    with pytest.raises(ValueError, match="temperature must be finite and non-negative"):
        tau2_evaluate.load_eval_config(args)


def test_build_eval_rows_expands_official_test_repeats(monkeypatch):
    """Official test rows are expanded into independently identified trials."""

    monkeypatch.setattr(
        tau2_evaluate, "get_tau2_dataset", lambda **kw: _official_rows(kw["domain"])
    )
    config = type(
        "Config",
        (),
        {
            "seed": 42,
            "experiment_mode": "formal",
            "domains": ["airline", "retail", "telecom"],
        },
    )()

    rows = tau2_evaluate.build_eval_rows(config, repeats=8)

    assert len(rows) == 800
    assert rows[0]["source_id"] == "airline:airline-0:trial0"
    assert rows[1]["source_id"] == "airline:airline-0:trial1"
    assert rows[7]["trial"] == 7
    assert rows[8]["source_id"] == "airline:airline-1:trial0"


def test_build_eval_rows_covers_only_configured_domains(monkeypatch):
    """Dropping Telecom shrinks the official test coverage to airline+retail."""

    monkeypatch.setattr(
        tau2_evaluate, "get_tau2_dataset", lambda **kw: _official_rows(kw["domain"])
    )
    config = SimpleNamespace(
        seed=42, experiment_mode="formal", domains=["airline", "retail"]
    )

    rows = tau2_evaluate.build_eval_rows(config, repeats=4)

    assert len(rows) == 240
    assert {row["domain"] for row in rows} == {"airline", "retail"}
    assert rows[3]["trial"] == 3


def _official_split_rows(domains, split):
    counts = {
        "test": {"airline": 20, "retail": 40, "telecom": 40},
        "train": {"airline": 30, "retail": 74, "telecom": 74},
    }[split]
    return [
        {"domain": domain, "task_id": f"{split}-{idx}", "split": split}
        for domain, count in counts.items()
        if domain in domains
        for idx in range(count)
    ]


def test_build_eval_rows_base_covers_train_and_test(monkeypatch):
    """Base evaluation is the official train+test union with split tags kept."""

    calls = []

    def fake_dataset(**kw):
        calls.append((kw["split"], kw["experiment_mode"]))
        return _official_split_rows(kw["domain"], kw["split"])

    monkeypatch.setattr(tau2_evaluate, "get_tau2_dataset", fake_dataset)
    monkeypatch.setattr(
        tau2_evaluate,
        "official_base_task_ids",
        lambda domain: {
            row["task_id"]
            for split in ("train", "test")
            for row in _official_split_rows((domain,), split)
        },
    )
    config = SimpleNamespace(
        seed=42, experiment_mode="tune", domains=["airline", "retail"]
    )

    rows = tau2_evaluate.build_eval_rows(config, repeats=4, task_set="base")

    assert len(rows) == (50 + 114) * 4
    assert {(row["domain"], row["split"]) for row in rows} == {
        ("airline", "test"),
        ("airline", "train"),
        ("retail", "test"),
        ("retail", "train"),
    }
    # The full train pool is used even when the resolved config is in tune mode.
    assert ("train", "formal") in calls
    assert len({row["source_id"] for row in rows}) == len(rows)


def test_build_eval_rows_base_mismatch_with_official_rejects(monkeypatch):
    """Base coverage is checked against the official base list, not just counts."""

    monkeypatch.setattr(
        tau2_evaluate,
        "get_tau2_dataset",
        lambda **kw: _official_split_rows(kw["domain"], kw["split"]),
    )
    monkeypatch.setattr(
        tau2_evaluate, "official_base_task_ids", lambda domain: {"other"}
    )
    config = SimpleNamespace(seed=42, experiment_mode="formal", domains=["airline"])

    with pytest.raises(ValueError, match="does not equal official base"):
        tau2_evaluate.build_eval_rows(config, repeats=1, task_set="base")


def test_aggregate_reports_base_scores_by_domain_split():
    records = [
        {
            "status": "completed",
            "domain": "retail",
            "trial": 0,
            "split": split,
            "official_score": score,
        }
        for split, score in (("test", 1.0), ("test", 0.0), ("train", 1.0))
    ]

    summary = tau2_evaluate.aggregate(records, planned=3, repeats=1)

    assert summary["by_domain_split"] == {
        "retail:test": {"episodes": 2, "pass1_mean": 0.5},
        "retail:train": {"episodes": 1, "pass1_mean": 1.0},
    }
    assert summary["by_domain_trial"] == {
        "retail:0": {"episodes": 3, "pass1_mean": 2 / 3}
    }


def test_parse_args_task_set_defaults_to_test_and_rejects_unknown():
    base = ["--config", "c.yaml", "--model-path", "m", "--output-root", "o"]
    assert tau2_evaluate.parse_args(base).task_set == "test"
    assert tau2_evaluate.parse_args(base + ["--task-set", "base"]).task_set == "base"
    with pytest.raises(SystemExit):
        tau2_evaluate.parse_args(base + ["--task-set", "train"])


def test_build_eval_rows_missing_domain_rows_rejects(monkeypatch):
    """A loader that drops a configured domain cannot pass as full coverage."""

    monkeypatch.setattr(
        tau2_evaluate, "get_tau2_dataset", lambda **_: _official_rows(("airline",))
    )
    config = SimpleNamespace(
        seed=42, experiment_mode="formal", domains=["airline", "retail"]
    )

    with pytest.raises(ValueError, match="coverage mismatch: 80 != 240"):
        tau2_evaluate.build_eval_rows(config, repeats=4)


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
            max_concurrent_rollouts=None,
        ),
        rollout=None,
        gconfig=gconfig,
        eval_gconfig=eval_gconfig,
    )

    bound = tau2_evaluate.bind_eval_config(
        config, model_path="/models/policy", output_root=tmp_path
    )

    assert Path(bound.cluster.name_resolve.nfs_record_root).is_dir()
    assert bound.eval_gconfig.temperature == 1.0
    assert bound.eval_gconfig.top_p == 1.0
    assert bound.eval_gconfig.n_samples == 1
    assert bound.eval_gconfig.seed is None
    assert bound.gconfig.temperature == 0.25
    assert tau2_evaluate.resolve_eval_concurrency(bound) == 1


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


def test_record_from_trajectory_rejects_missing_reward():
    row = {"source_id": "airline:a:trial0"}

    with pytest.raises(RuntimeError, match="Missing rewards"):
        tau2_evaluate.record_from_trajectory(row, {})


class RollingController:
    def __init__(self):
        self.dispatcher = self
        self.pending = {}
        self.submissions = []
        self.max_pending = 0
        self.waits = 0

    def submit(self, row, **kwargs):
        assert kwargs["group_size"] == 1
        assert kwargs["is_eval"] is True
        task_id = len(self.submissions)
        self.submissions.append(row)
        self.pending[task_id] = row
        self.max_pending = max(self.max_pending, len(self.pending))
        return task_id

    def wait_results(self, count, timeout=None):
        assert count == 1
        assert timeout is None
        self.waits += 1
        # Task zero is a straggler. Every newer task can finish first.
        task_id = max(self.pending)
        self.pending.pop(task_id)
        return [SimpleNamespace(task_id=task_id, trajectory={"task_id": task_id})]


def _rolling_rows(count=8):
    return [
        {"source_id": str(i), "trial": i // 4, "domain": "retail", "task_id": str(i)}
        for i in range(count)
    ]


def test_rolling_refills_without_trial_or_submission_order_barrier():
    controller = RollingController()
    rows = _rolling_rows()
    results = tau2_evaluate.iter_completed_rollouts(
        controller, rows, workflow_kwargs={}, max_inflight=2
    )
    first, trajectory = next(results)
    assert first == rows[1]
    assert trajectory["task_id"] == 1
    assert len(controller.submissions) == 2
    second, _ = next(results)
    assert second == rows[2]
    assert len(controller.submissions) == 3
    completed = [first, second] + [row for row, _ in results]
    assert [row["source_id"] for row in completed] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "0",
    ]
    assert controller.max_pending == 2
    assert controller.waits == 8
    assert len({row["source_id"] for row in completed}) == len(rows)
    assert controller.submissions == rows


@pytest.mark.parametrize(
    "bad_result, message",
    [
        ([], "coverage mismatch"),
        ([None], "unbound failed"),
        ([SimpleNamespace(task_id=100, trajectory={})], "Unexpected rollout"),
        ([SimpleNamespace(task_id=1, trajectory={})], "Duplicate rollout task result"),
    ],
)
def test_rolling_rejects_missing_unbound_unexpected_and_duplicate_results(
    bad_result, message
):
    controller = RollingController()
    results = tau2_evaluate.iter_completed_rollouts(
        controller, _rolling_rows(), workflow_kwargs={}, max_inflight=2
    )
    next(results)
    controller.wait_results = lambda *args, **kwargs: bad_result
    with pytest.raises(RuntimeError, match=message):
        next(results)


def test_rolling_rejects_duplicate_submission_ids():
    controller = RollingController()
    controller.submit = lambda *args, **kwargs: 7
    with pytest.raises(RuntimeError, match="Duplicate rollout task IDs"):
        list(
            tau2_evaluate.iter_completed_rollouts(
                controller, _rolling_rows(), workflow_kwargs={}, max_inflight=2
            )
        )


def test_rolling_rejects_duplicate_episode_ids_before_submitting():
    controller = RollingController()
    rows = _rolling_rows(1) * 2
    with pytest.raises(ValueError, match="Duplicate evaluation source IDs"):
        list(
            tau2_evaluate.iter_completed_rollouts(
                controller, rows, workflow_kwargs={}, max_inflight=2
            )
        )
    assert not controller.submissions


def test_collect_persists_each_result_before_refill_and_retains_partial_on_failure(
    tmp_path, monkeypatch
):
    controller = RollingController()
    original_submit = controller.submit
    original_wait = controller.wait_results

    def submit(row, **kwargs):
        if len(controller.submissions) == 2:
            summary = json.loads((tmp_path / "summary.json").read_text())
            assert summary["completed_episodes"] == 1
            assert len((tmp_path / "episodes.jsonl").read_text().splitlines()) == 1
        return original_submit(row, **kwargs)

    def wait(*args, **kwargs):
        if controller.waits:
            raise ConnectionError("provider disconnected")
        return original_wait(*args, **kwargs)

    controller.submit = submit
    controller.wait_results = wait
    monkeypatch.setattr(
        tau2_evaluate,
        "record_from_trajectory",
        lambda row, _: {**row, "status": "completed", "official_score": 1.0},
    )
    with pytest.raises(ConnectionError, match="provider disconnected"):
        tau2_evaluate.collect_eval_records(
            controller,
            _rolling_rows(),
            workflow_kwargs={},
            max_inflight=2,
            output_root=tmp_path,
            repeats=2,
        )
    summary = json.loads((tmp_path / "summary.json").read_text())
    records = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
    ]
    assert summary["status"] == "partial"
    assert summary["completed_episodes"] == 1
    assert summary["metrics"]["eval/pass1_mean"] == 1.0
    assert "provider disconnected" in summary["error"]
    assert len(records) == 1
    assert records[0]["source_id"] == "1"


def test_rolling_covers_all_800_episodes_with_64_inflight(monkeypatch):
    monkeypatch.setattr(
        tau2_evaluate, "get_tau2_dataset", lambda **kw: _official_rows(kw["domain"])
    )
    rows = tau2_evaluate.build_eval_rows(
        SimpleNamespace(
            seed=42,
            experiment_mode="formal",
            domains=["airline", "retail", "telecom"],
        ),
        repeats=8,
    )
    controller = RollingController()
    completed = list(
        tau2_evaluate.iter_completed_rollouts(
            controller, rows, workflow_kwargs={}, max_inflight=64
        )
    )
    assert len(completed) == 800
    assert {row["source_id"] for row, _ in completed} == {
        row["source_id"] for row in rows
    }
    assert controller.max_pending == 64
    assert not controller.pending


def test_collect_missing_trajectory_retains_failure_without_reward(tmp_path):
    controller = RollingController()
    controller.wait_results = lambda *args, **kwargs: [
        SimpleNamespace(task_id=0, trajectory=None)
    ]
    with pytest.raises(RuntimeError, match="Missing trajectory"):
        tau2_evaluate.collect_eval_records(
            controller,
            _rolling_rows(1),
            workflow_kwargs={},
            max_inflight=1,
            output_root=tmp_path,
            repeats=1,
        )
    record = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert record["status"] == "infra_failed"
    assert "reward" not in record
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "partial"
    assert summary["completed_episodes"] == 0
    assert summary["infra_failed_episodes"] == 1


@pytest.mark.parametrize(
    "configured, batch_size, expected", [(None, 1, 1), (None, 32, 32), (64, 1, 64)]
)
def test_resolve_eval_concurrency_matches_native_fallback(
    configured, batch_size, expected
):
    config = SimpleNamespace(
        evaluation_rollout=SimpleNamespace(
            max_concurrent_rollouts=configured, consumer_batch_size=batch_size
        )
    )
    assert tau2_evaluate.resolve_eval_concurrency(config) == expected
