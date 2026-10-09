# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
            "task_id": str(index),
            "trial": 0,
            "split": split,
            "official_score": score,
        }
        for index, (split, score) in enumerate(
            (("test", 1.0), ("test", 0.0), ("train", 1.0))
        )
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


def test_write_record_appends_jsonl(tmp_path):
    path = tmp_path / "episodes.jsonl"

    tau2_evaluate.write_record(path, {"b": 2, "a": 1})
    tau2_evaluate.write_record(path, {"a": 3})

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"a": 1, "b": 2}, {"a": 3}]


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


def test_summarize_user_usage_keeps_last_retry_and_counts_missing(tmp_path):
    path = tmp_path / "user-usage.jsonl"
    lines = [
        {
            "source_id": "a",
            "user_llm_calls": 9,
            "user_prompt_tokens": 900,
            "user_completion_tokens": 90,
        },
        {
            "source_id": "a",
            "user_llm_calls": 2,
            "user_prompt_tokens": 200,
            "user_completion_tokens": 20,
        },
        {
            "source_id": "orphan",
            "user_llm_calls": 1,
            "user_prompt_tokens": 5,
            "user_completion_tokens": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines))
    records = [
        {"source_id": "a", "status": "completed"},
        {"source_id": "b", "status": "completed"},
        {"source_id": "c", "status": "failed"},
    ]

    usage = tau2_evaluate.summarize_user_usage(path, records)

    assert usage["episodes_with_usage"] == 1
    assert usage["episodes_missing_usage"] == 1
    assert usage["user_prompt_tokens"] == 200
    assert usage["mean_completion_tokens_per_episode"] == 20


def test_load_eval_config_applies_hydra_overrides(monkeypatch):
    seen = {}
    config = SimpleNamespace(
        gconfig=SimpleNamespace(temperature=1.0),
        eval_gconfig=SimpleNamespace(temperature=1.0, seed=None),
    )

    def fake_load(argv, _cls):
        seen["argv"] = argv
        return config, "c.yaml"

    monkeypatch.setattr(tau2_evaluate, "load_expr_config", fake_load)
    monkeypatch.setattr(tau2_evaluate, "validate_tau2_recipe", lambda _: None)
    args = tau2_evaluate.parse_args(
        [
            "--config",
            "c.yaml",
            "--model-path",
            "m",
            "--output-root",
            "o",
            "econfig.user_llm=doubao",
            "~econfig.user_llm_args.max_completion_tokens",
        ]
    )

    tau2_evaluate.load_eval_config(args)

    assert seen["argv"] == [
        "--config",
        "c.yaml",
        "econfig.user_llm=doubao",
        "~econfig.user_llm_args.max_completion_tokens",
    ]


def test_aggregate_excludes_infra_errors_like_official_and_reports_them():
    records = [
        {
            "status": "completed",
            "domain": "airline",
            "task_id": "a",
            "trial": 0,
            "official_score": 1.0,
        },
        {
            "status": "infrastructure_error",
            "domain": "airline",
            "task_id": "b",
            "trial": 0,
            "source_id": "airline:b:trial0",
            "attempts": 4,
            "error_type": "EmptyUserResponseError",
            "error": "Empty user response",
        },
    ]

    summary = tau2_evaluate.aggregate(records, planned=2, repeats=1)

    # tau2 run finishes, scores the evaluated episodes and counts the excluded one.
    assert summary["status"] == "complete"
    assert summary["completed_episodes"] == 1
    assert summary["infra_excluded_episodes"] == 1
    assert summary["metrics"]["eval/avg_reward"] == 1.0
    assert summary["metrics"]["eval/pass^1"] == 1.0
    assert summary["infra_excluded"] == [
        {
            "source_id": "airline:b:trial0",
            "attempts": 4,
            "error_type": "EmptyUserResponseError",
            "error": "Empty user response",
        }
    ]
    assert tau2_evaluate.aggregate(records[:1], planned=2, repeats=1)["status"] == (
        "partial"
    )


def test_select_tasks_keeps_named_subset_and_rejects_unknown():
    rows = [
        {"domain": "retail", "task_id": "51", "trial": 0},
        {"domain": "retail", "task_id": "38", "trial": 0},
    ]
    assert tau2_evaluate.select_tasks(rows, None) == rows
    assert tau2_evaluate.select_tasks(rows, "retail:38") == rows[1:]
    with pytest.raises(ValueError, match="Unknown tasks"):
        tau2_evaluate.select_tasks(rows, "retail:38,airline:1")
