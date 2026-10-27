# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the saved PPO checkpoint evaluation protocol."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.tau2 import evaluation, ppo_data
from examples.tau2.evaluation import Tau2PPOTrainer
from examples.tau2.utils import Tau2PPOConfig
from tests.test_tau2_evaluate import tau2_evaluate
from tests.test_tau2_ppo import _config

from areal.api.cli_args import load_expr_config


def test_shipped_ppo_recipe_accepts_plain_offline_override(monkeypatch, tmp_path):
    _config(monkeypatch, tmp_path)
    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_ppo.yaml", "offline_eval_only=true"],
        Tau2PPOConfig,
    )
    assert config.offline_eval_only is True


def _args(tmp_path, *extra):
    return [
        "--config",
        "unused.yaml",
        "--model-path",
        "saved-actor",
        "--output-root",
        str(tmp_path),
        "--task-set",
        "ppo",
        *extra,
    ]


def test_checkpoint_eval_rows_match_training_validation_set(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    expected = ppo_data.build_ppo_eval_dataset(
        config, {"seed": config.seed, "variant": "fixed"}
    )
    rows = tau2_evaluate.build_eval_rows(config, 4, "ppo")
    identities = {(row["domain"], row["task_id"]) for row in expected}
    assert len(rows) == len({row["source_id"] for row in rows}) == 400
    assert Counter(row["domain"] for row in rows) == {
        "airline": 80,
        "retail": 160,
        "telecom": 160,
    }
    for trial in range(4):
        assert {
            (row["domain"], row["task_id"]) for row in rows if row["trial"] == trial
        } == identities


@pytest.mark.parametrize(
    "extra",
    [
        ["--repeats", "2"],
        ["--temperature", "1"],
        ["--tasks", "retail:1"],
        ["--task-pack", "pack"],
    ],
)
def test_checkpoint_protocol_rejects_sampling_or_subset_changes(tmp_path, extra):
    with pytest.raises(SystemExit):
        tau2_evaluate.parse_args(_args(tmp_path, *extra))


def test_checkpoint_eval_dry_run_preserves_user_judge_and_loads_no_engine(
    monkeypatch, tmp_path
):
    config = _config(monkeypatch, tmp_path)
    config.offline_eval_only = True
    config.econfig.user_llm = "caller-user"
    config.econfig.nl_judge_llm = "caller-judge"
    config.econfig.user_llm_args = {"temperature": 0.3}
    config.econfig.nl_judge_llm_args = {"temperature": 0.2}
    original = deepcopy(config.econfig)
    monkeypatch.setattr(
        tau2_evaluate, "load_eval_config", lambda _: (config, "synthetic.yaml")
    )
    # Nothing below the dry-run boundary may create a worker or collect an episode.
    scheduler = Mock(side_effect=AssertionError("must not launch scheduler"))
    collect = Mock(side_effect=AssertionError("must not evaluate"))
    monkeypatch.setattr(tau2_evaluate, "LocalScheduler", scheduler)
    monkeypatch.setattr(tau2_evaluate, "collect_eval_records", collect)
    result = tau2_evaluate.run(_args(tmp_path, "--check-config"))
    assert result["manifest"]["planned_episodes"] == 400
    assert result["manifest"]["domains"] == ["airline", "retail", "telecom"]
    assert result["manifest"]["temperature"] == 0
    assert result["manifest"]["repeats"] == 4
    assert result["summary"]["status"] == "check_config"
    bound = tau2_evaluate.bind_eval_config(
        config, model_path="saved-actor", output_root=tmp_path
    )
    assert bound.econfig._quota_breaker_state_path == str(
        tmp_path / "quota-breaker.json"
    )
    original._quota_breaker_state_path = bound.econfig._quota_breaker_state_path
    assert bound.econfig == original
    assert (
        bound.actor.path
        == bound.tokenizer_path
        == bound.sglang.model_path
        == "saved-actor"
    )
    assert config.actor.path != "saved-actor"
    scheduler.assert_not_called()
    collect.assert_not_called()


def test_offline_and_training_eval_write_identical_checkpoint_metrics(
    monkeypatch, tmp_path
):
    records = [
        {
            "status": "completed",
            "domain": "airline",
            "task_id": "a",
            "trial": trial,
            "official_score": float(trial % 2),
            "stop_reason": "context_limit" if trial == 0 else "user_stop",
            "policy_turns": 1,
        }
        for trial in range(4)
    ] + [
        {
            "status": "infrastructure_error",
            "domain": "retail",
            "task_id": "r",
            "trial": 0,
            "source_id": "retail:r:trial0",
        }
    ]
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(
        experiment_mode="formal",
        eval_gconfig=SimpleNamespace(n_samples=4),
        evaluation_rollout=SimpleNamespace(max_concurrent_rollouts=2),
        experiment_name="test",
        trial_name="test",
        cluster=SimpleNamespace(fileroot=str(tmp_path)),
    )
    trainer.valid_dataloader = [[{"domain": "airline", "task_id": "a"}]]
    trainer._async_eval_rollout = Mock()
    monkeypatch.setattr(evaluation, "collect_episodes", lambda *_, **__: records)
    online = trainer._evaluate_dataset("workflow", {})
    offline = tau2_evaluate.aggregate(records, planned=len(records), repeats=4)
    assert online["metrics"]
    for key, value in online["metrics"].items():
        assert offline["metrics"][key] == value


def test_checkpoint_eval_accepts_resolved_actor_path_without_relaxing_training(
    monkeypatch, tmp_path
):
    from examples.tau2.train import validate_tau2_recipe

    config = _config(monkeypatch, tmp_path)
    config.actor.path = str(tmp_path / "resolved-local-model")
    monkeypatch.setattr(
        tau2_evaluate,
        "load_expr_config",
        lambda *_: (config, "synthetic-resolved.yaml"),
    )
    args = tau2_evaluate.parse_args(_args(tmp_path))
    loaded, _ = tau2_evaluate.load_eval_config(args)
    assert loaded.actor.path == config.actor.path
    with pytest.raises(ValueError, match="pinned Qwen3"):
        validate_tau2_recipe(loaded)
