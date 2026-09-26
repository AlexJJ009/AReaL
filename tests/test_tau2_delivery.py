"""Delivery-focused τ² tests for dataset, GRPO config, retry, and eval failure gates."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from examples.tau2.evaluation import Tau2AsyncEvalTrainer
from examples.tau2.train import get_tau2_dataset, validate_tau2_recipe
from examples.tau2.utils import Tau2PPOConfig

from areal.api.cli_args import load_expr_config
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def grpo_config(monkeypatch, tmp_path) -> Tau2PPOConfig:
    monkeypatch.setenv("TAU2_TRIAL_NAME", "delivery-unit")
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        ["--config", str(REPO_ROOT / "examples/tau2/config_grpo.yaml")],
        Tau2PPOConfig,
    )
    return config


def _source_ids(dataset) -> set[str]:
    return {str(row["source_id"]) for row in dataset}


@pytest.mark.skipif(
    not os.getenv("TAU2_DATA_DIR"), reason="TAU2_DATA_DIR is required for official data"
)
def test_formal_and_tuning_splits_use_official_tau2_data_without_test_selection():
    domains = ("airline", "retail", "telecom")

    formal_train = get_tau2_dataset(
        domains,
        split="train",
        algorithm="grpo",
        experiment_mode="formal",
        seed=42,
    )
    formal_test = get_tau2_dataset(
        domains,
        split="test",
        algorithm="grpo",
        experiment_mode="formal",
        seed=42,
    )
    tune_train = get_tau2_dataset(
        domains,
        split="train",
        algorithm="grpo",
        experiment_mode="tune",
        seed=42,
    )
    tune_dev = get_tau2_dataset(
        domains,
        split="dev",
        algorithm="grpo",
        experiment_mode="tune",
        seed=42,
    )

    assert len(formal_train) == 178
    assert len(formal_test) == 100
    assert len(tune_train) == 142
    assert len(tune_dev) == 36
    assert _source_ids(tune_train).isdisjoint(_source_ids(tune_dev))
    assert _source_ids(tune_train) | _source_ids(tune_dev) == _source_ids(formal_train)
    assert _source_ids(formal_test).isdisjoint(_source_ids(formal_train))


def test_config_grpo_freezes_4_3_1_async_grpo_contract(grpo_config: Tau2PPOConfig):
    assert validate_tau2_recipe(grpo_config) == ("airline", "retail", "telecom")

    assert grpo_config.actor.backend == "fsdp:d4p1t1"
    assert grpo_config.rollout.backend == "sglang:d3p1t1"
    assert grpo_config.evaluation_rollout.backend == "sglang:d1p1t1"
    assert grpo_config.critic is None
    assert grpo_config.ref is None
    assert grpo_config.teacher is None

    assert grpo_config.gconfig.n_samples == 8
    assert grpo_config.train_batch_episodes == 64
    assert grpo_config.train_dataset.batch_size == 8
    assert grpo_config.gconfig.max_tokens == 32768
    assert grpo_config.gconfig.max_new_tokens == 4096
    assert grpo_config.actor.use_decoupled_loss is True
    assert grpo_config.actor.recompute_logprob is True
    assert grpo_config.actor.eps_clip_higher == 0.28
    assert grpo_config.gconfig.seed is None
    assert grpo_config.eval_gconfig.seed is None
    assert grpo_config.actor.reward_norm is not None
    assert grpo_config.actor.reward_norm.group_size == grpo_config.gconfig.n_samples
    assert grpo_config.saver.mode == "sync"
    assert grpo_config.saver.freq_steps == grpo_config.evaluator.freq_steps
    assert grpo_config.saver.freq_steps == 20


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: setattr(config.actor.reward_norm, "group_size", 7),
            "group reward normalization",
        ),
        (
            lambda config: setattr(config.actor, "use_decoupled_loss", False),
            "decoupled loss",
        ),
    ],
)
def test_config_grpo_rejects_malformed_norm_and_non_decoupled_loss(
    grpo_config: Tau2PPOConfig, mutate, message: str
):
    broken = copy.deepcopy(grpo_config)
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        validate_tau2_recipe(broken)


class _TransientEpisodeError(RuntimeError):
    pass


class _DeterministicEpisodeError(RuntimeError):
    pass


class _RetryAgent:
    infra_retries = 1

    async def run(self, data, **kwargs):
        raise AssertionError("_run_agent should not be reached")

    @staticmethod
    def should_retry_episode(exc: Exception) -> bool:
        return isinstance(exc, _TransientEpisodeError)


@pytest.fixture()
def retry_workflow(monkeypatch) -> OpenAIProxyWorkflow:
    workflow = OpenAIProxyWorkflow(mode="inline", agent=_RetryAgent())

    async def _no_retry_backoff(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "areal.experimental.openai.proxy.workflow.asyncio.sleep",
        _no_retry_backoff,
    )
    monkeypatch.setattr(
        "areal.experimental.openai.proxy.workflow.stats_tracker.get",
        lambda *_args, **_kwargs: SimpleNamespace(scalar=lambda **_kw: None),
    )
    return workflow


@pytest.mark.asyncio
async def test_proxy_workflow_retries_transient_episode_from_fresh_wrapper(
    retry_workflow: OpenAIProxyWorkflow,
):
    calls = []

    async def _arun_episode(self, engine, data):
        calls.append((engine, data))
        if len(calls) == 1:
            raise _TransientEpisodeError("capacity blip")
        return {"ok": object()}

    retry_workflow._arun_episode = MethodType(_arun_episode, retry_workflow)

    result = await retry_workflow.arun_episode("engine", {"task_id": "x"})

    assert result is not None
    assert list(result) == ["ok"]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_proxy_workflow_does_not_retry_deterministic_or_cancelled_errors(
    retry_workflow: OpenAIProxyWorkflow,
):
    deterministic_calls = 0

    async def deterministic(self, engine, data):
        nonlocal deterministic_calls
        deterministic_calls += 1
        raise _DeterministicEpisodeError("bad task")

    retry_workflow._arun_episode = MethodType(deterministic, retry_workflow)
    with pytest.raises(_DeterministicEpisodeError):
        await retry_workflow.arun_episode("engine", {"task_id": "x"})
    assert deterministic_calls == 1

    cancelled_calls = 0

    async def cancelled(self, engine, data):
        nonlocal cancelled_calls
        cancelled_calls += 1
        raise asyncio.CancelledError()

    retry_workflow._arun_episode = MethodType(cancelled, retry_workflow)
    with pytest.raises(asyncio.CancelledError):
        await retry_workflow.arun_episode("engine", {"task_id": "x"})
    assert cancelled_calls == 1


def test_tau2_eval_wait_none_writes_failed_status_not_completed(tmp_path):
    trainer = object.__new__(Tau2AsyncEvalTrainer)
    statuses: list[dict] = []
    trainer.config = SimpleNamespace(
        eval_gconfig=SimpleNamespace(n_samples=1),
        experiment_mode="formal",
    )
    trainer.valid_dataloader = [[{"domain": "airline", "task_id": "0"}]]
    trainer._async_eval_rollout = SimpleNamespace(
        submit=lambda *args, **kwargs: None,
        wait=lambda count, timeout=None: None,
    )
    trainer._load_eval_checkpoint = lambda checkpoint_path, version: None
    trainer._write_eval_status = lambda version, payload: statuses.append(
        {"version": version, **payload}
    )

    with pytest.raises(TypeError):
        trainer._run_eval_job(5, tmp_path / "checkpoint", "workflow", {}, False)

    assert json.loads(json.dumps(statuses[-1]))["status"] == "failed"
    assert all(status["status"] != "completed" for status in statuses)


@pytest.mark.parametrize("global_step,forced", [(44, False), (45, True)])
def test_final_grpo_checkpoint_is_forced_off_regular_cadence(
    monkeypatch, global_step, forced
):
    from scripts.sao.async_eval import AsyncEvalPPOTrainer

    calls = []
    monkeypatch.setattr(
        AsyncEvalPPOTrainer,
        "_save_training_state",
        lambda self, **kwargs: calls.append(kwargs),
    )
    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.config = SimpleNamespace(total_train_steps=46)
    trainer._save_training_state(epoch=1, epoch_step=22, global_step=global_step)
    assert calls[0]["force"] is forced


def test_final_grpo_evaluation_uses_final_checkpoint_once(tmp_path):
    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.config = SimpleNamespace(
        total_train_steps=46,
        experiment_name="tau2",
        trial_name="test",
        cluster=SimpleNamespace(fileroot=str(tmp_path)),
    )
    trainer.valid_dataloader = [object()]
    calls = []
    trainer._enqueue_eval = lambda **kwargs: calls.append(kwargs)
    trainer._evaluate("workflow", {}, epoch=1, epoch_step=22, global_step=45)
    assert len(calls) == 1
    assert calls[0]["version"] == 46
    assert calls[0]["checkpoint_path"].endswith("epoch1epochstep22globalstep45")


@pytest.mark.parametrize("recovering", [False, True])
def test_initial_eval_consumes_trigger_without_requesting_unsaved_step1(
    monkeypatch, grpo_config, recovering
):
    from scripts.sao.async_eval import AsyncEvalPPOTrainer

    from areal.api import FinetuneSpec
    from areal.utils.evaluator import Evaluator

    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.config = grpo_config
    trainer.recover_info = object() if recovering else None
    trainer.valid_dataloader = [object()]
    trainer.evaluator = Evaluator(
        grpo_config.evaluator,
        FinetuneSpec(total_train_epochs=2, dataset_size=178, train_batch_size=8),
    )
    if recovering:
        trainer.evaluator.freq_ctl.check(epochs=0, steps=0)
    queued = []
    trainer._evaluate_fn = lambda *args: queued.append("backbone")
    trainer._enqueue_eval = lambda **kwargs: queued.append(kwargs["version"])
    trainer.check_evaluation = lambda: None
    monkeypatch.setattr(AsyncEvalPPOTrainer, "train", lambda *args, **kwargs: None)
    trainer.train(eval_workflow="workflow", eval_workflow_kwargs={})
    for step in range(20):
        trainer._evaluate("workflow", {}, 0, step, step)
        if step < 19:
            assert queued == ([] if recovering else ["backbone"])
    assert queued == ([20] if recovering else ["backbone", 20])
