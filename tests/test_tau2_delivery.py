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
from examples.tau2.train import (
    get_tau2_dataset,
    run_fixed_policy_collection,
    validate_tau2_recipe,
)
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


def test_context_error_wrapped_as_rate_limit_does_not_retry():
    import litellm

    from examples.tau2.agent import Tau2AgentWorkflow, Tau2InfrastructureError

    cause = litellm.RateLimitError(
        message="areal_context_limit: exhausted", model="dummy", llm_provider="openai"
    )
    failure = Tau2InfrastructureError(str(cause))
    failure.__cause__ = cause
    assert not Tau2AgentWorkflow.should_retry_episode(failure)
    transient = litellm.RateLimitError(
        message="Too many requests", model="dummy", llm_provider="openai"
    )
    failure = Tau2InfrastructureError(str(transient))
    failure.__cause__ = transient
    assert Tau2AgentWorkflow.should_retry_episode(failure)


def test_fixed_policy_collection_rolls_finite_tail_without_updates():
    forbidden_calls = []

    class ForbiddenActor:
        def __init__(self):
            self.cleared = []

        def compute_advantages(self, *args, **kwargs):
            forbidden_calls.append("compute_advantages")
            raise AssertionError("collector must not compute advantages")

        def ppo_update(self, *args, **kwargs):
            forbidden_calls.append("ppo_update")
            raise AssertionError("collector must not update actor")

        def update_weights(self, *args, **kwargs):
            forbidden_calls.append("update_weights")
            raise AssertionError("collector must not sync weights")

        def clear_batches(self, *targets):
            self.cleared.append(targets)

    class FakeRollout:
        def __init__(self):
            self.calls = []
            self.consumed_without_update = 0
            self.versions = []

        def rollout_batch(self, rows, **kwargs):
            self.calls.append((list(rows), kwargs))
            return [
                {"source_id": row["source_id"], "remote": index}
                for index, row in enumerate(rows)
            ]

        def on_batch_consumed_without_update(self):
            self.consumed_without_update += 1

        def set_version(self, version):
            self.versions.append(version)

    rows = [{"source_id": f"task-{index}"} for index in range(178)]
    dataloader = [rows[index : index + 16] for index in range(0, len(rows), 16)]
    rollout = FakeRollout()
    actor = ForbiddenActor()
    lifecycle = []
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            total_train_steps=12,
            gconfig=SimpleNamespace(
                n_samples=1,
                reward_normalization=False,
                drop_incomplete_group=False,
            ),
        ),
        train_dataloader=dataloader,
        rollout=rollout,
        actor=actor,
        _should_offload_rollout=True,
        _requires_proxy_workflow=lambda workflow: True,
        _ensure_proxy_started=lambda: lifecycle.append("proxy"),
        _onload_rollout=lambda: lifecycle.append("onload"),
        _offload_rollout=lambda: lifecycle.append("offload"),
    )

    collected = run_fixed_policy_collection(
        trainer,
        workflow="examples.tau2.agent.Tau2AgentWorkflow",
        workflow_kwargs={"econfig": {}},
    )

    assert collected == 12
    assert lifecycle == ["proxy", "onload", "offload"]
    assert len(rollout.calls) == 12
    assert sum(len(batch) for batch, _ in rollout.calls) == 178
    assert len(rollout.calls[-1][0]) == 2
    assert rollout.consumed_without_update == 12
    assert len(actor.cleared) == 12
    assert len(actor.cleared[-1]) == 2
    assert rollout.versions == []
    assert forbidden_calls == []
    assert rollout.calls[0][1]["workflow"] == "examples.tau2.agent.Tau2AgentWorkflow"
    assert rollout.calls[0][1]["group_size"] == 1


def test_fixed_policy_collection_rejects_incomplete_rollout_batch_and_clears_results():
    class FakeActor:
        def __init__(self):
            self.cleared = []

        def clear_batches(self, *targets):
            self.cleared.append(targets)

    class RejectingRollout:
        def __init__(self):
            self.consumed_without_update = 0

        def rollout_batch(self, rows, **kwargs):
            return [{"source_id": rows[0]["source_id"]}]

        def on_batch_consumed_without_update(self):
            self.consumed_without_update += 1

    actor = FakeActor()
    rollout = RejectingRollout()
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            total_train_steps=None,
            gconfig=SimpleNamespace(
                n_samples=1,
                reward_normalization=False,
                drop_incomplete_group=False,
            ),
        ),
        train_dataloader=[[{"source_id": "a"}, {"source_id": "b"}]],
        rollout=rollout,
        actor=actor,
        _should_offload_rollout=False,
        _requires_proxy_workflow=lambda workflow: False,
        _ensure_proxy_started=lambda: None,
    )

    with pytest.raises(RuntimeError, match="incomplete rollout batch"):
        run_fixed_policy_collection(
            trainer,
            workflow="examples.tau2.agent.Tau2AgentWorkflow",
            workflow_kwargs={},
        )

    assert rollout.consumed_without_update == 0
    assert len(actor.cleared) == 1
    assert actor.cleared[0] == ({"source_id": "a"},)


def test_tail_dispatch_preserves_native_token_mean_loss_and_gradient():
    import torch

    from examples.tau2.evaluation import repeat_groups_for_dispatch

    from areal.infra.controller.train_controller import _dispatch_tensors
    from areal.utils.functional import ppo_actor_loss_fn

    masks = [torch.ones(8, 3, dtype=torch.bool), torch.ones(8, 3, dtype=torch.bool)]
    masks[1][:, -1] = False
    groups = [
        {
            "attention_mask": mask,
            "loss_mask": mask,
            "feature": torch.arange(24, dtype=torch.float64).reshape(8, 3) / 24 + value,
        }
        for mask, value in zip(masks, (0.5, 1.5))
    ]
    physical, factor = repeat_groups_for_dispatch(groups, 4)
    assert factor == 2 and len(physical) == 4
    assert physical[0] is not physical[2]
    assert physical[0]["feature"] is physical[2]["feature"]
    shards, _ = _dispatch_tensors(physical, dp_size=4)
    assert [len(shard) for shard in shards] == [1, 1, 1, 1]
    weight = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    def loss(rows):
        features = torch.cat([row["feature"] for row in rows])
        mask = torch.cat([row["loss_mask"] for row in rows])
        logp = weight * features - 1.0
        advantage = (
            torch.arange(8, dtype=torch.float64)
            .sub(3.5)[:, None]
            .expand(8, 3)
            .repeat(len(rows), 1)
        )
        value, _ = ppo_actor_loss_fn(
            logprobs=logp,
            proximal_logprobs=torch.full_like(logp, -1.0),
            old_logprobs=torch.full_like(logp, -1.1),
            advantages=advantage,
            eps_clip=0.2,
            eps_clip_higher=0.28,
            loss_mask=mask,
        )
        return value

    expected = loss(groups)
    total_tokens = sum(row["loss_mask"].sum() for row in physical)
    # FSDP scales by world size / global token weight, then averages gradients.
    actual = (
        sum(
            loss(shard) * shard[0]["loss_mask"].sum() / total_tokens * 4
            for shard in shards
        )
        / 4
    )
    expected_grad = torch.autograd.grad(expected, weight, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, weight)[0]
    assert expected_grad.abs() > 1e-8
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


def test_grpo_and_critic_recipes_disable_warmup(grpo_config, monkeypatch, tmp_path):
    from areal.api.cli_args import PPOConfig

    monkeypatch.setenv("TAU2_ACTOR_PATH", "Qwen/Qwen3.5-4B")
    monkeypatch.setenv("TAU2_CRITIC_INIT_PATH", "Qwen/Qwen3.5-4B")
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    critic, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"], PPOConfig
    )
    for optimizer in (grpo_config.actor.optimizer, critic.critic.optimizer):
        assert optimizer.warmup_steps == 0
        assert optimizer.warmup_steps_proportion == 0.0
        assert optimizer.lr_scheduler_type == "constant"


def test_finite_rollout_restarts_for_second_epoch_without_resampling_tail():
    from collections import deque

    from torch.utils.data import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    from examples.tau2.utils import FiniteEpochBatcher

    from areal.infra.workflow_executor import _RecoveryInputGenerator

    rows = [{"id": i} for i in range(5)]
    sampler = DistributedSampler(rows, num_replicas=1, rank=0, shuffle=False)
    loader = StatefulDataLoader(
        rows, batch_size=3, sampler=sampler, collate_fn=lambda x: x
    )

    class Rollout:
        def prepare_batch(self, dataloader, **kwargs):
            assert kwargs == {"finite_epoch": True, "fail_on_rejection": True}
            if not hasattr(self, "data_generator"):
                self.data_generator = _RecoveryInputGenerator(
                    dataloader,
                    True,
                    deque(),
                    lambda row: SimpleNamespace(task_id=row["id"], data=row),
                )
            batch = []
            for _ in range(dataloader.batch_size):
                try:
                    item = next(self.data_generator)
                except StopIteration:
                    break
                batch.append(item.data)
                self.data_generator.acknowledge_submission(item)
            return batch

    prepare = FiniteEpochBatcher(Rollout())
    batches = [prepare(dataloader=loader) for _ in range(4)]
    assert [len(batch) for batch in batches] == [3, 2, 3, 2]
    assert [
        [row["id"] for batch in batches[i : i + 2] for row in batch] for i in (0, 2)
    ] == [list(range(5)), list(range(5))]
    assert sampler.epoch == 1
