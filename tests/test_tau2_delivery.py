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
    assert validate_tau2_recipe(grpo_config) == ("airline", "retail")

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
    assert grpo_config.saver.freq_steps == 5
    assert grpo_config.recover.freq_steps == 5
    assert grpo_config.eval_gconfig.n_samples == 4
    assert grpo_config.eval_gconfig.temperature == 0.0
    assert grpo_config.rollout.agent.drop_retry_orphans
    assert grpo_config.evaluation_rollout.agent.drop_retry_orphans


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


class _DefaultAgent:
    async def run(self, data, **kwargs):
        raise AssertionError("_run_agent should not be reached")


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
@pytest.mark.parametrize("reject_exhausted", [False, True])
async def test_proxy_workflow_exhausted_retries_lose_only_this_member(
    retry_workflow: OpenAIProxyWorkflow, monkeypatch, reject_exhausted
):
    calls = []

    async def _arun_episode(self, engine, data):
        calls.append(data)
        raise _TransientEpisodeError("provider down")

    retry_workflow._arun_episode = MethodType(_arun_episode, retry_workflow)
    monkeypatch.setattr(
        retry_workflow.agent,
        "reject_exhausted_episode",
        reject_exhausted,
        raising=False,
    )

    if reject_exhausted:
        assert await retry_workflow.arun_episode("engine", {"task_id": "x"}) is None
    else:
        with pytest.raises(_TransientEpisodeError):
            await retry_workflow.arun_episode("engine", {"task_id": "x"})
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_proxy_workflow_never_drops_a_deterministic_error(
    retry_workflow: OpenAIProxyWorkflow, monkeypatch
):
    calls = []

    # The deterministic error lands on the last attempt, after a transient one.
    async def _arun_episode(self, engine, data):
        calls.append(data)
        if len(calls) == 1:
            raise _TransientEpisodeError("provider down")
        raise _DeterministicEpisodeError("bug")

    retry_workflow._arun_episode = MethodType(_arun_episode, retry_workflow)
    monkeypatch.setattr(
        retry_workflow.agent, "reject_exhausted_episode", True, raising=False
    )

    with pytest.raises(_DeterministicEpisodeError):
        await retry_workflow.arun_episode("engine", {"task_id": "x"})


@pytest.mark.asyncio
async def test_proxy_export_validation_is_optional_for_default_agents():
    workflow = OpenAIProxyWorkflow(mode="inline", agent=_DefaultAgent())

    await workflow._validate_exported_interactions(
        {"leaf-1": object(), "leaf-2": object()},
        {"task_id": "x"},
    )


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


@pytest.mark.asyncio
async def test_tau2_proxy_export_validation_rejects_multiple_concat_leaves():
    from examples.tau2.agent import (
        Tau2AgentWorkflow,
        Tau2ExportCardinalityError,
        Tau2InfrastructureError,
    )

    workflow = OpenAIProxyWorkflow(
        mode="inline",
        agent=Tau2AgentWorkflow(),
        export_style="concat",
    )

    with pytest.raises(Tau2InfrastructureError, match="exactly one") as caught:
        await workflow._validate_exported_interactions(
            {"leaf-1": object(), "leaf-2": object()},
            {"domain": "telecom", "task_id": "branched"},
        )
    assert isinstance(caught.value.__cause__, Tau2ExportCardinalityError)
    assert Tau2AgentWorkflow.should_retry_episode(caught.value)


def test_tau2_concat_export_cardinality_guard_allows_single_or_non_concat():
    from examples.tau2.agent import Tau2AgentWorkflow

    Tau2AgentWorkflow.validate_exported_interactions(
        interactions={"leaf-1": object()},
        export_style="concat",
        data={"domain": "telecom", "task_id": "ok"},
    )
    # The user simulator may stop before any policy call; official τ² scores it.
    Tau2AgentWorkflow.validate_exported_interactions(
        interactions={},
        export_style="concat",
        data={"domain": "retail", "task_id": "policy-free"},
    )
    Tau2AgentWorkflow.validate_exported_interactions(
        interactions={"leaf-1": object(), "leaf-2": object()},
        export_style="individual",
        data={"domain": "telecom", "task_id": "legacy"},
    )


class _OfficialEvalController:
    """Dedicated eval controller whose episodes end as scripted per source_id."""

    def __init__(self, outcomes):
        self.dispatcher = self
        self.outcomes = outcomes
        self.pending = []
        self.kwargs = []

    def submit(self, row, workflow, workflow_kwargs, **kwargs):
        import torch

        assert kwargs["group_size"] == 1 and kwargs["is_eval"]
        self.kwargs.append(workflow_kwargs)
        task_id = len(self.kwargs) - 1
        outcome = self.outcomes[row["source_id"]]
        root = Path(workflow_kwargs["trajectory_dir"])
        if outcome == "policy_free":
            directory = root / row["source_id"]
            directory.mkdir(parents=True)
            (directory / "a.json").write_text(
                json.dumps(
                    {
                        "recorded_at": 1.0,
                        "status": "completed",
                        "policy_turns": 0,
                        "official_score": 1.0,
                        "stop_reason": "user_stop",
                    }
                )
            )
            result = None
        elif outcome == "infra":
            result = None
        else:
            result = SimpleNamespace(
                trajectory={
                    "rewards": torch.tensor([outcome]),
                    "task_budget_failure": torch.tensor([outcome == 0.0]),
                }
            )
        self.pending.append((task_id, result))
        return task_id

    def wait_results(self, count, timeout=None, with_task_ids=False):
        assert count == 1 and with_task_ids
        return [self.pending.pop()]


def _official_eval_trainer(tmp_path, outcomes, tasks):
    trainer = object.__new__(Tau2AsyncEvalTrainer)
    statuses: list[dict] = []
    trainer.config = SimpleNamespace(
        eval_gconfig=SimpleNamespace(n_samples=2),
        evaluation_rollout=SimpleNamespace(
            max_concurrent_rollouts=8, consumer_batch_size=1
        ),
        experiment_mode="formal",
        experiment_name="exp",
        trial_name="trial",
        cluster=SimpleNamespace(fileroot=str(tmp_path)),
    )
    trainer.valid_dataloader = [tasks]
    trainer._async_eval_rollout = _OfficialEvalController(outcomes)
    trainer._load_eval_checkpoint = lambda checkpoint_path, version: None
    trainer._write_eval_status = lambda version, payload: statuses.append(
        {"version": version, **payload}
    )
    return trainer, statuses


def test_tau2_in_training_eval_scores_and_excludes_like_official(tmp_path):
    outcomes = {
        "airline:44:trial0": 0.0,  # context-limited training-style export
        "airline:44:trial1": "policy_free",
        "retail:1:trial0": 1.0,
        "retail:1:trial1": "infra",
    }
    trainer, statuses = _official_eval_trainer(
        tmp_path,
        outcomes,
        [{"domain": "airline", "task_id": "44"}, {"domain": "retail", "task_id": "1"}],
    )

    trainer._run_eval_job(5, tmp_path / "checkpoint", "workflow", {"timeout": 9}, False)

    status = statuses[-1]
    assert status["status"] == "completed"
    assert status["domains"]["airline"]["episodes"] == 2
    assert status["domains"]["airline"]["reward_mean"] == 0.5
    assert status["domains"]["airline"]["policy_free_episodes"] == 1
    # The exhausted episode is excluded from the denominator and counted.
    assert status["domains"]["retail"]["episodes"] == 1
    assert status["domains"]["retail"]["reward_mean"] == 1.0
    assert status["domains"]["retail"]["infra_excluded_episodes"] == 1
    assert status["metrics"]["eval/infra_excluded_episodes"] == 1
    assert status["metrics"]["eval/reward_mean"] == pytest.approx(2 / 3)
    assert [r["source_id"] for r in status["infra_excluded"]] == ["retail:1:trial1"]
    kwargs = trainer._async_eval_rollout.kwargs
    assert all(k["evaluation"] and k["timeout"] == 9 for k in kwargs)
    assert len({k["trajectory_dir"] for k in kwargs}) == 1


def test_tau2_in_training_eval_failure_marks_status_failed(tmp_path):
    trainer, statuses = _official_eval_trainer(
        tmp_path, {}, [{"domain": "airline", "task_id": "0"}]
    )
    with pytest.raises(KeyError):
        trainer._run_eval_job(5, tmp_path / "checkpoint", "workflow", {}, False)
    assert json.loads(json.dumps(statuses[-1]))["status"] == "failed"
    assert all(status["status"] != "completed" for status in statuses)


@pytest.mark.parametrize("global_step,forced", [(44, False), (45, True)])
def test_final_grpo_checkpoint_is_forced_off_regular_cadence(
    monkeypatch, global_step, forced
):
    from areal import PPOTrainer

    calls = []
    monkeypatch.setattr(
        PPOTrainer,
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
    from areal import PPOTrainer
    from areal.api import FinetuneSpec
    from areal.utils.evaluator import Evaluator
    from areal.utils.recover import StepInfo

    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.config = grpo_config
    trainer.recover_info = (
        SimpleNamespace(
            last_step_info=StepInfo(
                epoch=0,
                epoch_step=4,
                global_step=4,
                steps_per_epoch=23,
            )
        )
        if recovering
        else None
    )
    trainer.valid_dataloader = [object()]
    trainer.evaluator = Evaluator(
        grpo_config.evaluator,
        FinetuneSpec(total_train_epochs=2, dataset_size=178, train_batch_size=8),
    )
    if recovering:
        trainer.evaluator.freq_ctl.check(epochs=0, steps=4)
        assert trainer.evaluator.freq_ctl.state_dict()["step"]["steps"] == 4
    queued = []
    trainer._evaluate_fn = lambda *args: queued.append("backbone")
    trainer._enqueue_eval = lambda **kwargs: queued.append(
        (kwargs["version"], str(kwargs["checkpoint_path"]))
    )
    trainer.check_evaluation = lambda: None
    monkeypatch.setattr(PPOTrainer, "train", lambda *args, **kwargs: None)
    trainer.train(eval_workflow="workflow", eval_workflow_kwargs={})
    if recovering:
        assert queued == [(5, queued[0][1])]
        assert queued[0][1].endswith("epoch0epochstep4globalstep4")
        assert trainer.evaluator.freq_ctl.state_dict()["step"]["steps"] == 5
        trainer.train(eval_workflow="workflow", eval_workflow_kwargs={})
        assert queued == [(5, queued[0][1])]
        assert trainer.evaluator.freq_ctl.state_dict()["step"]["steps"] == 5
        trainer._evaluate("workflow", {}, 0, 5, 5)
        assert queued == [(5, queued[0][1])]
        return
    for step in range(5):
        trainer._evaluate("workflow", {}, 0, step, step)
        if step < 4:
            assert queued == ["backbone"]
    assert queued == ["backbone", (5, queued[1][1])]


def test_recovered_non_frequency_step_catches_up_counter_without_eval(
    monkeypatch, grpo_config
):
    from areal import PPOTrainer
    from areal.api import FinetuneSpec
    from areal.utils.evaluator import Evaluator
    from areal.utils.recover import StepInfo

    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.config = grpo_config
    trainer.recover_info = SimpleNamespace(
        last_step_info=StepInfo(
            epoch=0,
            epoch_step=18,
            global_step=18,
            steps_per_epoch=23,
        )
    )
    trainer.valid_dataloader = [object()]
    trainer.evaluator = Evaluator(
        grpo_config.evaluator,
        FinetuneSpec(total_train_epochs=2, dataset_size=178, train_batch_size=8),
    )
    trainer.evaluator.freq_ctl.check(epochs=0, steps=18)
    queued = []
    trainer._enqueue_eval = lambda **kwargs: queued.append(kwargs["version"])
    trainer.check_evaluation = lambda: None
    monkeypatch.setattr(PPOTrainer, "train", lambda *args, **kwargs: None)

    trainer.train(eval_workflow="workflow", eval_workflow_kwargs={})

    assert queued == []
    assert trainer.evaluator.freq_ctl.state_dict()["step"]["steps"] == 19


def test_tau2_episode_retry_covers_any_exception_like_official_runner():
    import asyncio

    import litellm

    from examples.tau2.agent import (
        OFFICIAL_RETRY_ATTEMPTS,
        OFFICIAL_RETRY_DELAY,
        Tau2AgentWorkflow,
        Tau2InfrastructureError,
    )

    # tau2.runner.progress retries every Exception, including context overflow
    # and deterministic errors; shutdown/cancellation is not an Exception.
    for failure in (
        Tau2InfrastructureError("areal_context_limit: exhausted"),
        litellm.RateLimitError(
            message="Too many requests", model="dummy", llm_provider="openai"
        ),
        ValueError("UserMessage must have either content or tool_calls"),
    ):
        assert Tau2AgentWorkflow.should_retry_episode(failure)
    assert not isinstance(asyncio.CancelledError(), Exception)
    workflow = Tau2AgentWorkflow()
    assert (workflow.infra_retries, workflow.retry_delay) == (3, 1.0)
    assert (OFFICIAL_RETRY_ATTEMPTS, OFFICIAL_RETRY_DELAY) == (3, 1.0)


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
    monkeypatch.setenv("TAU2_ACTOR_PATH", "Qwen/Qwen3.5-4B")
    monkeypatch.setenv("TAU2_CRITIC_INIT_PATH", "Qwen/Qwen3.5-4B")
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    critic, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"], Tau2PPOConfig
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


@pytest.mark.asyncio
async def test_proxy_replays_tau2_episode_three_more_times_with_official_delay(
    monkeypatch,
):
    from examples.tau2.agent import Tau2AgentWorkflow

    sleeps = []

    async def _record_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(
        "areal.experimental.openai.proxy.workflow.asyncio.sleep", _record_sleep
    )
    monkeypatch.setattr(
        "areal.experimental.openai.proxy.workflow.stats_tracker.get",
        lambda *_args, **_kwargs: SimpleNamespace(scalar=lambda **_kw: None),
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=Tau2AgentWorkflow())
    calls = []

    async def _arun_episode(self, engine, data):
        calls.append(data)
        raise ValueError("UserMessage must have either content or tool_calls")

    workflow._arun_episode = MethodType(_arun_episode, workflow)
    # Exhausted retries lose this episode from its group (ADR 0011).
    assert await workflow.arun_episode("engine", {"task_id": "x"}) is None
    assert len(calls) == 4
    assert sleeps == [1.0, 1.0, 1.0]


def _thinking_eval(config: Tau2PPOConfig, template: str, export: str):
    config = copy.deepcopy(config)
    config.econfig.enable_thinking = True
    config.rollout.agent.chat_template_type = template
    config.rollout.agent.export_style = export
    return config


def test_thinking_is_evaluation_only_and_uses_the_full_template(
    grpo_config: Tau2PPOConfig,
):
    thinking = _thinking_eval(grpo_config, "hf", "individual")
    assert validate_tau2_recipe(thinking, evaluation=True) == ("airline", "retail")
    with pytest.raises(ValueError, match="training requires enable_thinking=false"):
        validate_tau2_recipe(thinking)
    # Concatenated tokens would keep every earlier reasoning block in context.
    with pytest.raises(ValueError, match="hf template"):
        validate_tau2_recipe(
            _thinking_eval(grpo_config, "concat", "concat"), evaluation=True
        )
    non_thinking_hf = copy.deepcopy(grpo_config)
    non_thinking_hf.rollout.agent.chat_template_type = "hf"
    with pytest.raises(ValueError, match="concat template"):
        validate_tau2_recipe(non_thinking_hf, evaluation=True)
