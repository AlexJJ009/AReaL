# SPDX-License-Identifier: Apache-2.0
"""Native metric ownership and checkpoint identity for asynchronous evaluation."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from examples.tau2.evaluation import Tau2AsyncEvalTrainer

from areal.trainer.async_eval import AsyncEvalPPOTrainer
from areal.utils.stats_logger import StatsLogger


def test_tau2_stats_extend_native_critic_collection(monkeypatch):
    """A scenario counter must not replace the native actor/critic exports."""
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)
    exports = []

    def engine(role):
        def export():
            exports.append(role)
            return {"loss": 2.0} if role == "critic" else {role: 1.0}

        return SimpleNamespace(export_stats=export)

    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.actor, trainer.critic, trainer.rollout = map(
        engine, ("actor", "critic", "rollout")
    )
    trainer.eval_rollout = None
    trainer.config = SimpleNamespace(num_critic_only_steps=0)
    trainer._batch_counts = {"tau2_batch/real_episodes": 16}
    committed = []
    trainer.stats_logger = SimpleNamespace(commit=lambda *args: committed.append(args))
    trainer._export_and_commit_stats(1, 22, 45)
    assert exports == ["actor", "critic", "rollout"]
    assert committed == [
        (
            1,
            22,
            45,
            {
                "actor": 1.0,
                "critic/loss": 2.0,
                "rollout": 1.0,
                "tau2_batch/real_episodes": 16,
            },
        )
    ]


def test_late_eval_keeps_checkpoint_axis_without_training_completion(monkeypatch):
    """A checkpoint-5 result arriving after update 10 is not update 11."""
    messages, uploads, axes = [], [], []
    monkeypatch.setattr("areal.utils.stats_logger.wandb.run", None)
    monkeypatch.setattr("areal.utils.stats_logger.logger.info", messages.append)
    monkeypatch.setattr(
        "areal.utils.stats_logger.wandb.log",
        lambda data, step: uploads.append((step, data)),
    )
    monkeypatch.setattr(
        "areal.utils.stats_logger.wandb.define_metric",
        lambda key, **kw: axes.append((key, kw)),
    )
    monkeypatch.setattr(
        "areal.utils.stats_logger.swanlab.log", lambda *args, **kw: None
    )
    logger = object.__new__(StatsLogger)
    logger._last_commit_step = -1
    logger.summary_writer = None
    logger.ft_spec = SimpleNamespace(
        total_train_epochs=2, steps_per_epoch=23, total_train_steps=46
    )
    logger.commit(0, 9, 9, {"actor/loss": 0.1})
    logger.log_metrics(
        {"eval/reward_mean": 0.5}, step=5, step_metric="eval/checkpoint_step"
    )
    logger.commit(0, 10, 10, {"actor/loss": 0.2})
    assert [step for step, _ in uploads] == [0, 1, 2]
    assert uploads[0][1]["train/global_step"] == 10
    assert uploads[1][1]["eval/checkpoint_step"] == 5
    assert uploads[2][1]["train/global_step"] == 11
    assert sum("Train step " in msg for msg in messages) == 2
    assert ("eval/reward_mean", {"step_metric": "eval/checkpoint_step"}) in axes
    assert logger.state_dict()["last_commit_step"] == 2


def test_completed_eval_is_logged_once_on_controller_thread():
    """Polling followed by close must not upload the same result twice."""
    trainer = object.__new__(AsyncEvalPPOTrainer)
    events = []
    trainer.stats_logger = SimpleNamespace(
        log_metrics=lambda data, **kw: events.append((data, kw))
    )
    future = Future()
    future.set_result({"version": 5, "metrics": {"eval/reward_mean": 0.75}})
    trainer._async_eval_futures = [future]
    trainer.check_evaluation()
    trainer._drain_evaluations()
    assert events == [
        ({"eval/reward_mean": 0.75}, {"step": 5, "step_metric": "eval/checkpoint_step"})
    ]


def test_failed_eval_does_not_replay_previous_success_on_close():
    trainer = object.__new__(AsyncEvalPPOTrainer)
    logged = []
    trainer.stats_logger = SimpleNamespace(
        log_metrics=lambda *args, **kw: logged.append(kw)
    )
    good, bad = Future(), Future()
    good.set_result({"version": 5, "metrics": {"eval/reward_mean": 1.0}})
    bad.set_exception(RuntimeError("incomplete evaluation"))
    trainer._async_eval_futures = [good, bad]
    with pytest.raises(RuntimeError, match="incomplete evaluation"):
        trainer.check_evaluation()
    trainer._drain_evaluations()
    assert len(logged) == 1


def test_native_eval_accepts_other_gpu_counts_and_deduplicates_colocation():
    trainer = object.__new__(AsyncEvalPPOTrainer)
    trainer.config = SimpleNamespace(
        evaluation_rollout=SimpleNamespace(backend="sglang:d1p1t1")
    )
    trainer.scheduler = SimpleNamespace(
        gpu_devices=[0, 1, 2, 3],
        _workers={
            "actor": [SimpleNamespace(gpu_devices=[0, 1])],
            "critic": [SimpleNamespace(gpu_devices=[0, 1])],
            "rollout": [SimpleNamespace(gpu_devices=[2])],
        },
    )
    trainer._assert_dedicated_eval_preconditions()


def test_resumed_wandb_upload_starts_after_existing_run_events(monkeypatch):
    uploads = []
    monkeypatch.setattr("areal.utils.stats_logger.wandb.run", SimpleNamespace(step=19))
    monkeypatch.setattr(
        "areal.utils.stats_logger.wandb.log",
        lambda data, step: uploads.append((step, data)),
    )
    monkeypatch.setattr(
        "areal.utils.stats_logger.wandb.define_metric", lambda *args, **kw: None
    )
    monkeypatch.setattr(
        "areal.utils.stats_logger.swanlab.log", lambda *args, **kw: None
    )
    logger = object.__new__(StatsLogger)
    logger._last_commit_step = 10
    logger.summary_writer = None
    logger.log_metrics(
        {"eval/reward_mean": 0.5}, step=5, step_metric="eval/checkpoint_step"
    )
    assert uploads == [(19, {"eval/reward_mean": 0.5, "eval/checkpoint_step": 5})]
    assert logger.state_dict()["last_commit_step"] == 19


@pytest.mark.parametrize("scheduler_type", ["ray", "slurm"])
def test_unsupported_scheduler_fails_before_worker_initialization(scheduler_type):
    trainer = object.__new__(AsyncEvalPPOTrainer)
    config = SimpleNamespace(scheduler=SimpleNamespace(type=scheduler_type))
    with pytest.raises(NotImplementedError, match="LocalScheduler"):
        trainer._init_impl(config)
