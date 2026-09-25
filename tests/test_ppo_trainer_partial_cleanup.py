# SPDX-License-Identifier: Apache-2.0
"""Regression tests for PPOTrainer cleanup after constructor failures."""

from __future__ import annotations

from typing import Any

import pytest

import areal.trainer.rl_trainer as rl_trainer_module
from areal.trainer.rl_trainer import PPOTrainer


class _CleanupProbe:
    def __init__(self, calls: list[str], name: str):
        self._calls = calls
        self._name = name

    def finalize(self) -> None:
        self._calls.append(f"{self._name}.finalize")

    def close(self) -> None:
        self._calls.append(f"{self._name}.close")

    def destroy(self) -> None:
        self._calls.append(f"{self._name}.destroy")


class _SchedulerProbe:
    def __init__(self, calls: list[str]):
        self._calls = calls

    def delete_workers(self, role: str | None, reverse_order: bool) -> None:
        self._calls.append(
            f"scheduler.delete_workers(role={role},reverse_order={reverse_order})"
        )


def test_constructor_failure_before_saver_cleans_existing_components_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure before saver initialization still tears down created components."""
    calls: list[str] = []
    instances: list[PPOTrainer] = []

    def fail_before_saver(
        self: PPOTrainer,
        config: Any,
        train_dataset: Any = None,
        valid_dataset: Any = None,
    ) -> None:
        del config, train_dataset, valid_dataset
        instances.append(self)
        self.scheduler = _SchedulerProbe(calls)
        self._train_rdataset = _CleanupProbe(calls, "train_rdataset")
        self._valid_rdataset = _CleanupProbe(calls, "valid_rdataset")
        self.data_controller = _CleanupProbe(calls, "data_controller")
        self.stats_logger = _CleanupProbe(calls, "stats_logger")
        self.eval_rollout = _CleanupProbe(calls, "eval_rollout")
        self.rollout = _CleanupProbe(calls, "rollout")
        self.teacher = _CleanupProbe(calls, "teacher")
        self.ref = _CleanupProbe(calls, "ref")
        self.critic = _CleanupProbe(calls, "critic")
        self.actor = _CleanupProbe(calls, "actor")
        raise RuntimeError("boom before saver")

    monkeypatch.setattr(PPOTrainer, "_init_impl", fail_before_saver)
    monkeypatch.setattr(
        rl_trainer_module.perf_tracer,
        "save",
        lambda **kwargs: calls.append(f"perf_tracer.save({kwargs['force']})"),
    )

    with pytest.raises(RuntimeError, match="boom before saver"):
        PPOTrainer(config=object())

    assert calls == [
        "train_rdataset.close",
        "valid_rdataset.close",
        "data_controller.destroy",
        "stats_logger.close",
        "eval_rollout.destroy",
        "rollout.destroy",
        "teacher.destroy",
        "ref.destroy",
        "critic.destroy",
        "actor.destroy",
        "scheduler.delete_workers(role=None,reverse_order=True)",
        "perf_tracer.save(True)",
    ]

    instances[0].close()
    assert calls == [
        "train_rdataset.close",
        "valid_rdataset.close",
        "data_controller.destroy",
        "stats_logger.close",
        "eval_rollout.destroy",
        "rollout.destroy",
        "teacher.destroy",
        "ref.destroy",
        "critic.destroy",
        "actor.destroy",
        "scheduler.delete_workers(role=None,reverse_order=True)",
        "perf_tracer.save(True)",
    ]


def test_constructor_failure_after_saver_cleans_only_present_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only initialized cleanup targets are called, each exactly once."""
    calls: list[str] = []
    instances: list[PPOTrainer] = []

    def fail_after_saver(
        self: PPOTrainer,
        config: Any,
        train_dataset: Any = None,
        valid_dataset: Any = None,
    ) -> None:
        del config, train_dataset, valid_dataset
        instances.append(self)
        self.saver = _CleanupProbe(calls, "saver")
        self.stats_logger = _CleanupProbe(calls, "stats_logger")
        self.actor = _CleanupProbe(calls, "actor")
        raise RuntimeError("boom after saver")

    monkeypatch.setattr(PPOTrainer, "_init_impl", fail_after_saver)
    monkeypatch.setattr(
        rl_trainer_module.perf_tracer,
        "save",
        lambda **kwargs: calls.append(f"perf_tracer.save({kwargs['force']})"),
    )

    with pytest.raises(RuntimeError, match="boom after saver"):
        PPOTrainer(config=object())

    assert calls == [
        "saver.finalize",
        "stats_logger.close",
        "actor.destroy",
        "perf_tracer.save(True)",
    ]

    instances[0].close()
    assert calls == [
        "saver.finalize",
        "stats_logger.close",
        "actor.destroy",
        "perf_tracer.save(True)",
    ]


@pytest.mark.parametrize("critic_only", [False, True])
def test_critic_finalize_memory_window_temporarily_offloads_only_for_sao(
    critic_only: bool,
) -> None:
    calls: list[str] = []
    trainer = object.__new__(PPOTrainer)
    trainer.actor = object()
    trainer.critic = object()
    trainer._should_offload_critic = True
    trainer._should_offload_actor = False
    trainer._offload_model = lambda engine, role: calls.append(f"offload.{role}")
    trainer._onload_model = lambda engine, role: calls.append(f"onload.{role}")

    actor_was_offloaded = trainer._begin_critic_finalize_memory_window(critic_only)
    trainer._end_critic_finalize_memory_window(actor_was_offloaded)

    if critic_only:
        assert not actor_was_offloaded
        assert calls == []
    else:
        assert actor_was_offloaded
        assert calls == ["offload.actor", "onload.actor"]
