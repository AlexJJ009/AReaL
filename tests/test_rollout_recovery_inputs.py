# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

from examples.tau2.evaluation import Tau2AsyncEvalTrainer

from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    TimedResult,
    _RecoveryInputGenerator,
)
from areal.trainer.rl_trainer import PPOTrainer
from areal.trainer.rollout_batch import FiniteEpochBatcher


@dataclass
class _Input:
    task_id: int
    data: dict


class _Staleness:
    def on_rollout_enqueued(self):
        pass

    def get_capacity(self):
        return 1

    def get_pending_limit(self):
        return 1


class _Loader:
    batch_size = 2

    def __iter__(self):
        return iter([[{"id": 0}, {"id": 1}], [{"id": 2}, {"id": 3}]])


def test_dispatcher_recovery_drops_only_consumed_results():
    dispatcher = BatchTaskDispatcher(
        max_queue_size=8,
        task_factory=lambda _item: None,
        staleness_manager=_Staleness(),
        deterministic_order=True,
    )
    dispatcher.submit_task_input(_Input(0, {"id": 0}))
    dispatcher.submit_task_input(_Input(1, {"id": 1}))
    dispatcher._pending_results[0] = TimedResult(0, "done-0", 0)

    assert dispatcher.get_input_recovery_state() == {
        "outstanding": [{"id": 0}, {"id": 1}]
    }

    assert dispatcher.wait_results(1, timeout=0.01) == ["done-0"]
    assert dispatcher.get_input_recovery_state() == {"outstanding": [{"id": 1}]}

    dispatcher._pending_results[1] = TimedResult(1, "done-1", 1)
    assert dispatcher.wait_results(1, timeout=0.01) == ["done-1"]
    assert dispatcher.get_input_recovery_state() == {"outstanding": []}


def test_recovery_state_replays_raw_inputs_not_finished_trajectory_outputs():
    dispatcher = BatchTaskDispatcher(
        max_queue_size=8,
        task_factory=lambda _item: None,
        staleness_manager=_Staleness(),
        deterministic_order=True,
    )
    dispatcher.submit_task_input(_Input(0, {"id": 0}))
    dispatcher._pending_results[0] = TimedResult(
        0,
        {
            "id": 0,
            "behavior_logprobs": [-0.1],
            "versions": [3],
        },
        0,
    )

    state = dispatcher.get_input_recovery_state()

    assert state == {"outstanding": [{"id": 0}]}
    assert "behavior_logprobs" not in state["outstanding"][0]
    assert "versions" not in state["outstanding"][0]


def test_partial_batch_buffer_keeps_unsubmitted_items_until_acknowledged():
    buffer = deque()
    next_id = 0

    def make_input(item):
        nonlocal next_id
        task_input = _Input(next_id, item)
        next_id += 1
        return task_input

    generator = _RecoveryInputGenerator(_Loader(), True, buffer, make_input)

    first = next(generator)
    assert first.data == {"id": 0}
    assert list(buffer) == [{"id": 0}, {"id": 1}]

    generator.acknowledge_submission(first)
    assert list(buffer) == [{"id": 1}]

    second = next(generator)
    assert second.data == {"id": 1}
    assert list(buffer) == [{"id": 1}]


def test_recovery_union_excludes_consumed_and_preserves_partial_batch_order():
    dispatcher = BatchTaskDispatcher(
        max_queue_size=8,
        task_factory=lambda _item: None,
        staleness_manager=_Staleness(),
        deterministic_order=True,
    )
    dispatcher.submit_task_input(_Input(0, {"id": 0}))
    dispatcher.submit_task_input(_Input(1, {"id": 1}))
    dispatcher._pending_results[0] = TimedResult(0, "done-0", 0)

    partial_batch = deque([{"id": 2}, {"id": 3}])
    state = dispatcher.get_input_recovery_state()
    recovered = state["outstanding"] + list(partial_batch)
    assert recovered == [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]

    dispatcher.wait_results(1, timeout=0.01)
    state = dispatcher.get_input_recovery_state()
    recovered = state["outstanding"] + list(partial_batch)
    assert recovered == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_fresh_stateful_loader_replays_prefetched_rows_before_restored_tail():
    rows = [{"id": i} for i in range(48)]
    loader = StatefulDataLoader(
        rows,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    iterator = iter(loader)
    for _ in range(9):
        next(iterator)
    loader_state = loader.state_dict()

    restored_loader = StatefulDataLoader(
        rows,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    restored_loader.load_state_dict(loader_state)

    replay = deque({"id": i} for i in range(20, 36))
    next_id = 0

    def make_input(item):
        nonlocal next_id
        task_input = _Input(next_id, item)
        next_id += 1
        return task_input

    generator = _RecoveryInputGenerator(
        restored_loader,
        True,
        replay,
        make_input,
    )
    recovered_ids = []
    for _ in range(28):
        task_input = next(generator)
        recovered_ids.append(task_input.data["id"])
        generator.acknowledge_submission(task_input)

    assert recovered_ids == list(range(20, 48))


class _EpochSampler:
    def __init__(self, loader):
        self._loader = loader
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        self._loader.reset()


class _StatefulLoaderProxy:
    batch_size = 8

    def __init__(self, rows):
        self.rows = rows
        self.sampler = _EpochSampler(self)
        self.reset()

    def reset(self):
        self._loader = StatefulDataLoader(
            self.rows,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

    def state_dict(self):
        return self._loader.state_dict()

    def load_state_dict(self, state):
        self._loader.load_state_dict(state)


class _FiniteRollout:
    def __init__(self, replay):
        self._input_recovery_buffer = deque(replay)
        self._next_task_id = 0

    def prepare_batch(self, dataloader, *args, finite_epoch=True, **kwargs):
        assert finite_epoch is True
        if not hasattr(self, "data_generator"):

            def make_input(item):
                task_input = _Input(self._next_task_id, item)
                self._next_task_id += 1
                return task_input

            self.data_generator = _RecoveryInputGenerator(
                dataloader,
                finite_epoch,
                self._input_recovery_buffer,
                make_input,
            )

        batch = []
        for _ in range(dataloader.batch_size):
            try:
                task_input = next(self.data_generator)
            except StopIteration:
                break
            batch.append(task_input.data)
            self.data_generator.acknowledge_submission(task_input)
        return batch


def test_tau2_finite_recovery_replays_outstanding_tail_then_next_epoch():
    """Recover logical step 20 after async prefetch reaches EOF."""
    rows = [{"id": i} for i in range(178)]
    prefetched_loader = _StatefulLoaderProxy(rows)
    iterator = iter(prefetched_loader)
    for _ in range(23):
        next(iterator)
    prefetched_state = prefetched_loader.state_dict()
    assert prefetched_state["_sampler_iter_state"]["samples_yielded"] == 178

    restored_loader = _StatefulLoaderProxy(rows)
    restored_loader.load_state_dict(prefetched_state)
    rollout = _FiniteRollout({"id": i} for i in range(160, 178))
    batcher = FiniteEpochBatcher(rollout, epoch=0)

    resumed_ids = []
    while len(resumed_ids) < 18 + 178:
        resumed_ids.extend(row["id"] for row in batcher(restored_loader))

    assert resumed_ids == list(range(160, 178)) + list(range(178))
    assert len(resumed_ids) + 160 == 356
    assert restored_loader.sampler.epochs == [1]


def test_tau2_recovery_policy_requires_raw_rollout_inputs():
    cfg = SimpleNamespace(num_critic_only_steps=0, critic_updates_before_actor=False)

    base = object.__new__(PPOTrainer)
    assert base._should_recover_rollout_inputs(cfg) is False
    assert base._expected_recovery_trainer_state(cfg)["rollout_recovery_policy"] == 0
    base._validate_recovered_rollout_input_state(None)

    tau2 = object.__new__(Tau2AsyncEvalTrainer)
    assert tau2._should_recover_rollout_inputs(cfg) is True
    assert (
        tau2._expected_recovery_trainer_state(cfg)["rollout_recovery_policy"]
        == "replay_raw_inputs_discard_generated_trajectories"
    )
    with pytest.raises(ValueError, match="missing rollout_input_state"):
        tau2._validate_recovered_rollout_input_state(None)


def test_lazy_proxy_start_inherits_recovered_version():
    import threading
    from unittest.mock import AsyncMock, Mock

    from areal.infra.controller.rollout_controller import RolloutController

    controller = RolloutController.__new__(RolloutController)
    controller._version_lock = threading.Lock()
    controller._version = 0
    controller._proxy_started = False
    controller.server_infos = [object()]
    controller._collective_rpc = Mock()
    controller._proxy_collective_rpc = Mock()
    controller._async_start_proxy = AsyncMock()
    controller.set_version(20)
    controller._proxy_collective_rpc.assert_not_called()
    controller.start_proxy()
    controller._proxy_collective_rpc.assert_called_once_with(
        "set_version", version=20, http_timeout=60.0
    )
    assert controller._proxy_started
