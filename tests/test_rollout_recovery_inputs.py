# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dataclasses import dataclass

from torchdata.stateful_dataloader import StatefulDataLoader

from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    TimedResult,
    _RecoveryInputGenerator,
)


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
