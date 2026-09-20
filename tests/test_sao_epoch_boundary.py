# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest

from areal.api.cli_args import TrainDatasetConfig
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher
from areal.utils.dataloader import create_dataloader


class _SampleIdDataset:
    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"sample_id": index}


@dataclass
class _Input:
    task_id: int
    data: dict[str, Any]


@dataclass
class _Result:
    task_id: int
    data: dict[str, Any]


class _VersionProvider:
    def get_version(self) -> int:
        return 0


def test_strict_epoch_fails_on_a_rejected_problem_instead_of_replacing_it():
    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=8,
        consumer_batch_size=4,
        max_staleness=2,
    )

    def factory(item):
        async def run():
            if item.task_id == 3:
                manager.on_rollout_rejected()
                return None
            manager.on_rollout_accepted()
            return _Result(item.task_id, item.data)

        return run

    dispatcher = BatchTaskDispatcher(
        max_queue_size=16,
        task_factory=factory,
        staleness_manager=manager,
        deterministic_order=True,
    )
    dispatcher.initialize(logger=logging.getLogger("StrictEpochTest"))
    try:
        with pytest.raises(RuntimeError, match="replacing or dropping"):
            dispatcher.active_submit_and_wait(
                (_Input(i, {"sample_id": i}) for i in range(8)),
                batch_size=4,
                finite_epoch=True,
                fail_on_rejection=True,
            )
    finally:
        dispatcher.destroy()


def test_native_single_controller_dataloader_keeps_full_epoch_tail():
    dataset = _SampleIdDataset(17157)
    dataloader = create_dataloader(
        dataset,
        rank=0,
        world_size=1,
        dataset_config=TrainDatasetConfig(
            path="unused",
            type="rl",
            batch_size=128,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        ),
    )

    batches = list(dataloader)

    assert len(batches) == 135
    assert len(batches[-1]) == 5
    assert [item["sample_id"] for item in batches[-1]] == list(range(17152, 17157))
    assert sum(len(batch) for batch in batches) == 17157
    assert len({item["sample_id"] for batch in batches for item in batch}) == 17157


def test_finite_epoch_returns_tail_partial_without_resubmitting_next_epoch():
    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=256,
        consumer_batch_size=128,
        max_staleness=2,
    )
    release = threading.Event()

    def task_factory(task_input: _Input):
        async def run() -> _Result:
            await asyncio.to_thread(release.wait)
            manager.on_rollout_accepted()
            return _Result(task_id=task_input.task_id, data=task_input.data)

        return run

    dispatcher = BatchTaskDispatcher[_Input, _Result](
        max_queue_size=512,
        task_factory=task_factory,
        staleness_manager=manager,
        deterministic_order=True,
    )
    dispatcher.initialize(logger=logging.getLogger("test_sao_epoch_boundary"))

    data = (_Input(task_id=i, data={"sample_id": i}) for i in range(133))
    first_batch: list[_Result] = []
    first_error: list[BaseException] = []

    def collect_first_batch() -> None:
        try:
            first_batch.extend(
                dispatcher.active_submit_and_wait(
                    data,
                    batch_size=128,
                    finite_epoch=True,
                )
            )
        except BaseException as exc:
            first_error.append(exc)

    worker = threading.Thread(target=collect_first_batch)
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stats = manager.get_stats()
            if stats.running + stats.enqueued >= 133:
                break
            time.sleep(0.01)
        else:
            stats = manager.get_stats()
            raise AssertionError(
                f"dispatcher did not prefetch the final partial epoch; stats={stats}"
            )

        release.set()
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert not first_error

        second_batch = dispatcher.active_submit_and_wait(
            data,
            batch_size=128,
            finite_epoch=True,
        )
        third_batch = dispatcher.active_submit_and_wait(
            data,
            batch_size=128,
            finite_epoch=True,
        )

        assert [r.data["sample_id"] for r in first_batch] == list(range(128))
        assert [r.data["sample_id"] for r in second_batch] == list(range(128, 133))
        assert third_batch == []

        stats = manager.get_stats()
        assert stats.accepted == 133
        assert stats.running == 0
        assert stats.enqueued == 0
    finally:
        release.set()
        worker.join(timeout=5.0)
        dispatcher.destroy()
