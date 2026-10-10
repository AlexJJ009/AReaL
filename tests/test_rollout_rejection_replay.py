# SPDX-License-Identifier: Apache-2.0
"""Finite-epoch replay of rejected rollouts (ADR 0011)."""

from __future__ import annotations

import itertools
import logging
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import pytest

from areal.api.cli_args import TrainDatasetConfig
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher, _RecoveryInputGenerator
from areal.utils.dataloader import create_dataloader


class _Dataset:
    def __len__(self) -> int:
        return 8

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


def _run_epoch(rejections: dict[int, int], max_skipped: int | None):
    """Collect one finite epoch; sample ``i`` is rejected ``rejections[i]`` times."""

    manager = StalenessManager(
        version_provider=_VersionProvider(),
        max_concurrent_rollouts=8,
        consumer_batch_size=4,
        max_staleness=2,
    )
    attempts: Counter[int] = Counter()

    def factory(item):
        async def run():
            sample = item.data["sample_id"]
            attempts[sample] += 1
            if attempts[sample] <= rejections.get(sample, 0):
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
    dispatcher.initialize(logger=logging.getLogger("RejectionReplayTest"))
    dataloader = create_dataloader(
        _Dataset(),
        rank=0,
        world_size=1,
        dataset_config=TrainDatasetConfig(
            path="unused", type="rl", batch_size=4, shuffle=False, num_workers=0
        ),
    )
    ids = itertools.count()
    generator = _RecoveryInputGenerator(
        dataloader,
        True,
        deque(),
        lambda item: _Input(next(ids), item),
        max_skipped_inputs=max_skipped,
    )
    accepted = []
    try:
        while batch := dispatcher.active_submit_and_wait(
            generator, batch_size=4, finite_epoch=True, fail_on_rejection=True
        ):
            accepted.extend(r.data["sample_id"] for r in batch)
    finally:
        dispatcher.destroy()
    return accepted, generator, attempts


def test_rejected_input_replays_once_at_the_end_of_the_epoch():
    accepted, generator, attempts = _run_epoch({3: 1}, max_skipped=0)

    assert sorted(accepted) == list(range(8))
    assert accepted[-1] == 3
    assert attempts[3] == 2 and generator.deferred_count == 1
    assert generator.skipped == [] and not generator.deferred


def test_input_rejected_twice_is_skipped_within_budget():
    accepted, generator, attempts = _run_epoch({3: 9}, max_skipped=1)

    assert sorted(accepted) == [0, 1, 2, 4, 5, 6, 7]
    assert attempts[3] == 2
    assert generator.skipped == [{"sample_id": 3}]


def test_skips_beyond_the_epoch_budget_abort():
    with pytest.raises(RuntimeError, match="max_skipped_inputs=1"):
        _run_epoch({3: 9, 5: 9}, max_skipped=1)


def test_without_a_budget_rejection_stays_fatal():
    with pytest.raises(RuntimeError, match="replacing or dropping"):
        _run_epoch({3: 1}, max_skipped=None)
