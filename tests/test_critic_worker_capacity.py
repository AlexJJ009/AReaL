# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.staleness_manager import StalenessManager


def manager():
    return StalenessManager(SimpleNamespace(get_version=lambda: 0), 128, 16, 2)


def test_frozen_worker_without_credit_stalls_after_48_prompt_groups():
    worker = manager()
    for _ in range(48):
        worker.on_rollout_enqueued()
        worker.on_rollout_submitted()
        worker.on_rollout_accepted()
    assert worker.get_capacity() == 0


def test_both_dispatch_layers_support_50_frozen_batches_without_version_bump():
    outer = manager()
    workers = [manager() for _ in range(4)]

    def rpc(method, **kwargs):
        assert method == "on_batch_consumed_without_update"
        for worker in workers:
            worker.on_batch_consumed_without_update()

    controller = SimpleNamespace(
        staleness_manager=outer, _proxy_started=False, _collective_rpc=rpc
    )
    for step in range(50):
        for prompt in range(16):
            worker = workers[(step * 16 + prompt) % 4]
            for sm in (outer, worker):
                assert sm.get_capacity() > 0
                sm.on_rollout_enqueued()
                sm.on_rollout_submitted()
                sm.on_rollout_accepted()
        RolloutController.on_batch_consumed_without_update(controller)
        assert outer.get_capacity() == 48
    assert outer.get_stats().accepted == 800
    assert sum(w.get_stats().accepted for w in workers) == 800
    assert all(w.version_provider.get_version() == 0 for w in workers)


def test_proxy_workers_also_receive_frozen_batch_credit():
    controller = SimpleNamespace(
        staleness_manager=manager(),
        _proxy_started=True,
        _collective_rpc=Mock(),
        _proxy_collective_rpc=Mock(),
    )
    RolloutController.on_batch_consumed_without_update(controller)
    controller._collective_rpc.assert_called_once_with(
        "on_batch_consumed_without_update", http_timeout=60.0
    )
    controller._proxy_collective_rpc.assert_called_once_with(
        "on_batch_consumed_without_update", http_timeout=60.0
    )
