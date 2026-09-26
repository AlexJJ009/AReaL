# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from areal.utils.stats_tracker import DistributedStatsTracker


def test_stats_tracker_scalar_reduce_group_override_used_on_export():
    # Arrange
    tracker = DistributedStatsTracker()
    default_group = object()
    override_group = object()

    tracker.scalar(score=1.0, reduce_group=override_group)

    # Act
    with patch("areal.utils.stats_tracker.dist.all_reduce") as mock_all_reduce:
        tracker.export(key="score", reduce_group=default_group)

    # Assert
    assert mock_all_reduce.call_count == 2
    assert all(
        call.kwargs["group"] is override_group for call in mock_all_reduce.mock_calls
    )


def test_stats_tracker_stat_reduce_group_override_used_on_export():
    # Arrange
    tracker = DistributedStatsTracker()
    default_group = object()
    override_group = object()

    tracker.denominator(mask=torch.tensor([True, False, True]))
    tracker.stat(
        "mask",
        loss=torch.tensor([1.0, 2.0, 3.0]),
        reduce_group=override_group,
    )

    # Act
    with patch("areal.utils.stats_tracker.dist.all_reduce") as mock_all_reduce:
        tracker.export(key="loss", reduce_group=default_group)

    # Assert
    assert mock_all_reduce.call_count == 4
    assert all(
        call.kwargs["group"] is override_group for call in mock_all_reduce.mock_calls
    )


def test_local_scalar_export_does_not_allocate_on_initialized_cuda(monkeypatch):
    from types import SimpleNamespace

    from areal.utils import stats_tracker

    monkeypatch.setattr(
        stats_tracker,
        "current_platform",
        SimpleNamespace(
            is_initialized=lambda: True,
            device_type="cuda",
            communication_backend="nccl",
        ),
    )
    original_tensor = torch.tensor
    devices = []

    def checked_tensor(*args, **kwargs):
        devices.append(kwargs.get("device", "cpu"))
        assert str(devices[-1]) == "cpu", "local metrics must not touch CUDA"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", checked_tensor)
    tracker = DistributedStatsTracker()
    tracker.scalar(latency=2.0)
    tracker.scalar(latency=4.0)
    result = tracker.export(reduce_group=None)
    assert result == {"latency": 3.0, "latency__count": 2}
    assert devices


def test_collective_scalar_placeholder_keeps_nccl_device(monkeypatch):
    from types import SimpleNamespace

    from areal.utils import stats_tracker

    group = object()
    monkeypatch.setattr(
        stats_tracker,
        "current_platform",
        SimpleNamespace(
            is_initialized=lambda: True,
            device_type="cuda",
            communication_backend="nccl",
        ),
    )
    monkeypatch.setattr(stats_tracker.dist, "get_backend", lambda candidate: "nccl")
    assert DistributedStatsTracker()._device_for_placeholder_tensor(group) == "cuda"
