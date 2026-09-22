# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from examples.swe.qwen38_flash_next.batch_snapshot import (
    capture_training_batches,
    save_batch_snapshot,
)

from areal.infra.rpc import rtensor
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo


def test_snapshot_preserves_pixels_logps_aliases_and_live_batch(tmp_path):
    pixels = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    remote = RTensor(TensorShardInfo("test-shard", "unused"), pixels)
    batch = [
        {
            "pixels": remote,
            "same_pixels": remote,
            "logprobs": torch.tensor([-1.2]),
            "version": 0,
        }
    ]
    path = tmp_path / "raw.pt"
    save_batch_snapshot(path, batch, {"stage": "raw"})
    saved = torch.load(path, weights_only=True)["batch"][0]
    assert batch[0]["pixels"] is remote
    assert remote.data is pixels
    assert saved["pixels"] is saved["same_pixels"]
    torch.testing.assert_close(saved["pixels"], pixels, rtol=0, atol=0)
    pixels.zero_()
    assert saved["pixels"].sum() > 0
    assert saved["version"] == 0
    torch.testing.assert_close(saved["logprobs"], torch.tensor([-1.2]))
    with pytest.raises(FileExistsError):
        save_batch_snapshot(path, batch, {})


def test_capture_snapshots_before_inplace_advantage_and_restores_on_failure(tmp_path):
    batch = [{"logprobs": torch.tensor([-2.0])}]

    def advantage(data):
        data[0]["logprobs"].zero_()
        return data

    actor = SimpleNamespace(prepare_batch=lambda: batch, compute_advantages=advantage)
    original = actor.prepare_batch
    with pytest.raises(RuntimeError, match="training failed"):
        with capture_training_batches(actor, tmp_path, {}):
            assert actor.prepare_batch() is batch
            assert actor.compute_advantages(data=batch) is batch
            raise RuntimeError("training failed")
    assert actor.prepare_batch is original
    assert actor.compute_advantages is advantage
    raw = torch.load(tmp_path / "prepare_batch-0000.output.pt", weights_only=True)
    before = torch.load(
        tmp_path / "compute_advantages-0000.input.pt", weights_only=True
    )
    after = torch.load(
        tmp_path / "compute_advantages-0000.output.pt", weights_only=True
    )
    assert raw["batch"][0]["logprobs"].item() == -2
    assert before["batch"][0]["logprobs"].item() == -2
    assert after["batch"][0]["logprobs"].item() == 0


def test_snapshot_failure_does_not_publish_partial_file(tmp_path):
    path = tmp_path / "bad.pt"
    with pytest.raises(TypeError, match="Unsupported"):
        save_batch_snapshot(path, {"unsupported": object()}, {})
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_remote_snapshot_fetches_once_without_localizing_live_wrappers(
    tmp_path, monkeypatch
):
    shard = TensorShardInfo(uuid4().hex, "unused")
    first = RTensor(shard, torch.empty(3, device="meta"))
    second = RTensor(shard, torch.empty(3, device="meta"))
    calls = []

    def fetch(shards):
        calls.append(shards)
        return [torch.arange(3)]

    monkeypatch.setattr(rtensor, "get_backend", lambda: SimpleNamespace(fetch=fetch))
    try:
        path = tmp_path / "remote.pt"
        save_batch_snapshot(path, {"a": first, "b": second}, {})
        saved = torch.load(path, weights_only=True)["batch"]
        assert first.data.is_meta and second.data.is_meta
        assert calls == [[shard]]
        assert saved["a"] is saved["b"]
        torch.testing.assert_close(saved["a"], torch.arange(3))
    finally:
        with rtensor._fetch_buffer_lock:
            rtensor._fetch_buffer.pop(shard.shard_id, None)
