# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from examples.swe.qwen38_flash_next.batch_snapshot import (
    capture_training_batches,
    load_batch_snapshot,
    save_batch_snapshot,
)

from areal.infra.rpc import rtensor
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.utils.data import RolloutGroup, TrajBatchMeta


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


def test_snapshot_real_rollout_metadata_roundtrips_without_mutating_batch(tmp_path):
    group = RolloutGroup((1, 2, 1, 1), (0.0, 1.0, 1.0, 1.0))
    pixels = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    batch = [
        {
            "rollout_group": group,
            "pixel_values": pixels,
            "alias": pixels,
            "logprobs": torch.zeros(5, 3),
            "attention_mask": torch.ones(5, 3),
        }
    ]
    actor = SimpleNamespace(
        prepare_batch=lambda: batch, compute_advantages=lambda data: data
    )
    with capture_training_batches(actor, tmp_path, {}):
        assert actor.prepare_batch() is batch
        assert actor.compute_advantages(batch) is batch
    path = tmp_path / "prepare_batch-0000.output.pt"
    assert torch.load(path, weights_only=True)["schema_version"] == 2
    restored = load_batch_snapshot(path)["batch"][0]
    assert restored["rollout_group"].validate_rows(5) == group
    assert batch[0]["rollout_group"] is group
    assert restored["pixel_values"] is restored["alias"]
    torch.testing.assert_close(restored["pixel_values"], pixels, rtol=0, atol=0)
    meta = TrajBatchMeta(1, [5], [3], [group])
    save_batch_snapshot(tmp_path / "meta.pt", {"meta": meta}, {})
    assert load_batch_snapshot(tmp_path / "meta.pt")["batch"]["meta"] == meta


def test_snapshot_unknown_metadata_and_reserved_keys_are_rejected(tmp_path):
    path = tmp_path / "invalid.pt"
    with pytest.raises(ValueError, match="Reserved"):
        save_batch_snapshot(path, {"__areal_snapshot_type__": "user-value"}, {})
    torch.save(
        {
            "schema_version": 2,
            "batch": {"__areal_snapshot_type__": "Unknown", "fields": {}},
        },
        path,
    )
    with pytest.raises(ValueError, match="Unsupported snapshot metadata"):
        load_batch_snapshot(path)


@pytest.mark.parametrize("call_index", [0, 1, 7])
def test_replay_supplies_visual_batch_once_without_live_rollout(tmp_path, call_index):
    from examples.swe.qwen38_flash_next.batch_snapshot import replay_training_batch

    def live(*args, **kwargs):
        pytest.fail("Replay called live rollout")

    actor = SimpleNamespace(prepare_batch=live, compute_advantages=lambda data: data)
    path = tmp_path / "source.pt"
    group = RolloutGroup((1, 1), (0.0, 1.0))
    save_batch_snapshot(
        path,
        [{"rollout_group": group, "pixel_values": torch.ones(3, 4)}],
        {"method": "prepare_batch", "call_index": call_index, "n_samples": 2},
    )
    with replay_training_batch(actor, path, {"n_samples": 2}):
        with capture_training_batches(actor, tmp_path / "captured", {}):
            batch = actor.prepare_batch(None, group_size=2)
            assert batch[0]["rollout_group"] == group
            torch.testing.assert_close(batch[0]["pixel_values"], torch.ones(3, 4))
            with pytest.raises(RuntimeError, match="already consumed"):
                actor.prepare_batch(None)
    assert actor.prepare_batch is live
    captured = load_batch_snapshot(tmp_path / "captured/prepare_batch-0000.output.pt")
    assert captured["batch"][0]["rollout_group"] == group
    with pytest.raises(ValueError, match="metadata mismatch"):
        with replay_training_batch(actor, path, {"n_samples": 4}):
            pytest.fail("Mismatched replay accepted")
    assert actor.prepare_batch is live


@pytest.mark.parametrize(
    "changes",
    [
        {"total_train_steps": 10},
        {"recover": SimpleNamespace(mode="auto")},
        {"evaluator": SimpleNamespace(eval_before_train=True)},
    ],
)
def test_replay_rejects_training_resume_and_evaluation(changes):
    from examples.swe.qwen38_flash_next.batch_snapshot import validate_diagnostic_replay

    config = dict(
        total_train_steps=1,
        recover=SimpleNamespace(mode="disabled"),
        evaluator=SimpleNamespace(eval_before_train=False),
    )
    validate_diagnostic_replay(SimpleNamespace(**config))
    config.update(changes)
    with pytest.raises(ValueError, match="Batch replay requires"):
        validate_diagnostic_replay(SimpleNamespace(**config))


@pytest.mark.parametrize("call_index", [-1, None, True, "1"])
def test_replay_rejects_invalid_source_call_index(tmp_path, call_index):
    from examples.swe.qwen38_flash_next.batch_snapshot import replay_training_batch

    actor = SimpleNamespace(prepare_batch=lambda: pytest.fail("Live rollout invoked"))
    original = actor.prepare_batch
    path = tmp_path / "invalid-index.pt"
    save_batch_snapshot(
        path,
        [{"input_ids": torch.tensor([[1, 2]])}],
        {"method": "prepare_batch", "call_index": call_index},
    )
    with pytest.raises(ValueError, match="valid call index"):
        with replay_training_batch(actor, path, {}):
            pytest.fail("Invalid source index accepted")
    assert actor.prepare_batch is original


def test_replay_sequence_preserves_order_and_restores_actor(tmp_path):
    from examples.swe.qwen38_flash_next.batch_snapshot import replay_training_batches

    def live():
        pytest.fail("Sequence replay called live rollout")

    actor = SimpleNamespace(prepare_batch=live)
    paths = []
    for index in range(2):
        path = tmp_path / f"batch-{index}.pt"
        save_batch_snapshot(
            path,
            [{"pixel_values": torch.full((2, 3), float(index))}],
            {
                "method": "prepare_batch",
                "call_index": index,
                "experiment": "exp",
                "trial": "source",
                "n_samples": 4,
            },
        )
        paths.append(path)
    with pytest.raises(RuntimeError, match="already consumed"):
        with replay_training_batches(actor, paths, {"n_samples": 4}):
            first = actor.prepare_batch()
            first[0]["pixel_values"].fill_(99)
            second = actor.prepare_batch()
            torch.testing.assert_close(second[0]["pixel_values"], torch.ones(2, 3))
            actor.prepare_batch()
    assert actor.prepare_batch is live
    # Loading another replay starts with pristine tensors, even after mutation.
    with replay_training_batches(actor, paths, {"n_samples": 4}):
        torch.testing.assert_close(
            actor.prepare_batch()[0]["pixel_values"], torch.zeros(2, 3)
        )


@pytest.mark.parametrize(
    "second_index,second_trial", [(0, "source"), (2, "source"), (1, "other")]
)
def test_replay_sequence_rejects_gaps_duplicates_and_mixed_trials(
    tmp_path, second_index, second_trial
):
    from examples.swe.qwen38_flash_next.batch_snapshot import replay_training_batches

    actor = SimpleNamespace(prepare_batch=lambda: pytest.fail("Live rollout invoked"))
    original = actor.prepare_batch
    paths = []
    for position, (index, trial) in enumerate(
        [(0, "source"), (second_index, second_trial)]
    ):
        path = tmp_path / f"batch-{position}.pt"
        save_batch_snapshot(
            path,
            [{"input_ids": torch.tensor([[1]])}],
            {
                "method": "prepare_batch",
                "call_index": index,
                "experiment": "exp",
                "trial": trial,
            },
        )
        paths.append(path)
    with pytest.raises(ValueError, match="contiguous"):
        with replay_training_batches(actor, paths, {}):
            pytest.fail("Invalid sequence accepted")
    assert actor.prepare_batch is original


def test_replay_sequence_configuration_and_path_selection():
    from pathlib import Path

    from examples.swe.qwen38_flash_next.batch_snapshot import (
        resolve_replay_paths,
        validate_diagnostic_replay,
    )

    config = SimpleNamespace(
        total_train_steps=2,
        recover=SimpleNamespace(mode="disabled"),
        evaluator=SimpleNamespace(eval_before_train=False),
    )
    validate_diagnostic_replay(config, 2)
    with pytest.raises(ValueError, match="snapshot count"):
        validate_diagnostic_replay(config, 1)
    assert resolve_replay_paths("one.pt", None) == [Path("one.pt")]
    assert resolve_replay_paths(None, '["zero.pt", "one.pt"]') == [
        Path("zero.pt"),
        Path("one.pt"),
    ]
    with pytest.raises(ValueError, match="only one"):
        resolve_replay_paths("one.pt", '["zero.pt"]')
    for invalid in ("[]", "{}", '"one.pt"', "[null]", '[""]'):
        with pytest.raises(ValueError, match="nonempty JSON array"):
            resolve_replay_paths(None, invalid)
