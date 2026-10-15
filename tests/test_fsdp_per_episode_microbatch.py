# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of the shared FSDP microbatch preparation path."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine


def _engine(monkeypatch, *, enabled=False, model_type="qwen3", granularity=1):
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.config = SimpleNamespace(
        per_episode_microbatching=enabled,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=32768, granularity=granularity),
        pad_to_maximum=False,
    )
    engine.model_config = SimpleNamespace(model_type=model_type)
    engine.enable_tree_training = False
    engine.logger = Mock()
    monkeypatch.setattr("areal.engine.fsdp_engine.dist.get_rank", lambda: 0)
    return engine


def _batch(lengths):
    ids = torch.arange(len(lengths) * max(lengths)).reshape(len(lengths), -1)
    return {
        "input_ids": ids,
        "attention_mask": torch.arange(max(lengths))[None, :]
        < torch.tensor(lengths)[:, None],
    }


@pytest.mark.parametrize("count", [1, 7, 31, 32, 33])
@pytest.mark.parametrize("granularity", [1, 4])
def test_per_episode_enabled_qwen3_varied_counts_keep_singletons(
    monkeypatch, count, granularity
):
    """The split follows actual local rows and overrides sample granularity."""
    engine = _engine(monkeypatch, enabled=True, granularity=granularity)
    lengths = [300 + i % 5 for i in range(count)]
    batch = _batch(lengths)

    result = engine._prepare_mb_list(batch)

    assert len(result.mbs) == count
    assert result.group_lens == lengths
    assert result.forward_indices == list(range(count))
    assert result.backward_indices == list(range(count))
    for i, mb in enumerate(result.mbs):
        assert mb["cu_seqlens"].tolist() == [0, lengths[i]]
        torch.testing.assert_close(
            mb["input_ids"].squeeze(0),
            batch["input_ids"][i, : lengths[i]],
            rtol=0,
            atol=0,
        )
    assert engine.config.mb_spec.granularity == granularity
    assert engine.config.mb_spec.n_mbs == 1


@pytest.mark.parametrize("has_switch", [False, True])
def test_per_episode_disabled_qwen3_keeps_ffd_packing(monkeypatch, has_switch):
    """Both an absent field and explicit false preserve the old FFD behavior."""
    engine = _engine(monkeypatch)
    if not has_switch:
        del engine.config.per_episode_microbatching

    result = engine._prepare_mb_list(_batch([7000, 8000, 9000]))

    assert result.group_lens == [24000]
    assert result.mbs[0]["cu_seqlens"].tolist() == [0, 7000, 15000, 24000]


def test_per_episode_disabled_qwen35_still_keeps_singletons(monkeypatch):
    """The Qwen3.5 recurrent-state safeguard remains unconditional."""
    engine = _engine(monkeypatch, model_type="qwen3_5")

    result = engine._prepare_mb_list(_batch([7000, 8000, 9000]))

    assert result.group_lens == [7000, 8000, 9000]


@pytest.mark.parametrize("length", [32767, 32768, 32769])
def test_per_episode_token_capacity_is_preserved(monkeypatch, length):
    """A full 32768-token episode is valid; exceeding the cap still raises."""
    engine = _engine(monkeypatch, enabled=True)
    if length > 32768:
        with pytest.raises(RuntimeError, match="larger than capacity"):
            engine._prepare_mb_list(_batch([length]))
    else:
        result = engine._prepare_mb_list(_batch([length]))
        assert result.group_lens == [length]
        assert result.padded_to_lengths == [32768]


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5"])
def test_per_episode_unequal_ranks_keep_existing_qwen35_error(monkeypatch, model_type):
    """Count sync cannot create 33 nonempty batches from 31 local episodes."""
    engine = _engine(monkeypatch, enabled=True, model_type=model_type)
    monkeypatch.setattr("areal.utils.data.dist.is_initialized", lambda: True)
    monkeypatch.setattr("areal.utils.data.dist.get_world_size", lambda group: 2)
    gathered = []

    def gather(output, value, group):
        gathered.append((value, group))
        output[:] = [31, 33]

    monkeypatch.setattr("areal.utils.data.dist.all_gather_object", gather)

    with pytest.raises(RuntimeError, match="31 is smaller than min_groups 33"):
        engine._prepare_mb_list(_batch([300] * 31))

    assert gathered == [(31, None)]


@pytest.mark.parametrize("count", [31, 33])
def test_per_episode_equal_rank_counts_sync_singletons(monkeypatch, count):
    """The existing count collective accepts changing but equal local counts."""
    engine = _engine(monkeypatch, enabled=True)
    monkeypatch.setattr("areal.utils.data.dist.is_initialized", lambda: True)
    monkeypatch.setattr("areal.utils.data.dist.get_world_size", lambda group: 2)
    gathered = []

    def gather(output, value, group):
        gathered.append((value, group))
        output[:] = [count, count]

    monkeypatch.setattr("areal.utils.data.dist.all_gather_object", gather)

    result = engine._prepare_mb_list(_batch([300] * count))

    assert result.group_lens == [300] * count
    assert gathered == [(count, None)]
