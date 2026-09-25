from types import SimpleNamespace

import pytest
import torch

from areal.engine.fsdp_engine import FSDPEngine
from areal.utils.data import MicroBatchItem


def _make_microbatch() -> MicroBatchItem:
    data = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "turn_ids": torch.tensor([[-1, 0, 0]], dtype=torch.int32),
        "action_origin_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
        "episode_ids": torch.full((1, 3), 17, dtype=torch.int64),
        "terminated": torch.tensor([True]),
        "truncated": torch.tensor([False]),
        "bootstrap_mask": torch.tensor([False]),
        "official_scores": torch.tensor([1.0]),
        "task_budget_failure": torch.tensor([False]),
    }
    return MicroBatchItem(
        orig_mb=data,
        padded_mb=data,
        padding_length=0,
        old_cu_seqlens=None,
    )


def test_fsdp_prepare_inputs_strips_turn_ids_without_mutating_context():
    """FSDP forwards model fields only while retaining algorithm metadata."""
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.parallel_helper = SimpleNamespace(sp_size=1)

    inputs, context = engine._prepare_mb_inputs(_make_microbatch())

    for key in (
        "turn_ids",
        "action_origin_mask",
        "episode_ids",
        "terminated",
        "truncated",
        "bootstrap_mask",
        "official_scores",
        "task_budget_failure",
    ):
        assert key not in inputs
        assert key in context.mb_input


def test_archon_prepare_inputs_strips_turn_ids_without_mutating_context():
    """Archon forwards model fields only while retaining algorithm metadata."""
    pytest.importorskip("triton", reason="Archon import requires Triton")
    from areal.experimental.engine.archon_engine import ArchonEngine

    engine = ArchonEngine.__new__(ArchonEngine)
    engine.enable_tree_training = False
    engine.parallel_dims = SimpleNamespace(cp_enabled=False)

    inputs, context = engine._prepare_mb_inputs(_make_microbatch())

    for key in (
        "turn_ids",
        "action_origin_mask",
        "episode_ids",
        "terminated",
        "truncated",
        "bootstrap_mask",
        "official_scores",
        "task_budget_failure",
    ):
        assert key not in inputs
        assert key in context.mb_input
