import hashlib

import pytest
import torch

from examples.swe.qwen38_flash_next import patch_sglang_qsa_topk as patch


def test_stable_topk_ties_respect_bounds_and_pad():
    logits = torch.tensor([[9.0, 3.0, 3.0, 3.0], [9.0, 8.0, 7.0, 6.0]])
    starts = torch.tensor([1, 2], dtype=torch.int32)
    ends = torch.tensor([4, 3], dtype=torch.int32)

    result = patch.stable_qsa_topk(logits, starts, ends, 2)

    torch.testing.assert_close(
        result, torch.tensor([[0, 1], [0, -1]], dtype=torch.int32), rtol=0, atol=0
    )


def test_stable_topk_empty_prefix_returns_padding():
    result = patch.stable_qsa_topk(
        torch.zeros(1, 2), torch.tensor([0]), torch.tensor([0]), 4
    )

    torch.testing.assert_close(
        result, torch.full((1, 4), -1, dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (0, 4)])
def test_stable_topk_invalid_bounds_rejected(start, end):
    with pytest.raises(ValueError, match="Invalid row bounds"):
        patch.stable_qsa_topk(
            torch.zeros(1, 3), torch.tensor([start]), torch.tensor([end]), 1
        )


def test_patch_unknown_source_rejected():
    with pytest.raises(ValueError, match="SHA256"):
        patch.patched_source("def qsa_fast_topk(): pass\n")


def test_patch_wrapper_preserves_native_and_enables_stable_selection(monkeypatch):
    source = (
        "import torch\n"
        "def qsa_fast_topk(logits, row_starts, row_ends, topk):\n"
        "    return torch.tensor([[2, 1]], dtype=torch.int32)\n"
    )
    monkeypatch.setattr(
        patch, "EXPECTED_SHA256", hashlib.sha256(source.encode()).hexdigest()
    )
    for key in (
        "QWEN_QSA_STABLE_TOPK",
        "QWEN_QSA_CANONICAL_TOPK",
        "QWEN_QSA_TOPK_DUMP_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    namespace = {}
    exec(patch.patched_source(source), namespace)
    inputs = (torch.ones(1, 3), torch.tensor([0]), torch.tensor([3]), 2)

    torch.testing.assert_close(
        namespace["qsa_fast_topk"](*inputs),
        torch.tensor([[2, 1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    monkeypatch.setenv("QWEN_QSA_STABLE_TOPK", "1")
    torch.testing.assert_close(
        namespace["qsa_fast_topk"](*inputs),
        torch.tensor([[0, 1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
