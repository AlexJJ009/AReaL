# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from examples.swe.qwen38_flash_next.gdn_cp_compat import patch_packed_cp_forward
from examples.swe.qwen38_flash_next.ple_chunked import chunked_ple
from examples.swe.qwen38_flash_next.train_rl import configure_training_rpc


def _causal_reference(h, k, v, nk, nq, nc, weight, n, eps, dilation, seq_len):
    x = (h * nk + k * nq + v * nc).reshape(-1, seq_len, h.shape[-1])
    x = x.transpose(1, 2)
    halo = (weight.shape[-1] - 1) * dilation
    return (
        F.conv1d(
            F.pad(x, (halo, 0)), weight.float(), groups=h.shape[-1], dilation=dilation
        )
        .transpose(1, 2)
        .reshape_as(h)
    )


@pytest.mark.parametrize("chunk_tokens,dilation", [(5, 1), (2, 3), (32, 1)])
def test_ple_chunks_preserve_rows_halo_and_gradients(chunk_tokens, dilation):
    torch.manual_seed(1234)
    shape = (26, 3)  # Two rows; the last chunk is shorter than the others.
    data = [torch.randn(shape) for _ in range(3)]
    data += [torch.randn(3).bfloat16() for _ in range(3)]
    data += [torch.randn(3, 1, 4).bfloat16()]
    actual_inputs = [x.clone().requires_grad_() for x in data]
    reference_inputs = [x.clone().requires_grad_() for x in data]
    actual = chunked_ple(
        _causal_reference,
        *actual_inputs,
        4,
        1e-6,
        dilation,
        13,
        chunk_tokens=chunk_tokens,
    )
    h, k, v, *weights = reference_inputs
    reference = _causal_reference(
        h, k, v, *(w.float() for w in weights), 4, 1e-6, dilation, 13
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs):
        torch.testing.assert_close(
            actual_input.grad, reference_input.grad, rtol=1e-4, atol=1e-4
        )


def _legacy_forward(self, cu_seqlens, *, cp_size=2):
    return cu_seqlens // self.cp_size


def _fixed_forward(self, cu_seqlens, *, cp_size=2):
    return cu_seqlens // cp_size


def _unknown_forward(self, cu_seqlens):
    return cu_seqlens


def test_packed_cp_uses_local_size_and_patch_is_idempotent():
    patched = patch_packed_cp_forward(_legacy_forward)
    # Legacy self.cp_size may be missing or stale.
    torch.testing.assert_close(
        patched(SimpleNamespace(), torch.tensor([0, 16, 40])),
        torch.tensor([0, 8, 20]),
        rtol=0,
        atol=0,
    )
    assert patch_packed_cp_forward(patched) is patched
    assert patch_packed_cp_forward(_fixed_forward) is _fixed_forward
    with pytest.raises(RuntimeError, match="divisor"):
        patch_packed_cp_forward(_unknown_forward)


def test_training_rpc_disables_optimizer_replay_only():
    calls = []

    async def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "ok"

    scheduler = configure_training_rpc(SimpleNamespace(async_call_engine=original))
    assert (
        asyncio.run(
            scheduler.async_call_engine(
                "actor/0", "ppo_update", http_timeout=1, max_retries=3
            )
        )
        == "ok"
    )
    assert calls[-1][1] == dict(http_timeout=28800, max_retries=1)
    asyncio.run(
        scheduler.async_call_engine(
            "actor/0", "get_version", http_timeout=17, max_retries=2
        )
    )
    assert calls[-1][1] == dict(http_timeout=17, max_retries=2)


def test_ple_chunk_propagates_kernel_fallback():
    calls = []

    def unavailable(*args):
        calls.append(args)
        return None

    x = torch.zeros(26, 3)
    weight = torch.ones(3)
    result = chunked_ple(
        unavailable,
        x,
        x,
        x,
        weight,
        weight,
        weight,
        torch.ones(3, 1, 4),
        4,
        1e-6,
        1,
        13,
        chunk_tokens=5,
    )
    assert result is None
    assert len(calls) == 1
