# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from examples.swe.qwen38_flash_next.grad_norm_guard import (
    guard_grad_norm,
    local_gradient_diagnostics,
)


def _optimizer(grad):
    return SimpleNamespace(get_main_grads_for_grad_norm=lambda: [grad])


def test_large_finite_gradients_diagnose_fp32_norm_overflow():
    grad = torch.tensor([1e20, -1e20], dtype=torch.float32)
    assert torch.isinf(grad.norm())
    before = grad.clone()
    optimizer = _optimizer(grad)
    report = local_gradient_diagnostics(optimizer, chunk_size=1)
    assert report["nonfinite_elements"] == 0
    assert report["finite_fp64_norm"] == pytest.approx(grad.double().norm().item())
    with pytest.raises(RuntimeError, match="before clipping"):
        guard_grad_norm(lambda _: grad.norm().item())(optimizer)
    torch.testing.assert_close(grad, before, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_elements_are_distinct_from_norm_overflow(bad):
    optimizer = _optimizer(torch.tensor([3.0, bad, 4.0]))
    report = local_gradient_diagnostics(optimizer, chunk_size=2)
    assert report["nonfinite_elements"] == 1
    assert report["finite_fp64_norm"] == 5.0
    assert report["first_nonfinite_gradient"]["gradient"] == 0
    with pytest.raises(RuntimeError, match="before clipping"):
        guard_grad_norm(lambda _: bad)(optimizer)


@pytest.mark.parametrize("norm", [0.0, 1.2, 1000.0])
def test_finite_norm_preserved_without_diagnostic_access(norm):
    wrapped = guard_grad_norm(lambda _: norm)
    assert wrapped(object()) == norm
    assert guard_grad_norm(wrapped) is wrapped


def test_diagnostic_failure_still_rejects_nonfinite_norm():
    with pytest.raises(RuntimeError, match="diagnostic_error"):
        guard_grad_norm(lambda _: float("inf"))(object())


def test_chained_local_diagnostics_include_all_children():
    optimizer = SimpleNamespace(
        chained_optimizers=[
            _optimizer(torch.tensor([3.0])),
            _optimizer(torch.tensor([4.0])),
        ]
    )
    assert local_gradient_diagnostics(optimizer)["finite_fp64_norm"] == 5.0
