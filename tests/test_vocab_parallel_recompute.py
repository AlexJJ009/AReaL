# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.utils.functional import vocab_parallel


@pytest.fixture(autouse=True)
def eager_chunk_kernels(monkeypatch):
    # CPU tests validate autograd semantics independently of compiler availability.
    for name in ("_gather_logprobs", "_gather_logprobs_entropy"):
        fn = getattr(vocab_parallel, name)
        monkeypatch.setattr(
            vocab_parallel, name, getattr(fn, "_torchdynamo_orig_callable", fn)
        )


@pytest.mark.parametrize("chunk_size", [1, 4, 17, 32])
@pytest.mark.parametrize("with_entropy", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_recomputed_chunks_match_full_outputs_and_gradients(
    chunk_size, with_entropy, dtype
):
    generator = torch.Generator().manual_seed(7)
    initial = torch.randn(17, 43, generator=generator).to(dtype)
    actual_logits = initial.clone().requires_grad_()
    expected_logits = initial.clone().requires_grad_()
    labels = torch.randint(43, (17,), generator=generator)
    weights = torch.randn(17, generator=generator)
    weights[::3] = 0
    temperature = 0.7
    expected_all = (expected_logits.float() / temperature).log_softmax(-1)
    expected_logprobs = expected_all.gather(-1, labels[:, None]).squeeze(-1)
    if with_entropy:
        expected_entropy = -(expected_all.exp() * expected_all).sum(-1)
        actual_logprobs, actual_entropy = vocab_parallel.gather_logprobs_entropy(
            actual_logits, labels, temperature, chunk_size=chunk_size
        )
        torch.testing.assert_close(actual_entropy, expected_entropy)
        actual_loss = (actual_logprobs * weights + 0.23 * actual_entropy).sum()
        expected_loss = (expected_logprobs * weights + 0.23 * expected_entropy).sum()
    else:
        actual_logprobs = vocab_parallel.gather_logprobs(
            actual_logits, labels, temperature, chunk_size=chunk_size
        )
        actual_loss = (actual_logprobs * weights).sum()
        expected_loss = (expected_logprobs * weights).sum()
    torch.testing.assert_close(actual_logprobs, expected_logprobs)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_no_grad_chunks_match_full_outputs(requires_grad):
    logits = torch.randn(11, 29, requires_grad=requires_grad)
    labels = torch.arange(11)
    with torch.no_grad():
        logprobs, entropy = vocab_parallel.gather_logprobs_entropy(
            logits, labels, temperature=1.3, chunk_size=4
        )
        reference = (logits.float() / 1.3).log_softmax(-1)
    torch.testing.assert_close(logprobs, reference.gather(-1, labels[:, None])[:, 0])
    torch.testing.assert_close(entropy, -(reference.exp() * reference).sum(-1))
    assert not logprobs.requires_grad
    assert not entropy.requires_grad


def test_differentiable_chunks_do_not_save_vocabulary_intermediates():
    logits = torch.randn(17, 43, requires_grad=True)
    labels = torch.arange(17)
    saved_storage = []

    def record(tensor):
        if tensor.numel():
            saved_storage.append(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        logprobs, entropy = vocab_parallel.gather_logprobs_entropy(
            logits, labels, chunk_size=4
        )
    assert set(saved_storage) <= {
        logits.untyped_storage().data_ptr(),
        labels.untyped_storage().data_ptr(),
    }
    (logprobs.sum() + 0.2 * entropy.sum()).backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("output_index", [0, 1])
def test_noncontiguous_input_and_unused_output_match_gradients(output_index):
    base = torch.randn(31, 13)
    actual = base.t().detach().requires_grad_()
    expected = base.t().detach().requires_grad_()
    assert not actual.is_contiguous()
    labels = torch.arange(13)
    outputs = vocab_parallel.gather_logprobs_entropy(
        actual, labels, temperature=1.7, chunk_size=5
    )
    reference_all = (expected.float() / 1.7).log_softmax(-1)
    reference = (
        reference_all.gather(-1, labels[:, None]).squeeze(-1),
        -(reference_all.exp() * reference_all).sum(-1),
    )
    torch.testing.assert_close(outputs[output_index], reference[output_index])
    outputs[output_index].sum().backward()
    reference[output_index].sum().backward()
    torch.testing.assert_close(actual.grad, expected.grad)
