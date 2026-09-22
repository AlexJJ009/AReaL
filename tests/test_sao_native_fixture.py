# SPDX-License-Identifier: Apache-2.0

"""Reject malformed native inputs without silently changing behavior probabilities."""

import copy

import pytest
import torch

from tests.torchrun.run_sao_qwen35_native import (
    _row_from_sample,
    apply_termination_contract,
    sample_indices,
)


def sample():
    return dict(
        input_tokens=[1, 2],
        output_tokens=[3, 4],
        behavior_logprobs=[-0.3, -0.7],
        behavior_versions=[2, 3],
        reward=0.0,
        stop_reason="length",
        truncated=True,
    )


def test_native_fixture_keeps_behavior_and_prefix():
    row = _row_from_sample(sample(), 0)
    assert row["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert row["versions"].tolist() == [[-1, -1, 2, 3]]
    sao = apply_termination_contract([copy.deepcopy(row)], "sao")[0]
    ppo = apply_termination_contract([copy.deepcopy(row)], "ppo")[0]
    assert sao["terminated"].item() and not sao["truncated"].item()
    assert ppo["truncated"].item() and not ppo["terminated"].item()
    assert sao["logprobs"].equal(ppo["logprobs"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "missing"])
def test_native_fixture_invalid_logprob_fails(bad):
    raw = sample()
    raw["behavior_logprobs"][0] = bad
    with pytest.raises((ValueError, TypeError)):
        _row_from_sample(raw, 0)


def test_native_fixture_cap_rejects_instead_of_cropping_prefix():
    with pytest.raises(ValueError, match="instead of truncating"):
        _row_from_sample(sample(), 3)


@pytest.mark.parametrize("numel", [0, 1, 3, 8, 97280000, 97280001, 2**32])
def test_native_parameter_sample_indices_stay_inside_large_shards(numel):
    actual = sample_indices(numel, 8, torch.device("cpu"))
    assert actual.dtype == torch.int64
    assert actual.numel() == min(numel, 8)
    assert torch.all((actual >= 0) & (actual < numel))
    if numel > 1:
        assert actual[0] == 0 and actual[-1] == numel - 1
        assert torch.all(actual[1:] > actual[:-1])


def test_fp32_linspace_negative_control_detects_out_of_range_endpoint():
    numel = 97280000
    unsafe = torch.linspace(0, numel - 1, steps=8, dtype=torch.float32).long()
    assert unsafe[-1] == numel
    assert sample_indices(numel, 8, torch.device("cpu"))[-1] == numel - 1
