# SPDX-License-Identifier: Apache-2.0

"""Reject malformed native inputs without silently changing behavior probabilities."""

import copy

import pytest

from tests.torchrun.run_sao_qwen35_native import (
    _row_from_sample,
    apply_termination_contract,
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
