# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.trainer.ppo.validation import verify_gamma_one_episodic_returns


def _return_probe_group(
    *, bad_padding_return: bool = False, no_valid_tokens: bool = False
) -> dict[str, torch.Tensor]:
    loss_mask = torch.tensor(
        [[0, 1, 1, 0, 0], [0, 1, 1, 1, 0]],
        dtype=torch.bool,
    )
    if no_valid_tokens:
        loss_mask = torch.zeros_like(loss_mask)
    returns = torch.tensor(
        [[1.0, 1.0, 1.0, 9.0, 9.0], [1.7, 1.7, 1.7, 1.7, 9.0]],
        dtype=torch.float32,
    )
    if bad_padding_return:
        returns[0, 1] = 1.7
    return {
        "values": torch.tensor(
            [[0.0, 0.0, 0.0, 0.7, 9.0], [0.0, 0.0, 0.0, 0.0, 0.7]],
            dtype=torch.float32,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
            dtype=torch.bool,
        ),
        "terminated": torch.tensor([True, False], dtype=torch.bool),
        "truncated": torch.tensor([False, True], dtype=torch.bool),
        "rewards": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "loss_mask": loss_mask,
        "returns": returns,
    }


def test_gamma_one_returns_accept_terminal_and_truncated_trajectories():
    report = verify_gamma_one_episodic_returns([_return_probe_group()])

    assert report["passed"] is True
    assert report["samples"] == 2
    assert report["terminated"] == 1
    assert report["truncated"] == 1
    assert report["tokens"] == 5
    assert report["max_abs_error"] == pytest.approx(0.0)


def test_gamma_one_returns_reject_padding_derived_terminal_return():
    with pytest.raises(AssertionError):
        verify_gamma_one_episodic_returns(
            [_return_probe_group(bad_padding_return=True)]
        )


def test_gamma_one_returns_reject_no_valid_tokens():
    with pytest.raises(RuntimeError, match="no valid samples/tokens"):
        verify_gamma_one_episodic_returns([_return_probe_group(no_valid_tokens=True)])
