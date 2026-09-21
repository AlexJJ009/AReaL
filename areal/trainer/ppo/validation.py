# SPDX-License-Identifier: Apache-2.0

"""Reusable validation primitives for PPO data and targets."""

from collections.abc import Mapping, Sequence

import torch


def verify_gamma_one_episodic_returns(
    groups: Sequence[Mapping[str, torch.Tensor]],
    *,
    reward_scaling: float = 1.0,
    reward_bias: float = 0.0,
    reward_clip: float = 20.0,
) -> dict[str, bool | int | float | str]:
    """Verify the gamma=lambda=1 episodic return contract on real PPO tensors.

    The contract assumes one scalar outcome reward per trajectory and no token-level
    reward shaping. Explicit bootstrap masks select whether a length cutoff has
    continuation value; absent masks retain the legacy length-bootstrap contract.
    """
    samples = terminated = truncated = tokens = bootstrapped = 0
    max_error = 0.0
    for group in groups:
        values = group["values"].float()
        lengths = group["attention_mask"].sum(-1).long()
        terminal = group["terminated"].bool()
        cutoff = group["truncated"].bool()
        if not torch.all(terminal ^ cutoff):
            raise RuntimeError("Invalid episode termination flags in return probe")

        bootstrap_mask = group.get("bootstrap_mask", cutoff)
        if bootstrap_mask.dtype != torch.bool or bootstrap_mask.shape != cutoff.shape:
            raise RuntimeError("Invalid bootstrap_mask in return probe")
        if torch.any(bootstrap_mask & ~cutoff):
            raise RuntimeError("Terminal episodes cannot bootstrap")
        bootstrapped += int(bootstrap_mask.sum())
        bootstrap = (
            values.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1) * bootstrap_mask
        )
        reward = ((group["rewards"].float() + reward_bias) * reward_scaling).clamp(
            min=-reward_clip, max=reward_clip
        )
        expected = (reward + bootstrap).unsqueeze(1)
        mask = group["loss_mask"].bool()
        actual = group["returns"].float()
        torch.testing.assert_close(
            actual[mask], expected.expand_as(actual)[mask], rtol=1e-4, atol=2e-4
        )

        if mask.any():
            max_error = max(max_error, float((actual - expected).abs()[mask].max()))
        samples += values.shape[0]
        terminated += int(terminal.sum())
        truncated += int(cutoff.sum())
        tokens += int(mask.sum())

    if not samples or not tokens:
        raise RuntimeError("Return probe has no valid samples/tokens")
    return {
        "passed": True,
        "samples": samples,
        "terminated": terminated,
        "truncated": truncated,
        "bootstrapped": bootstrapped,
        "tokens": tokens,
        "max_abs_error": max_error,
        "rtol": 1e-4,
        "atol": 2e-4,
        "oracle": "gamma=lambda=1: transformed_reward + bootstrap_mask * V(real final token)",
        "reward_scaling": reward_scaling,
        "reward_bias": reward_bias,
        "reward_clip": reward_clip,
    }
