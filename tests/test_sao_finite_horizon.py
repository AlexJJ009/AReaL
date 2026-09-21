# SPDX-License-Identifier: Apache-2.0
"""Finite-horizon SAO PPO return semantics."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.lambda_fn import resolve_gae_lambda_fn
from areal.trainer.ppo.validation import verify_gamma_one_episodic_returns
from areal.utils.data import KLEstimator


def _make_actor() -> PPOActor:
    config = PPOActorConfig(
        kl_ctl=0.0,
        discount=1.0,
        gae_lambda=1.0,
        gae_timestep_unit="token",
        adv_norm=None,
        reward_norm=None,
        reward_scaling=10.0,
        reward_bias=-0.5,
        reward_clip=20.0,
        use_decoupled_loss=False,
        recompute_logprob=False,
        rejection_sampling=None,
        importance_sampling_level="token",
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        overlong_reward_penalty=False,
    )
    actor = PPOActor.__new__(PPOActor)
    actor.config = config
    actor.reward_bias = config.reward_bias
    actor.reward_scaling = config.reward_scaling
    actor.reward_clip = config.reward_clip
    actor.reward_norm = None
    actor.adv_norm = None
    actor.kl_ctl = 0.0
    actor.kl_estimator = KLEstimator("k1")
    actor.discount = 1.0
    actor.gae_lambda = 1.0
    actor.gae_lambda_fn, actor._gae_lambda_is_custom = resolve_gae_lambda_fn(1.0)
    actor.gae_lambda_kwargs = {}
    actor.gae_timestep_unit = "token"
    actor.mask_no_eos_with_zero = False
    actor.m2_threshold = None
    return actor


def _batch(
    *,
    rewards: list[float],
    final_values: list[float],
    terminated: list[bool],
    truncated: list[bool],
    bootstrap_mask: list[bool] | None = None,
) -> dict[str, torch.Tensor]:
    batch_size = len(rewards)
    width = 6
    seq_len = 5
    values = torch.zeros(batch_size, width, dtype=torch.float32)
    values[:, seq_len - 1] = torch.tensor(final_values, dtype=torch.float32)
    loss_mask = torch.zeros(batch_size, width, dtype=torch.float32)
    loss_mask[:, 2:seq_len] = 1.0
    batch = {
        "input_ids": torch.arange(width).expand(batch_size, -1),
        "attention_mask": (torch.arange(width).expand(batch_size, -1) < seq_len),
        "loss_mask": loss_mask,
        "logprobs": torch.zeros(batch_size, width, dtype=torch.float32),
        "values": values,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "terminated": torch.tensor(terminated, dtype=torch.bool),
        "truncated": torch.tensor(truncated, dtype=torch.bool),
    }
    if bootstrap_mask is not None:
        batch["bootstrap_mask"] = torch.tensor(bootstrap_mask, dtype=torch.bool)
    return batch


def _active_returns(result: dict[str, torch.Tensor], row: int) -> torch.Tensor:
    return result["returns"][row, result["loss_mask"][row].bool()]


def test_actor_finite_horizon_cutoff_ignores_large_tail_values() -> None:
    """SAO finite-horizon cutoffs keep truncated metadata without bootstrapping."""
    actor = _make_actor()
    batch = _batch(
        rewards=[0.0, 1.0],
        final_values=[101.0, 203.0],
        terminated=[False, False],
        truncated=[True, True],
        bootstrap_mask=[False, False],
    )

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(
        _active_returns(result, 0),
        torch.full((3,), -5.0, dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        _active_returns(result, 1),
        torch.full((3,), 5.0, dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_actor_explicit_true_bootstrap_mask_retains_value_bootstrap() -> None:
    """A true bootstrap mask preserves the previous truncated-bootstrap contract."""
    actor = _make_actor()
    batch = _batch(
        rewards=[1.0],
        final_values=[7.0],
        terminated=[False],
        truncated=[True],
        bootstrap_mask=[True],
    )

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(
        _active_returns(result, 0),
        torch.full((3,), 12.0, dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_actor_missing_bootstrap_mask_falls_back_to_truncated_bootstrap() -> None:
    """Old callers that omit bootstrap_mask still bootstrap truncated trajectories."""
    actor = _make_actor()
    batch = _batch(
        rewards=[1.0],
        final_values=[7.0],
        terminated=[False],
        truncated=[True],
    )

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(
        _active_returns(result, 0),
        torch.full((3,), 12.0, dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda batch: batch.__setitem__(
            "bootstrap_mask", torch.tensor([False], dtype=torch.bool).view(1, 1)
        ),
        lambda batch: batch.__setitem__(
            "bootstrap_mask", torch.tensor([0], dtype=torch.int64)
        ),
        lambda batch: batch.__setitem__(
            "bootstrap_mask", torch.tensor([True], dtype=torch.bool)
        ),
    ],
)
def test_actor_rejects_invalid_bootstrap_mask(
    mutate: Callable[[dict[str, torch.Tensor]], None],
) -> None:
    actor = _make_actor()
    batch = _batch(
        rewards=[1.0],
        final_values=[7.0],
        terminated=[True],
        truncated=[False],
        bootstrap_mask=[False],
    )
    mutate(batch)

    with pytest.raises((ValueError, RuntimeError), match="bootstrap_mask"):
        actor._compute_advantages(batch)


def test_return_probe_honors_finite_horizon_bootstrap_mask() -> None:
    group = {
        "values": torch.tensor([[0.0, 0.0, 0.0, 0.0, 101.0]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "terminated": torch.tensor([False]),
        "truncated": torch.tensor([True]),
        "bootstrap_mask": torch.tensor([False]),
        "rewards": torch.tensor([1.0]),
        "returns": torch.full((1, 5), 5.0, dtype=torch.float32),
        "loss_mask": torch.ones(1, 5, dtype=torch.bool),
    }

    report = verify_gamma_one_episodic_returns(
        [group],
        reward_scaling=10.0,
        reward_bias=-0.5,
        reward_clip=20.0,
    )

    assert report["passed"] is True
    assert report["truncated"] == 1


def test_return_probe_defaults_to_truncated_bootstrap_without_mask() -> None:
    group = {
        "values": torch.tensor([[0.0, 0.0, 0.0, 0.0, 7.0]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "terminated": torch.tensor([False]),
        "truncated": torch.tensor([True]),
        "rewards": torch.tensor([1.0]),
        "returns": torch.full((1, 5), 12.0, dtype=torch.float32),
        "loss_mask": torch.ones(1, 5, dtype=torch.bool),
    }

    report = verify_gamma_one_episodic_returns(
        [group],
        reward_scaling=10.0,
        reward_bias=-0.5,
        reward_clip=20.0,
    )

    assert report["passed"] is True
    assert report["truncated"] == 1
