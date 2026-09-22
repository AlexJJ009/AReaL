# SPDX-License-Identifier: Apache-2.0
"""Critic-return GAE targets for PPO actor advantage computation."""

import torch

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.lambda_fn import resolve_gae_lambda_fn
from areal.utils.data import KLEstimator, Normalization


def _make_actor(
    *,
    gae_lambda: float = 0.95,
    critic_gae_lambda: float | None = None,
    adv_norm: NormConfig | None = None,
) -> PPOActor:
    config = PPOActorConfig(
        kl_ctl=0.0,
        discount=1.0,
        gae_lambda=gae_lambda,
        critic_gae_lambda=critic_gae_lambda,
        gae_timestep_unit="token",
        adv_norm=adv_norm,
        reward_norm=None,
        use_decoupled_loss=False,
        recompute_logprob=False,
        rejection_sampling=None,
        importance_sampling_level="token",
        overlong_reward_penalty=False,
    )
    actor = PPOActor.__new__(PPOActor)
    actor.config = config
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 20.0
    actor.reward_norm = None
    actor.adv_norm = Normalization(adv_norm) if adv_norm else None
    actor.kl_ctl = 0.0
    actor.kl_estimator = KLEstimator("k1")
    actor.discount = 1.0
    actor.gae_lambda = gae_lambda
    actor.gae_lambda_fn, actor._gae_lambda_is_custom = resolve_gae_lambda_fn(gae_lambda)
    actor.gae_lambda_kwargs = {}
    actor.critic_gae_lambda = critic_gae_lambda
    actor.gae_timestep_unit = "token"
    actor.mask_no_eos_with_zero = False
    actor.m2_threshold = None
    return actor


def _terminal_batch(values: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    width = 5
    batch = {
        "input_ids": torch.arange(width).view(1, width),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.float32),
        "logprobs": torch.zeros(1, width, dtype=torch.float32),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }
    if values is not None:
        batch["values"] = values
    return batch


def test_critic_gae_lambda_one_keeps_mc_target_separate_from_actor_lambda():
    """Actor advantages can use lambda=.95 while critic returns use MC lambda=1."""
    actor = _make_actor(gae_lambda=0.95, critic_gae_lambda=1.0)
    values = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0]], dtype=torch.float32)

    result = actor._compute_advantages(_terminal_batch(values))

    torch.testing.assert_close(
        result["returns"],
        torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[0.82675, 0.765, 0.7, 0.0, 0.0]], dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_critic_gae_lambda_zero_respects_mask_and_truncated_bootstrap():
    """Critic returns reuse the same mask and bootstrap values as actor GAE."""
    actor = _make_actor(gae_lambda=1.0, critic_gae_lambda=0.0)
    batch = {
        "input_ids": torch.arange(6).view(1, 6),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32),
        "logprobs": torch.zeros(1, 6, dtype=torch.float32),
        "values": torch.tensor([[0.0, 2.0, 3.0, 4.0, 7.0, 0.0]]),
        "rewards": torch.tensor([2.0], dtype=torch.float32),
        "terminated": torch.tensor([False]),
        "truncated": torch.tensor([True]),
        "bootstrap_mask": torch.tensor([True]),
    }

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(
        result["loss_mask"],
        torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result["returns"],
        torch.tensor([[1.0, 3.0, 4.0, 9.0, 7.0, 0.0]], dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_advantage_normalization_does_not_change_critic_returns():
    """adv_norm only touches actor advantages; critic targets stay raw."""
    actor = _make_actor(
        gae_lambda=0.95,
        critic_gae_lambda=1.0,
        adv_norm=NormConfig(mean_level="batch", std_level="batch", group_size=1),
    )
    values = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0]], dtype=torch.float32)

    result = actor._compute_advantages(_terminal_batch(values))

    torch.testing.assert_close(
        result["returns"],
        torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert not torch.allclose(
        result["advantages"],
        torch.tensor([[0.82675, 0.765, 0.7, 0.0, 0.0]], dtype=torch.float32),
    )


def test_default_none_preserves_actor_lambda_returns():
    """Leaving critic_gae_lambda unset keeps legacy actor-lambda returns."""
    actor = _make_actor(gae_lambda=0.95, critic_gae_lambda=None)
    values = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0]], dtype=torch.float32)

    result = actor._compute_advantages(_terminal_batch(values))

    torch.testing.assert_close(
        result["returns"],
        torch.tensor([[0.92675, 0.965, 1.0, 0.0, 0.0]], dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
