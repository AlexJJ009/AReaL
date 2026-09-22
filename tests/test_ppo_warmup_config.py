# SPDX-License-Identifier: Apache-2.0
"""PPO warmup and critic-return config validation."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from areal.api.cli_args import PPOActorConfig, PPOConfig, PPOCriticConfig


def test_ppo_actor_config_defaults_keep_critic_returns_compatible():
    config = PPOActorConfig()

    assert config.critic_gae_lambda is None
    assert OmegaConf.structured(config).critic_gae_lambda is None


@pytest.mark.parametrize(
    "critic_gae_lambda",
    [True, "0.5", -0.1, 1.1, float("nan"), float("inf")],
)
def test_ppo_actor_config_rejects_invalid_critic_gae_lambda(critic_gae_lambda):
    with pytest.raises(ValueError, match="critic_gae_lambda"):
        PPOActorConfig(critic_gae_lambda=critic_gae_lambda)


@pytest.mark.parametrize("critic_gae_lambda", [0, 0.0, 0.5, 1, 1.0])
def test_ppo_actor_config_accepts_static_critic_gae_lambda(critic_gae_lambda):
    config = PPOActorConfig(critic_gae_lambda=critic_gae_lambda)

    assert config.critic_gae_lambda == critic_gae_lambda


def test_ppo_config_defaults_disable_critic_only_warmup():
    config = PPOConfig()

    assert config.num_critic_only_steps == 0
    assert OmegaConf.structured(config).num_critic_only_steps == 0


@pytest.mark.parametrize("num_critic_only_steps", [True, 1.5, -1])
def test_ppo_config_rejects_invalid_critic_only_warmup_steps(
    num_critic_only_steps,
):
    with pytest.raises(ValueError, match="num_critic_only_steps"):
        PPOConfig(num_critic_only_steps=num_critic_only_steps)


@pytest.mark.parametrize(
    "critic",
    [None, SimpleNamespace(is_critic=False), PPOCriticConfig(is_critic=False)],
)
def test_ppo_config_rejects_critic_only_warmup_without_critic(critic):
    with pytest.raises(ValueError, match="requires a critic config"):
        PPOConfig(num_critic_only_steps=1, critic=critic)


@pytest.mark.parametrize(
    "critic",
    [None, SimpleNamespace(is_critic=False), PPOCriticConfig(is_critic=False)],
)
def test_ppo_config_rejects_critic_returns_without_critic(critic):
    actor = PPOActorConfig(critic_gae_lambda=1.0)

    with pytest.raises(ValueError, match="actor.critic_gae_lambda"):
        PPOConfig(actor=actor, critic=critic)


def test_ppo_config_accepts_critic_returns_with_critic_model():
    config = PPOConfig(
        actor=PPOActorConfig(critic_gae_lambda=1.0),
        critic=PPOCriticConfig(is_critic=True),
    )

    assert config.actor.critic_gae_lambda == 1.0


def test_ppo_config_accepts_critic_only_warmup_with_critic_model():
    config = PPOConfig(
        num_critic_only_steps=5,
        total_train_steps=1,
        critic=PPOCriticConfig(is_critic=True),
    )

    assert config.num_critic_only_steps == 5
    assert config.total_train_steps == 1
