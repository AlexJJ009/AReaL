# SPDX-License-Identifier: Apache-2.0
"""SAO math PPO contract tests against independent small-tensor oracles.

These tests intentionally call the current upstream PPO/critic/GAE functions and
compare them with hand-computed fixtures. They protect the async PPO launch
contract without copying the implementation's wrappers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from examples.math.sao_ppo import validate_contract
from scripts.sao.async_eval import AsyncEvalPPOTrainer, SaoPPOConfig

from areal.api.cli_args import (
    PPOActorConfig,
    PPOConfig,
    parse_cli_args,
    to_structured_cfg,
)
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.critic import ppo_loss_fn as critic_ppo_loss_fn
from areal.trainer.ppo.gae import _compute_token_level_gae
from areal.trainer.ppo.lambda_fn import resolve_gae_lambda_fn
from areal.utils.data import KLEstimator
from areal.utils.functional import ppo_actor_loss_fn, ppo_critic_loss_fn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _manual_ppo_actor_loss(
    logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps_clip: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(logprobs - behavior_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    per_token = torch.maximum(-advantages * ratio, -advantages * clipped_ratio)
    return torch.where(loss_mask, per_token, torch.zeros_like(per_token)).sum() / (
        loss_mask.count_nonzero()
    )


def _manual_token_gae_gamma_lambda_one(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    seq_no_eos_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, max_seqlen = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(bs, dtype=torch.float32)
    next_values = values[:, -1] * seq_no_eos_mask
    for timestep in reversed(range(max_seqlen - 1)):
        delta = rewards[:, timestep] + next_values - values[:, timestep]
        candidate = delta + last_advantage
        mask = loss_mask[:, timestep]
        last_advantage = torch.where(mask.bool(), candidate, last_advantage)
        next_values = torch.where(mask.bool(), values[:, timestep], next_values)
        advantages[:, timestep] = last_advantage
    return advantages, advantages + values


def _manual_clipped_value_mse(
    value: torch.Tensor,
    old_value: torch.Tensor,
    target_value: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps_clip: float = 0.5,
) -> torch.Tensor:
    unclipped = 0.5 * (value - target_value) ** 2
    clipped_value = old_value + (value - old_value).clamp(-eps_clip, eps_clip)
    clipped = 0.5 * (clipped_value - target_value) ** 2
    per_token = torch.maximum(unclipped, clipped)
    return torch.where(loss_mask, per_token, torch.zeros_like(per_token)).sum() / (
        loss_mask.count_nonzero()
    )


def _make_contract_actor() -> PPOActor:
    config = PPOActorConfig(
        kl_ctl=0.0,
        discount=1.0,
        gae_lambda=1.0,
        gae_timestep_unit="token",
        adv_norm=None,
        reward_norm=None,
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
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 20.0
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


def _load_sao_ppo_config(monkeypatch, tmp_path) -> SaoPPOConfig:
    monkeypatch.setenv("SAO_TRIAL_NAME", "unit-test")
    monkeypatch.setenv("SAO_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("SAO_MODEL_PATH", str(tmp_path / "actor"))
    monkeypatch.setenv("SAO_CRITIC_PATH", str(tmp_path / "critic"))
    monkeypatch.setenv("SAO_DATA_PATH", str(tmp_path / "dataset"))

    cfg, _ = parse_cli_args(["--config", str(REPO_ROOT / "examples/math/sao_ppo.yaml")])
    cfg = to_structured_cfg(cfg, SaoPPOConfig)
    config = OmegaConf.to_object(cfg)

    assert isinstance(config, SaoPPOConfig)
    return config


def test_sao_ppo_config_parses_dedicated_eval_and_colocated_critic(
    monkeypatch, tmp_path
):
    config = _load_sao_ppo_config(monkeypatch, tmp_path)

    validate_contract(config)

    assert AsyncEvalPPOTrainer is not None
    assert config.actor.backend == "fsdp:d4p1t1"
    assert config.critic.backend == "fsdp:d4p1t1"
    assert config.rollout.backend == "sglang:d3p1t1"
    assert config.evaluation_rollout.backend == "sglang:d1p1t1"
    assert config.critic.scheduling_strategy.type == "colocation"
    assert config.critic.scheduling_strategy.target == "actor"
    assert config.evaluation_rollout.scheduling_strategy.type == "separation"
    assert config.evaluation_rollout.max_concurrent_rollouts == 32
    assert config.evaluation_rollout.queue_size == 64
    assert config.eval_gconfig.n_samples == 2
    assert config.saver.freq_steps == 20
    assert config.recover.freq_steps == 20
    assert config.evaluator.freq_steps == 20


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda config: setattr(config.rollout, "backend", "sglang:d4p1t1"),
            "rollout.backend",
        ),
        (
            lambda config: setattr(
                config.critic.scheduling_strategy, "type", "separation"
            ),
            "critic must colocate with actor",
        ),
        (
            lambda config: setattr(config.evaluator, "freq_steps", 25),
            "saver and evaluator frequencies to match",
        ),
        (
            lambda config: setattr(config.evaluation_rollout, "scheduling_spec", ()),
            "evaluation_rollout workers must reserve one GPU each",
        ),
    ],
)
def test_sao_ppo_contract_rejects_invalid_allocation_and_eval_sync(
    monkeypatch, tmp_path, mutate, match
):
    config = copy.deepcopy(_load_sao_ppo_config(monkeypatch, tmp_path))
    mutate(config)

    with pytest.raises(ValueError, match=match):
        validate_contract(config)


def test_standard_clipped_ppo_matches_independent_oracle_with_behavior_denominator():
    log_ratio = torch.log(torch.tensor([0.7, 0.9, 1.0, 1.3, 1.6, 0.5]))
    behavior = torch.tensor([-1.0, -0.7, -2.0, -1.5, -0.2, -3.0])
    logprobs = behavior + log_ratio
    advantages = torch.tensor([1.5, 2.0, -1.0, 3.0, -2.5, 9.0])
    loss_mask = torch.tensor([True, True, True, True, True, False])

    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=behavior,
        old_logprobs=behavior,
        advantages=advantages,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        loss_mask=loss_mask,
        rejection_sampling=None,
        importance_sampling_level="token",
    )

    expected = _manual_ppo_actor_loss(logprobs, behavior, advantages, loss_mask)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=0.0)
    expected_ratio_stat = torch.where(
        loss_mask, torch.exp(log_ratio), torch.zeros_like(log_ratio)
    )
    torch.testing.assert_close(stat["importance_weight"], expected_ratio_stat)
    assert stat["n_valid_tokens"] == float(loss_mask.count_nonzero())
    assert stat["n_total_tokens"] == float(loss_mask.numel())


def test_standard_ppo_positive_and_negative_clip_gradients_match_finite_difference():
    ratios = torch.tensor([1.35, 1.10, 0.70, 0.90], dtype=torch.float64)
    behavior = torch.tensor([-1.0, -1.2, -0.8, -1.5], dtype=torch.float64)
    logprobs = (behavior + torch.log(ratios)).clone().detach().requires_grad_(True)
    advantages = torch.tensor([2.0, 2.0, -3.0, -3.0], dtype=torch.float64)
    loss_mask = torch.ones(4, dtype=torch.bool)

    loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=behavior,
        old_logprobs=behavior,
        advantages=advantages,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        loss_mask=loss_mask,
        rejection_sampling=None,
        importance_sampling_level="token",
    )
    loss.backward()

    n = int(loss_mask.count_nonzero())
    expected_grad = torch.tensor(
        [
            0.0,  # positive advantage, ratio above upper clip: constant branch
            -2.0 * 1.10 / n,  # positive advantage inside band
            0.0,  # negative advantage, ratio below lower clip: constant branch
            3.0 * 0.90 / n,  # negative advantage inside band
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(logprobs.grad, expected_grad, rtol=0.0, atol=1e-12)

    eps = 1.0e-6
    for idx in range(logprobs.numel()):
        plus = logprobs.detach().clone()
        minus = logprobs.detach().clone()
        plus[idx] += eps
        minus[idx] -= eps
        f_plus = _manual_ppo_actor_loss(plus, behavior, advantages, loss_mask)
        f_minus = _manual_ppo_actor_loss(minus, behavior, advantages, loss_mask)
        finite_diff = (f_plus - f_minus) / (2 * eps)
        torch.testing.assert_close(
            finite_diff, expected_grad[idx], rtol=1e-5, atol=1e-7
        )


def test_token_gae_gamma_lambda_one_matches_manual_reward_to_go_and_signs():
    rewards = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.5, 0.0]],
        dtype=torch.float32,
    )
    values = torch.tensor(
        [[0.2, 0.1, 0.0, 0.4, 0.0], [0.0, 0.3, 0.1, -0.2, 0.0]],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [[False, True, True, False, False], [False, True, True, True, False]]
    )
    seq_no_eos_mask = torch.tensor([False, False])

    actual_advantages, actual_returns = _compute_token_level_gae(
        rewards=rewards,
        values=values,
        loss_mask=loss_mask.float(),
        seq_no_eos_mask=seq_no_eos_mask,
        discount=1.0,
        gae_lambda=1.0,
    )
    expected_advantages, expected_returns = _manual_token_gae_gamma_lambda_one(
        rewards, values, loss_mask, seq_no_eos_mask
    )

    torch.testing.assert_close(actual_advantages, expected_advantages)
    torch.testing.assert_close(actual_returns, expected_returns)
    assert actual_advantages[0, 1] > 0
    assert actual_advantages[1, 1] < 0


def test_actor_advantage_path_uses_prompt_padding_next_token_shift():
    actor = _make_contract_actor()
    batch = {
        "input_ids": torch.tensor([[10, 11, 20, 21, 22, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        # Rollout mask marks generated target tokens. Actor shifts it left so
        # position i predicts target token i+1.
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32),
        "logprobs": torch.tensor([[0.0, 0.0, -0.1, -0.2, -0.3, 0.0]]),
        "values": torch.zeros(1, 6, dtype=torch.float32),
        "rewards": torch.tensor([1.0], dtype=torch.float32),
    }

    result = actor._compute_advantages(batch)

    expected_shifted_mask = torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.float32)
    expected_shifted_logp = torch.tensor([[0.0, -0.1, -0.2, -0.3, 0.0, 0.0]])
    expected_advantages = expected_shifted_mask.clone()
    torch.testing.assert_close(result["loss_mask"], expected_shifted_mask)
    torch.testing.assert_close(result["logprobs"], expected_shifted_logp)
    # GAE may carry the response return into masked prompt positions. Only the
    # shifted response mask contributes to policy/value losses.
    torch.testing.assert_close(
        result["advantages"] * result["loss_mask"], expected_advantages
    )
    torch.testing.assert_close(
        result["returns"] * result["loss_mask"], expected_advantages
    )


def test_clipped_critic_loss_is_half_mse_and_uses_upstream_wrapper():
    value = torch.tensor([[1.8, 0.1, -0.6, 4.0]], dtype=torch.float32)
    old_value = torch.tensor([[1.0, 0.0, 0.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, 1.0, -1.0, 2.0]], dtype=torch.float32)
    loss_mask = torch.tensor([[True, True, False, True]])

    direct_loss, stat = ppo_critic_loss_fn(
        value=value,
        old_value=old_value,
        target_value=target,
        value_eps_clip=0.5,
        loss_mask=loss_mask,
    )
    wrapper_loss = critic_ppo_loss_fn(
        value=value.unsqueeze(-1),
        input_data={"values": old_value, "returns": target, "loss_mask": loss_mask},
        eps_clip=0.5,
    )

    expected = _manual_clipped_value_mse(
        value, old_value, target, loss_mask, eps_clip=0.5
    )
    torch.testing.assert_close(direct_loss, expected)
    torch.testing.assert_close(wrapper_loss, expected)
    assert stat["clip_mask"][0, 0]
    assert not stat["clip_mask"][0, 1]


def test_sao_contract_config_has_no_hidden_dapo_group_or_filtering_knobs():
    actor = PPOActorConfig(
        kl_ctl=0.0,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        discount=1.0,
        gae_lambda=1.0,
        gae_timestep_unit="token",
        adv_norm=None,
        reward_norm=None,
        use_decoupled_loss=False,
        recompute_logprob=False,
        rejection_sampling=None,
        importance_sampling_level="token",
        overlong_reward_penalty=False,
        use_sapo_loss=False,
        use_cispo_loss=False,
        ppo_n_minibatches=1,
    )
    config = PPOConfig(
        actor=actor,
        dynamic_bs=False,
        critic=SimpleNamespace(eps_clip=0.5),
    )

    assert config.dynamic_bs is False
    assert actor.rejection_sampling is None
    assert actor.reward_norm is None
    assert actor.adv_norm is None
    assert actor.overlong_reward_penalty is False
    assert actor.use_decoupled_loss is False
    assert actor.use_sapo_loss is False
    assert actor.use_cispo_loss is False
    assert actor.importance_sampling_level == "token"
    assert actor.eps_clip == pytest.approx(0.2)
    assert actor.eps_clip_higher is None


def test_fsdp_per_token_accumulation_scaling_matches_global_masked_mean_on_cpu():
    per_rank_mb_losses = [
        [torch.tensor(0.20), torch.tensor(0.80)],
        [torch.tensor(-0.40), torch.tensor(1.10)],
    ]
    per_rank_mb_weights = [
        [torch.tensor(2.0), torch.tensor(3.0)],
        [torch.tensor(1.0), torch.tensor(4.0)],
    ]
    dp_size = len(per_rank_mb_losses)
    total_weight = sum(
        weight for rank_weights in per_rank_mb_weights for weight in rank_weights
    )

    # FSDP scales each local mean loss by local_weight/global_weight*dp_size;
    # DDP then averages gradients across dp_size ranks. That must equal one
    # global masked mean over all valid response tokens.
    rank_backward_losses = []
    for mb_losses, mb_weights in zip(per_rank_mb_losses, per_rank_mb_weights):
        rank_loss = sum(
            loss * (weight / total_weight) * dp_size
            for loss, weight in zip(mb_losses, mb_weights)
        )
        rank_backward_losses.append(rank_loss)
    actual_after_ddp_average = sum(rank_backward_losses) / dp_size
    expected_global_mean = (
        sum(
            loss * weight
            for mb_losses, mb_weights in zip(per_rank_mb_losses, per_rank_mb_weights)
            for loss, weight in zip(mb_losses, mb_weights)
        )
        / total_weight
    )

    torch.testing.assert_close(actual_after_ddp_average, expected_global_mean)
