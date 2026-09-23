# SPDX-License-Identifier: Apache-2.0
"""PPO loss reduction contracts for token and equal-sequence means."""

from __future__ import annotations

import math

import pytest
import torch

from areal.api.cli_args import (
    PPOActorConfig,
    PPOCriticConfig,
    RejectionSamplingConfig,
)
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.trainer.ppo.critic import ppo_loss_fn as critic_wrapper_loss_fn
from areal.utils.functional import (
    loss_reduction_weight,
    ppo_actor_loss_fn,
    ppo_critic_loss_fn,
)


def _manual_sequence_mean(
    per_token_loss: torch.Tensor,
    numerator_mask: torch.Tensor,
    denominator_mask: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if denominator_mask is None:
        denominator_mask = numerator_mask
    masked = torch.where(numerator_mask, per_token_loss, 0.0)
    if masked.ndim == 2:
        denom = denominator_mask.sum(dim=-1).clamp(min=1).to(masked.dtype)
        return (masked.sum(dim=-1) / denom).mean()

    assert cu_seqlens is not None
    parts = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        seq_loss = masked[start:end]
        seq_den = denominator_mask[start:end].sum().clamp(min=1).to(masked.dtype)
        parts.append(seq_loss.sum() / seq_den)
    return torch.stack(parts).mean()


def _actor_loss_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    proximal = torch.zeros(3, 4, dtype=torch.float32)
    log_ratio = torch.tensor(
        [[0.0, math.log(1.4), 0.0, 0.0], [math.log(0.6), 0.0, 0.0, 0.0], [0.0] * 4],
        dtype=torch.float32,
    )
    logprobs = (proximal + log_ratio).detach().clone().requires_grad_(True)
    return logprobs, proximal, log_ratio


def test_config_defaults_preserve_token_mean_and_reject_invalid_values() -> None:
    """The new config field is backward-compatible and validates typos."""
    assert PPOActorConfig().loss_reduction == "token_mean"
    assert PPOCriticConfig().loss_reduction == "token_mean"
    assert PPOActorConfig(loss_reduction="sequence_mean").loss_reduction == (
        "sequence_mean"
    )
    assert PPOCriticConfig(loss_reduction="sequence_mean").loss_reduction == (
        "sequence_mean"
    )

    with pytest.raises(ValueError, match="loss_reduction"):
        PPOActorConfig(loss_reduction="bad")
    with pytest.raises(ValueError, match="loss_reduction"):
        PPOCriticConfig(loss_reduction="bad")


def test_actor_sequence_mean_matches_manual_padded_oracle_and_gradients() -> None:
    """Padded actor loss averages each answer, including all-masked answers."""
    logprobs, proximal, log_ratio = _actor_loss_inputs()
    old = proximal.clone()
    advantages = torch.tensor(
        [[-2.0, 1.0, -3.0, 5.0], [4.0, -6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False], [False] * 4]
    )

    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=old,
        advantages=advantages,
        eps_clip=0.2,
        eps_clip_higher=None,
        loss_mask=loss_mask,
        importance_sampling_level="token",
        loss_reduction="sequence_mean",
    )

    ratio = torch.exp(log_ratio)
    clipped = ratio.clamp(0.8, 1.2)
    per_token = torch.maximum(-advantages * ratio, -advantages * clipped)
    expected = _manual_sequence_mean(per_token, loss_mask)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=1e-6)
    expected_stat_ratio = torch.where(loss_mask, ratio, 0.0)
    torch.testing.assert_close(
        stat["importance_weight"], expected_stat_ratio, rtol=0.0, atol=1e-6
    )

    loss.backward()
    expected_grad = torch.zeros_like(logprobs)
    # Token (0,0): negative advantage inside band, sequence 0 has two valid tokens.
    expected_grad[0, 0] = 2.0 / (2 * 3)
    # Token (0,1): positive advantage above upper clip -> clipped constant branch.
    expected_grad[0, 1] = 0.0
    # Token (1,0): positive advantage below lower clip -> unclipped branch.
    expected_grad[1, 0] = -4.0 * 0.6 / 3
    torch.testing.assert_close(logprobs.grad, expected_grad, rtol=0.0, atol=1e-6)


def test_actor_sequence_mean_matches_miles_length_oracle_and_backward() -> None:
    """Lengths 2/4/0 give sequence_mean=7/3 while token_mean stays 4."""
    per_token_loss = torch.tensor(
        [[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 6.0, 8.0], [99.0, 99.0, 99.0, 99.0]]
    )
    advantages = -per_token_loss
    proximal = torch.zeros_like(per_token_loss)
    logprobs = torch.zeros_like(per_token_loss, requires_grad=True)
    loss_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, True], [False] * 4]
    )

    sequence_loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=proximal,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
        loss_reduction="sequence_mean",
    )
    token_loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=proximal,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
        loss_reduction="token_mean",
    )

    torch.testing.assert_close(
        sequence_loss, torch.tensor(7.0 / 3.0), rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(token_loss, torch.tensor(4.0), rtol=0.0, atol=1e-6)
    sequence_loss.backward()
    expected_grad = torch.tensor(
        [
            [1.0 / 6.0, 3.0 / 6.0, 0.0, 0.0],
            [2.0 / 12.0, 4.0 / 12.0, 6.0 / 12.0, 8.0 / 12.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(logprobs.grad, expected_grad, rtol=0.0, atol=1e-6)


def test_actor_sequence_mean_uses_token_ratios_not_gspo_sequence_ratios() -> None:
    """loss_reduction=sequence_mean is independent of GSPO ratio aggregation."""
    logprobs, proximal, log_ratio = _actor_loss_inputs()
    advantages = torch.full_like(logprobs, -1.0)
    loss_mask = torch.tensor(
        [[True, True, False, False], [True, False, False, False], [False] * 4]
    )

    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=proximal,
        advantages=advantages,
        eps_clip=1.0,
        eps_clip_higher=10.0,
        loss_mask=loss_mask,
        importance_sampling_level="token",
        loss_reduction="sequence_mean",
    )

    ratio = torch.exp(log_ratio)
    expected = _manual_sequence_mean(ratio, loss_mask)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=1e-6)
    expected_stat_ratio = torch.where(loss_mask, ratio, 0.0)
    torch.testing.assert_close(
        stat["importance_weight"], expected_stat_ratio, rtol=0.0, atol=1e-6
    )


def test_actor_sequence_mean_packed_matches_manual_oracle() -> None:
    """Packed 1D losses use cu_seqlens as real sequence boundaries."""
    cu_seqlens = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    proximal = torch.zeros(6, dtype=torch.float32)
    logprobs = torch.zeros(6, dtype=torch.float32, requires_grad=True)
    advantages = -torch.tensor([1.0, 3.0, 9.0, 2.0, 4.0, 99.0])
    loss_mask = torch.tensor([True, False, True, True, True, False])

    loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=proximal,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
        cu_seqlens=cu_seqlens,
        loss_reduction="sequence_mean",
    )

    per_token = -advantages
    expected = _manual_sequence_mean(per_token, loss_mask, cu_seqlens=cu_seqlens)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=1e-6)
    assert loss_reduction_weight(loss_mask, "sequence_mean", cu_seqlens).item() == 3


def test_actor_sequence_mean_rejection_sampling_keeps_pre_rejection_denominator() -> (
    None
):
    """Rejected tokens are zeroed without renormalizing the answer denominator."""
    proximal = torch.zeros(2, 4, dtype=torch.float32)
    old = torch.tensor(
        [[0.0, 0.0, -math.log(2.0), -math.log(2.0)], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    logprobs = proximal.detach().clone().requires_grad_(True)
    advantages = -torch.tensor([[1.0, 3.0, 99.0, 99.0], [4.0, 6.0, 88.0, 88.0]])
    original_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    config = RejectionSamplingConfig(
        level="token", action="mask", metric="ratio", upper=1.5
    )

    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=old,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=original_mask,
        rejection_sampling=config,
        loss_reduction="sequence_mean",
    )

    kept_mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    per_token = -advantages
    expected = _manual_sequence_mean(
        per_token, kept_mask, denominator_mask=original_mask
    )
    torch.testing.assert_close(expected, torch.tensor(3.0), rtol=0.0, atol=0.0)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(stat["behave_mask"], kept_mask, rtol=0.0, atol=0.0)
    loss.backward()
    expected_grad = torch.tensor(
        [[1.0 / 8, 3.0 / 8, 0.0, 0.0], [4.0 / 4, 6.0 / 4, 0.0, 0.0]]
    )
    torch.testing.assert_close(logprobs.grad, expected_grad, rtol=0.0, atol=1e-6)


def test_all_masked_real_microbatch_keeps_sequence_weight_and_zero_gradient() -> None:
    """A real zero-loss answer still counts, with a differentiable zero loss."""
    logprobs = torch.zeros(3, requires_grad=True)
    mask = torch.zeros(3, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=torch.zeros(3),
        old_logprobs=torch.zeros(3),
        advantages=torch.ones(3),
        eps_clip=0.2,
        loss_mask=mask,
        cu_seqlens=cu_seqlens,
        loss_reduction="sequence_mean",
    )
    assert loss_reduction_weight(mask, "sequence_mean", cu_seqlens).item() == 1
    assert loss_reduction_weight(mask, "token_mean", cu_seqlens).item() == 0
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0.0, atol=0.0)
    loss.backward()
    torch.testing.assert_close(logprobs.grad, torch.zeros(3), rtol=0.0, atol=0.0)


def test_critic_sequence_mean_matches_half_mse_value_clip_oracle() -> None:
    """Critic sequence_mean preserves half-MSE and value clipping semantics."""
    value = torch.tensor([[1.8, 0.1, -0.6, 4.0], [2.0, 3.0, 9.0, 10.0]])
    old_value = torch.tensor([[1.0, 0.0, 0.0, 3.0], [0.0, 3.0, 1.0, 10.0]])
    target = torch.tensor([[2.0, 1.0, -1.0, 2.0], [1.0, 5.0, 8.0, 10.0]])
    loss_mask = torch.tensor([[True, True, False, True], [False, True, True, False]])

    loss, stat = ppo_critic_loss_fn(
        value=value,
        old_value=old_value,
        target_value=target,
        value_eps_clip=0.5,
        loss_mask=loss_mask,
        loss_reduction="sequence_mean",
    )

    unclipped = 0.5 * (value - target).square()
    clipped_value = old_value + (value - old_value).clamp(-0.5, 0.5)
    clipped = 0.5 * (clipped_value - target).square()
    per_token = torch.maximum(unclipped, clipped)
    expected = _manual_sequence_mean(per_token, loss_mask)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=1e-6)
    assert stat["clip_mask"][0, 0]

    wrapper_loss = critic_wrapper_loss_fn(
        value=value.unsqueeze(-1),
        input_data={"values": old_value, "returns": target, "loss_mask": loss_mask},
        eps_clip=0.5,
        loss_reduction="sequence_mean",
    )
    torch.testing.assert_close(wrapper_loss, expected, rtol=0.0, atol=1e-6)


def test_sequence_mean_microbatch_weighting_matches_full_batch() -> None:
    """Engine-style local sequence weights reconstruct the full sequence mean."""
    proximal = torch.zeros(3, 3, dtype=torch.float32)
    logprobs = torch.zeros_like(proximal)
    advantages = -torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [4.0, 6.0, 8.0]])
    loss_mask = torch.tensor(
        [[True, True, False], [True, False, False], [True, True, True]]
    )

    full_loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=proximal,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=loss_mask,
        loss_reduction="sequence_mean",
    )
    weighted_parts = []
    total_weight = torch.tensor(0.0)
    for rows in (slice(0, 2), slice(2, 3)):
        mb = {
            "logprobs": logprobs[rows],
            "proximal": proximal[rows],
            "advantages": advantages[rows],
            "loss_mask": loss_mask[rows],
        }
        mb_loss, _ = ppo_actor_loss_fn(
            logprobs=mb["logprobs"],
            proximal_logprobs=mb["proximal"],
            old_logprobs=mb["proximal"],
            advantages=mb["advantages"],
            eps_clip=0.2,
            loss_mask=mb["loss_mask"],
            loss_reduction="sequence_mean",
        )
        weight = loss_reduction_weight(mb["loss_mask"], "sequence_mean").float()
        weighted_parts.append((mb_loss, weight))
        total_weight = total_weight + weight
    reconstructed = sum(loss * weight / total_weight for loss, weight in weighted_parts)

    torch.testing.assert_close(reconstructed, full_loss, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"use_sapo_loss": True}, "SAPO"),
        ({"use_cispo_loss": True, "eps_clip_higher": 0.28}, "CISPO"),
        ({"m2_threshold": 0.1}, "M2PO"),
    ],
)
def test_grpo_sequence_mean_fails_closed_for_unsupported_actor_paths(
    kwargs: dict[str, object], error: str
) -> None:
    """Unsupported actor objectives must not silently use sequence_mean."""
    input_data = {
        "logprobs": torch.zeros(1, 2),
        "prox_logp": torch.zeros(1, 2),
        "advantages": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    call_kwargs = dict(kwargs)
    eps_clip_higher = call_kwargs.pop("eps_clip_higher", None)

    with pytest.raises(ValueError, match=error):
        grpo_loss_fn(
            logprobs=torch.zeros(1, 2),
            entropy=torch.zeros(1, 2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=eps_clip_higher,
            c_clip=None,
            loss_reduction="sequence_mean",
            **call_kwargs,
        )


def test_grpo_sequence_mean_fails_closed_for_teacher_distillation() -> None:
    """Joint distillation has its own token denominator and is not supported."""
    input_data = {
        "logprobs": torch.zeros(1, 2),
        "prox_logp": torch.zeros(1, 2),
        "advantages": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "teacher_logp": torch.zeros(1, 2),
    }

    with pytest.raises(ValueError, match="teacher distillation"):
        grpo_loss_fn(
            logprobs=torch.zeros(1, 2),
            entropy=torch.zeros(1, 2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            loss_reduction="sequence_mean",
        )
