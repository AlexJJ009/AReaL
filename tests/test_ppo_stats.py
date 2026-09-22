from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.trainer.ppo.actor import _infer_prompt_lens, grpo_loss_fn
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.trainer.ppo.stats import (
    derive_critic_update_metrics,
    infer_token_denominator,
)
from areal.utils.stats_tracker import DistributedStatsTracker


def test_infer_token_denominator_prefers_attention_mask():
    input_data = {
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "input_ids": torch.tensor([[11, 12], [13, 14]]),
    }

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(5))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_uses_input_ids_when_attention_mask_missing():
    input_data = {"input_ids": torch.tensor([[11, 12, 13], [14, 15, 16]])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(2, 3))

    assert n_tokens.shape == torch.Size([2, 3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_for_padded_tree_input_ids():
    input_data = {"input_ids": torch.tensor([11, 12, 13, 0])}

    n_tokens = infer_token_denominator(input_data, fallback=torch.zeros(3))

    assert n_tokens.shape == torch.Size([3])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_infer_token_denominator_falls_back_when_metadata_is_missing():
    fallback = torch.zeros(4)

    n_tokens = infer_token_denominator({"logprobs": torch.zeros(2)}, fallback=fallback)

    assert n_tokens.shape == torch.Size([4])
    assert n_tokens.dtype == torch.bool
    assert torch.all(n_tokens)


def test_grpo_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "logprobs": torch.zeros(2),
        "advantages": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
        "prox_logp": torch.zeros(2),
        "versions": torch.zeros(2, dtype=torch.int32),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()
        mock_tracker.scope = MagicMock()
        mock_tracker.scope.return_value.__enter__ = MagicMock()
        mock_tracker.scope.return_value.__exit__ = MagicMock()

        grpo_loss_fn(
            logprobs=torch.zeros(2),
            entropy=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )

    n_tokens = next(
        call.kwargs["n_tokens"]
        for call in mock_tracker.denominator.call_args_list
        if "n_tokens" in call.kwargs
    )
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def test_critic_loss_fn_uses_full_cu_seqlens_for_n_tokens():
    input_data = {
        "input_ids": torch.tensor([11, 12]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "values": torch.zeros(2),
        "returns": torch.ones(2),
        "loss_mask": torch.ones(2, dtype=torch.bool),
    }

    with patch("areal.trainer.ppo.critic.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()

        ppo_loss_fn(
            value=torch.zeros(2),
            input_data=input_data,
            eps_clip=0.2,
        )

    n_tokens = mock_tracker.denominator.call_args.kwargs["n_tokens"]
    assert n_tokens.shape == torch.Size([4])
    assert torch.all(n_tokens)


def _collect_critic_stats(
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict:
    tracker = DistributedStatsTracker()
    with patch("areal.trainer.ppo.critic.stats_tracker", tracker):
        with tracker.scope("ppo_critic"):
            for value, target, loss_mask in batches:
                ppo_loss_fn(
                    value=value.unsqueeze(0),
                    input_data={
                        "values": value.clone().unsqueeze(0),
                        "returns": target.unsqueeze(0),
                        "loss_mask": loss_mask.unsqueeze(0),
                    },
                    eps_clip=10.0,
                )
    return derive_critic_update_metrics(tracker.export(reset=True))


def test_critic_update_metrics_ignore_masked_padding():
    value = torch.tensor([0.0, 1.0, 2.0, 999.0])
    target = torch.tensor([0.0, 2.0, 4.0, -999.0])
    loss_mask = torch.tensor([True, True, True, False])

    stats = _collect_critic_stats([(value, target, loss_mask)])

    assert stats["ppo_critic/critic_residual"] == -1.0
    assert stats["ppo_critic/critic_mse"] == pytest.approx(5.0 / 3.0)
    assert stats["ppo_critic/critic_target"] == 2.0
    assert stats["ppo_critic/critic_target_second_moment"] == pytest.approx(20.0 / 3.0)
    assert stats["ppo_critic/critic_target_variance"] == pytest.approx(8.0 / 3.0)
    assert stats["ppo_critic/critic_residual_variance"] == pytest.approx(2.0 / 3.0)
    assert stats["ppo_critic/critic_explained_variance"] == pytest.approx(0.75)
    assert stats["ppo_critic/critic_explained_variance_defined"] == 1.0


def test_critic_update_metrics_weight_uneven_microbatches_by_tokens():
    batches = [
        (
            torch.tensor([0.0, 100.0]),
            torch.tensor([0.0, -100.0]),
            torch.tensor([True, False]),
        ),
        (
            torch.tensor([10.0, 20.0, 0.0]),
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([True, True, True]),
        ),
    ]

    stats = _collect_critic_stats(batches)

    assert stats["ppo_critic/critic_target"] == 15.0
    assert stats["ppo_critic/critic_target_second_moment"] == 350.0
    assert stats["ppo_critic/critic_target_variance"] == 125.0
    assert stats["ppo_critic/critic_mse"] == 225.0
    assert stats["ppo_critic/critic_residual_variance"] == 168.75
    assert stats["ppo_critic/critic_explained_variance"] == pytest.approx(-0.35)
    assert stats["ppo_critic/critic_explained_variance_defined"] == 1.0


def test_critic_explained_variance_uses_residual_variance_not_mse():
    value = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([0.0, 1.0, 2.0])
    loss_mask = torch.tensor([True, True, True])

    stats = _collect_critic_stats([(value, target, loss_mask)])

    assert stats["ppo_critic/critic_mse"] == 1.0
    assert stats["ppo_critic/critic_residual"] == 1.0
    assert stats["ppo_critic/critic_residual_variance"] == 0.0
    assert stats["ppo_critic/critic_target_variance"] == pytest.approx(2.0 / 3.0)
    assert stats["ppo_critic/critic_explained_variance"] == 1.0


def test_critic_update_metrics_leave_explained_variance_undefined_for_zero_variance():
    value = torch.tensor([1.0, 2.0, 3.0])
    target = torch.zeros(3)
    loss_mask = torch.tensor([True, True, True])

    stats = _collect_critic_stats([(value, target, loss_mask)])

    assert stats["ppo_critic/critic_target_variance"] == 0.0
    assert stats["ppo_critic/critic_explained_variance"] is None
    assert stats["ppo_critic/critic_explained_variance_defined"] == 0.0
    assert stats["ppo_critic/critic_mse"] == pytest.approx(14.0 / 3.0)


def test_grpo_loss_fn_uses_packed_denominator_for_tree_vocab_stats():
    tracker = DistributedStatsTracker()
    input_data = {
        "input_ids": torch.tensor([11, 12, 13, 0]),
        "logprobs": torch.zeros(3),
        "advantages": torch.ones(3),
        "loss_mask": torch.ones(3, dtype=torch.bool),
        "prox_logp": torch.zeros(3),
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        grpo_loss_fn(
            logprobs=torch.zeros(3),
            entropy=torch.zeros(3),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            vocab_min_logits=torch.zeros(3),
            vocab_max_logits=torch.zeros(3),
        )

    stats = tracker.export(reset=True)
    assert "n_tokens" in stats


def _rolled_loss_mask(prompt_len: int, answer_len: int) -> torch.Tensor:
    """Build the loss_mask as _ppo_update sees it: rolled left by one."""
    mask = torch.tensor([[0] * prompt_len + [1] * answer_len], dtype=torch.float)
    return torch.roll(mask, shifts=-1, dims=-1)


def test_infer_prompt_lens_recovers_the_prompt_length():
    prompt_len, answer_len = 5, 4
    loss_mask = _rolled_loss_mask(prompt_len, answer_len)
    attention_mask = torch.ones(1, prompt_len + answer_len, dtype=torch.long)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == [prompt_len]


def test_infer_prompt_lens_matches_the_sum_formula_without_rejection():
    prompt_len, answer_len = 3, 7
    loss_mask = _rolled_loss_mask(prompt_len, answer_len)
    attention_mask = torch.ones(1, prompt_len + answer_len, dtype=torch.long)

    legacy = attention_mask.sum(-1) - loss_mask.sum(-1)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == legacy.tolist()


def test_infer_prompt_lens_handles_a_batch_of_mixed_prompt_lengths():
    specs = [(2, 6), (5, 3), (1, 7)]
    masks = torch.cat([_rolled_loss_mask(p, a) for p, a in specs])
    attention_mask = torch.ones(len(specs), 8, dtype=torch.long)

    got = _infer_prompt_lens(attention_mask, masks).tolist()

    assert got == [p for p, _ in specs]


def test_infer_prompt_lens_falls_back_to_seqlen_when_nothing_is_trained():
    loss_mask = torch.zeros(1, 9)
    attention_mask = torch.ones(1, 9, dtype=torch.long)

    assert _infer_prompt_lens(attention_mask, loss_mask).tolist() == [9]
