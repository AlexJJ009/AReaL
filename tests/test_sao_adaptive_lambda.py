"""M2: paper length rule reaches the actual GAE and optimizer consumer."""

import copy

import pytest
import torch

from tests.test_sao_dual_lambda import make_actor

from areal.trainer.ppo.lambda_fn import sao_length_adaptive_gae

LAMBDA_PATH = "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"


def mixed_rollout():
    ids = torch.arange(12).repeat(3, 1)
    mask = torch.zeros(3, 12)
    attention = torch.zeros_like(mask)
    for row, length in enumerate([1, 2, 8]):
        mask[row, 2 : 2 + length] = 1
        attention[row, : 2 + length] = 1
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "loss_mask": mask,
        "rewards": torch.tensor([1.0, 1.0, 1.0]),
        "values": torch.zeros_like(mask),
        "logprobs": torch.full_like(mask, -2.77),
    }


def test_adaptive_lambda_mixed_lengths_match_independent_reward_decay():
    """The terminal-reward polynomial independently fixes every advantage."""
    actor = make_actor(LAMBDA_PATH, lambda_kwargs={"alpha": 1.5})
    data = actor._compute_advantages(mixed_rollout())
    lambdas = actor._compute_gae_lambda(data["loss_mask"], None)
    torch.testing.assert_close(
        lambdas, torch.tensor([1 / 3, 2 / 3, 11 / 12]), rtol=1e-6, atol=1e-7
    )
    for row, length in enumerate([1, 2, 8]):
        expected = torch.tensor(
            [(1 - 1 / (1.5 * length)) ** power for power in reversed(range(length))]
        )
        active = data["loss_mask"][row].bool()
        torch.testing.assert_close(
            data["advantages"][row, active], expected, rtol=1e-6, atol=1e-7
        )
        torch.testing.assert_close(
            data["returns"][row, active], torch.ones(length), rtol=0, atol=0
        )
    before = actor.engine.model.weight.detach().clone()
    actor.ppo_update([copy.deepcopy(data)])
    assert not torch.equal(actor.engine.model.weight, before)


def test_adaptive_lambda_prompt_padding_changes_do_not_change_action_length():
    """Insertion outside action positions must not change the per-action decay."""
    data = mixed_rollout()
    changed = {}
    for key, value in data.items():
        changed[key] = (
            torch.nn.functional.pad(value, (3, 4)) if value.ndim == 2 else value.clone()
        )
    changed["attention_mask"][:, :3] = 1
    a = make_actor(LAMBDA_PATH)._compute_advantages(data)
    b = make_actor(LAMBDA_PATH)._compute_advantages(changed)
    torch.testing.assert_close(
        a["advantages"][a["loss_mask"].bool()],
        b["advantages"][b["loss_mask"].bool()],
        rtol=1e-6,
        atol=1e-7,
    )


def test_wrong_length_or_short_sequence_override_is_detected():
    """Negative controls model prompt-inclusive L and Miles' short-L override."""
    actor = make_actor(LAMBDA_PATH)
    data = actor._compute_advantages(mixed_rollout())
    correct = actor._compute_gae_lambda(data["loss_mask"], None)
    for broken in (torch.ones(3), 1 - 1 / (1.5 * data["attention_mask"].sum(-1))):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(correct, broken, rtol=1e-6, atol=1e-7)


def test_empty_action_trajectory_rejected():
    """An empty trajectory cannot be repaired by inventing L=1."""
    with pytest.raises(RuntimeError, match="nonempty"):
        sao_length_adaptive_gae({"effective_token_lengths": torch.tensor([0])})
