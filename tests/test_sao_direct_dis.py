"""M6 independent gradients and real production loss dispatch."""

import copy
import math

import pytest
import torch

from tests.test_sao_dual_lambda import make_actor, rollout

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.trainer.ppo.dis import DirectDISLoss
from areal.trainer.ppo.update import actor_update_completed


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_direct_dis_gradient_matches_independent_score_function(dtype):
    # Exact boundary equality is tested separately in log space; low precision
    # logprobs round before entering the loss, so use well-separated samples.
    ratios = [0.5, 0.8, 1.0, 2.0, 7.0]
    current = torch.tensor(
        [math.log(r) for r in ratios], dtype=dtype, requires_grad=True
    )
    behavior = torch.zeros(5, dtype=dtype, requires_grad=True)
    advantage = torch.tensor([1, -2, 3, -4, 5], dtype=dtype, requires_grad=True)
    data = {
        "logprobs": behavior,
        "advantages": advantage,
        "loss_mask": torch.ones(5, dtype=torch.bool),
    }
    loss = DirectDISLoss()(current, torch.zeros_like(current), data)
    loss.backward()
    actual_ratios = current.detach().double().exp()
    expected = torch.tensor(
        [0]
        + [
            -float(actual_ratios[i]) * float(advantage[i].detach()) / 5
            for i in (1, 2, 3)
        ]
        + [0],
        dtype=dtype,
    )
    torch.testing.assert_close(
        current.grad,
        expected,
        rtol=0.01 if dtype == torch.bfloat16 else 1e-6,
        atol=1e-7,
    )
    assert behavior.grad is None and advantage.grad is None


def test_direct_dis_strict_boundaries_and_original_token_denominator():
    logs = torch.tensor(
        [math.log1p(-0.3), math.log1p(5), 0.0], dtype=torch.float64, requires_grad=True
    )
    data = {
        "logprobs": torch.zeros(3, dtype=torch.float64),
        "advantages": torch.ones(3, dtype=torch.float64),
        "loss_mask": torch.ones(3, dtype=torch.bool),
    }
    loss = DirectDISLoss()(logs, torch.zeros_like(logs), data)
    loss.backward()
    torch.testing.assert_close(
        logs.grad, torch.tensor([0, 0, -1 / 3], dtype=torch.float64), rtol=0, atol=0
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            logs.grad, torch.tensor([0, 0, -1.0], dtype=torch.float64), rtol=0, atol=0
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1000.0])
def test_nonfinite_inputs_or_ratio_rejected(bad):
    data = {
        "logprobs": torch.zeros(1),
        "advantages": torch.ones(1),
        "loss_mask": torch.ones(1, dtype=torch.bool),
    }
    with pytest.raises(RuntimeError, match="Non-finite"):
        DirectDISLoss()(torch.tensor([bad], requires_grad=True), torch.zeros(1), data)


def test_no_actions_rejected_and_all_masked_gradient_zero():
    current = torch.tensor([-3.0, -3.0], requires_grad=True)
    data = {
        "logprobs": torch.zeros(2),
        "advantages": torch.ones(2),
        "loss_mask": torch.zeros(2, dtype=torch.bool),
    }
    with pytest.raises(RuntimeError, match="valid action"):
        DirectDISLoss()(current, current.detach(), data)
    data["loss_mask"][:] = True
    dis = DirectDISLoss()
    loss = dis(current, current.detach(), data)
    loss.backward()
    assert dis.kept_tokens == 0 and torch.count_nonzero(current.grad) == 0


def test_real_actor_dispatch_avoids_ppo_and_preserves_behavior(monkeypatch):
    actor = make_actor(1.0)
    actor.config.use_direct_dis_loss = True
    actor.config.__post_init__()
    assert actor.config.should_compute_prox_logp() is False

    def forbidden(*args, **kwargs):
        raise AssertionError("PPO/proximal path called")

    monkeypatch.setattr("areal.trainer.ppo.actor.grpo_loss_fn", forbidden)
    monkeypatch.setattr(actor, "compute_logp", forbidden)
    row = rollout()
    row["values"] = row["values"].detach()
    row["logprobs"][0, 1:] = torch.tensor([-2.77, -10, -2.77, -2.77])
    expected_behavior = row["logprobs"].roll(-1, -1) * row["loss_mask"].roll(-1, -1)
    data = actor._compute_advantages(row)
    torch.testing.assert_close(data["logprobs"], expected_behavior, rtol=0, atol=0)
    before = actor.engine.model.weight.detach().clone()
    actor.ppo_update([copy.deepcopy(data)])
    assert not torch.equal(actor.engine.model.weight, before)
    gradients, _ = actor.engine.output_gradients[0]
    assert torch.count_nonzero(gradients[0, 1]) == 0


@pytest.mark.parametrize(
    "conflict",
    [
        {"recompute_logprob": True},
        {"use_decoupled_loss": True},
        {"kl_ctl": 0.1},
        {"use_sapo_loss": True},
        {"use_cispo_loss": True},
    ],
)
def test_dis_rejects_conflicting_ppo_or_kl_config(conflict):
    config = dict(
        backend="fsdp:d1",
        use_direct_dis_loss=True,
        recompute_logprob=False,
        use_decoupled_loss=False,
        kl_ctl=0,
    )
    config.update(conflict)
    with pytest.raises(ValueError, match="Direct DIS"):
        PPOActorConfig(**config)


def test_plain_value_mse_no_clip_gradient():
    value = torch.tensor([[2.0, -1.0]], requires_grad=True)
    loss = ppo_loss_fn(
        value.unsqueeze(-1),
        {
            "values": torch.zeros_like(value),
            "returns": torch.tensor([[1.0, 1.0]]),
            "loss_mask": torch.ones_like(value),
        },
        eps_clip=None,
    )
    loss.backward()
    torch.testing.assert_close(value.grad, torch.tensor([[1.0, -2.0]]), rtol=0, atol=0)


def test_dis_non_fsdp_and_unreported_update_rejected():
    with pytest.raises(ValueError, match="FSDP only"):
        PPOActorConfig(backend="megatron:d1", use_direct_dis_loss=True)
    with pytest.raises(RuntimeError, match="did not complete"):
        actor_update_completed(None)
    skipped = {
        "attempted": 1,
        "successful": 0,
        "skipped": 1,
        "effective": 0,
        "all_masked": 1,
    }
    assert actor_update_completed([skipped, skipped]) is False
