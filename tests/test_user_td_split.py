# SPDX-License-Identifier: Apache-2.0
"""CPU oracles for actor-only user TD splitting and independent critic points."""

import copy
import json
from pathlib import Path

import pytest
import torch

from tests.test_ppo_critic_returns import _make_actor
from tests.test_sao_dual_lambda import TinyTrainEngine

from areal.api.cli_args import PPOCriticConfig
from areal.trainer.ppo.critic import PPOCritic, ppo_loss_fn
from areal.trainer.ppo.lambda_fn import resolve_gae_lambda_fn
from areal.utils import stats_tracker

ADAPTIVE = "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"
ACTIVE = [0, 1, 4, 5, 8, 9, 12, 13]
U_POINTS = [2, 10, 14]


def batch(*, enabled=False, beta=1.0):
    """Two internal users, one tool and a final user; all values are deliberate."""
    data = {
        "input_ids": torch.arange(16).view(1, -1),
        "attention_mask": torch.tensor([[1] * 15 + [0]], dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]]),
        "values": torch.tensor(
            [
                [
                    0.125,
                    0.25,
                    0.375,
                    9,
                    0.5,
                    0.625,
                    8,
                    9,
                    0.75,
                    0.875,
                    1,
                    9,
                    1.125,
                    1.25,
                    1.375,
                    0,
                ]
            ]
        ),
        "logprobs": torch.zeros(1, 16),
        "rewards": torch.tensor([2.0]),
        "terminated": torch.tensor([True]),
        "truncated": torch.tensor([False]),
        "bootstrap_mask": torch.tensor([False]),
    }
    if enabled:
        types = torch.zeros(1, 16, dtype=torch.int32)
        types[0, [2, 6, 10, 14]] = torch.tensor([1, 2, 1, 3], dtype=torch.int32)
        data["user_boundary_types"] = types
        data["user_td_beta"] = torch.tensor([beta])
    return data


def actor(lam):
    result = _make_actor(gae_lambda=1.0, critic_gae_lambda=1.0)
    result.gae_lambda = lam
    result.gae_lambda_fn, result._gae_lambda_is_custom = resolve_gae_lambda_fn(lam)
    result.gae_lambda_kwargs = {"alpha": 1.5} if lam == ADAPTIVE else {}
    return result


@pytest.mark.parametrize("lam", [1.0, ADAPTIVE])
def test_disabled_outputs_match_base_commit_bit_for_bit(lam):
    """Golden tensors/losses were captured from unmodified 9626bc6b, not this code."""
    golden = json.loads(
        (Path(__file__).parent / "fixtures/user_td_split_base.json").read_text()
    )
    assert golden["commit"].startswith("9626bc6b")
    expected = golden[str(lam)]
    data = actor(lam)._compute_advantages(batch())
    for key in ("advantages", "returns", "loss_mask"):
        torch.testing.assert_close(
            data[key], torch.tensor(expected[key]), rtol=0, atol=0
        )
    assert "critic_loss_mask" not in data
    for clip in (None, 0.5):
        loss = ppo_loss_fn(data["values"] + 0.125, data, eps_clip=clip)
        assert loss.item() == expected[f"loss_{clip}"]


@pytest.mark.parametrize("lam", [1.0, ADAPTIVE])
def test_beta_one_keeps_actor_and_adds_only_exact_u_points(lam):
    """Extra critic states never become policy actions or SAO length steps."""
    off = actor(lam)._compute_advantages(batch())
    on = actor(lam)._compute_advantages(batch(enabled=True))
    torch.testing.assert_close(on["advantages"], off["advantages"], rtol=0, atol=0)
    torch.testing.assert_close(on["loss_mask"], off["loss_mask"], rtol=0, atol=0)
    extra = on["critic_loss_mask"].bool() & ~on["loss_mask"].bool()
    assert extra.nonzero()[:, 1].tolist() == U_POINTS
    torch.testing.assert_close(
        on["returns"][:, ACTIVE], off["returns"][:, ACTIVE], rtol=0, atol=0
    )
    torch.testing.assert_close(
        on["returns"][:, U_POINTS], torch.full((1, 3), 2.0), rtol=0, atol=0
    )
    torch.testing.assert_close(on["values"], batch()["values"], rtol=0, atol=0)


@pytest.mark.parametrize("beta", [0.0, 0.5])
@pytest.mark.parametrize("lam", [1.0, ADAPTIVE])
def test_user_split_matches_hand_residual_polynomial(beta, lam):
    """Expand eight residuals by hand; the tool residual remains 0.125."""
    result = actor(lam)._compute_advantages(batch(enabled=True, beta=beta))
    decay = 1.0 if lam == 1.0 else float(torch.tensor(1 - 1 / (1.5 * 8)))
    deltas = [
        0.125,
        0.125 + beta * 0.125,
        0.125,
        0.125,
        0.125,
        0.125 + beta * 0.125,
        0.125,
        0.125 + beta * 0.625,
    ]
    expected = [
        sum(decay ** (j - i) * deltas[j] for j in range(i, 8)) for i in range(8)
    ]
    torch.testing.assert_close(
        result["advantages"][0, ACTIVE], torch.tensor(expected), rtol=1e-6, atol=1e-7
    )
    reference = actor(lam)._compute_advantages(batch(enabled=True, beta=1.0))
    torch.testing.assert_close(result["returns"], reference["returns"], rtol=0, atol=0)


@pytest.mark.parametrize("clip", [None, 0.5])
def test_critic_loss_and_microbatch_weight_include_u_points(clip):
    """Real backward must train U but not tool endpoints, and normalize by 11."""
    data = actor(1.0)._compute_advantages(batch(enabled=True))
    predicted = data["returns"].clone()
    predicted[0, U_POINTS] += 2.0
    predicted.requires_grad_()
    loss = ppo_loss_fn(predicted, data, eps_clip=clip)
    # Make clipping irrelevant while keeping the production clipped reducer.
    if clip is not None:
        data["values"] = predicted.detach().clone()
        loss = ppo_loss_fn(predicted, data, eps_clip=clip)
    torch.testing.assert_close(
        loss, torch.tensor((12 if clip is None else 6) / 11), rtol=1e-6, atol=1e-7
    )
    loss.backward()
    assert predicted.grad[0, U_POINTS].count_nonzero() == 3
    assert predicted.grad[0, 6] == 0
    engine = TinyTrainEngine(critic=True)
    observed = []
    original = engine.train_batch

    def train_batch(data, loss_fn, loss_weight_fn):
        observed.append(loss_weight_fn(data).item())
        return original(data, loss_fn, loss_weight_fn)

    engine.train_batch = train_batch
    PPOCritic(PPOCriticConfig(ppo_n_minibatches=1, eps_clip=clip), engine).ppo_update(
        [copy.deepcopy(data)]
    )
    assert observed == [11]


def test_batch_diagnostics_use_initial_u_values_and_boundary_denominators():
    """Population moments, including empty subsets, are logged only pre-fit."""
    stats_tracker.export_all()
    rows = [actor(1.0)._compute_advantages(batch(enabled=True)) for _ in range(2)]
    rows[1]["returns"].zero_()
    rows[1]["values"] *= 0.5
    critic = PPOCritic(
        PPOCriticConfig(ppo_n_minibatches=1), TinyTrainEngine(critic=True)
    )
    critic.ppo_update(copy.deepcopy(rows), critic_update_index=1)
    critic.ppo_update(copy.deepcopy(rows), critic_update_index=2)
    stats = stats_tracker.export_all()
    root = "ppo_critic/critic_update_1/user_boundary"
    assert stats[f"{root}/u_points"] == 6
    values = torch.cat([r["values"][:, U_POINTS].flatten() for r in rows])
    targets = torch.tensor([2.0] * 3 + [0.0] * 3)
    residuals = values - targets
    expected_ev = 1 - residuals.var(unbiased=False) / targets.var(unbiased=False)
    assert stats[f"{root}/critic_explained_variance_pre_update"] == pytest.approx(
        expected_ev.item()
    )
    gap = torch.tensor([0.125, 0.125, 0.625, 0.0625, 0.0625, -0.6875])
    assert stats[f"{root}/user_segment_mean"] == pytest.approx(gap.mean().item())
    assert stats[f"{root}/user_segment_std"] == pytest.approx(
        gap.std(unbiased=False).item()
    )
    assert stats[f"{root}/internal/u_points"] == 4
    assert stats[f"{root}/terminal/u_points"] == 2
    assert not any("critic_update_2/user_boundary" in k for k in stats)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_sao_two_critic_steps_keep_targets_and_use_refreshed_unmasked_u(beta):
    """Production critic-first orchestration preserves extra states across both passes."""
    from tests.test_sao_critic_updates import TrainRole

    from areal.trainer.ppo.update import update_critic_before_actor

    events = []
    policy, critic = TrainRole(events), TrainRole(events, critic=True)
    raw = [batch(enabled=True, beta=beta)]
    raw[0]["values"] = critic.compute_values(raw)[0]
    fixed = policy.compute_advantages(copy.deepcopy(raw))
    fixed_targets = fixed[0]["returns"].clone()
    report = update_critic_before_actor(policy, critic, raw, fixed, 2)
    assert critic.optimizer_steps == 2
    assert policy.optimizer_steps == 1
    assert report["actor_updated"]
    for consumed in critic.engine.consumed:
        torch.testing.assert_close(consumed, fixed_targets, rtol=0, atol=0)
    assert all(
        (grad[0, U_POINTS, 0].abs() > 0).all()
        for grad, _ in critic.engine.output_gradients
    )
    assert all(
        not grad[0, U_POINTS].any() for grad, _ in policy.engine.output_gradients
    )
    refreshed = copy.deepcopy(raw)
    refreshed[0]["values"] = critic.predictions[-1][0]
    oracle = policy.compute_advantages(refreshed)[0]
    torch.testing.assert_close(
        policy.advantages[0][0], oracle["advantages"], rtol=0, atol=0
    )
    if beta < 1:
        changed = copy.deepcopy(refreshed)
        changed[0]["values"][0, U_POINTS] += 1
        changed_adv = policy.compute_advantages(changed)[0]["advantages"]
        assert not torch.equal(changed_adv[:, ACTIVE], oracle["advantages"][:, ACTIVE])


def test_packed_loss_and_grad_norm_weight_preserve_critic_eligibility():
    """Packing and the diagnostics path must use the same eleven critic points."""
    from areal.utils.data import pack_tensor_dict

    data = actor(1.0)._compute_advantages(batch(enabled=True))
    packed = pack_tensor_dict(data)
    predicted = packed["returns"].clone()
    predicted[U_POINTS] += 2
    loss = ppo_loss_fn(predicted, packed, eps_clip=None)
    torch.testing.assert_close(loss, torch.tensor(12 / 11), rtol=1e-6, atol=1e-7)
    observed = []

    class GradEngine:
        def train(self):
            pass

        def grad_norm_batch(self, data, loss_fn, loss_weight_fn):
            observed.append(loss_weight_fn(data).item())
            return {}

    critic = PPOCritic(PPOCriticConfig(), GradEngine())
    critic.grad_norm([copy.deepcopy(data)])
    assert observed == [11]


def test_no_user_boundary_metrics_are_undefined_and_do_not_invent_points():
    """A tool-only or agent-stop ending still transports enabled knobs safely."""
    stats_tracker.export_all()
    data = batch(enabled=True, beta=0.0)
    data["user_boundary_types"][0, U_POINTS] = 4
    result = actor(1.0)._compute_advantages(data)
    torch.testing.assert_close(
        result["critic_loss_mask"], result["loss_mask"].bool(), rtol=0, atol=0
    )
    critic = PPOCritic(
        PPOCriticConfig(ppo_n_minibatches=1), TinyTrainEngine(critic=True)
    )
    critic.ppo_update([result])
    stats = stats_tracker.export_all()
    prefix = "ppo_critic/user_boundary"
    assert stats[f"{prefix}/u_points"] == 0
    assert stats[f"{prefix}/user_segment_mean"] is None
    assert stats[f"{prefix}/user_segment_std"] is None
    assert stats[f"{prefix}/critic_explained_variance_pre_update"] is None
    assert stats[f"{prefix}/critic_explained_variance_pre_update_defined"] == 0
