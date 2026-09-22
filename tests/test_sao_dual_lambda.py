"""M1: independent targets through production GAE and optimizer consumers.

CPU float32, seed 17, tensor tolerances rtol=1e-6 / atol=1e-7.
The tiny engine supplies a trainable model; PPOActor/PPOCritic select the actual
production losses. This is component acceptance, not FSDP qualification.
"""

import copy

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import NormConfig, PPOActorConfig, PPOCriticConfig
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.critic import PPOCritic


class TinyTrainEngine:
    """Test transport for real forward/backward/SGD through supplied losses."""

    def __init__(self, *, critic: bool = False, device: str = "cpu"):
        torch.manual_seed(17)
        self.critic = critic
        self.model = torch.nn.Embedding(16, 1 if critic else 16).to(device)
        torch.nn.init.uniform_(self.model.weight, -0.1, 0.1)
        # Test-only LR makes deltas readily measurable, not the online SAO LR.
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.consumed = []
        self.losses = []
        self.output_gradients = []

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, input_, aggregate_fn):
        return aggregate_fn([self.model(input_["input_ids"])])

    def get_version(self):
        return 0

    def train_batch(self, data, loss_fn, loss_weight_fn):
        assert loss_weight_fn(data) > 0
        self.optimizer.zero_grad()
        output = self.model(data["input_ids"])
        output.retain_grad()
        if self.critic:
            loss = loss_fn(output, data)
        else:
            logp = output.log_softmax(-1)
            labels = data["input_ids"].roll(-1, -1)
            selected = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            loss = loss_fn(selected, -(logp.exp() * logp).sum(-1), data)
        self.consumed.append(data["returns"].detach().clone())
        assert not data["returns"].requires_grad
        assert not data["advantages"].requires_grad
        loss.backward()
        self.output_gradients.append(
            (output.grad.detach().clone(), data["loss_mask"].bool().clone())
        )
        assert torch.isfinite(self.model.weight.grad).all()
        assert self.model.weight.grad.abs().sum() > 0
        self.optimizer.step()
        self.losses.append(loss.detach().clone())
        return {
            "update_successful": 1.0,
            "grad_norm": float(self.model.weight.grad.norm()),
            "lr": self.optimizer.param_groups[0]["lr"],
        }


def rollout(device="cpu"):
    # Two terminal samples with 3 actions; final padding avoids the upstream
    # padding-derived truncation heuristic, which M1 does not repair.
    ids = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 8, 0]], device=device)
    return {
        "input_ids": ids,
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * 2, device=device),
        "loss_mask": torch.tensor([[0, 1, 1, 1, 0]] * 2, device=device),
        "logprobs": torch.full((2, 5), -2.77, device=device),
        "rewards": torch.tensor([1.0, 0.0], device=device),
        "values": torch.tensor(
            [[0.2, 0.4, 0.6, 7, 9], [0.3, 0.5, 0.7, 8, 10]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        ),
    }


def make_actor(
    lam, *, critic_lam=1.0, normalize=False, device="cpu", lambda_kwargs=None
):
    # Exercise structured config -> override -> dataclass -> production consumer.
    cfg = OmegaConf.structured(PPOActorConfig())
    cfg = OmegaConf.merge(
        cfg,
        {
            "experiment_name": "sao-m1-test",
            "trial_name": "dual-lambda",
            "backend": "fsdp:d1",
            "gae_lambda": lam,
            "gae_lambda_kwargs": lambda_kwargs or {},
            "critic_gae_lambda": critic_lam,
            "discount": 1.0,
            "kl_ctl": 0.0,
            "ppo_n_minibatches": 1,
            "recompute_logprob": False,
            "use_decoupled_loss": False,
        },
    )
    config = OmegaConf.to_object(cfg)
    config.adv_norm = NormConfig() if normalize else None
    return PPOActor(config, TinyTrainEngine(device=device))


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_dual_lambda_actor_change_preserves_critic_targets(lam, normalize):
    """Lambda-V=1 equals the episodic outcome, independently of actor lambda."""
    actor = make_actor(lam, normalize=normalize)
    data = actor._compute_advantages(rollout())
    expected = torch.tensor([[1.0] * 3, [0.0] * 3])
    torch.testing.assert_close(data["returns"][:, :3], expected, rtol=1e-6, atol=1e-7)
    assert not data["returns"].requires_grad
    assert not data["advantages"].requires_grad
    if not normalize:
        # Independently expanded three-step TD-residual polynomial.
        deltas = torch.tensor([[0.2, 0.2, 0.4], [0.2, 0.2, -0.7]])
        oracle = torch.stack(
            [
                deltas[:, 0] + lam * deltas[:, 1] + lam**2 * deltas[:, 2],
                deltas[:, 1] + lam * deltas[:, 2],
                deltas[:, 2],
            ],
            dim=1,
        )
        torch.testing.assert_close(
            data["advantages"][:, :3], oracle, rtol=1e-6, atol=1e-7
        )


def test_legacy_single_lambda_is_detected_by_critic_oracle():
    """Negative control: switching off separation must violate lambda-V=1."""
    broken = make_actor(0.0, critic_lam=None)._compute_advantages(rollout())
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            broken["returns"][:, :3],
            torch.tensor([[1.0] * 3, [0.0] * 3]),
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dual_lambda_production_updates_consume_detached_target(device):
    """Real PPO update dispatch changes weights and preserves critic targets."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    actors = [make_actor(lam, normalize=True, device=device) for lam in (0.0, 1.0)]
    critic_weights = []
    for actor in actors:
        data = actor._compute_advantages(rollout(device))
        # Old values belong to the rollout snapshot, not a live critic graph.
        data["values"] = data["values"].detach()
        critic_engine = TinyTrainEngine(critic=True, device=device)
        critic = PPOCritic(PPOCriticConfig(ppo_n_minibatches=1), critic_engine)
        actor_before = actor.engine.model.weight.detach().clone()
        critic_before = critic_engine.model.weight.detach().clone()
        actor.ppo_update([copy.deepcopy(data)])
        critic.ppo_update([copy.deepcopy(data)])
        assert not torch.equal(actor.engine.model.weight, actor_before)
        assert not torch.equal(critic_engine.model.weight, critic_before)
        torch.testing.assert_close(
            critic_engine.consumed[0][:, :3],
            torch.tensor([[1.0] * 3, [0.0] * 3], device=device),
            rtol=1e-6,
            atol=1e-7,
        )
        critic_weights.append(critic_engine.model.weight.detach().clone())
    torch.testing.assert_close(critic_weights[0], critic_weights[1], rtol=0, atol=0)
    assert not torch.equal(actors[0].engine.model.weight, actors[1].engine.model.weight)


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan"), float("inf"), True, "1"])
def test_invalid_critic_lambda_rejected(invalid):
    """Invalid weights cannot silently enter target estimation."""
    with pytest.raises(ValueError, match="critic_gae_lambda"):
        PPOActorConfig(critic_gae_lambda=invalid)
