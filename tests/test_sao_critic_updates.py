"""M4: real optimizers and critic snapshots through the production orchestrator."""

import copy

import pytest
import torch

from tests.test_sao_adaptive_lambda import LAMBDA_PATH
from tests.test_sao_dual_lambda import TinyTrainEngine, make_actor
from tests.test_sao_skip_observation import tool_rollout

from areal.api.cli_args import PPOConfig, PPOCriticConfig
from areal.api.io_struct import FinetuneSpec
from areal.trainer.ppo.critic import PPOCritic
from areal.trainer.ppo.update import (
    critic_optimizer_spec,
    require_single_update,
    summarize_updates,
    update_critic_before_actor,
)


class TrainRole:
    """Local facade over actual actor/critic components and tiny models."""

    def __init__(self, events, *, critic=False):
        self.events = events
        self.name = "critic" if critic else "actor"
        if critic:
            self.engine = TinyTrainEngine(critic=True)
            self.component = PPOCritic(
                PPOCriticConfig(ppo_n_minibatches=1, eps_clip=1000), self.engine
            )
        else:
            self.component = make_actor(LAMBDA_PATH)
            self.engine = self.component.engine
        self.optimizer_steps = 0
        self.engine.optimizer.register_step_post_hook(self._stepped)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(
            self.engine.optimizer, factor=1.0, total_iters=100
        )
        self.advantages = []
        self.predictions = []

    def _stepped(self, optimizer, args, kwargs):
        self.optimizer_steps += 1
        self.events.append(f"{self.name}.optimizer.{self.optimizer_steps}")

    def ppo_update(self, data):
        if self.name == "actor":
            self.advantages.append([row["advantages"].clone() for row in data])
        return self.component.ppo_update(data)

    def step_lr_scheduler(self):
        self.events.append(f"{self.name}.scheduler")
        self.scheduler.step()

    def compute_values(self, data):
        self.events.append(f"critic.forward.{self.optimizer_steps}")
        values = self.component.compute_values(data)
        self.predictions.append([v.clone() for v in values])
        return values

    def compute_advantages(self, data):
        self.events.append("actor.advantages")
        return self.component.compute_advantages(data)


def setup_batch():
    from areal.utils import stats_tracker

    stats_tracker.export_all()
    events = []
    actor, critic = TrainRole(events), TrainRole(events, critic=True)
    raw = [tool_rollout(truncated=True)]
    for row, value in zip(raw, critic.compute_values(raw)):
        row["values"] = value
    fixed = actor.compute_advantages(copy.deepcopy(raw))
    return actor, critic, raw, fixed, events


def test_two_critic_updates_refresh_values_before_one_actor_update():
    """Post-step hooks count actual updates; old-value reuse fails the oracle."""
    actor, critic, raw, fixed, events = setup_batch()
    targets = fixed[0]["returns"].clone()
    result = update_critic_before_actor(actor, critic, raw, fixed, updates=2)
    assert events == [
        "critic.forward.0",
        "actor.advantages",
        "critic.optimizer.1",
        "critic.scheduler",
        "critic.optimizer.2",
        "critic.scheduler",
        "critic.forward.2",
        "actor.advantages",
        "actor.optimizer.1",
        "actor.scheduler",
    ]
    assert critic.optimizer_steps == 2 and actor.optimizer_steps == 1
    assert len(result["critic"]) == 2
    assert result["actor"]["effective"] == 1
    for consumed in critic.engine.consumed:
        torch.testing.assert_close(consumed, targets, rtol=0, atol=0)
    refreshed = copy.deepcopy(raw)
    refreshed[0]["values"] = critic.predictions[-1][0]
    oracle = actor.component.compute_advantages(refreshed)[0]["advantages"]
    torch.testing.assert_close(actor.advantages[0][0], oracle, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            actor.advantages[0][0], fixed[0]["advantages"], rtol=1e-6, atol=1e-7
        )
    # Original rollout and fixed target remain bound to the old snapshot.
    torch.testing.assert_close(
        raw[0]["values"], critic.predictions[0][0], rtol=0, atol=0
    )
    torch.testing.assert_close(fixed[0]["returns"], targets, rtol=0, atol=0)


def test_skipped_critic_update_blocks_actor_and_refresh(monkeypatch):
    actor, critic, raw, fixed, events = setup_batch()

    def skipped(*args, **kwargs):
        return {"attempted": 1.0, "successful": 0.0, "skipped": 1.0, "effective": 0.0}

    monkeypatch.setattr(critic, "ppo_update", skipped)
    with pytest.raises(RuntimeError, match="critic did not complete"):
        update_critic_before_actor(actor, critic, raw, fixed, 2)
    assert critic.optimizer_steps == 0 and actor.optimizer_steps == 0
    assert events == ["critic.forward.0", "actor.advantages"]


def test_minibatches_cannot_masquerade_as_one_critic_update():
    actor, critic, raw, fixed, events = setup_batch()
    critic.component.config.ppo_n_minibatches = 2
    # Two real minibatch optimizer steps must be rejected as a logical update.
    with pytest.raises(RuntimeError, match="critic did not complete"):
        update_critic_before_actor(actor, critic, raw * 2, fixed * 2, 2)
    assert critic.optimizer_steps == 2 and actor.optimizer_steps == 0


@pytest.mark.parametrize("updates", [-1, True, 1.5])
def test_invalid_critic_update_config_rejected(updates):
    with pytest.raises(ValueError, match="critic_updates_before_actor"):
        PPOConfig(critic_updates_before_actor=updates)


def test_critic_update_config_enforces_one_complete_batch_step():
    actor = make_actor(LAMBDA_PATH).config
    critic = PPOCriticConfig(ppo_n_minibatches=1, backend="fsdp:d1", is_critic=True)
    config = PPOConfig(actor=actor, critic=critic, critic_updates_before_actor=2)
    assert config.critic_updates_before_actor == 2
    critic.ppo_n_minibatches = 2
    with pytest.raises(ValueError, match="ppo_n_minibatches=1"):
        PPOConfig(actor=actor, critic=critic, critic_updates_before_actor=2)


def test_unsupported_backend_rejected_before_training():
    actor = make_actor(LAMBDA_PATH).config
    actor.backend = "megatron:d1"
    with pytest.raises(ValueError, match="FSDP only"):
        PPOConfig(
            actor=actor,
            critic=PPOCriticConfig(
                ppo_n_minibatches=1, backend="fsdp:d1", is_critic=True
            ),
            critic_updates_before_actor=2,
        )


def test_zero_gradient_is_visible_and_nonfinite_cannot_claim_success():
    report = summarize_updates(
        [{"update_successful": 1.0, "grad_norm": 0.0, "lr": 1e-6}]
    )
    require_single_update(report, "actor")
    assert report["successful"] == 1 and report["effective"] == 0
    nonfinite = summarize_updates(
        [{"update_successful": 1.0, "grad_norm": float("nan"), "lr": 1e-6}]
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        require_single_update(nonfinite, "critic")


def test_critic_scheduler_covers_same_training_batches_with_two_updates():
    from transformers import get_linear_schedule_with_warmup

    spec = FinetuneSpec(total_train_epochs=3, dataset_size=5, train_batch_size=2)
    critic_spec = critic_optimizer_spec(spec, 2)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 0, critic_spec.total_train_steps
    )
    # Half of the six actor batches = six critic steps, so half the LR remains.
    for _ in range(6):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == 0.5
    assert critic_optimizer_spec(spec, 0).total_train_steps == spec.total_train_steps
