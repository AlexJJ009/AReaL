"""M3 action-subsequence oracle, explicit episode boundaries and real losses."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.test_sao_adaptive_lambda import LAMBDA_PATH
from tests.test_sao_dual_lambda import TinyTrainEngine, make_actor

from areal.api import ModelResponse
from areal.api.cli_args import GenerationHyperparameters, PPOCriticConfig
from areal.trainer.ppo.critic import PPOCritic
from areal.workflow.rlvr import RLVRWorkflow


def tool_rollout(gap=2, padding=3, *, truncated=False, pseudo_value=100.0):
    # Values are prefix-state aligned: the last observation token's output is
    # the value BEFORE the next action, so it is a valid state, not a fake value.
    tokens = [1, 2, 3] + [9] * gap + [4, 5]
    width = len(tokens) + padding
    raw_mask = torch.zeros(1, width)
    action_indices = [1, 2, 3 + gap, 4 + gap]
    raw_mask[0, action_indices] = 1
    states = [x - 1 for x in action_indices]
    values = torch.full((1, width), pseudo_value)
    values[0, states] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    values[0, len(tokens) - 1] = 2.0
    rewards = torch.zeros(1, width)
    rewards[0, action_indices] = torch.tensor([0.0, 0.5, 0.0, 1.0])
    return {
        "input_ids": torch.tensor([tokens + [0] * padding]),
        "attention_mask": torch.tensor([[1] * len(tokens) + [0] * padding]),
        "loss_mask": raw_mask,
        "logprobs": torch.full((1, width), -2.77),
        "values": values,
        "rewards": torch.tensor([0.0]),
        "token_rewards": rewards,
        "terminated": torch.tensor([not truncated]),
        "truncated": torch.tensor([truncated]),
        "episode_ids": torch.full((1, width), 12, dtype=torch.int64),
    }


def action_oracle(lam, bootstrap=0.0):
    # A separate four-action episode: observations do not appear in the oracle.
    rewards, values = [0, 0.5, 0, 1], [0.1, 0.2, 0.3, 0.4]
    output, tail = [], 0.0
    for index in reversed(range(4)):
        next_value = values[index + 1] if index < 3 else bootstrap
        tail = rewards[index] + 0.9 * next_value - values[index] + 0.9 * lam * tail
        output.insert(0, tail)
    return torch.tensor(output)


@pytest.mark.parametrize("gap,padding,pseudo", [(0, 0, 100), (2, 3, -999), (7, 0, 999)])
@pytest.mark.parametrize("truncated", [False, True])
def test_skip_observation_matches_action_oracle(gap, padding, pseudo, truncated):
    actor = make_actor(LAMBDA_PATH)
    actor.discount = 0.9  # Test-only discount exposes accidental observation steps.
    data = actor._compute_advantages(
        tool_rollout(gap, padding, truncated=truncated, pseudo_value=float(pseudo))
    )
    active = data["loss_mask"].bool()
    torch.testing.assert_close(
        data["advantages"][active],
        action_oracle(5 / 6, 2.0 if truncated else 0),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        data["returns"][active],
        action_oracle(1.0, 2.0 if truncated else 0)
        + torch.tensor([0.1, 0.2, 0.3, 0.4]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_episode_batch_isolation_and_public_training_consumers():
    actor = make_actor(LAMBDA_PATH)
    actor.discount = 0.9
    rows = [tool_rollout(gap=0), tool_rollout(gap=4, truncated=True)]
    rows[1]["episode_ids"].fill_(42)
    # Public concat/unpack preserves distinct bootstrap and boundaries.
    batch = actor.compute_advantages(copy.deepcopy(rows))
    for original, row in zip(rows, batch):
        single = actor._compute_advantages(copy.deepcopy(original))
        active = row["loss_mask"].bool()
        torch.testing.assert_close(
            row["advantages"][active], single["advantages"][active], rtol=0, atol=0
        )
    critic_engine = TinyTrainEngine(critic=True)
    critic = PPOCritic(
        PPOCriticConfig(ppo_n_minibatches=1, eps_clip=1000), critic_engine
    )
    actor.ppo_update(copy.deepcopy(batch))
    critic.ppo_update(copy.deepcopy(batch))
    for engine in (actor.engine, critic_engine):
        assert len(engine.losses) == 1
        grad, mask = engine.output_gradients[0]
        assert torch.count_nonzero(grad[~mask]) == 0
        assert torch.count_nonzero(grad[mask]) > 0


def test_observation_reward_and_packed_episode_are_rejected():
    actor = make_actor(LAMBDA_PATH)
    bad_reward = tool_rollout()
    bad_reward["token_rewards"][0, 3] = 1.0
    with pytest.raises(RuntimeError, match="Map observation rewards"):
        actor._compute_advantages(bad_reward)
    packed = tool_rollout()
    packed["episode_ids"][0, 5:] = 43
    with pytest.raises(RuntimeError, match="Separate packed episodes"):
        actor._compute_advantages(packed)
    ambiguous = tool_rollout()
    ambiguous["truncated"][:] = True
    with pytest.raises(RuntimeError, match="XOR"):
        actor._compute_advantages(ambiguous)


def test_wrong_observation_time_and_padding_bootstrap_are_detected():
    actor = make_actor(LAMBDA_PATH)
    actor.discount = 0.9
    row = tool_rollout(truncated=True)
    good = actor._compute_advantages(copy.deepcopy(row))
    # Mutated mask incorrectly treats observations as actions / time steps.
    row["loss_mask"][0, 3:5] = 1
    wrong = actor._compute_advantages(row)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            good["advantages"][good["loss_mask"].bool()],
            wrong["advantages"][good["loss_mask"].bool()],
            rtol=1e-6,
            atol=1e-7,
        )

    # A padded terminal heuristic would zero the true truncation bootstrap.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            good["advantages"][good["loss_mask"].bool()],
            action_oracle(5 / 6, 0),
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,budget_terminal,terminal",
    [("stop", False, True), ("length", True, True), ("length", False, False)],
)
async def test_workflow_stop_metadata_reaches_gae(reason, budget_terminal, terminal):
    """Synthetic inference, real workflow packing and production GAE consumer."""
    tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=5, decode=lambda ids: "math"
    )
    workflow = RLVRWorkflow(
        reward_fn=lambda *args, **kwargs: 1.0,
        gconfig=GenerationHyperparameters(n_samples=1),
        tokenizer=tokenizer,
        get_input_ids_fn=lambda *args: [1, 2],
        length_stop_is_terminal=budget_terminal,
    )

    async def reward(*args, **kwargs):
        return 1.0

    workflow.async_reward_fn = reward

    class Inference:
        async def agenerate(self, request):
            assert request.gconfig.n_samples == 1
            return ModelResponse(
                input_tokens=[1, 2],
                output_tokens=[3, 5],
                output_logprobs=[-2.0, -3.0],
                output_versions=[7, 8],
                stop_reason=reason,
            )

    row = await workflow.arun_episode(Inference(), {"messages": []})
    assert row["terminated"].item() == terminal
    assert row["truncated"].item() != terminal
    assert row["versions"].tolist() == [[-1, -1, 7, 8]]
    row["values"] = torch.tensor([[0.0, 0.1, 0.2, 2.0]])
    data = make_actor(LAMBDA_PATH)._compute_advantages(row)
    torch.testing.assert_close(
        data["returns"][data["loss_mask"].bool()],
        torch.full((2,), 1.0 if terminal else 3.0),
        rtol=1e-6,
        atol=1e-7,
    )
