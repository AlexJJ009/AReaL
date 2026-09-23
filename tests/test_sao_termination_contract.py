# SPDX-License-Identifier: Apache-2.0
"""SAO termination semantics for PPO bootstrap targets.

These tests protect the formal PPO contract that EOS-completed episodes are
terminal, while response-length capped episodes bootstrap from the true final
sequence value. The expectations are independent finite-horizon returns for the
gamma=lambda=1 case; they intentionally do not reimplement the runtime GAE loop.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from areal.api import ModelResponse
from areal.api.cli_args import GenerationHyperparameters, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.gae import _compute_token_level_gae
from areal.trainer.ppo.lambda_fn import resolve_gae_lambda_fn
from areal.utils.data import KLEstimator
from areal.workflow.rlvr import RLVRWorkflow

SAO_PPO_PATH = Path(__file__).resolve().parents[1] / "examples/math/sao_ppo.py"
SAO_PPO_SPEC = importlib.util.spec_from_file_location(
    "sao_ppo_under_test", SAO_PPO_PATH
)
assert SAO_PPO_SPEC is not None and SAO_PPO_SPEC.loader is not None
sao_ppo = importlib.util.module_from_spec(SAO_PPO_SPEC)
SAO_PPO_SPEC.loader.exec_module(sao_ppo)


def _active_returns_equal(
    returns: torch.Tensor,
    loss_mask: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        returns[loss_mask.bool()],
        expected,
        rtol=0.0,
        atol=1.0e-6,
    )


def _make_actor() -> PPOActor:
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


def _single_traj_batch(
    *,
    width: int,
    seq_len: int,
    value_last: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, torch.Tensor]:
    values = torch.zeros(1, width, dtype=torch.float32)
    values[0, seq_len - 1] = value_last
    loss_mask = torch.zeros(1, width, dtype=torch.float32)
    loss_mask[0, 2:seq_len] = 1.0
    return {
        "input_ids": torch.arange(width).unsqueeze(0),
        "attention_mask": torch.arange(width).unsqueeze(0) < seq_len,
        "loss_mask": loss_mask,
        "logprobs": torch.zeros(1, width, dtype=torch.float32),
        "values": values,
        "rewards": torch.tensor([1.0], dtype=torch.float32),
        "terminated": torch.tensor([terminated], dtype=torch.bool),
        "truncated": torch.tensor([truncated], dtype=torch.bool),
    }


def _legacy_padding_derived_returns(width: int, *, seq_len: int = 4) -> torch.Tensor:
    rewards = torch.zeros(1, width, dtype=torch.float32)
    values = torch.zeros(1, width, dtype=torch.float32)
    loss_mask = torch.zeros(1, width, dtype=torch.float32)
    rewards[0, seq_len - 2] = 1.0
    values[0, seq_len - 1] = 0.7
    loss_mask[0, 1 : seq_len - 1] = 1.0
    seq_no_eos_mask = torch.tensor([seq_len == width])

    _, returns = _compute_token_level_gae(
        rewards=rewards,
        values=values,
        loss_mask=loss_mask,
        seq_no_eos_mask=seq_no_eos_mask,
        discount=1.0,
        gae_lambda=1.0,
    )
    return returns[0, loss_mask[0].bool()]


def test_legacy_padding_width_rule_would_change_same_terminal_return():
    """Document the original bug without using it as the expected contract."""
    no_padding = _legacy_padding_derived_returns(width=4)
    with_one_pad = _legacy_padding_derived_returns(width=5)

    torch.testing.assert_close(
        no_padding, torch.tensor([1.7, 1.7]), rtol=0.0, atol=1.0e-6
    )
    torch.testing.assert_close(
        with_one_pad, torch.tensor([1.0, 1.0]), rtol=0.0, atol=1.0e-6
    )


def test_token_gae_uses_explicit_bootstrap_not_padded_last_position():
    rewards = torch.zeros(2, 6, dtype=torch.float32)
    values = torch.zeros_like(rewards)
    loss_mask = torch.zeros_like(rewards)

    rewards[0, 2] = 1.0
    loss_mask[0, 1:3] = 1.0
    values[0, 3] = 0.7

    rewards[1, 3] = 1.0
    loss_mask[1, 2:4] = 1.0
    values[1, 4] = 0.7

    _, returns = _compute_token_level_gae(
        rewards=rewards,
        values=values,
        loss_mask=loss_mask,
        seq_no_eos_mask=torch.tensor([False, True]),
        discount=1.0,
        gae_lambda=1.0,
        bootstrap_values=torch.tensor([0.0, 0.7]),
    )

    _active_returns_equal(
        returns[0],
        loss_mask[0],
        torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    _active_returns_equal(
        returns[1],
        loss_mask[1],
        torch.tensor([1.7, 1.7], dtype=torch.float32),
    )


def test_actor_returns_are_invariant_to_cobatch_padding_for_eos_terminal():
    actor = _make_actor()
    single = _single_traj_batch(
        width=4,
        seq_len=4,
        value_last=0.7,
        terminated=True,
        truncated=False,
    )
    cobatch_first = _single_traj_batch(
        width=6,
        seq_len=4,
        value_last=0.7,
        terminated=True,
        truncated=False,
    )
    cobatch_second = _single_traj_batch(
        width=6,
        seq_len=5,
        value_last=0.7,
        terminated=False,
        truncated=True,
    )
    cobatch = {
        key: torch.cat([cobatch_first[key], cobatch_second[key]], dim=0)
        for key in cobatch_first
    }

    single_result = actor._compute_advantages(single)
    cobatch_result = actor._compute_advantages(cobatch)

    single_active = single_result["loss_mask"][0].bool()
    cobatch_active = cobatch_result["loss_mask"][0].bool()
    torch.testing.assert_close(
        single_result["returns"][0, single_active],
        cobatch_result["returns"][0, cobatch_active],
        rtol=0.0,
        atol=1.0e-6,
    )
    torch.testing.assert_close(
        single_result["returns"][0, single_active],
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_actor_bootstraps_true_length_capped_response_even_when_not_longest():
    actor = _make_actor()
    truncated = _single_traj_batch(
        width=7,
        seq_len=5,
        value_last=0.7,
        terminated=False,
        truncated=True,
    )
    eos_longest = _single_traj_batch(
        width=7,
        seq_len=7,
        value_last=9.0,
        terminated=True,
        truncated=False,
    )
    eos_longest["rewards"] = torch.tensor([2.0], dtype=torch.float32)
    batch = {
        key: torch.cat([truncated[key], eos_longest[key]], dim=0) for key in truncated
    }

    result = actor._compute_advantages(batch)

    _active_returns_equal(
        result["returns"][0],
        result["loss_mask"][0],
        torch.tensor([1.7, 1.7, 1.7], dtype=torch.float32),
    )
    _active_returns_equal(
        result["returns"][1],
        result["loss_mask"][1],
        torch.full((5,), 2.0, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda batch: batch.pop("truncated"),
        lambda batch: batch.__setitem__(
            "terminated", torch.tensor([True, True], dtype=torch.bool)
        ),
        lambda batch: batch.__setitem__(
            "terminated", torch.tensor([True], dtype=torch.bool).view(1, 1)
        ),
        lambda batch: batch.__setitem__(
            "truncated", torch.tensor([True], dtype=torch.bool)
        ),
    ],
)
def test_actor_rejects_missing_or_invalid_termination_metadata_pair(mutate):
    actor = _make_actor()
    batch = _single_traj_batch(
        width=4,
        seq_len=4,
        value_last=0.0,
        terminated=True,
        truncated=False,
    )
    mutate(batch)

    with pytest.raises(
        (ValueError, RuntimeError), match="terminated|truncated|episode"
    ):
        actor._compute_advantages(batch)


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _Engine:
    def __init__(self, stop_reason: str):
        self.stop_reason = stop_reason

    async def agenerate(self, req):
        output_tokens = [5]
        if self.stop_reason == "stop":
            output_tokens.append(req.tokenizer.eos_token_id)
        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=output_tokens,
            output_logprobs=[-0.1] * len(output_tokens),
            output_versions=[0] * len(output_tokens),
            stop_reason=self.stop_reason,
            tokenizer=req.tokenizer,
        )


def _reward_fn(*args, **kwargs) -> float:
    return 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected_terminated", "expected_truncated"),
    [("stop", True, False), ("length", False, True)],
)
async def test_rlvr_include_termination_emits_stop_reason_metadata(
    stop_reason,
    expected_terminated,
    expected_truncated,
):
    gconfig = GenerationHyperparameters(max_new_tokens=8, greedy=False)
    workflow = RLVRWorkflow(
        reward_fn=_reward_fn,
        gconfig=gconfig,
        tokenizer=_Tokenizer(),
        enable_thinking=False,
        include_termination=True,
        get_input_ids_fn=lambda data, tokenizer, enable_thinking: [1, 2],
        data_extract_prompt_fn=lambda data: data,
    )

    result = await workflow.arun_episode(_Engine(stop_reason), {"messages": []})

    assert result["terminated"].shape == (1,)
    assert result["truncated"].shape == (1,)
    assert result["terminated"].dtype == torch.bool
    assert result["truncated"].dtype == torch.bool
    assert bool(result["terminated"][0]) is expected_terminated
    assert bool(result["truncated"][0]) is expected_truncated


def test_rlvr_include_termination_defaults_to_backward_compatible_false():
    gconfig = GenerationHyperparameters(max_new_tokens=8, greedy=False)
    workflow = RLVRWorkflow(
        reward_fn=_reward_fn,
        gconfig=gconfig,
        tokenizer=_Tokenizer(),
        enable_thinking=False,
        get_input_ids_fn=lambda data, tokenizer, enable_thinking: [1, 2],
        data_extract_prompt_fn=lambda data: data,
    )

    assert workflow.include_termination is False


@pytest.mark.parametrize("shape", [(), (1,), (2, 1), (3,)])
def test_explicit_bootstrap_rejects_broadcastable_wrong_shape(shape):
    values = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="shape"):
        _compute_token_level_gae(
            values,
            values,
            values,
            torch.zeros(2, dtype=torch.bool),
            1.0,
            1.0,
            bootstrap_values=torch.zeros(shape),
        )


@pytest.mark.parametrize(
    "field,value",
    [("discount", 0.99), ("gae_lambda", 1.0), ("gae_timestep_unit", "turn")],
)
def test_entry_rejects_overrides_inconsistent_with_return_oracle(field, value):
    from areal.api.cli_args import PPOConfig

    config = PPOConfig()
    setattr(config.actor, field, value)
    with pytest.raises(ValueError, match=f"actor.{field}"):
        sao_ppo.validate_contract(config, preflight=True)
