# SPDX-License-Identifier: Apache-2.0

"""Replay sealed τ² episodes without re-running tools or the user simulator."""

from __future__ import annotations

from typing import Any

import torch

from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.types import InteractionWithTokenLogpReward


def _one_dimensional(
    data: dict[str, Any], key: str, dtype: torch.dtype
) -> torch.Tensor:
    value = torch.as_tensor(data[key], dtype=dtype)
    if value.ndim != 1:
        raise ValueError(f"Sealed critic row {key} must be one-dimensional")
    return value


class Tau2CriticReplayWorkflow(RolloutWorkflow):
    """Return a pre-tokenized episode as the normal PPO rollout tensor contract."""

    async def arun_episode(
        self, engine: Any, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward]:
        _ = engine
        input_ids = _one_dimensional(data, "input_ids", torch.long)
        attention_mask = _one_dimensional(data, "attention_mask", torch.bool)
        loss_mask = _one_dimensional(data, "loss_mask", torch.bool)
        action_origin_mask = _one_dimensional(data, "action_origin_mask", torch.bool)
        behavior_logprobs = _one_dimensional(data, "behavior_logprobs", torch.float32)
        versions = _one_dimensional(data, "versions", torch.long)
        turn_ids = _one_dimensional(data, "turn_ids", torch.int32)
        width = input_ids.numel()
        aligned = {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "action_origin_mask": action_origin_mask,
            "behavior_logprobs": behavior_logprobs,
            "versions": versions,
            "turn_ids": turn_ids,
        }
        mismatched = {
            name: int(value.numel())
            for name, value in aligned.items()
            if value.numel() != width
        }
        if mismatched:
            raise ValueError(
                f"Sealed critic row tensors are not token aligned: width={width}, "
                f"mismatched={mismatched}"
            )
        if not torch.all(torch.isfinite(behavior_logprobs)):
            raise ValueError("Sealed critic behavior logprobs must be finite")
        if not torch.any(loss_mask):
            raise ValueError("Sealed critic episode has no action tokens")
        if not torch.equal(loss_mask, action_origin_mask):
            raise ValueError(
                "Sealed critic loss_mask must match independent action provenance"
            )
        if torch.any(loss_mask & (versions < 0)):
            raise ValueError("Every sealed action token requires a behavior version")
        if torch.any(loss_mask & (turn_ids < 0)):
            raise ValueError("Every sealed action token requires a turn ID")

        terminated = bool(data["terminated"])
        truncated = bool(data["truncated"])
        if terminated == truncated:
            raise ValueError("Sealed critic episode requires terminated XOR truncated")
        bootstrap_mask = bool(data.get("bootstrap_mask", False))
        if bootstrap_mask and terminated:
            raise ValueError("bootstrap_mask may only be true for truncated episodes")
        reward = float(data["reward"])
        if reward not in (0.0, 1.0):
            raise ValueError("First τ² critic recipe requires binary outcome rewards")
        episode_id = int(data["episode_tensor_id"])
        cache = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "loss_mask": loss_mask.unsqueeze(0),
            "action_origin_mask": action_origin_mask.unsqueeze(0),
            "logprobs": behavior_logprobs.unsqueeze(0),
            "versions": versions.unsqueeze(0),
            "turn_ids": turn_ids.unsqueeze(0),
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "original_rewards": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor([terminated], dtype=torch.bool),
            "truncated": torch.tensor([truncated], dtype=torch.bool),
            "bootstrap_mask": torch.tensor([bootstrap_mask], dtype=torch.bool),
            "episode_ids": torch.full((1, width), episode_id, dtype=torch.int64),
        }
        interaction_id = str(data["episode_id"])
        return {
            interaction_id: InteractionWithTokenLogpReward(
                reward=reward,
                original_reward=reward,
                _interaction_id=interaction_id,
                _cache=cache,
            )
        }
