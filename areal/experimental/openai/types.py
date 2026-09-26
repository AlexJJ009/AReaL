# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam

from areal.api import ModelResponse

CONTEXT_LENGTH_EXCEEDED_MARKER = "areal_context_limit"


class ContextLengthExceededError(ValueError):
    """Raised before generation when the prompt leaves no room for output."""

    code = "context_length_exceeded"
    marker = CONTEXT_LENGTH_EXCEEDED_MARKER

    def __init__(self, *, prompt_len: int, limit_name: str, limit: int):
        self.prompt_len = prompt_len
        self.limit_name = limit_name
        self.limit = limit
        super().__init__(
            f"{self.marker}: prompt_tokens={prompt_len} exceeds "
            f"{limit_name}={limit}; max_new_tokens<=0 before generation"
        )


@dataclass(frozen=True)
class AgentWorkflowResult:
    """Reward plus explicit episode semantics returned by an agent workflow."""

    reward: float | dict[str, float]
    terminated: bool
    truncated: bool
    bootstrap_mask: bool = False
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.terminated == self.truncated:
            raise ValueError("AgentWorkflowResult requires terminated XOR truncated")
        if self.bootstrap_mask and self.terminated:
            raise ValueError("bootstrap_mask may only be true for truncated episodes")


class ApiType(str, Enum):
    """API type for interaction."""

    COMPLETION = "completion"
    RESPONSE = "response"
    NONE = "none"


class InputName(str, Enum):
    """Input name used for logging."""

    MESSAGES = "messages"
    INPUT_DATA = "input_data"
    NONE = "none"


@dataclass
class InteractionWithTokenLogpReward:
    """Internal structure to store completions/responses with their rewards."""

    # Common
    model_response: ModelResponse | None = None
    reward: float | None = None
    original_reward: float | None = None
    parent: InteractionWithTokenLogpReward | None = None
    chat_template_type: str = "hf"
    _cache: dict[str, torch.Tensor] | None = None

    # Fields used for parent-child relationship resolving
    messages: list[dict] = field(default_factory=list)
    output_message_list: list[dict] | None = None

    # Completion fields (optional for response)
    completion: ChatCompletion | None = None

    # Response fields (optional for completion)
    response: Response | None = None
    input_data: str | ResponseInputParam = field(default_factory=lambda: "")

    # Interaction ID cache (used for deserialization)
    _interaction_id: str | None = None

    # Explicit episode metadata.  These fields are populated by the proxy
    # workflow after the environment finishes, then materialized in the tensor
    # payload consumed by PPO/SAO.
    terminated: bool | None = None
    truncated: bool | None = None
    bootstrap_mask: bool | None = None
    episode_id: int | None = None
    episode_stop_reason: str | None = None
    episode_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tensor_data(self) -> bool:
        return self.model_response is not None or self._cache is not None

    @property
    def is_completion(self) -> bool:
        return self.completion is not None

    @property
    def is_response(self) -> bool:
        return self.response is not None

    @property
    def api_type(self) -> ApiType:
        """API type (completion/response)."""
        if self.is_completion:
            return ApiType.COMPLETION
        elif self.is_response:
            return ApiType.RESPONSE
        else:
            return ApiType.NONE

    @property
    def input_name_for_logging(self) -> InputName:
        """Input name used for logging."""
        if self.is_completion:
            return InputName.MESSAGES
        elif self.is_response:
            return InputName.INPUT_DATA
        else:
            return InputName.NONE

    @property
    def current_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.is_completion:
            return self.messages
        elif self.is_response:
            return self.input_data
        else:
            return None

    @property
    def parent_data(self) -> list[dict] | str | ResponseInputParam | None:
        if self.parent is None:
            return None
        return self.parent.current_data

    @property
    def interaction_id(self) -> str | None:
        if self.is_completion:
            return self.completion.id
        elif self.is_response:
            return self.response.id
        elif self._interaction_id is not None:
            return self._interaction_id
        else:
            return None

    @interaction_id.setter
    def interaction_id(self, value):
        if self.is_completion or self.is_response:
            raise ValueError("Cannot set ID for completion or responses")
        self._interaction_id = value

    @property
    def created_at(self) -> float | None:
        if self.is_completion:
            return float(self.completion.created)
        elif self.is_response:
            return float(self.response.created_at)
        else:
            return None

    @property
    def remaining_messages(self) -> list[dict]:
        if self.parent is None:
            return self.messages
        assert self.parent.output_message_list is not None, (
            "Parent output message is not set."
        )
        parent_len = len(self.parent.messages + self.parent.output_message_list)
        return self.messages[parent_len:]

    def to_tensor_dict(self) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache
        resp = self.model_response
        assert resp is not None, "Model response is not set."
        self.seq_tokens = seq = resp.input_tokens + resp.output_tokens
        if self.chat_template_type == "concat" and self.parent is not None:
            parent_res = self.parent.to_tensor_dict()
            parent_logprobs = parent_res["logprobs"].squeeze(0).tolist()
            parent_loss_mask = parent_res["loss_mask"].squeeze(0).tolist()
            parent_action_origin_mask = (
                parent_res["action_origin_mask"].squeeze(0).tolist()
            )
            parent_versions = parent_res["versions"].squeeze(0).tolist()
            parent_turn_ids = parent_res["turn_ids"].squeeze(0).tolist()
            parent_len = len(parent_logprobs)
            assert (
                parent_len
                == len(parent_loss_mask)
                == len(parent_action_origin_mask)
                == len(parent_versions)
                == len(parent_turn_ids)
            )
            valid_parent_turn_ids = [tid for tid in parent_turn_ids if tid >= 0]
            own_turn_id = max(valid_parent_turn_ids) + 1 if valid_parent_turn_ids else 0
            if resp.input_len > parent_len:
                parent_tokens = parent_res["input_ids"].squeeze(0).tolist()
                if resp.input_tokens[:parent_len] != parent_tokens:
                    raise ValueError(
                        "Concat child input tokens do not preserve the complete "
                        "parent trajectory; refusing to fabricate a training prefix"
                    )
                logprobs = (
                    parent_logprobs
                    + [0.0] * (resp.input_len - parent_len)
                    + resp.output_logprobs
                )
                loss_mask = (
                    parent_loss_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                action_origin_mask = (
                    parent_action_origin_mask
                    + [0] * (resp.input_len - parent_len)
                    + [1] * resp.output_len
                )
                versions = (
                    parent_versions
                    + [-1] * (resp.input_len - parent_len)
                    + resp.output_versions
                )
                turn_ids = (
                    parent_turn_ids
                    + [-1] * (resp.input_len - parent_len)
                    + [own_turn_id] * resp.output_len
                )
            else:
                api_type = self.api_type
                input_name = self.input_name_for_logging
                raise ValueError(
                    f"Concat child {api_type} input length {resp.input_len} is not "
                    f"greater than parent trajectory length {parent_len}; refusing "
                    f"to ignore parent {api_type}. Parent {input_name} and child "
                    f"{input_name} must form a strict trajectory prefix."
                )
        else:
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            action_origin_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            turn_ids = [-1] * resp.input_len + [0] * resp.output_len
        reward = self.reward if self.reward is not None else 0.0
        original_reward = (
            self.original_reward if self.original_reward is not None else reward
        )
        result = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            # Derived from generation boundaries, independently of the trainer
            # loss mask.  Offline sealing compares the two so a context/tool
            # token cannot be relabeled as an action by deriving both views
            # from the same potentially-corrupt mask.
            action_origin_mask=torch.tensor(
                action_origin_mask, dtype=torch.bool
            ).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            turn_ids=torch.tensor(turn_ids, dtype=torch.int32).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            # reward
            rewards=torch.tensor([float(reward)]),
            original_rewards=torch.tensor([float(original_reward)]),
        )
        self._add_episode_tensors(result)
        self._cache = result
        return result

    def _add_episode_tensors(self, result: dict[str, torch.Tensor]) -> None:
        metadata = (self.terminated, self.truncated, self.bootstrap_mask)
        if all(value is None for value in metadata):
            return
        if any(value is None for value in metadata):
            raise ValueError("Episode termination metadata must be complete")
        assert self.terminated is not None
        assert self.truncated is not None
        assert self.bootstrap_mask is not None
        if self.terminated == self.truncated:
            raise ValueError("Episode metadata requires terminated XOR truncated")
        if self.bootstrap_mask and self.terminated:
            raise ValueError("bootstrap_mask may only be true for truncated episodes")
        result["terminated"] = torch.tensor([self.terminated], dtype=torch.bool)
        result["truncated"] = torch.tensor([self.truncated], dtype=torch.bool)
        result["bootstrap_mask"] = torch.tensor([self.bootstrap_mask], dtype=torch.bool)
        if self.episode_id is not None:
            width = int(result["input_ids"].shape[-1])
            result["episode_ids"] = torch.full(
                (1, width), self.episode_id, dtype=torch.int64
            )
        official_score = self.episode_metadata.get("official_score")
        if isinstance(official_score, int | float):
            result["official_scores"] = torch.tensor(
                [float(official_score)], dtype=torch.float32
            )
        result["task_budget_failure"] = torch.tensor(
            [self.episode_metadata.get("failure_class") == "task_budget"],
            dtype=torch.bool,
        )

    def apply_episode_result(
        self,
        result: AgentWorkflowResult,
        *,
        episode_id: int,
    ) -> None:
        """Attach environment outcome metadata to a generated trajectory."""

        self.terminated = result.terminated
        self.truncated = result.truncated
        self.bootstrap_mask = result.bootstrap_mask
        self.episode_id = episode_id
        self.episode_stop_reason = result.stop_reason
        self.episode_metadata = dict(result.metadata)
        if self._cache is not None:
            self._add_episode_tensors(self._cache)


def normalize_group_rewards(
    results: list[dict[str, InteractionWithTokenLogpReward] | None],
) -> bool:
    """Normalize one scalar reward per rollout while preserving raw rewards."""
    if not results:
        return False

    reward_per_result: list[float | None] = []
    for result in results:
        if not result:
            reward_per_result.append(None)
            continue
        last_id = next(reversed(result))
        reward_per_result.append(result[last_id].reward)

    if any(reward is None for reward in reward_per_result):
        return False

    rewards = torch.tensor(reward_per_result, dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std(unbiased=False) if rewards.numel() > 1 else torch.tensor(1.0)
    normalized_rewards = ((rewards - mean) / (std + 1e-8)).tolist()

    for result, normalized_reward in zip(results, normalized_rewards):
        assert result is not None
        for interaction in result.values():
            if interaction.reward is None:
                continue
            interaction.original_reward = interaction.reward
            interaction.reward = normalized_reward
            if interaction._cache is not None:
                interaction._cache["rewards"] = torch.tensor([float(normalized_reward)])
                interaction._cache["original_rewards"] = torch.tensor(
                    [float(interaction.original_reward)]
                )
    return True


def concat_string_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict[str, list[dict]]:
    """Concat interactions that lack tensor data (e.g. external API mode).

    Returns a dict with an ``"interactions"`` key containing a list of
    ``{"request": ..., "response": ..., "reward": ...}`` dicts, one per
    interaction.  This is the counterpart of
    :func:`~areal.utils.data.concat_padded_tensors` for string-only
    trajectories.
    """
    return {
        "interactions": [
            {
                "request": v.messages,
                "response": (
                    v.output_message_list[0]["content"] if v.output_message_list else ""
                ),
                "reward": v.reward,
            }
            for v in interactions.values()
        ]
    }
