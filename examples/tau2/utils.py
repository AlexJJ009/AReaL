"""Utilities for Tau2 benchmark training with AReaL."""

import sys
from dataclasses import dataclass, field

import tau2.utils.llm_utils
import yaml
from litellm import completion_cost
from litellm.main import ModelResponse
from loguru import logger
from pydantic import BaseModel
from tau2.data_model.message import Message
from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import Task

from scripts.sao.async_eval import SaoPPOConfig

_POLICY_SESSION_KEY_ENV_NAMES = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AREAL_PROXY_SESSION_API_KEY",
    }
)


class FiniteEpochBatcher:
    """Restart the native finite iterator only after its epoch is fully drained."""

    def __init__(self, rollout, epoch: int = 0):
        self.rollout = rollout
        self.prepare = rollout.prepare_batch
        self.epoch = epoch

    def __call__(self, *args, **kwargs):
        kwargs.update(finite_epoch=True, fail_on_rejection=True)
        batch = self.prepare(*args, **kwargs)
        if not batch:
            # A finite empty result means all submitted inputs/results have
            # drained. Preserve controller task IDs and staleness state.
            del self.rollout.data_generator
            self.epoch += 1
            dataloader = args[0] if args else kwargs["dataloader"]
            dataloader.sampler.set_epoch(self.epoch)
            batch = self.prepare(*args, **kwargs)
        if not batch:
            raise RuntimeError("Finite tau2 training dataset is empty")
        return batch


@dataclass
class Tau2EnvConfig:
    """Environment configuration for Tau2 benchmark."""

    domain: str = field(
        default="airline",
        metadata={
            "help": "The tau2 domain name, e.g., 'retail', 'airline', 'telecom'."
        },
    )
    max_steps: int = field(
        default=100, metadata={"help": "Maximum number of steps per episode."}
    )
    add_thinking_tool: bool = field(
        default=False, metadata={"help": "Whether to add a thinking tool."}
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether the policy chat template enables thinking."},
    )
    context_window_tokens: int = field(
        default=32768,
        metadata={"help": "Total rendered prompt plus completion token budget."},
    )
    max_completion_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens for one assistant response."},
    )
    solo_mode: bool = field(
        default=False, metadata={"help": "Whether to use solo mode."}
    )
    user_llm_base_url: str | None = field(
        default=None,
        metadata={"help": "The base URL of the user LLM."},
    )
    user_llm: str | None = field(
        default=None,
        metadata={"help": "The user LLM to use, default to the gpt-4.1 model."},
    )
    user_llm_api_key_env: str = field(
        default="DEEPSEEK_API_KEY",
        metadata={"help": "Environment variable containing the user LLM API key."},
    )
    user_llm_args: dict | None = field(
        default=None, metadata={"help": "The arguments for the user LLM."}
    )
    user_llm_num_retries: int = field(
        default=3,
        metadata={"help": "Maximum retries for one user-simulator request."},
    )
    turn_discount: float = field(
        default=1.0, metadata={"help": "Discount factor for turn-based learning."}
    )
    invalid_format_penalty: float = field(
        default=0.1, metadata={"help": "Penalty for invalid format in completions."}
    )

    def __post_init__(self) -> None:
        if self.domain not in ("airline", "retail", "telecom", "mixed"):
            raise ValueError(f"Unsupported τ² domain: {self.domain}")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.context_window_tokens != 32768:
            raise ValueError("The τ² SAO experiment requires a 32768-token context")
        if not 0 < self.max_completion_tokens <= 4096:
            raise ValueError("max_completion_tokens must be in [1, 4096]")
        if self.enable_thinking:
            raise ValueError("The τ² SAO experiment requires enable_thinking=false")
        if self.user_llm_api_key_env in _POLICY_SESSION_KEY_ENV_NAMES:
            raise ValueError(
                "user_llm_api_key_env must not alias a policy proxy session key"
            )
        if self.user_llm_num_retries < 0:
            raise ValueError("user_llm_num_retries must be non-negative")


@dataclass
class Tau2PPOConfig(SaoPPOConfig):
    """PPO configuration with Tau2-specific settings."""

    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    # OmegaConf structured configs cannot materialize ``Literal`` annotations
    # in the AReaL version used by the launcher, so validate the enum here.
    algorithm: str = "grpo"
    domains: list[str] = field(default_factory=lambda: ["airline"])
    train_batch_episodes: int | None = None
    effective_episodes: int | None = None
    domain_effective_episodes: dict[str, int] = field(default_factory=dict)
    dynamic_group_filter: bool = False
    episode_timeout_seconds: float = 600.0
    experiment_mode: str = "qualification"
    critic_dev_fraction: float = 0.2
    critic_tasks_per_domain: int | None = None
    critic_rollouts_per_task: int = 1
    infra_retries: int = 1
    task_limit: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.algorithm not in ("grpo", "sao", "critic"):
            raise ValueError(f"Unsupported τ² algorithm: {self.algorithm}")
        if self.episode_timeout_seconds <= 0:
            raise ValueError("episode_timeout_seconds must be positive")
        if self.experiment_mode not in ("qualification", "tune", "formal"):
            raise ValueError("experiment_mode must be qualification, tune or formal")
        if self.infra_retries < 0 or self.critic_rollouts_per_task < 1:
            raise ValueError("infra_retries must be non-negative and rollouts positive")


# Configure loguru logger for tau2-bench package
# This runs at import time, so workers will also have this configuration
logger.remove()
# Log to stderr by default, will be captured by the worker's log system
logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")


def _get_response_cost_silent(response: ModelResponse) -> float:
    """Get cost from response, silently returning 0.0 for unmapped models.

    This is a patched version of tau2.utils.llm_utils.get_response_cost that
    suppresses the error log when LiteLLM doesn't have pricing info for a model
    (e.g., self-hosted models like 'openai/self-hosted-Qwen2.5-72B').

    The original function logs an error via logger.error(e) which is noisy.
    This version silently returns 0.0 for unmapped models.
    """
    # Parse fine-tuned model names (reuse tau2's helper)
    response.model = tau2.utils.llm_utils._parse_ft_model_name(response.model)
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        # Silently return 0.0 for unmapped models (e.g., self-hosted models)
        return 0.0
    return cost


# Patch tau2.utils.llm_utils.get_response_cost with our silent version
tau2.utils.llm_utils.get_response_cost = _get_response_cost_silent


class Tau2RunInfo(BaseModel):
    """Information about a Tau2 simulation run."""

    reward: float
    agent_time: list[float]
    user_time: list[float]
    messages: list[Message]
    task: Task
    reward_info: RewardInfo | None = None
    error: str | None = None
    error_type: str | None = None
    terminated: bool = True
    truncated: bool = False
    stop_reason: str | None = None

    def __str__(self):
        s = f"[REWARD]: {self.reward}\n\n"
        s += "[TASK]\n"
        s += yaml.dump(self.task.model_dump()) + "\n"
        if self.reward_info:
            s += "[REWARD_INFO]\n"
            s += yaml.dump(self.reward_info.model_dump()) + "\n"
        s += f"[TURNS COUNT]: {len(self.messages)}\n"
        s += "[MESSAGES]\n"
        for message in self.messages:
            turn_idx = message.turn_idx
            role = message.role
            content = message.content or ""
            usage = getattr(message, "usage", {})
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                content += "\n[TOOL_CALLS]\n"
                content += yaml.dump(
                    [tool_call.model_dump() for tool_call in tool_calls]
                )
            s += f"[{turn_idx}][{role}]: {content}\n"
            if usage:
                s += f"[{turn_idx}][{role}][USAGE]: {yaml.dump(usage)}\n"
        if len(self.agent_time):
            s += f"[AGENT_TIME]: total {sum(self.agent_time)}, avg {sum(self.agent_time) / len(self.agent_time)}\n"
        if len(self.user_time):
            s += f"[USER_TIME]: total {sum(self.user_time)}, avg {sum(self.user_time) / len(self.user_time)}\n"
        if self.error:
            s += f"[ERROR]: {self.error}\n"
        return s
