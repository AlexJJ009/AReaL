"""Tau2 Agent Workflow for AReaL proxy mode.

This module implements a Tau2 agent that uses the AReaL proxy server
for OpenAI-compatible API calls during RL training.
"""

import asyncio
import os
from collections.abc import Awaitable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

import litellm
import tau2.evaluator.evaluator_nl_assertions as tau2_nl_evaluator
from litellm import register_model
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser
from tau2.user.user_simulator_base import HalfDuplexUser

# Import utilities (also patches tau2.utils.llm_utils)
from examples.tau2.contracts import bind_policy_request
from examples.tau2.user_simulator import RetryingUserSimulator
from examples.tau2.utils import Tau2EnvConfig, Tau2RunInfo

from areal.experimental.openai.types import (
    CONTEXT_LENGTH_EXCEEDED_MARKER,
    AgentWorkflowResult,
)
from areal.utils import logging

logger = logging.getLogger("Tau2Agent")
_T = TypeVar("_T")


class Tau2InfrastructureError(RuntimeError):
    """Unexpected environment/evaluator failure that must not become reward zero."""


def is_context_budget_error(exc: BaseException) -> bool:
    """Classify known policy context exhaustion without retrying as infrastructure."""

    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current).lower()
        if CONTEXT_LENGTH_EXCEEDED_MARKER in message:
            return True
        if (
            "requested token count exceeds" in message
            and "maximum context length" in message
            and "endpoint: /generate" in message
            and "max_new_tokens" in message
            and "return_logprob" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


async def _await_without_orphaning(awaitable: Awaitable[_T]) -> _T:
    """Delay cancellation until a worker-thread operation has actually stopped."""

    task = asyncio.create_task(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc

    if cancellation is not None:
        try:
            task.result()
        except Exception:
            logger.warning(
                "τ² episode failed while cancellation waited for worker cleanup",
                exc_info=True,
            )
        raise cancellation
    return task.result()


# Silence litellm verbose output (Provider List messages)
litellm.suppress_debug_info = True

# Register dummy model for litellm
register_model(
    {
        "dummy": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    }
)


def _get_task(domain: str, task_id: str, split: str | None = None) -> Task:
    """Get a task by ID from the tau2 registry."""
    tasks: list[Task] = registry.get_tasks_loader(domain)(split)
    for task in tasks:
        if task.id == task_id:
            return task
    raise ValueError(f"No task found with id {task_id} for domain {domain}")


def think(thoughts: str):
    """Use this tool to think. The thoughts will be visible in the history.
    Only use this tool to think when necessary.
    """
    return "Your thoughts are recorded. Please continue your work."


def _litellm_user_model(model: str) -> str:
    """Route a bare OpenAI-compatible model ID through LiteLLM's OpenAI adapter."""

    return model if "/" in model else f"openai/{model}"


class Tau2Runner:
    """Run the official synchronous τ² orchestrator on a worker thread."""

    def __init__(
        self,
        econfig: Tau2EnvConfig,
        gen_args: dict,
        agent_base_url: str,
        agent_api_key: str,
        user_api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.econfig = econfig
        self.gen_args = gen_args
        self.agent_base_url = agent_base_url
        self.agent_api_key = agent_api_key
        self.user_api_key = user_api_key
        self.timeout = timeout
        self.domain = econfig.domain
        self.solo_mode = econfig.solo_mode

    def _get_environment(self) -> Environment:
        environment_constructor = registry.get_env_constructor(self.domain)
        return environment_constructor(solo_mode=self.solo_mode)

    def _agent_llm_args(self) -> dict[str, Any]:
        args = bind_policy_request(
            self.gen_args,
            enable_thinking=self.econfig.enable_thinking,
        )
        args.update(
            api_base=self.agent_base_url,
            api_key=self.agent_api_key,
            timeout=min(self.timeout, 120.0),
            num_retries=0,
        )
        # SGLang rejects prompt + max_new_tokens >= context_length. Keep the
        # server's 32K window, reserving its final slot at the policy consumer.
        args["max_total_tokens"] = min(
            args.get("max_total_tokens", self.econfig.context_window_tokens),
            self.econfig.context_window_tokens - 1,
        )
        return args

    def _user_llm_args(self) -> dict[str, Any]:
        if not self.econfig.user_llm_base_url or not self.user_api_key:
            raise ValueError("User simulator endpoint and credential are required")
        args = dict(self.econfig.user_llm_args or {})
        args.setdefault("top_p", 1.0)
        args.update(
            api_base=self.econfig.user_llm_base_url,
            api_key=self.user_api_key,
            timeout=min(self.timeout, 120.0),
            num_retries=self.econfig.user_llm_num_retries,
        )
        return args

    def _bind_nl_assertion_evaluator(self) -> None:
        """Use the qualified DeepSeek route for official NL-assertion checks."""

        if not self.econfig.user_llm:
            raise ValueError("User simulator model is required for NL evaluation")
        tau2_nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS = _litellm_user_model(
            self.econfig.user_llm
        )
        tau2_nl_evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = self._user_llm_args()

    def _get_agent_and_user(self, task: Task, env: Environment, run_info: Tau2RunInfo):
        agent_policy_doc = env.get_policy()
        tools: list[Tool] = env.get_tools()
        try:
            user_tools = env.get_user_tools()
        except Exception:
            user_tools = []
        if self.econfig.add_thinking_tool:
            tools.append(Tool(think))

        if self.solo_mode:
            agent = LLMSoloAgent(
                tools=tools + user_tools,
                domain_policy=agent_policy_doc,
                llm="openai/dummy",
                llm_args=self._agent_llm_args(),
                task=task,
            )
            user = DummyUser()
        else:
            agent = LLMAgent(
                tools=tools,
                domain_policy=agent_policy_doc,
                llm="openai/dummy",
                llm_args=self._agent_llm_args(),
            )
            user = RetryingUserSimulator(
                debug_dir=Path(os.environ.get("TAU2_RUN_ROOT", "."))
                / "tau2-user-failures",
                task_id=task.id,
                domain=self.domain,
                tools=user_tools if len(user_tools) > 0 else None,
                instructions=str(task.user_scenario),
                llm=_litellm_user_model(self.econfig.user_llm),
                llm_args=self._user_llm_args(),
            )
        return agent, user

    def _get_orchestrator(
        self,
        agent: HalfDuplexAgent[LLMAgentState],
        user: HalfDuplexUser,
        env: Environment,
        task: Task,
    ) -> Orchestrator:
        return Orchestrator(
            domain=self.domain,
            agent=agent,
            user=user,
            environment=env,
            task=task,
            max_steps=self.econfig.max_steps,
            solo_mode=self.solo_mode,
            timeout=self.timeout,
        )

    async def run(self, task: Task) -> Tau2RunInfo:
        """Run a simulation for the given task."""
        domain = self.domain
        solo_mode = self.solo_mode
        logger.info(
            f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, "
            f"Solo Mode: {solo_mode}"
        )

        env = self._get_environment()
        run_info = Tau2RunInfo(
            reward=0.0,
            task=task,
            messages=[],
            agent_time=[],
            user_time=[],
            reward_info=None,
            error=None,
        )
        agent, user = self._get_agent_and_user(task=task, env=env, run_info=run_info)
        orchestrator = self._get_orchestrator(
            agent=agent, user=user, env=env, task=task
        )

        try:
            simulation = await asyncio.to_thread(orchestrator.run)
            run_info.messages = simulation.messages
        except Exception as e:
            run_info.messages = orchestrator.get_trajectory()
            run_info.error = str(e)
            if is_context_budget_error(e):
                run_info.error_type = "task_budget"
                run_info.terminated = False
                run_info.truncated = True
                run_info.stop_reason = "context_limit"
                logger.info(
                    "FINISHED SIMULATION: Domain: %s, Task: %s, Reward: 0.0, "
                    "Stop reason: context_limit",
                    domain,
                    task.id,
                )
                return run_info
            raise Tau2InfrastructureError(
                f"τ² simulation failed for {domain}/{task.id}: {type(e).__name__}: {e}"
            ) from e

        try:
            self._bind_nl_assertion_evaluator()
            reward_info = evaluate_simulation(
                domain=domain,
                task=task,
                simulation=simulation,
                evaluation_type=EvaluationType.ALL,
                solo_mode=solo_mode,
            )
            run_info.reward_info = reward_info
            run_info.reward = reward_info.reward
        except Exception as e:
            raise Tau2InfrastructureError(
                f"τ² evaluator failed for {domain}/{task.id}: {type(e).__name__}: {e}"
            ) from e

        termination_reason = str(
            getattr(simulation, "termination_reason", "completed")
        ).lower()
        budget_markers = (
            "max_step",
            "max step",
            "budget",
            "context_limit",
            "timeout",
        )
        run_info.truncated = any(
            marker in termination_reason for marker in budget_markers
        )
        run_info.terminated = not run_info.truncated
        run_info.stop_reason = termination_reason

        logger.info(
            f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, "
            f"Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. "
            f"Reward: {reward_info.reward}"
        )
        return run_info


class Tau2AgentWorkflow:
    """Tau2 agent workflow for AReaL proxy mode.

    This workflow runs a Tau2 customer service simulation using the proxy server
    for OpenAI-compatible API calls. It supports both standard multi-turn mode
    and solo mode where the agent handles both agent and user roles.

    Args:
        econfig: Tau2 environment configuration
        gen_args: Generation arguments (temperature, max_tokens, etc.)
        timeout: Maximum time allowed for a single episode (default: 600s)
    """

    def __init__(
        self,
        econfig: Tau2EnvConfig | dict | None = None,
        gen_args: dict | None = None,
        timeout: float = 600.0,
        infra_retries: int = 1,
    ):
        if econfig is None:
            econfig = Tau2EnvConfig()
        elif isinstance(econfig, dict):
            econfig = Tau2EnvConfig(**econfig)
        self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = timeout
        self.infra_retries = infra_retries

    @staticmethod
    def should_retry_episode(exc: Exception) -> bool:
        """Retry transient provider failures, never scored model failures or bugs."""
        if not isinstance(exc, Tau2InfrastructureError):
            return False
        if is_context_budget_error(exc):
            return False
        # Provider context/config failures are deterministic, even when an
        # OpenAI-compatible wrapper labels them as rate limiting.
        message = str(exc).lower()
        if "maximum context length" in message or "context_length_exceeded" in message:
            return False
        cause = exc.__cause__
        return isinstance(
            cause,
            (
                litellm.APIConnectionError,
                litellm.Timeout,
                litellm.RateLimitError,
                litellm.InternalServerError,
            ),
        )

    async def run(
        self, data: dict[str, Any], **extra_kwargs: Any
    ) -> AgentWorkflowResult:
        """Run a Tau2 simulation episode.

        Args:
            data: Input data containing task_id, split, and optional econfig/gconfig
            **extra_kwargs: Additional kwargs including:
                - base_url: Proxy server URL for agent LLM
                - api_key: Session-wise API key for proxy server authentication

        Returns:
            AgentWorkflowResult: Reward plus explicit episode end metadata.
        """
        # Get proxy URL from workflow context
        base_url: str | None = extra_kwargs.get("base_url", None) or os.getenv(
            "OPENAI_BASE_URL"
        )
        api_key: str | None = extra_kwargs.get("api_key", None) or os.getenv(
            "AREAL_PROXY_SESSION_API_KEY"
        )
        if base_url is None:
            raise ValueError("base_url is required for Tau2AgentWorkflow")
        if not api_key:
            raise ValueError("api_key is required for Tau2AgentWorkflow policy session")

        # Override econfig from data if provided
        econfig = self.econfig
        if "econfig" in data:
            econfig = Tau2EnvConfig(**data["econfig"])
        data_domain = data.get("domain")
        if data_domain is not None:
            if econfig.domain != "mixed" and data_domain != econfig.domain:
                raise ValueError(
                    f"Dataset domain {data_domain} conflicts with fixed domain "
                    f"{econfig.domain}"
                )
            econfig = replace(econfig, domain=str(data_domain))

        # Override gen_args from data if provided
        gen_args = self.gen_args.copy()
        if "gconfig" in data:
            gen_args.update(data["gconfig"])
        requested_completion = int(
            gen_args.get("max_completion_tokens", econfig.max_completion_tokens)
        )
        gen_args["max_completion_tokens"] = min(
            requested_completion, econfig.max_completion_tokens
        )
        gen_args["max_total_tokens"] = econfig.context_window_tokens

        # Get task information
        domain = econfig.domain
        split = data.get("split", "train")
        task_id = data["task_id"]
        task = _get_task(domain=domain, task_id=task_id, split=split)

        # The official pinned τ² orchestrator is synchronous. Tau2Runner executes
        # it in a worker thread and binds two independent OpenAI-compatible
        # LiteLLM routes for the policy and user simulator.
        user_api_key = None
        if not econfig.solo_mode:
            if not econfig.user_llm_base_url or not econfig.user_llm:
                raise ValueError(
                    "user_llm_base_url and user_llm are required outside solo mode"
                )
            user_api_key = os.getenv(econfig.user_llm_api_key_env)
            if not user_api_key:
                raise ValueError(
                    "Missing user simulator credential environment variable "
                    f"{econfig.user_llm_api_key_env}"
                )
        # Create runner and execute
        runner = Tau2Runner(
            econfig=econfig,
            gen_args=gen_args,
            agent_base_url=base_url,
            agent_api_key=api_key,
            user_api_key=user_api_key,
            timeout=self.timeout,
        )

        # asyncio cannot kill a thread created by to_thread(). The official
        # orchestrator and per-request LiteLLM timeouts provide the cooperative
        # bound; if the caller cancels, join the episode before the proxy session
        # is allowed to close so no calls or tool effects survive the session.
        run_info = await _await_without_orphaning(runner.run(task))

        reward_info = (
            run_info.reward_info.model_dump()
            if run_info.reward_info is not None
            else None
        )
        return AgentWorkflowResult(
            reward=float(run_info.reward),
            terminated=run_info.terminated,
            truncated=run_info.truncated,
            bootstrap_mask=False,
            stop_reason=run_info.stop_reason,
            metadata={
                "domain": domain,
                "task_id": task_id,
                "official_score": float(run_info.reward),
                "reward_info": reward_info,
                "failure_class": run_info.error_type,
            },
        )
