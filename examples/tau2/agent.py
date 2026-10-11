"""Tau2 Agent Workflow for AReaL proxy mode.

This module implements a Tau2 agent that uses the AReaL proxy server
for OpenAI-compatible API calls during RL training.
"""

import asyncio
import json
import os
import time
from collections.abc import Awaitable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import litellm
import tau2.evaluator.evaluator_nl_assertions as tau2_nl_evaluator
from litellm import register_model
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent
from tau2.config import DEFAULT_RETRY_ATTEMPTS as OFFICIAL_RETRY_ATTEMPTS
from tau2.config import DEFAULT_RETRY_MIN_WAIT as OFFICIAL_RETRY_DELAY
from tau2.data_model.message import AssistantMessage, Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser
from tau2.user.user_simulator_base import HalfDuplexUser

# Import utilities (also patches tau2.utils.llm_utils)
from examples.tau2 import benchmark_variant, task_pack
from examples.tau2.contracts import bind_policy_request, strip_policy_reasoning
from examples.tau2.user_simulator import RecordingUserSimulator
from examples.tau2.utils import Tau2EnvConfig, Tau2RunInfo, trajectory_record_dir

from areal.experimental.openai.types import (
    CONTEXT_LENGTH_EXCEEDED_MARKER,
    AgentWorkflowResult,
    InteractionWithTokenLogpReward,
)
from areal.utils import logging

logger = logging.getLogger("Tau2Agent")
_T = TypeVar("_T")


class Tau2InfrastructureError(RuntimeError):
    """Unexpected environment/evaluator failure that must not become reward zero."""


class Tau2ExportCardinalityError(RuntimeError):
    """Retryable proxy export shape failure before tensors enter training."""


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


def user_llm_usage(messages: list[Any]) -> dict[str, int]:
    """Sum the user-simulator tokens tau2 records on each accepted user turn.

    Discarded retry attempts and NL-judge calls are not attached to messages.
    """

    usage = {"user_llm_calls": 0, "user_prompt_tokens": 0, "user_completion_tokens": 0}
    for message in messages:
        if not isinstance(message, UserMessage) or not message.usage:
            continue
        usage["user_llm_calls"] += 1
        usage["user_prompt_tokens"] += int(message.usage.get("prompt_tokens") or 0)
        usage["user_completion_tokens"] += int(
            message.usage.get("completion_tokens") or 0
        )
    return usage


def append_usage_record(path: str, record: dict[str, Any]) -> None:
    """Append one episode line; single small appends keep concurrent writers whole."""

    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        logger.warning(
            "Could not append user-simulator usage to %s", path, exc_info=True
        )


def policy_turns(messages: list[Message]) -> int:
    """Count policy completions; the orchestrator's fixed greeting has no raw_data."""

    return sum(
        isinstance(m, AssistantMessage) and m.raw_data is not None for m in messages
    )


def write_trajectory_record(root: str, record: dict[str, Any]) -> None:
    """Persist one episode attempt; never mask the episode's own outcome."""

    try:
        directory = trajectory_record_dir(root, record["source_id"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{record['attempt']}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, default=str)
    except Exception:
        logger.warning("Could not write τ² trajectory record", exc_info=True)


def record_evaluator_incident(
    domain: str, task_id: str, exc: json.JSONDecodeError
) -> None:
    """Keep the malformed NL-judge reply that failed an evaluation attempt."""

    debug_dir = Path(os.environ.get("TAU2_RUN_ROOT", ".")) / "tau2-evaluator-failures"
    path = debug_dir / f"{uuid4().hex}.json"
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "domain": domain,
                    "task_id": task_id,
                    "pid": os.getpid(),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "doc": exc.doc,
                        "pos": exc.pos,
                    },
                },
                stream,
                ensure_ascii=False,
                indent=2,
            )
    except OSError:
        logger.warning("Could not write evaluator incident %s", path, exc_info=True)
        return
    logger.warning(
        "Malformed evaluator JSON for %s/%s; evidence %s", domain, task_id, path
    )


class ReasoningStrippedLLMAgent(LLMAgent):
    """Official ``LLMAgent`` whose replies drop the thinking block.

    The full reply stays in ``raw_data``. Without this, the user simulator reads
    the reasoning, and a tool call arrives with text, which the orchestrator
    rejects as an agent error.
    """

    def _generate_next_message(self, message, state) -> AssistantMessage:
        reply = super()._generate_next_message(message, state)
        return reply.model_copy(
            update={"content": strip_policy_reasoning(reply.content)}
        )


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
        evaluation: bool = False,
        task_db: str | None = None,
    ):
        self.econfig = econfig
        self.gen_args = gen_args
        self.agent_base_url = agent_base_url
        self.agent_api_key = agent_api_key
        self.user_api_key = user_api_key
        self.timeout = timeout
        # Evaluation reproduces `tau2 run` outcomes; training keeps ADR 0002/0007.
        self.evaluation = evaluation
        self.domain = econfig.domain
        self.solo_mode = econfig.solo_mode
        # A task-pack task's own database; None uses the domain's default.
        self.task_db = task_db
        self._env_kwargs = {} if task_db is None else {"task_db": task_db}
        # Kept on the runner so a failed episode's partial trajectory survives.
        self.messages: list[Message] = []

    def _get_environment(self) -> Environment:
        benchmark_variant.activate(self.econfig.benchmark_variant, [self.domain])
        environment_constructor = registry.get_env_constructor(self.domain)
        return environment_constructor(solo_mode=self.solo_mode, **self._env_kwargs)

    def _agent_llm_args(self) -> dict[str, Any]:
        args = bind_policy_request(
            self.gen_args,
            enable_thinking=self.econfig.enable_thinking,
        )
        args.update(
            api_base=self.agent_base_url,
            api_key=self.agent_api_key,
            timeout=self.timeout,
            num_retries=0,
        )
        # The AReaL client reserves SGLang's final context slot when computing
        # max_new_tokens. Pass the full window here so that margin is applied once.
        args["max_total_tokens"] = min(
            args.get("max_total_tokens", self.econfig.context_window_tokens),
            self.econfig.context_window_tokens,
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
            agent_class = (
                ReasoningStrippedLLMAgent if self.econfig.enable_thinking else LLMAgent
            )
            agent = agent_class(
                tools=tools,
                domain_policy=agent_policy_doc,
                llm="openai/dummy",
                llm_args=self._agent_llm_args(),
            )
            user = RecordingUserSimulator(
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

        def run_simulation():
            started = time.monotonic()
            simulation = orchestrator.run()
            return simulation, time.monotonic() - started

        try:
            simulation, elapsed = await asyncio.to_thread(run_simulation)
            run_info.messages = self.messages = simulation.messages
        except Exception as e:
            run_info.messages = self.messages = orchestrator.get_trajectory()
            run_info.error = str(e)
            # Official τ² has no context-window outcome: the provider error is
            # retried and finally excluded, so evaluation raises it instead.
            if is_context_budget_error(e) and not self.evaluation:
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

        reason = getattr(simulation, "termination_reason", "completed")
        termination_reason = str(getattr(reason, "value", reason)).lower()
        # Evaluation scores TIMEOUT through the official evaluator (reward 0) and
        # keeps episodes that finished, like `tau2 run --timeout`. Training keeps
        # an over-budget episode as truncated with reward 0, like context_limit:
        # replaying it would drop slow trajectories from the training data.
        # Official termination checks are cooperative and can skip the deadline
        # when the last message already ended the episode, so it is enforced here.
        if not self.evaluation and (
            termination_reason == "timeout" or elapsed >= self.timeout
        ):
            run_info.error_type = "task_budget"
            run_info.terminated = False
            run_info.truncated = True
            run_info.stop_reason = "timeout"
            logger.info(
                "FINISHED SIMULATION: Domain: %s, Task: %s, Reward: 0.0, "
                "Stop reason: timeout",
                domain,
                task.id,
            )
            return run_info

        try:
            self._bind_nl_assertion_evaluator()
            reward_info = evaluate_simulation(
                domain=domain,
                task=task,
                simulation=simulation,
                evaluation_type=EvaluationType.ALL,
                solo_mode=solo_mode,
                env_kwargs=dict(self._env_kwargs),
            )
            if self.econfig.benchmark_variant == "fixed":
                reward_info = benchmark_variant.apply_transfer_rule(
                    task, simulation.messages, reward_info, domain
                )
        except Exception as e:
            # Official τ² scores inside the simulation run, so a scoring failure
            # replays the whole episode through the episode-level retry.
            if isinstance(e, json.JSONDecodeError):
                record_evaluator_incident(domain, task.id, e)
            raise Tau2InfrastructureError(
                f"τ² evaluator failed for {domain}/{task.id}: {type(e).__name__}: {e}"
            ) from e
        run_info.reward_info = reward_info
        run_info.reward = reward_info.reward

        budget_markers = (
            "max_step",
            "max step",
            "budget",
            "context_limit",
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
        infra_retries: int = OFFICIAL_RETRY_ATTEMPTS,
        evaluation: bool = False,
        trajectory_dir: str | None = None,
    ):
        if econfig is None:
            econfig = Tau2EnvConfig()
        elif isinstance(econfig, dict):
            econfig = Tau2EnvConfig(**econfig)
        self.econfig = econfig
        self.gen_args = gen_args or {}
        self.timeout = timeout
        self.infra_retries = infra_retries
        # Fixed pause between episode attempts, as in `tau2 run --retry-delay`.
        self.retry_delay = OFFICIAL_RETRY_DELAY
        # An episode that still fails after the official retries is lost from its
        # group (official exclusion), not a failure of the other members.
        self.reject_exhausted_episode = True
        self.evaluation = evaluation
        self.trajectory_dir = trajectory_dir
        from areal.utils.runtime_audit import write_runtime_audit

        write_runtime_audit(
            "tau2-workflow-initialized",
            {
                "environment": self.econfig,
                "generation": self.gen_args,
                "timeout": self.timeout,
                "infra_retries": self.infra_retries,
                "retry_delay": self.retry_delay,
                "evaluation": self.evaluation,
                "trajectory_dir": self.trajectory_dir,
            },
        )

    @staticmethod
    def should_retry_episode(exc: Exception) -> bool:
        """Replay any failed attempt, as `tau2.runner.progress` does.

        Cancellation is a BaseException and never reaches this check.
        """
        return isinstance(exc, Exception)

    @staticmethod
    def validate_exported_interactions(
        *,
        interactions: dict[str, InteractionWithTokenLogpReward],
        export_style: str,
        data: dict[str, Any],
    ) -> None:
        if export_style != "concat":
            return
        # Zero leaves means the policy was never called (e.g. the official user
        # simulator stopped on its first turn). Official τ² scores that episode
        # normally; it is not a transient failure and must not be replayed.
        if len(interactions) <= 1:
            return
        task_id = data.get("task_id", "<unknown>")
        domain = data.get("domain", "<unknown>")
        cause = Tau2ExportCardinalityError(
            f"τ² concat export requires exactly one trajectory for "
            f"{domain}/{task_id}, got {len(interactions)}: "
            f"{sorted(interactions.keys())}"
        )
        raise Tau2InfrastructureError(str(cause)) from cause

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
        if self.evaluation:
            # Official τ² sets no per-reply cap; only the remaining context
            # bounds a reply. The 4096 cap is a training rollout budget.
            gen_args["max_completion_tokens"] = econfig.context_window_tokens
        else:
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
        pack = data.get("task_pack")
        if pack:
            task_pack.install(domain)
            task = task_pack.get_task(pack, domain, task_id)
            task_db = task_pack.task_db(pack, domain, task_id)
        else:
            task = _get_task(domain=domain, task_id=task_id, split=split)
            task_db = None

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
            evaluation=self.evaluation,
            task_db=task_db,
        )

        # asyncio cannot kill a thread created by to_thread(). The official
        # orchestrator and per-request LiteLLM timeouts provide the cooperative
        # bound; if the caller cancels, join the episode before the proxy session
        # is allowed to close so no calls or tool effects survive the session.
        trajectory_root = self.trajectory_dir
        record = {
            "source_id": data.get("source_id") or f"{domain}:{task_id}",
            "attempt": uuid4().hex,
            "domain": domain,
            "task_id": task_id,
            "split": split,
            "trial": data.get("trial"),
            "user_llm": econfig.user_llm,
        }
        try:
            run_info = await _await_without_orphaning(runner.run(task))
        except BaseException as exc:
            # Record interrupted and failed attempts too, with whatever
            # trajectory the official orchestrator produced before failing.
            if trajectory_root:
                write_trajectory_record(
                    trajectory_root,
                    {
                        **record,
                        "recorded_at": time.time(),
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "policy_turns": policy_turns(runner.messages),
                        "messages": [
                            m.model_dump(mode="json") for m in runner.messages
                        ],
                    },
                )
            raise
        if trajectory_root:
            write_trajectory_record(
                trajectory_root,
                {
                    **record,
                    "recorded_at": time.time(),
                    "status": "completed",
                    "stop_reason": run_info.stop_reason,
                    "official_score": float(run_info.reward),
                    "task_budget_failure": run_info.error_type == "task_budget",
                    "reward_info": run_info.reward_info.model_dump(mode="json")
                    if run_info.reward_info is not None
                    else None,
                    "policy_turns": policy_turns(run_info.messages),
                    "messages": [m.model_dump(mode="json") for m in run_info.messages],
                },
            )
        usage_log = os.environ.get("TAU2_USAGE_LOG")
        if usage_log:
            append_usage_record(
                usage_log,
                {
                    "source_id": data.get("source_id"),
                    "domain": domain,
                    "task_id": task_id,
                    "trial": data.get("trial"),
                    "user_llm": econfig.user_llm,
                    "official_score": float(run_info.reward),
                    **user_llm_usage(run_info.messages),
                    "stop_reason": run_info.stop_reason,
                },
            )

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
