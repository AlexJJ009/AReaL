"""CPU qualification against the pinned official τ² package and data assets."""

import asyncio
import os

import pytest

if not os.getenv("TAU2_DATA_DIR"):
    pytest.skip(
        "TAU2_DATA_DIR is required for official τ² tests", allow_module_level=True
    )

pytest.importorskip("tau2")

from litellm import ModelResponse
from tau2.evaluator import evaluator_nl_assertions
from tau2.user.user_simulator_base import STOP
from tau2.utils import llm_utils

from examples.tau2.agent import (
    Tau2AgentWorkflow,
    Tau2Runner,
    _await_without_orphaning,
    _get_task,
    _litellm_user_model,
)
from examples.tau2.contracts import (
    OFFICIAL_SPLIT_SNAPSHOTS,
    OFFICIAL_TAU2_REVISION,
    SUPPORTED_DOMAINS,
    validate_installed_tau2_revision,
)
from examples.tau2.train import get_tau2_dataset
from examples.tau2.utils import Tau2EnvConfig


def test_official_loader_matches_all_three_domains():
    assert validate_installed_tau2_revision()["revision"] == OFFICIAL_TAU2_REVISION
    for domain in SUPPORTED_DOMAINS:
        train = get_tau2_dataset(domain, split="train")
        test = get_tau2_dataset(domain, split="test")
        assert len(train) == OFFICIAL_SPLIT_SNAPSHOTS[domain]["train"]
        assert len(test) == OFFICIAL_SPLIT_SNAPSHOTS[domain]["test"]


def test_collect_schedule_marks_task_disjoint_train_and_dev_per_domain():
    rows = get_tau2_dataset(
        ["airline", "retail", "telecom"],
        split="train",
        domain_effective_episodes={"airline": 2, "retail": 2, "telecom": 2},
        algorithm="collect",
        seed=42,
    )

    for domain in SUPPORTED_DOMAINS:
        selected = [row for row in rows if row["domain"] == domain]
        assert sorted(row["critic_split"] for row in selected) == ["dev", "train"]
        assert len({row["task_id"] for row in selected}) == 2


@pytest.mark.parametrize(
    "env_name",
    ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AREAL_PROXY_SESSION_API_KEY"],
)
def test_user_simulator_key_cannot_alias_policy_session_key(env_name):
    with pytest.raises(ValueError, match="must not alias"):
        Tau2EnvConfig(user_llm_api_key_env=env_name)


def test_bare_user_model_uses_openai_compatible_litellm_route():
    assert _litellm_user_model("deepseek-flash") == "openai/deepseek-flash"
    assert _litellm_user_model("openai/test-user") == "openai/test-user"


def test_nl_assertion_evaluator_reuses_qualified_user_route(monkeypatch):
    monkeypatch.setattr(evaluator_nl_assertions, "DEFAULT_LLM_NL_ASSERTIONS", "unbound")
    monkeypatch.setattr(evaluator_nl_assertions, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", {})
    runner = Tau2Runner(
        econfig=Tau2EnvConfig(
            user_llm_base_url="https://api.deepseek.com",
            user_llm="deepseek-flash",
            user_llm_args={"temperature": 0.0, "max_completion_tokens": 512},
        ),
        gen_args={},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="policy-session-key",
        user_api_key="deepseek-provider-key",
    )

    runner._bind_nl_assertion_evaluator()

    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS == (
        "openai/deepseek-flash"
    )
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS == {
        "temperature": 0.0,
        "max_completion_tokens": 512,
        "top_p": 1.0,
        "api_base": "https://api.deepseek.com",
        "api_key": "deepseek-provider-key",
        "timeout": 120.0,
        "num_retries": 3,
    }


@pytest.mark.asyncio
async def test_official_telecom_episode_reaches_evaluator_without_network(monkeypatch):
    """Use fake LLM replies but the real official task, env, orchestrator and scorer."""

    calls: list[dict] = []

    def fake_completion(**kwargs):
        user_call_count = sum(call["model"] == "openai/test-user" for call in calls)
        calls.append(
            {
                "model": kwargs["model"],
                "api_base": kwargs.get("api_base"),
                "extra_body": kwargs.get("extra_body"),
            }
        )
        if kwargs["model"] == "openai/test-user":
            content = "Please help me." if user_call_count == 0 else STOP
        else:
            content = "I can help with that."
        return ModelResponse(
            model=kwargs["model"],
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    econfig = Tau2EnvConfig(
        domain="telecom",
        max_steps=4,
        user_llm_base_url="https://user.invalid/v1",
        user_llm="openai/test-user",
        user_llm_args={"temperature": 0.0},
    )
    runner = Tau2Runner(
        econfig=econfig,
        gen_args={"temperature": 0.7, "max_completion_tokens": 4096},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="policy-session-key",
        user_api_key="user-provider-key",
        timeout=30.0,
    )
    telecom_task_id = next(
        row["task_id"]
        for row in get_tau2_dataset("telecom", split="train")
        if row["domain"] == "telecom"
    )
    task = _get_task("telecom", telecom_task_id, "train")

    result = await runner.run(task)

    assert result.terminated is True
    assert result.truncated is False
    assert result.reward_info is not None
    assert [call["model"] for call in calls] == [
        "openai/test-user",
        "openai/dummy",
        "openai/test-user",
    ]
    assert calls[0]["api_base"] == "https://user.invalid/v1"
    assert calls[1]["api_base"] == "http://policy.invalid/v1"
    assert calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.asyncio
async def test_workflow_rejects_missing_policy_session_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")
    monkeypatch.delenv("AREAL_PROXY_SESSION_API_KEY", raising=False)
    workflow = Tau2AgentWorkflow()

    with pytest.raises(ValueError, match="api_key.*policy session"):
        await workflow.run({}, base_url="http://policy.invalid/v1")


@pytest.mark.asyncio
async def test_cancellation_waits_for_episode_cleanup():
    started = asyncio.Event()
    release = asyncio.Event()

    async def bounded_episode():
        started.set()
        await release.wait()
        return "finished"

    workflow_task = asyncio.create_task(_await_without_orphaning(bounded_episode()))
    await started.wait()
    workflow_task.cancel()
    await asyncio.sleep(0)
    workflow_task.cancel()
    await asyncio.sleep(0)

    assert workflow_task.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await workflow_task


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_failure", [True, False])
async def test_context_exhaustion_is_policy_failure_not_simulator_failure(
    monkeypatch, policy_failure
):
    from litellm import BadRequestError

    from examples.tau2.agent import Tau2InfrastructureError

    def fake_completion(**kwargs):
        if kwargs["model"] == "openai/dummy" or not policy_failure:
            raise BadRequestError(
                message=(
                    "areal_context_limit: context_length_exceeded"
                    if policy_failure
                    else "user simulator exceeds max_total_tokens"
                ),
                model=kwargs["model"],
                llm_provider="openai",
            )
        return ModelResponse(
            model=kwargs["model"],
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Help me."},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    runner = Tau2Runner(
        econfig=Tau2EnvConfig(
            domain="telecom",
            user_llm_base_url="https://user.invalid/v1",
            user_llm="openai/test-user",
        ),
        gen_args={"max_total_tokens": 32768, "max_completion_tokens": 4096},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="session",
        user_api_key="user",
        timeout=30.0,
    )
    assert runner._agent_llm_args()["max_total_tokens"] == 32767
    task = _get_task(
        "telecom", get_tau2_dataset("telecom", split="train")[0]["task_id"], "train"
    )
    if not policy_failure:
        with pytest.raises(Tau2InfrastructureError):
            await runner.run(task)
        return
    result = await runner.run(task)
    assert result.reward == 0.0
    assert result.truncated and not result.terminated
    assert result.error_type == "task_budget"
    assert result.stop_reason == "context_limit"
    assert result.reward_info is None
