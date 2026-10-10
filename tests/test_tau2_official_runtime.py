"""CPU qualification against the pinned official τ² package and data assets."""

import asyncio
import json
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


@pytest.fixture(autouse=True)
def isolate_user_debug_files(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))


def test_official_loader_matches_all_three_domains():
    assert validate_installed_tau2_revision()["revision"] == OFFICIAL_TAU2_REVISION
    for domain in SUPPORTED_DOMAINS:
        train = get_tau2_dataset(domain, split="train")
        test = get_tau2_dataset(domain, split="test")
        assert len(train) == OFFICIAL_SPLIT_SNAPSHOTS[domain]["train"]
        assert len(test) == OFFICIAL_SPLIT_SNAPSHOTS[domain]["test"]


def test_fixed_variant_trains_without_the_test_twins():
    train = get_tau2_dataset("retail", split="train", variant="fixed")
    test = get_tau2_dataset("retail", split="test", variant="fixed")

    twins = {"37", "41", "67", "72", "96"}
    assert len(train) == OFFICIAL_SPLIT_SNAPSHOTS["retail"]["train"] - len(twins)
    assert not twins & {row["task_id"] for row in train}
    assert len(test) == OFFICIAL_SPLIT_SNAPSHOTS["retail"]["test"]


def test_online_task_split_is_disjoint_per_domain():
    kwargs = dict(
        domain=["airline", "retail", "telecom"],
        algorithm="sao",
        experiment_mode="tune",
        critic_tasks_per_domain=2,
        seed=42,
    )
    rows = list(get_tau2_dataset(**kwargs, split="train")) + list(
        get_tau2_dataset(**kwargs, split="dev")
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


@pytest.mark.parametrize("episode_timeout", [1200.0, 30.0])
def test_policy_transport_uses_episode_budget_without_changing_user_or_orchestrator(
    monkeypatch, episode_timeout
):
    import examples.tau2.agent as module

    runner = Tau2Runner(
        econfig=Tau2EnvConfig(
            max_steps=200,
            user_llm_base_url="https://user.invalid/v1",
            user_llm="openai/test-user",
        ),
        gen_args={},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="session",
        user_api_key="user",
        timeout=episode_timeout,
    )
    monkeypatch.setattr(module, "Orchestrator", lambda **kwargs: kwargs)

    assert runner._agent_llm_args()["timeout"] == episode_timeout
    assert runner._agent_llm_args()["num_retries"] == 0
    assert runner._user_llm_args()["timeout"] == min(episode_timeout, 120.0)
    assert runner._user_llm_args()["num_retries"] == 3
    orchestrator = runner._get_orchestrator(object(), object(), object(), object())
    assert orchestrator["timeout"] == episode_timeout
    assert orchestrator["max_steps"] == 200


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
@pytest.mark.parametrize(
    "policy_failure,policy_message",
    [
        (True, "areal_context_limit: context_length_exceeded"),
        (
            True,
            "OpenAIException - Error code: 500 - {'detail': "
            '"RuntimeError: Failed after 3 retries each. Payload: '
            "{'sampling_params': {'max_new_tokens': 3773}, "
            "'return_logprob': True}. Endpoint: /generate. Last error: "
            "Requested token count exceeds the model's maximum context length "
            "of 32768 tokens. You requested a total of 32768 tokens: "
            "28995 tokens from the input messages and 3773 tokens for the "
            'completion."}',
        ),
        (False, "user simulator exceeds max_total_tokens"),
    ],
)
async def test_context_exhaustion_is_policy_failure_not_simulator_failure(
    monkeypatch, policy_failure, policy_message
):
    from litellm import BadRequestError

    from examples.tau2.agent import Tau2InfrastructureError

    def fake_completion(**kwargs):
        if kwargs["model"] == "openai/dummy" or not policy_failure:
            raise BadRequestError(
                message=policy_message,
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
    assert runner._agent_llm_args()["max_total_tokens"] == 32768
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


@pytest.mark.parametrize("debug_write_fails", [True, False])
def test_empty_user_response_fails_attempt_once_with_evidence(
    monkeypatch, tmp_path, debug_write_fails
):
    import json

    from tau2.data_model.message import AssistantMessage

    from examples.tau2.user_simulator import (
        EmptyUserResponseError,
        RecordingUserSimulator,
    )

    requests = []
    debug_dir = tmp_path / "debug"
    if debug_write_fails:
        debug_dir.write_text("not a directory")

    def completion(**kwargs):
        requests.append(kwargs["messages"])
        return ModelResponse(
            id="provider-1",
            model="test",
            choices=[
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Full debugging content",
                    },
                }
            ],
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    simulator = RecordingUserSimulator(
        debug_dir=debug_dir,
        task_id="task",
        domain="telecom",
        instructions="Test scenario",
        tools=None,
        llm="openai/test",
        llm_args={"api_key": "test-secret", "num_retries": 3},
    )
    incoming = AssistantMessage(role="assistant", content="Please describe the issue")
    # Official τ² fails the attempt (UserMessage.validate raises ValueError); the
    # episode-level retry replays it. Nothing is re-requested in place.
    with pytest.raises(ValueError, match="Empty user response") as caught:
        simulator.generate_next_message(incoming, simulator.get_init_state())
    assert isinstance(caught.value, EmptyUserResponseError)
    assert len(requests) == 1
    if debug_write_fails:
        assert debug_dir.is_file()
        return
    (record,) = [json.loads(p.read_text()) for p in debug_dir.glob("*.json")]
    assert record["response"]["id"] == "provider-1"
    assert (
        record["response"]["choices"][0]["message"]["reasoning_content"]
        == "Full debugging content"
    )
    assert record["request"]["messages"] == requests[0]
    assert "api_key" not in record["request"]["args"]


def test_malformed_user_tool_json_fails_attempt_and_records_decode_error(
    monkeypatch, tmp_path
):
    import json

    from tau2.data_model.message import AssistantMessage

    from examples.tau2.user_simulator import RecordingUserSimulator

    requests = []

    def completion(**kwargs):
        requests.append(kwargs["messages"])
        return ModelResponse(
            model="test",
            choices=[
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool-1",
                                "type": "function",
                                "function": {
                                    "name": "verify_phone",
                                    "arguments": '{"phone":',
                                },
                            }
                        ],
                    },
                }
            ],
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    simulator = RecordingUserSimulator(
        debug_dir=tmp_path,
        task_id="task",
        domain="telecom",
        instructions="Test scenario",
        tools=None,
        llm="openai/test",
    )
    with pytest.raises(json.JSONDecodeError):
        simulator.generate_next_message(
            AssistantMessage(role="assistant", content="Check your phone"),
            simulator.get_init_state(),
        )
    assert len(requests) == 1
    (record,) = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert record["error"]["type"] == "JSONDecodeError"
    assert record["error"]["doc"] == '{"phone":'
    assert record["error"]["pos"] == len('{"phone":')


def test_user_simulator_transport_error_is_recorded_and_raised(monkeypatch, tmp_path):
    import json

    from tau2.data_model.message import AssistantMessage

    from examples.tau2.user_simulator import RecordingUserSimulator

    calls = []

    def completion(**kwargs):
        calls.append(kwargs["messages"])
        raise RuntimeError("transport failed")

    monkeypatch.setattr(llm_utils, "completion", completion)
    simulator = RecordingUserSimulator(
        debug_dir=tmp_path,
        task_id="task",
        domain="telecom",
        instructions="Test scenario",
        tools=None,
        llm="openai/test",
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        simulator.generate_next_message(
            AssistantMessage(role="assistant", content="Check your phone"),
            simulator.get_init_state(),
        )
    assert len(calls) == 1
    (record,) = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert record["error"]["message"] == "transport failed"


def test_user_tool_only_reply_is_valid_without_record(monkeypatch, tmp_path):
    from tau2.data_model.message import AssistantMessage

    from examples.tau2.user_simulator import RecordingUserSimulator

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return ModelResponse(
            model="test",
            choices=[
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool-1",
                                "type": "function",
                                "function": {
                                    "name": "check_phone",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                }
            ],
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    simulator = RecordingUserSimulator(
        debug_dir=tmp_path,
        task_id="task",
        domain="telecom",
        instructions="Test scenario",
        tools=None,
        llm="openai/test",
    )
    response, _ = simulator.generate_next_message(
        AssistantMessage(role="assistant", content="Check your phone"),
        simulator.get_init_state(),
    )
    response.validate()
    assert response.tool_calls[0].name == "check_phone"
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed", [1199.0, 1200.0, 1201.0])
async def test_terminal_episode_deadline_is_checked_before_scoring(
    monkeypatch, elapsed
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import examples.tau2.agent as module

    runner = Tau2Runner(
        econfig=Tau2EnvConfig(domain="telecom"),
        gen_args={},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="session",
        timeout=1200.0,
    )
    times = iter([100.0, 100.0 + elapsed])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(times)))
    monkeypatch.setattr(runner, "_get_environment", lambda: object())
    monkeypatch.setattr(runner, "_get_agent_and_user", lambda **kwargs: (None, None))
    monkeypatch.setattr(runner, "_bind_nl_assertion_evaluator", lambda: None)
    monkeypatch.setattr(
        runner,
        "_get_orchestrator",
        lambda **kwargs: SimpleNamespace(
            run=lambda: SimpleNamespace(messages=[], termination_reason="completed")
        ),
    )
    evaluator = Mock(return_value=SimpleNamespace(reward=1.0))
    monkeypatch.setattr(module, "evaluate_simulation", evaluator)
    task = _get_task(
        "telecom", get_tau2_dataset("telecom", split="train")[0]["task_id"], "train"
    )

    if elapsed >= 1200.0:
        result = await runner.run(task)
        assert (result.reward, result.truncated, result.terminated) == (
            0.0,
            True,
            False,
        )
        assert (result.stop_reason, result.error_type) == ("timeout", "task_budget")
        evaluator.assert_not_called()
    else:
        result = await runner.run(task)
        assert result.reward == 1.0
        evaluator.assert_called_once()


@pytest.mark.asyncio
async def test_training_episode_timeout_is_kept_as_truncated_zero(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tau2.data_model.simulation import TerminationReason

    import examples.tau2.agent as module

    runner = Tau2Runner(
        econfig=Tau2EnvConfig(domain="telecom"),
        gen_args={},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="session",
        user_api_key="user",
        timeout=1200.0,
    )
    monkeypatch.setattr(runner, "_get_environment", lambda: object())
    monkeypatch.setattr(
        runner, "_get_agent_and_user", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr(
        runner,
        "_get_orchestrator",
        lambda **kwargs: SimpleNamespace(
            run=lambda: SimpleNamespace(
                messages=[], termination_reason=TerminationReason.TIMEOUT
            )
        ),
    )
    evaluator = Mock(side_effect=AssertionError("Timeout must not be scored"))
    monkeypatch.setattr(module, "evaluate_simulation", evaluator)
    task = _get_task(
        "telecom", get_tau2_dataset("telecom", split="train")[0]["task_id"], "train"
    )
    # Replaying a timed-out episode would drop slow trajectories from training.
    result = await runner.run(task)
    evaluator.assert_not_called()
    assert (result.reward, result.truncated, result.terminated) == (0.0, True, False)
    assert (result.stop_reason, result.error_type) == ("timeout", "task_budget")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, error_type",
    [("malformed", json.JSONDecodeError), (0.0, None), ("unrelated", RuntimeError)],
)
async def test_evaluator_failure_fails_attempt_for_episode_retry(
    monkeypatch, tmp_path, outcome, error_type
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import examples.tau2.agent as module

    runner = Tau2Runner(
        econfig=Tau2EnvConfig(domain="retail"),
        gen_args={},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="session-secret",
        user_api_key="user-secret",
        timeout=1200.0,
    )
    task = _get_task("retail", "68", "test")
    simulation = SimpleNamespace(messages=[], termination_reason="completed")
    orchestrator = SimpleNamespace(run=Mock(return_value=simulation))
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(runner, "_get_environment", lambda: object())
    monkeypatch.setattr(runner, "_get_agent_and_user", lambda **kwargs: (None, None))
    monkeypatch.setattr(runner, "_get_orchestrator", lambda **kwargs: orchestrator)
    monkeypatch.setattr(runner, "_bind_nl_assertion_evaluator", lambda: None)
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        if outcome == "malformed":
            raise json.JSONDecodeError("Expecting value", "not JSON", 0)
        if outcome == "unrelated":
            raise RuntimeError("unrelated scorer failure")
        return SimpleNamespace(reward=outcome)

    monkeypatch.setattr(module, "evaluate_simulation", evaluate)
    if error_type:
        # Official τ² scores inside the simulation run: a scoring failure fails
        # the attempt and the whole episode is replayed, never re-scored in place.
        with pytest.raises(module.Tau2InfrastructureError) as caught:
            await runner.run(task)
        assert isinstance(caught.value.__cause__, error_type)
        assert Tau2AgentWorkflow.should_retry_episode(caught.value)
    else:
        result = await runner.run(task)
        assert result.reward == outcome
    assert len(calls) == 1
    paths = list((tmp_path / "tau2-evaluator-failures").glob("*.json"))
    assert len(paths) == (outcome == "malformed")
    for path in paths:
        raw = path.read_text()
        record = json.loads(raw)
        assert record["error"]["doc"] == "not JSON"
        assert record["error"]["pos"] == 0
        assert record["task_id"] == "68"
        assert "session-secret" not in raw and "user-secret" not in raw


def _scripted_runner(monkeypatch, *, evaluation, termination_reason, elapsed):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import examples.tau2.agent as module

    runner = Tau2Runner(
        econfig=Tau2EnvConfig(domain="telecom"),
        gen_args={},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="session",
        user_api_key="user",
        timeout=1800.0,
        evaluation=evaluation,
    )
    times = iter([100.0, 100.0 + elapsed])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(times)))
    monkeypatch.setattr(runner, "_get_environment", lambda: object())
    monkeypatch.setattr(runner, "_get_agent_and_user", lambda **kwargs: (None, None))
    monkeypatch.setattr(runner, "_bind_nl_assertion_evaluator", lambda: None)
    monkeypatch.setattr(
        runner,
        "_get_orchestrator",
        lambda **kwargs: SimpleNamespace(
            run=lambda: SimpleNamespace(
                messages=[], termination_reason=termination_reason
            )
        ),
    )
    reward = 0.0 if termination_reason != "completed" else 1.0
    evaluator = Mock(return_value=SimpleNamespace(reward=reward))
    monkeypatch.setattr(module, "evaluate_simulation", evaluator)
    return runner, evaluator


@pytest.mark.asyncio
async def test_evaluation_scores_official_timeout_as_counted_zero(monkeypatch):
    from tau2.data_model.simulation import TerminationReason

    runner, evaluator = _scripted_runner(
        monkeypatch,
        evaluation=True,
        termination_reason=TerminationReason.TIMEOUT,
        elapsed=1801.0,
    )
    task = _get_task(
        "telecom", get_tau2_dataset("telecom", split="train")[0]["task_id"], "train"
    )
    result = await runner.run(task)
    evaluator.assert_called_once()
    assert result.reward == 0.0
    assert result.stop_reason == "timeout"


@pytest.mark.asyncio
async def test_evaluation_keeps_finished_episode_past_wall_clock(monkeypatch):
    runner, evaluator = _scripted_runner(
        monkeypatch, evaluation=True, termination_reason="completed", elapsed=2400.0
    )
    task = _get_task(
        "telecom", get_tau2_dataset("telecom", split="train")[0]["task_id"], "train"
    )
    result = await runner.run(task)
    evaluator.assert_called_once()
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_evaluation_context_exhaustion_fails_attempt_like_official(monkeypatch):
    from litellm import BadRequestError

    from examples.tau2.agent import Tau2InfrastructureError

    def fake_completion(**kwargs):
        if kwargs["model"] == "openai/dummy":
            raise BadRequestError(
                message="areal_context_limit: context_length_exceeded",
                model=kwargs["model"],
                llm_provider="openai",
            )
        return ModelResponse(
            model="test",
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
        evaluation=True,
    )
    task = _get_task(
        "telecom", get_tau2_dataset("telecom", split="train")[0]["task_id"], "train"
    )
    # Official τ² has no context-window outcome: the attempt fails, is retried,
    # and is finally excluded as an infrastructure error.
    with pytest.raises(Tau2InfrastructureError) as caught:
        await runner.run(task)
    assert Tau2AgentWorkflow.should_retry_episode(caught.value)


def test_user_llm_usage_sums_only_user_turns():
    from tau2.data_model.message import AssistantMessage, UserMessage

    from examples.tau2.agent import user_llm_usage

    messages = [
        UserMessage(
            role="user",
            content="hi",
            usage={"prompt_tokens": 100, "completion_tokens": 10},
        ),
        AssistantMessage(
            role="assistant",
            content="ok",
            usage={"prompt_tokens": 999, "completion_tokens": 99},
        ),
        UserMessage(
            role="user",
            content="bye",
            usage={"prompt_tokens": 150, "completion_tokens": None},
        ),
        UserMessage(role="user", content="no usage"),
    ]

    assert user_llm_usage(messages) == {
        "user_llm_calls": 2,
        "user_prompt_tokens": 250,
        "user_completion_tokens": 10,
    }


def _workflow_with_fake_runner(monkeypatch, tmp_path, run):
    import examples.tau2.agent as agent_mod
    from examples.tau2.agent import Tau2AgentWorkflow

    class FakeRunner:
        def __init__(self, **kwargs):
            self.messages = []

        async def run(self, task):
            return await run(self, task)

    monkeypatch.setattr(agent_mod, "Tau2Runner", FakeRunner)
    monkeypatch.setattr(agent_mod, "_get_task", lambda **kwargs: object())
    monkeypatch.delenv("TAU2_USAGE_LOG", raising=False)
    monkeypatch.setenv("TAU2_TEST_USER_KEY", "user-secret")
    return Tau2AgentWorkflow(
        econfig=Tau2EnvConfig(
            domain="retail",
            user_llm="deepseek-flash",
            user_llm_base_url="http://user.invalid/v1",
            user_llm_api_key_env="TAU2_TEST_USER_KEY",
        ),
        trajectory_dir=str(tmp_path),
    )


def _read_records(tmp_path, source_id):
    return [
        json.loads(path.read_text()) for path in (tmp_path / source_id).glob("*.json")
    ]


def test_workflow_records_policy_free_episode_trajectory(monkeypatch, tmp_path):
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

    from examples.tau2.utils import Tau2RunInfo

    async def run(runner, task):
        messages = [
            DEFAULT_FIRST_AGENT_MESSAGE.model_copy(deep=True),
            UserMessage(role="user", content=f"bye {STOP}"),
        ]
        return Tau2RunInfo.model_construct(
            reward=0.0,
            messages=messages,
            reward_info=None,
            error_type=None,
            stop_reason="user_stop",
        )

    workflow = _workflow_with_fake_runner(monkeypatch, tmp_path, run)
    result = asyncio.run(
        workflow.run(
            {"task_id": "69", "split": "test", "source_id": "retail:69:trial0"},
            base_url="http://policy.invalid/v1",
            api_key="session-secret",
        )
    )

    assert result.stop_reason == "user_stop"
    (record,) = _read_records(tmp_path, "retail:69:trial0")
    assert record["status"] == "completed"
    assert record["policy_turns"] == 0
    assert record["official_score"] == 0.0
    assert record["messages"][-1]["content"] == f"bye {STOP}"
    # The fixed greeting is not a policy call; a generated reply is.
    assert isinstance(DEFAULT_FIRST_AGENT_MESSAGE, AssistantMessage)


@pytest.mark.parametrize("evaluation, expected", [(False, 4096), (True, 32768)])
def test_workflow_caps_policy_replies_only_in_training(
    monkeypatch, tmp_path, evaluation, expected
):
    import examples.tau2.agent as agent_mod

    seen = {}

    class CapturingRunner:
        def __init__(self, **kwargs):
            seen.update(kwargs["gen_args"])
            raise RuntimeError("stop after capturing gen_args")

    workflow = _workflow_with_fake_runner(monkeypatch, tmp_path, None)
    monkeypatch.setattr(agent_mod, "Tau2Runner", CapturingRunner)
    workflow.evaluation = evaluation
    workflow.gen_args = {"max_completion_tokens": 4096}

    with pytest.raises(RuntimeError, match="capturing"):
        asyncio.run(
            workflow.run(
                {"task_id": "69", "split": "test"},
                base_url="http://policy.invalid/v1",
                api_key="session-secret",
            )
        )

    # Official τ² sets no reply cap: evaluation is bounded by the 32K context only.
    assert seen["max_completion_tokens"] == expected
    assert seen["max_total_tokens"] == 32768


def test_workflow_records_failed_attempt_with_partial_trajectory(monkeypatch, tmp_path):
    from tau2.data_model.message import AssistantMessage, UserMessage

    from examples.tau2.agent import Tau2InfrastructureError

    async def run(runner, task):
        runner.messages = [
            UserMessage(role="user", content="hello"),
            AssistantMessage(role="assistant", content="hi", raw_data={"id": "x"}),
        ]
        raise Tau2InfrastructureError("user simulator exploded")

    workflow = _workflow_with_fake_runner(monkeypatch, tmp_path, run)
    with pytest.raises(Tau2InfrastructureError):
        asyncio.run(
            workflow.run(
                {"task_id": "7", "split": "test", "source_id": "retail:7:trial1"},
                base_url="http://policy.invalid/v1",
                api_key="session-secret",
            )
        )

    (record,) = _read_records(tmp_path, "retail:7:trial1")
    assert record["status"] == "error"
    assert record["error_type"] == "Tau2InfrastructureError"
    assert record["policy_turns"] == 1
    assert [m["content"] for m in record["messages"]] == ["hello", "hi"]
    raw = (next((tmp_path / "retail:7:trial1").glob("*.json"))).read_text()
    assert "session-secret" not in raw and "user-secret" not in raw
