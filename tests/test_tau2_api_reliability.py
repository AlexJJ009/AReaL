"""Offline regression tests for provider failures, retry bounds and fatal delivery."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from examples.tau2.agent import Tau2AgentWorkflow, Tau2InfrastructureError, Tau2Runner
from examples.tau2.api_reliability import (
    QuotaCircuitBreaker,
    bounded_call,
    classify_api_error,
    train_with_quota_exit,
)
from examples.tau2.utils import Tau2EnvConfig

from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.infra.workflow_executor import BatchTaskDispatcher, WorkflowExecutor
from areal.utils.rollout_error import QUOTA_EXIT_CODE, FatalRolloutError


def provider_error(code="AccountQuotaExceeded", message=None):
    """Same nested body as 642.log, with a synthetic request ID."""
    body = {
        "error": {
            "code": code,
            "message": message
            or (
                "You have exceeded the weekly usage quota. It will reset at "
                "2026-10-26 00:00:00 +0800 CST. We recommend upgrading your plan "
                "for more quota, or waiting for the reset. Request id: test"
            ),
            "param": "",
            "type": "TooManyRequests",
        }
    }
    return openai.RateLimitError(
        "Error code: 429 - " + str(body),
        response=httpx.Response(
            429, request=httpx.Request("POST", "https://mock.invalid")
        ),
        body=body,
    )


@pytest.mark.parametrize(
    "code,message,expected",
    [
        ("AccountQuotaExceeded", None, "quota"),
        ("AccountRateLimitExceeded", "RPM exceeded; try again", "transient"),
        ("insufficient_quota", "No credit", "quota"),
        ("OtherProviderCode", "You have exceeded your usage quota", "quota"),
        ("OtherProviderCode", "Quota might be low", "transient"),
        ("AccountRateLimitExceeded", "Quota per minute exceeded", "transient"),
    ],
)
def test_provider_body_classifies_code_and_conservative_fallback(
    code, message, expected
):
    assert classify_api_error(provider_error(code, message)) == expected


def test_wrapped_and_litellm_string_errors_classify_without_body():
    error = RuntimeError("litellm.RateLimitError: " + str(provider_error()))
    assert classify_api_error(error) == "quota"
    wrapped = Tau2InfrastructureError("episode failed")
    wrapped.__cause__ = error
    assert classify_api_error(wrapped) == "quota"
    assert classify_api_error(ValueError("quota calculation failed")) == "other"
    assert classify_api_error(TimeoutError()) == "transient"


def make_config(tmp_path, **kwargs):
    return Tau2EnvConfig(
        _quota_breaker_state_path=str(tmp_path / "breaker.json"), **kwargs
    )


@pytest.mark.parametrize(
    "status,body,message",
    [
        (402, {"error": {"message": "Insufficient Balance"}}, "Insufficient Balance"),
        (402, {}, "Payment Required"),
        (429, {"error": {"code": "insufficient_balance"}}, "No credit"),
        (429, {"error": {"code": "InsufficientBalance"}}, "No credit"),
        (429, {"error": {"code": "AccountQuotaExceeded"}}, "Account quota exceeded"),
    ],
)
def test_official_account_exhaustion_trips_shared_breaker_without_retry(
    tmp_path, status, body, message
):
    """Official user and judge share bounded_call and the same fatal state."""
    config = make_config(tmp_path)
    error = openai.APIStatusError(
        message,
        response=httpx.Response(
            status, request=httpx.Request("POST", "https://mock.invalid")
        ),
        body=body,
    )
    assert classify_api_error(error) == "quota"
    call = Mock(side_effect=error)
    for _ in range(config.quota_breaker_threshold - 1):
        with pytest.raises(openai.APIStatusError):
            bounded_call(call, config=config, num_retries=120, sleep=Mock())
    with pytest.raises(FatalRolloutError, match="provider reset: not supplied"):
        bounded_call(call, config=config, num_retries=120, sleep=Mock())
    assert call.call_count == config.quota_breaker_threshold
    with pytest.raises(FatalRolloutError):
        bounded_call(call, config=config)
    assert call.call_count == config.quota_breaker_threshold
    assert QuotaCircuitBreaker(config).snapshot()["provider_reset_time"] == (
        "not supplied by provider"
    )


@pytest.mark.parametrize(
    "message,expected",
    [
        ("litellm.APIError: Error code: 402 - Payment Required", "quota"),
        ("litellm.APIError: Insufficient Balance", "quota"),
        ("account_quota_exceeded", "quota"),
        ("Tokens per minute quota exceeded", "other"),
    ],
)
def test_official_wrapped_string_retains_account_exhaustion_classification(
    message, expected
):
    assert classify_api_error(RuntimeError(message)) == expected


def test_breaker_preserves_reset_from_earlier_quota_response(tmp_path):
    config = make_config(tmp_path)
    breaker = QuotaCircuitBreaker(config)
    breaker.record("quota", provider_error(), now=1)
    breaker.record("quota", RuntimeError("quota exhausted"), now=2)
    with pytest.raises(FatalRolloutError, match="2026-10-26 00:00:00"):
        breaker.record("quota", RuntimeError("quota exhausted"), now=3)
    assert breaker.snapshot()["provider_reset_time"] == "2026-10-26 00:00:00 +0800 CST"


def test_quota_is_one_request_and_never_replays_episode(tmp_path):
    config = make_config(tmp_path)
    call = Mock(side_effect=provider_error())
    with pytest.raises(openai.RateLimitError) as caught:
        bounded_call(call, config=config, num_retries=120, timeout=120)
    assert call.call_count == 1
    wrapped = Tau2InfrastructureError("simulation failed")
    wrapped.__cause__ = caught.value
    assert not Tau2AgentWorkflow.should_retry_episode(wrapped)
    assert QuotaCircuitBreaker(config).snapshot()["quota_errors"] == 1


@pytest.mark.asyncio
async def test_transient_bound_then_episode_rejection_and_next_success(tmp_path):
    config = make_config(tmp_path)
    call = Mock(side_effect=provider_error("AccountRateLimitExceeded", "RPM exceeded"))
    sleep = Mock()
    with pytest.raises(openai.RateLimitError):
        bounded_call(call, config=config, num_retries=2, timeout=120, sleep=sleep)
    assert call.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]
    assert QuotaCircuitBreaker(config).snapshot()["transient_api_errors"] == 3
    agent = SimpleNamespace(
        run=AsyncMock(),
        infra_retries=1,
        retry_delay=0,
        reject_exhausted_episode=True,
        should_retry_episode=Tau2AgentWorkflow.should_retry_episode,
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent)
    attempts = Mock(
        side_effect=[
            Tau2InfrastructureError("transient"),
            Tau2InfrastructureError("transient"),
            {"ok": "golden"},
        ]
    )

    async def run(*args):
        return attempts()

    workflow._arun_episode = run
    assert await workflow.arun_episode(None, {}) is None
    assert await workflow.arun_episode(None, {}) == {"ok": "golden"}
    assert attempts.call_count == 3


def test_breaker_window_and_cross_instance_trip(tmp_path):
    config = make_config(tmp_path)
    breaker = QuotaCircuitBreaker(config)
    for now in (0, 301, 602):
        breaker.record("quota", provider_error(), now=now)
    assert not breaker.snapshot()["fatal"]
    QuotaCircuitBreaker(config).record("quota", provider_error(), now=603)
    with pytest.raises(FatalRolloutError, match="2026-10-26"):
        breaker.record("quota", provider_error(), now=604)
    with pytest.raises(FatalRolloutError):
        QuotaCircuitBreaker(config).check()


def test_concurrent_episodes_share_exact_threshold(tmp_path):
    config = make_config(tmp_path)

    def record(_):
        try:
            QuotaCircuitBreaker(config).record("quota", provider_error(), now=1)
        except FatalRolloutError:
            return "fatal"
        return "quota"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(record, range(8)))
    assert outcomes.count("quota") == 2
    assert QuotaCircuitBreaker(config).snapshot()["fatal"]


def test_timeout_and_inner_retry_are_passed_to_generate(tmp_path):
    config = make_config(tmp_path)
    call = Mock(return_value="same reply")
    assert (
        bounded_call(
            call,
            config=config,
            num_retries=8,
            timeout=37,
            retry_strategy="exponential_backoff_retry",
        )
        == "same reply"
    )
    assert call.call_args.kwargs["timeout"] == 37
    assert call.call_args.kwargs["num_retries"] == 0
    assert "retry_strategy" not in call.call_args.kwargs


def test_request_budget_bounds_slow_requests_even_with_legacy_120(tmp_path):
    config = make_config(tmp_path, api_retry_budget_seconds=600)
    now = [0.0]

    def call(**kwargs):
        now[0] += kwargs["timeout"]
        raise TimeoutError()

    def sleep(delay):
        now[0] += delay

    with pytest.raises(TimeoutError):
        bounded_call(
            call,
            config=config,
            num_retries=120,
            timeout=120,
            clock=lambda: now[0],
            sleep=sleep,
        )
    assert now[0] == 600


def test_runner_routes_use_configured_timeout(tmp_path):
    config = make_config(
        tmp_path,
        user_llm="mock",
        user_llm_base_url="https://mock.invalid",
        user_llm_request_timeout=42,
        nl_judge_llm_request_timeout=37,
    )
    runner = Tau2Runner(
        config, {}, "http://policy.invalid", "mock", user_api_key="mock"
    )
    assert runner._user_llm_args()["timeout"] == 42
    with runner._nl_assertion_evaluation():
        from tau2.evaluator import evaluator_nl_assertions as evaluator

        assert evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS["timeout"] == 37


def test_fatal_dispatcher_reaches_trainer_dedicated_exit():
    dispatcher = BatchTaskDispatcher.__new__(BatchTaskDispatcher)
    import threading

    dispatcher._thread_exception_lock = threading.Lock()
    dispatcher._thread_exception = FatalRolloutError("quota reset at tomorrow")
    trainer = SimpleNamespace(
        train=Mock(side_effect=lambda **kwargs: dispatcher._check_thread_exception())
    )
    with pytest.raises(SystemExit) as caught:
        train_with_quota_exit(trainer, workflow="mock")
    assert caught.value.code == QUOTA_EXIT_CODE
    assert QUOTA_EXIT_CODE not in (0, 1)


@pytest.mark.asyncio
async def test_executor_fatal_is_delivered_instead_of_rejected():
    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor._staleness_manager = Mock()

    async def fatal(*args):
        raise FatalRolloutError("quota reset at tomorrow")

    executor.inference_engine = None
    pending = SimpleNamespace(
        task_id=1,
        is_eval=False,
        should_accept_fn=None,
        workflow=SimpleNamespace(arun_episode=fatal),
        data={},
    )
    with pytest.raises(FatalRolloutError, match="quota reset"):
        await executor._create_workflow_task(pending)()
    executor.staleness_manager.on_rollout_rejected.assert_not_called()


def test_real_litellm_openai_adapter_honors_timeout_and_disables_sdk_retries(tmp_path):
    """Exercise installed LiteLLM + SDK with an in-memory HTTP transport only."""
    import litellm

    received = []

    def respond(request):
        received.append(request)
        return httpx.Response(429, json=provider_error().body)

    client = openai.OpenAI(
        api_key="mock-only",
        base_url="https://mock.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    config = make_config(tmp_path)
    with pytest.raises(litellm.RateLimitError):
        bounded_call(
            litellm.completion,
            config=config,
            model="openai/mock",
            messages=[{"role": "user", "content": "mock"}],
            client=client,
            api_base="https://mock.invalid/v1",
            api_key="mock-only",
            num_retries=120,
            timeout=37,
            sleep=Mock(),
        )
    assert len(received) == 1
    assert received[0].extensions["timeout"]["read"] == 37
    assert client.max_retries == 0
    client.close()


def test_monitor_trips_trainer_wait_and_prevents_partial_batch_update(tmp_path):
    """The real dispatcher wait wakes from the run-wide file, without a callback."""
    import threading

    config = make_config(tmp_path, quota_breaker_threshold=1)
    dispatcher = BatchTaskDispatcher(8, Mock(), Mock())
    entered = threading.Event()
    trainer = SimpleNamespace(
        config=SimpleNamespace(econfig=config),
        rollout=SimpleNamespace(dispatcher=dispatcher),
        eval_rollout=None,
        updates=[],
    )

    def train(**kwargs):
        entered.set()
        dispatcher.wait_results(2, timeout=5)
        trainer.updates.append("must never update or checkpoint this partial batch")

    trainer.train = train

    def trip():
        assert entered.wait(3)
        with pytest.raises(FatalRolloutError):
            QuotaCircuitBreaker(config).record("quota", provider_error())

    thread = threading.Thread(target=trip)
    thread.start()
    try:
        with pytest.raises(SystemExit) as caught:
            train_with_quota_exit(trainer)
        assert caught.value.code == QUOTA_EXIT_CODE
        assert dispatcher.runner.paused.is_set()
        assert trainer.updates == []
        with pytest.raises(FatalRolloutError):
            dispatcher.submit_task_input(SimpleNamespace(task_id=2))
    finally:
        thread.join(timeout=3)
    assert not thread.is_alive()


def test_pretripped_run_has_dedicated_exit_before_trainer_called(tmp_path):
    config = make_config(tmp_path, quota_breaker_threshold=1)
    with pytest.raises(FatalRolloutError):
        QuotaCircuitBreaker(config).record("quota", provider_error())
    trainer = SimpleNamespace(config=SimpleNamespace(econfig=config), train=Mock())
    with pytest.raises(SystemExit) as caught:
        train_with_quota_exit(trainer)
    assert caught.value.code == QUOTA_EXIT_CODE
    trainer.train.assert_not_called()


def test_stats_are_step_deltas_and_include_subprocess_errors(tmp_path):
    from examples.tau2.api_reliability import Tau2ApiStatsMixin

    class Base:
        def _additional_stats(self):
            return {"existing": 7}

    class Trainer(Tau2ApiStatsMixin, Base):
        pass

    config = make_config(tmp_path)
    trainer = Trainer()
    trainer.config = SimpleNamespace(econfig=config)
    trainer.rollout = SimpleNamespace(
        staleness_manager=SimpleNamespace(get_stats=lambda: SimpleNamespace(rejected=4))
    )
    breaker = QuotaCircuitBreaker(config)
    breaker.record("quota", provider_error())
    breaker.record("transient", TimeoutError())
    for _ in range(4):
        breaker.record_rejection()
    assert trainer._additional_stats() == {
        "existing": 7,
        "rollout/quota_errors": 1,
        "rollout/transient_api_errors": 1,
        "rollout/rejected_episodes": 4,
    }
    assert trainer._additional_stats() == {
        "existing": 7,
        "rollout/quota_errors": 0,
        "rollout/transient_api_errors": 0,
        "rollout/rejected_episodes": 0,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("quota_breaker_threshold", 0),
        ("quota_breaker_window_seconds", -1),
        ("user_llm_request_timeout", 0),
        ("nl_judge_llm_request_timeout", float("inf")),
        ("api_retry_budget_seconds", float("nan")),
    ],
)
def test_config_invalid_breaker_and_timeout_values_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        Tau2EnvConfig(**{field: value})


@pytest.mark.asyncio
async def test_quota_has_no_fresh_session_retry_in_real_proxy_wrapper():
    workflow = OpenAIProxyWorkflow(
        mode="inline",
        agent=SimpleNamespace(
            run=AsyncMock(),
            infra_retries=3,
            reject_exhausted_episode=True,
            should_retry_episode=Tau2AgentWorkflow.should_retry_episode,
        ),
    )
    wrapped = Tau2InfrastructureError("simulation failed")
    wrapped.__cause__ = provider_error()
    workflow._arun_episode = AsyncMock(side_effect=wrapped)
    with pytest.raises(Tau2InfrastructureError):
        await workflow.arun_episode(None, {})
    assert workflow._arun_episode.await_count == 1


@pytest.mark.parametrize("route", ["user", "judge"])
def test_real_route_binding_quota_one_transport_call(monkeypatch, tmp_path, route):
    from tau2.data_model.message import AssistantMessage
    from tau2.evaluator import evaluator_nl_assertions as evaluator
    from tau2.utils import llm_utils

    from examples.tau2.user_simulator import RecordingUserSimulator

    config = make_config(
        tmp_path, user_llm="mock", user_llm_base_url="https://mock.invalid"
    )
    completion = Mock(side_effect=provider_error())
    monkeypatch.setattr(llm_utils, "completion", completion)
    incoming = AssistantMessage(role="assistant", content="Please describe the issue")
    if route == "user":
        simulator = RecordingUserSimulator(
            debug_dir=tmp_path / "incidents",
            task_id="task",
            domain="telecom",
            instructions="Test scenario",
            tools=None,
            llm="openai/mock",
            retry_config=config,
            llm_args={"api_key": "mock", "num_retries": 120, "timeout": 41},
        )
        with pytest.raises(openai.RateLimitError):
            simulator.generate_next_message(incoming, simulator.get_init_state())
        expected_timeout = 41
    else:
        runner = Tau2Runner(
            config, {}, "http://policy.invalid", "mock", user_api_key="mock"
        )
        with runner._nl_assertion_evaluation(), pytest.raises(openai.RateLimitError):
            evaluator.generate(
                model=evaluator.DEFAULT_LLM_NL_ASSERTIONS,
                messages=[incoming],
                **evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
            )
        expected_timeout = 120
    assert completion.call_count == 1
    assert completion.call_args.kwargs["num_retries"] == 0
    assert completion.call_args.kwargs["timeout"] == expected_timeout


def _child_trainer_waits_for_quota(config, entered):
    """Forked process stands in for the trainer, without creating any engines."""
    dispatcher = BatchTaskDispatcher(8, Mock(), Mock())

    def train(**kwargs):
        entered.set()
        dispatcher.wait_results(2, timeout=10)
        raise AssertionError("partial batch reached the update")

    trainer = SimpleNamespace(
        config=SimpleNamespace(econfig=config),
        rollout=SimpleNamespace(dispatcher=dispatcher),
        eval_rollout=None,
        train=train,
    )
    train_with_quota_exit(trainer)


def _child_records_quota(config):
    QuotaCircuitBreaker(config).record("quota", provider_error())


def test_cross_process_workers_trip_actual_trainer_exit_code(tmp_path):
    """Independent processes share the count; the trainer really exits with 86."""
    import multiprocessing

    context = multiprocessing.get_context("fork")
    config = make_config(tmp_path)
    entered = context.Event()
    trainer_process = context.Process(
        target=_child_trainer_waits_for_quota, args=(config, entered)
    )
    trainer_process.start()
    try:
        assert entered.wait(5)
        workers = [
            context.Process(target=_child_records_quota, args=(config,))
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
            assert worker.exitcode == 0
        with pytest.raises(FatalRolloutError):
            QuotaCircuitBreaker(config).record("quota", provider_error())
        trainer_process.join(timeout=5)
        assert trainer_process.exitcode == QUOTA_EXIT_CODE
        assert QuotaCircuitBreaker(config).snapshot()["quota_errors"] == 3
    finally:
        if trainer_process.is_alive():
            trainer_process.terminate()
            trainer_process.join(timeout=3)


def test_cleanup_error_after_quota_preserves_exit_86(tmp_path):
    from examples.tau2.api_reliability import quota_exit_guard

    config = make_config(tmp_path, quota_breaker_threshold=1)

    class CleanupRaises:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise RuntimeError("evaluator RPC failed during cleanup")

    with pytest.raises(SystemExit) as caught:
        with quota_exit_guard(config), CleanupRaises():
            with pytest.raises(FatalRolloutError):
                QuotaCircuitBreaker(config).record("quota", provider_error())
            raise SystemExit(QUOTA_EXIT_CODE)
    assert caught.value.code == QUOTA_EXIT_CODE


@pytest.mark.asyncio
async def test_rejected_episode_metric_counts_once_after_fresh_retries(tmp_path):
    config = make_config(tmp_path)
    agent = Tau2AgentWorkflow.__new__(Tau2AgentWorkflow)
    agent.econfig = config
    agent.infra_retries = 2
    agent.retry_delay = 0
    agent.reject_exhausted_episode = True
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent)
    workflow._arun_episode = AsyncMock(side_effect=Tau2InfrastructureError("transient"))
    assert await workflow.arun_episode(None, {}) is None
    assert workflow._arun_episode.await_count == 3
    assert QuotaCircuitBreaker(config).snapshot()["rejected_episodes"] == 1
    workflow._arun_episode.side_effect = provider_error()
    with pytest.raises(openai.RateLimitError):
        await workflow.arun_episode(None, {})
    assert QuotaCircuitBreaker(config).snapshot()["rejected_episodes"] == 2
    workflow._arun_episode.side_effect = FatalRolloutError("run stopped")
    with pytest.raises(FatalRolloutError):
        await workflow.arun_episode(None, {})
    assert QuotaCircuitBreaker(config).snapshot()["rejected_episodes"] == 2
