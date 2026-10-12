"""Independent NL judge routing and per-episode evidence, without network calls."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

if not os.getenv("TAU2_DATA_DIR"):
    pytest.skip("TAU2_DATA_DIR is required for tau2 tests", allow_module_level=True)

pytest.importorskip("tau2")

from litellm import ModelResponse
from tau2.evaluator import evaluator_nl_assertions as evaluator
from tau2.utils import llm_utils

import examples.tau2.agent as agent_module
from examples.tau2.agent import Tau2Runner
from examples.tau2.utils import Tau2EnvConfig, nl_judge_metadata


@pytest.fixture
def judge_config(monkeypatch):
    """Different user and judge routes and credentials make cross-routing visible."""
    monkeypatch.setenv("TEST_NL_JUDGE_KEY", "judge-secret")
    return Tau2EnvConfig(
        user_llm="doubao-user",
        user_llm_base_url="https://ark.invalid/api/v3",
        user_llm_args={"temperature": 0.7, "reasoning_effort": "medium"},
        user_llm_num_retries=2,
        nl_judge_llm="deepseek-flash",
        nl_judge_llm_base_url="https://api.deepseek.com",
        nl_judge_llm_api_key_env="TEST_NL_JUDGE_KEY",
        nl_judge_llm_num_retries=1,
    )


def make_runner(config):
    return Tau2Runner(
        econfig=config,
        gen_args={},
        agent_base_url="http://policy.invalid/v1",
        agent_api_key="policy-secret",
        user_api_key="user-secret",
    )


@pytest.mark.parametrize("solo_mode", [False, True])
def test_configured_judge_independent_of_user_uses_own_route(
    monkeypatch, judge_config, solo_mode
):
    """Judge transport never inherits the user key, endpoint, or sampling args."""
    config = replace(judge_config, solo_mode=solo_mode)
    runner = make_runner(config)
    with runner._nl_assertion_evaluation():
        assert evaluator.DEFAULT_LLM_NL_ASSERTIONS == "openai/deepseek-flash"
        assert evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS == {
            "api_base": "https://api.deepseek.com",
            "api_key": "judge-secret",
            "temperature": 0.0,
            "top_p": 1.0,
            "timeout": 120.0,
            "num_retries": 1,
        }
    assert runner._user_llm_args()["api_key"] == "user-secret"
    assert runner._user_llm_args()["temperature"] == 0.7


def test_configured_judge_explicit_args_override_temperature(judge_config):
    """Sampling can be overridden, while transport fields retain ownership."""
    runner = make_runner(
        replace(
            judge_config,
            nl_judge_llm_args={
                "temperature": 0.25,
                "api_key": "wrong-key",
                "api_base": "https://wrong.invalid",
                "num_retries": 99,
            },
        )
    )
    with runner._nl_assertion_evaluation():
        args = evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS
        assert args["temperature"] == 0.25
        assert args["api_key"] == "judge-secret"
        assert args["api_base"] == "https://api.deepseek.com"
        assert args["num_retries"] == 1


def test_unset_judge_falls_back_to_user_and_warns_once(monkeypatch, judge_config):
    """Legacy routing keeps user arguments and emits one clear process warning."""
    monkeypatch.setattr(agent_module, "_WARNED_USER_JUDGE_FALLBACK", False)
    warnings = []
    monkeypatch.setattr(agent_module.logger, "warning", warnings.append)
    runner = make_runner(replace(judge_config, nl_judge_llm=None))
    for _ in range(2):
        with runner._nl_assertion_evaluation():
            assert evaluator.DEFAULT_LLM_NL_ASSERTIONS == "openai/doubao-user"
            assert evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS == runner._user_llm_args()
    assert len(warnings) == 1
    assert "falls back" in warnings[0]
    assert nl_judge_metadata(runner.econfig) == {
        "model": "openai/doubao-user",
        "base_url_host": "ark.invalid",
        "source": "user_fallback",
    }


@pytest.mark.parametrize(
    "key_env", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AREAL_PROXY_SESSION_API_KEY"]
)
def test_judge_key_alias_policy_session_rejected(key_env):
    """The judge cannot consume a policy proxy session credential."""
    with pytest.raises(ValueError, match="nl_judge_llm_api_key_env must not alias"):
        Tau2EnvConfig(nl_judge_llm_api_key_env=key_env)


def test_judge_negative_retries_rejected():
    """Invalid retry budgets fail config validation."""
    with pytest.raises(
        ValueError, match="nl_judge_llm_num_retries must be non-negative"
    ):
        Tau2EnvConfig(nl_judge_llm_num_retries=-1)


@pytest.mark.parametrize("key_value", [None, ""])
def test_configured_judge_missing_own_key_rejected(
    monkeypatch, judge_config, key_value
):
    """A valid user key cannot satisfy the configured judge's credential check."""
    if key_value is None:
        monkeypatch.delenv("TEST_NL_JUDGE_KEY")
    else:
        monkeypatch.setenv("TEST_NL_JUDGE_KEY", key_value)
    with pytest.raises(
        ValueError, match="Missing NL judge credential.*TEST_NL_JUDGE_KEY"
    ):
        make_runner(judge_config)


def test_configured_judge_missing_endpoint_rejected(judge_config):
    """The judge cannot borrow a user endpoint when its endpoint is absent."""
    with pytest.raises(ValueError, match="nl_judge_llm_base_url is required"):
        make_runner(replace(judge_config, nl_judge_llm_base_url=None))


def test_solo_judge_without_user_configuration_uses_own_credentials(judge_config):
    """Solo episodes can score NL assertions without configuring a simulator."""
    runner = make_runner(
        replace(judge_config, solo_mode=True, user_llm=None, user_llm_base_url=None)
    )
    with runner._nl_assertion_evaluation():
        assert evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_key"] == "judge-secret"


@pytest.mark.parametrize("configured", [True, False])
def test_real_evaluator_calls_are_counted_on_effective_transport(
    monkeypatch, judge_config, configured
):
    """Observe real tau2 generate-to-completion arguments, not a config echo."""
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return ModelResponse(
            model=kwargs["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "results": [
                                    {
                                        "expectedOutcome": "helped",
                                        "metExpectation": True,
                                        "reasoning": "yes",
                                    }
                                ]
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    runner = make_runner(
        judge_config if configured else replace(judge_config, nl_judge_llm=None)
    )
    saved = (
        evaluator.DEFAULT_LLM_NL_ASSERTIONS,
        evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        evaluator.generate,
    )
    with runner._nl_assertion_evaluation():
        for _ in range(2):
            checks = evaluator.NLAssertionsEvaluator.evaluate_nl_assertions(
                [], ["helped"]
            )
            assert len(checks) == 1 and checks[0].met
    assert len(calls) == 2
    expected_model = "openai/deepseek-flash" if configured else "openai/doubao-user"
    expected_host = "api.deepseek.com" if configured else "ark.invalid"
    assert all(call["model"] == expected_model for call in calls)
    assert all(
        call["api_key"] == ("judge-secret" if configured else "user-secret")
        for call in calls
    )
    evidence = runner.llm_route_evidence()
    assert evidence["nl_judge_llm_calls"] == 2
    assert all(
        call["model"] == expected_model
        and call["base_url_host"] == expected_host
        and call["status"] == "completed"
        for call in evidence["nl_judge_calls"]
    )
    assert "secret" not in json.dumps(evidence)
    assert saved == (
        evaluator.DEFAULT_LLM_NL_ASSERTIONS,
        evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        evaluator.generate,
    )


def test_failed_judge_call_counted_and_global_binding_restored(
    monkeypatch, judge_config
):
    """A failed judge call stays attributable; it cannot contaminate the next runner."""

    def fail(**kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(evaluator, "generate", fail)
    runner = make_runner(judge_config)
    saved_model = evaluator.DEFAULT_LLM_NL_ASSERTIONS
    with pytest.raises(RuntimeError, match="provider failed"):
        with runner._nl_assertion_evaluation():
            evaluator.NLAssertionsEvaluator.evaluate_nl_assertions([], ["helped"])
    assert evaluator.generate is fail
    assert evaluator.DEFAULT_LLM_NL_ASSERTIONS == saved_model
    assert runner.nl_judge_calls[0]["status"] == "error"
    assert runner.nl_judge_calls[0]["error_type"] == "RuntimeError"
    assert len(runner.nl_judge_calls) == 1


def test_scoring_without_judge_invocations_records_zero_calls(judge_config):
    """Binding alone does not claim a network call occurred."""
    runner = make_runner(judge_config)
    with runner._nl_assertion_evaluation():
        pass
    assert runner.llm_route_evidence()["nl_judge_llm_calls"] == 0
    assert runner.nl_judge_calls == []


def test_concurrent_runners_keep_judge_routes_and_counts_separate(
    monkeypatch, judge_config
):
    """Scoring in different worker threads cannot exchange judge routes."""
    monkeypatch.setattr(evaluator, "generate", lambda **kwargs: kwargs["model"])
    runners = [
        make_runner(replace(judge_config, nl_judge_llm=f"judge-{i}")) for i in range(2)
    ]

    def score(runner):
        with runner._nl_assertion_evaluation():
            return evaluator.generate(
                model=evaluator.DEFAULT_LLM_NL_ASSERTIONS,
                **evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(score, runners))
    assert results == ["openai/judge-0", "openai/judge-1"]
    assert [r.nl_judge_calls[0]["model"] for r in runners] == results
    assert all(len(r.nl_judge_calls) == 1 for r in runners)
