# SPDX-License-Identifier: Apache-2.0

import importlib.util
import time
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "areal/reward/math_prd.py"
SPEC = importlib.util.spec_from_file_location("math_prd_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
math_prd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math_prd)

MathPRDTimeoutError = math_prd.MathPRDTimeoutError
MathPRDDataContractError = math_prd.MathPRDDataContractError
math_prd_reward_fn = math_prd.math_prd_reward_fn
score_math_prd = math_prd.score_math_prd


@pytest.mark.parametrize(
    ("predicted", "answer", "expected"),
    [
        (" 42 ", "42", True),
        (r"\dfrac{1}{2}", r"\frac{1}{2}", True),
        (r"\tfrac{1}{2}", r"\frac{1}{2}", True),
        (r"\left(1\right)", "(1)", True),
        ("33", "033", False),
    ],
)
def test_legacy_strip_equivalence_regression_cases(predicted, answer, expected):
    assert math_prd._legacy_is_equiv(predicted, answer) is expected


@pytest.mark.parametrize(
    ("completion", "answer"),
    [
        (r"The final answer is \boxed{33}.", "033"),
        (r"The final answer is \boxed{43}.", "043"),
        (r"The final answer is \boxed{2/4}.", r"\frac{1}{2}"),
        (r"The final answer is \boxed{0.5}.", r"\frac{1}{2}"),
        (r"The final answer is \boxed{2\sqrt{2}}.", r"\sqrt{8}"),
        (r"The final answer is \boxed{even}.", r"\text{even}"),
        (r"The final answer is \fbox{42}.", "42"),
        (r"The final answer is \boxed{7}.", r"7\text{ hours}."),
    ],
)
def test_math_prd_reward_accepts_expected_semantic_matches(completion, answer):
    assert score_math_prd(completion, answer) == 1.0


@pytest.mark.parametrize(
    ("completion", "answer"),
    [
        ("Answer: 42", "42"),
        (r"First \boxed{42}, final \boxed{41}.", "42"),
        (r"First \boxed{42}, final \boxed{41.", "42"),
        (r"The final answer is \boxed{42 or 41}.", "42"),
        (r"The final answer is \boxed{X}.", "x"),
        (r"The final answer is \boxed{0.5}.", r"50\%"),
        (r"The final answer is \boxed{10}.", "10_2"),
        (r"The final answer is \boxed{10_3}.", "10_2"),
        (r"The final answer is \boxed{1,-2,-1/2}.", r"1,1,-2,-1/2"),
        (
            r"The final answer is \boxed{\frac{\sum_{k=1}^{7671}\varphi(k)-1}{2}}.",
            "14711060",
        ),
    ],
)
def test_math_prd_reward_rejects_false_positives(completion, answer):
    assert score_math_prd(completion, answer) == 0.0


def test_math_prd_reward_function_uses_rlvr_signature():
    assert (
        math_prd_reward_fn(
            prompt="problem",
            completions=r"The final answer is \boxed{43}.",
            prompt_ids=[],
            completion_ids=[],
            answer="043",
            source_id="case-1",
            benchmark="aime24",
        )
        == 1.0
    )


def test_math_prd_reward_missing_answer_is_data_contract_error():
    with pytest.raises(MathPRDDataContractError):
        math_prd_reward_fn(
            prompt="problem",
            completions=r"The final answer is \boxed{43}.",
            prompt_ids=[],
            completion_ids=[],
            answer=None,
        )


def test_math_prd_timeout_is_classified(monkeypatch):
    def stuck_verify(*args, **kwargs):
        time.sleep(10)

    monkeypatch.setattr(math_prd.math_prd_worker, "verify_semantic_boxed", stuck_verify)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_RETRY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_TERMINATE_GRACE_SECONDS", 0.2)
    start = time.monotonic()
    with pytest.raises(MathPRDTimeoutError):
        score_math_prd(r"The final answer is \boxed{a+b}.", "b+a")
    assert time.monotonic() - start < 1.0


class TimeoutException(Exception):
    pass


TimeoutException.__module__ = "math_verify.errors"


def _read_counter(path):
    if not path.exists():
        return 0
    return int(path.read_text())


def _write_counter(path, value):
    path.write_text(str(value))


def _patch_semantic_timeouts(monkeypatch, *, initial=0.5, retry=0.5):
    monkeypatch.setattr(math_prd, "find_spec", lambda name: object())
    monkeypatch.setattr(math_prd, "MATH_VERIFY_TIMEOUT_SECONDS", initial)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_RETRY_TIMEOUT_SECONDS", retry)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_TERMINATE_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_PARSE_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_VERIFY_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_RETRY_PARSE_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr(math_prd, "MATH_VERIFY_RETRY_VERIFY_TIMEOUT_SECONDS", 4)


def test_math_prd_transient_timeout_retries_same_inputs_with_larger_budget(
    monkeypatch, tmp_path
):
    calls_file = tmp_path / "calls.txt"
    seen_file = tmp_path / "seen.txt"

    def flaky_verify(ground_truth_boxed, answer_boxed, parse_timeout, verify_timeout):
        call_index = _read_counter(calls_file)
        _write_counter(calls_file, call_index + 1)
        with seen_file.open("a") as handle:
            handle.write(
                f"{call_index}|{ground_truth_boxed}|{answer_boxed}|"
                f"{parse_timeout}|{verify_timeout}\n"
            )
        if call_index == 0:
            raise TimeoutException("internal budget expired")
        return True

    _patch_semantic_timeouts(monkeypatch)
    monkeypatch.setattr(math_prd.math_prd_worker, "verify_semantic_boxed", flaky_verify)

    assert score_math_prd(r"The final answer is \boxed{a+b}.", "b+a") == 1.0

    assert _read_counter(calls_file) == 2
    lines = seen_file.read_text().splitlines()
    assert lines == [
        r"0|\boxed{b+a}|\boxed{a+b}|1|1",
        r"1|\boxed{b+a}|\boxed{a+b}|3|4",
    ]


def test_math_prd_permanent_timeout_raises_and_cleans_processes(monkeypatch):
    def stuck_verify(*args, **kwargs):
        time.sleep(10)

    _patch_semantic_timeouts(monkeypatch, initial=0.05, retry=0.05)
    monkeypatch.setattr(math_prd.math_prd_worker, "verify_semantic_boxed", stuck_verify)

    start = time.monotonic()
    with pytest.raises(MathPRDTimeoutError) as exc_info:
        score_math_prd(r"The final answer is \boxed{a+b}.", "b+a")
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    message = str(exc_info.value)
    assert "timed out after retry" in message
    assert "gold_sha256=" in message
    assert "pred_sha256=" in message
    assert "gold_preview=" in message
    assert "pred_preview=" in message
    assert "boxed{b+a}" in message
    assert "boxed{a+b}" in message
    assert not [
        child
        for child in math_prd.multiprocessing.active_children()
        if child.is_alive()
    ]


def test_math_prd_persistent_worker_error_includes_bounded_diagnostics(monkeypatch):
    def failing_verify(*args, **kwargs):
        raise ValueError("synthetic parser failure")

    _patch_semantic_timeouts(monkeypatch)
    monkeypatch.setattr(
        math_prd.math_prd_worker, "verify_semantic_boxed", failing_verify
    )

    with pytest.raises(math_prd.MathPRDRewardError) as exc_info:
        score_math_prd(r"The final answer is \boxed{a+b}.", "b+a")

    message = str(exc_info.value)
    assert "synthetic parser failure" in message
    assert "gold_sha256=" in message
    assert "pred_sha256=" in message
    assert "gold_preview=" in message
    assert "pred_preview=" in message
    assert "boxed{b+a}" in message
    assert "boxed{a+b}" in message
