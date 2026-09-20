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
    monkeypatch.setattr(math_prd, "MATH_VERIFY_TERMINATE_GRACE_SECONDS", 0.2)
    start = time.monotonic()
    with pytest.raises(MathPRDTimeoutError):
        score_math_prd(r"The final answer is \boxed{2/4}.", r"\frac{1}{2}")
    assert time.monotonic() - start < 1.0
