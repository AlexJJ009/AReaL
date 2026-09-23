# SPDX-License-Identifier: Apache-2.0

"""Standalone PRD-style semantic boxed-answer math reward."""

from __future__ import annotations

import hashlib
import importlib.util
import multiprocessing
import queue
import re
import sys
from collections import Counter
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import Any, NamedTuple

try:
    from areal.reward import math_prd_worker
except Exception:  # pragma: no cover - used by file-level smoke tests in partial envs.
    worker_path = Path(__file__).with_name("math_prd_worker.py")
    spec = importlib.util.spec_from_file_location("math_prd_worker", worker_path)
    if spec is None or spec.loader is None:
        raise
    math_prd_worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = math_prd_worker
    spec.loader.exec_module(math_prd_worker)

MATH_PRD_REWARD_VERSION = "math-prd-semantic-boxed-v2-standalone"
MATH_VERIFY_TIMEOUT_SECONDS = 30.0
MATH_VERIFY_TERMINATE_GRACE_SECONDS = 1.0
MATH_VERIFY_PARSE_TIMEOUT_SECONDS = 5
MATH_VERIFY_VERIFY_TIMEOUT_SECONDS = 5
MATH_VERIFY_RETRY_TIMEOUT_SECONDS = 60.0
MATH_VERIFY_RETRY_PARSE_TIMEOUT_SECONDS = 15
MATH_VERIFY_RETRY_VERIFY_TIMEOUT_SECONDS = 15
MATH_VERIFY_DIAGNOSTIC_PREVIEW_CHARS = 160

_BOX_COMMANDS = ("\\boxed", "\\fbox")
_BASE_SUBSCRIPT_RE = re.compile(r"(?<![A-Za-z\\])\d+_\{?\d+\}?")
_NUMERIC_BASE_LITERAL_RE = re.compile(
    r"^\s*(?P<digits>\d+)_(?:\{(?P<braced_base>\d+)\}|(?P<plain_base>\d+))\s*$"
)
_CLOCK_LITERAL_RE = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})$")
_TEXT_COMMAND_RE = re.compile(r"^\\text\{(?P<text>.*)\}$", re.DOTALL)
_OR_GUESS_RE = re.compile(
    r"(?:^|[\s{},])(?:or|\\text\{\s*or\s*\})(?:$|[\s{},])", re.IGNORECASE
)
_THOUSANDS_NUMBER_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


class BoxStatus(Enum):
    ABSENT = "absent"
    COMPLETE = "complete"
    MALFORMED = "malformed"


class BoxExtraction(NamedTuple):
    status: BoxStatus
    answer: str | None = None


class MathPRDRewardError(RuntimeError):
    """Base class for reward infrastructure errors."""


class MathPRDTimeoutError(MathPRDRewardError):
    """Raised when semantic verification times out."""


class MathPRDDataContractError(MathPRDRewardError):
    """Raised when required reward data is absent or malformed."""


def _last_box_command_index(text: str) -> int:
    return max(text.rfind(command) for command in _BOX_COMMANDS)


def extract_last_complete_boxed_answer(text: str) -> BoxExtraction:
    command_idx = _last_box_command_index(text)
    if command_idx < 0:
        return BoxExtraction(BoxStatus.ABSENT)

    cursor = command_idx
    command = None
    for candidate in _BOX_COMMANDS:
        if text.startswith(candidate, cursor):
            command = candidate
            break
    if command is None:
        return BoxExtraction(BoxStatus.MALFORMED)

    cursor += len(command)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return BoxExtraction(BoxStatus.MALFORMED)

    cursor += 1
    answer_start = cursor
    depth = 1
    escaped = False
    while cursor < len(text):
        char = text[cursor]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return BoxExtraction(BoxStatus.COMPLETE, text[answer_start:cursor])
        cursor += 1
    return BoxExtraction(BoxStatus.MALFORMED)


def _boxed(content: str) -> str:
    return "\\boxed{" + content + "}"


def _strip_string(value: str) -> str:
    # Preserve the previous PRD scorer's Apache-2.0 normalization from
    # verl/utils/reward_score/math_reward.py (EleutherAI MATH normalization).
    text = value.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    text = text.replace("dfrac", "frac").replace("tfrac", "frac")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "")
    if "\\text{ " in text:
        parts = text.split("\\text{ ")
        if len(parts) != 2:
            raise ValueError("Ambiguous trailing unit text")
        text = parts[0]
    text = text.replace("\\\\%", "").replace("\\%", "")
    text = text.replace(" .", " 0.").replace("{.", "{0.")
    if not text:
        return text
    if text[0] == ".":
        text = "0" + text
    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=")[1]
    parts = text.split("\\sqrt")
    text = parts[0]
    for part in parts[1:]:
        text += "\\sqrt" + (part if part[0] == "{" else "{" + part[0] + "}" + part[1:])
    text = text.replace(" ", "")
    parts = text.split("\\frac")
    fixed = parts[0]
    for part in parts[1:]:
        fixed += "\\frac"
        if part[0] == "{":
            fixed += part
        elif len(part) < 2:
            fixed = text
            break
        elif part[1] != "{":
            fixed += "{" + part[0] + "}{" + part[1] + "}" + part[2:]
        else:
            fixed += "{" + part[0] + "}" + part[1:]
    text = fixed
    if text == "0.5":
        text = "\\frac{1}{2}"
    if text.count("/") == 1:
        a, b = text.split("/")
        try:
            a, b = int(a), int(b)
            if text == f"{a}/{b}":
                text = f"\\frac{{{a}}}{{{b}}}"
        except ValueError:
            pass
    return text


def _legacy_is_equiv(answer: str, ground_truth: str) -> bool:
    try:
        return _strip_string(answer) == _strip_string(ground_truth)
    except (ValueError, IndexError):
        return answer == ground_truth


def _strip_text_command(value: str) -> str:
    stripped = value.strip()
    match = _TEXT_COMMAND_RE.match(stripped)
    if match:
        stripped = match.group("text")
    return " ".join(stripped.strip().lower().split())


def _has_text_command(value: str) -> bool:
    return bool(_TEXT_COMMAND_RE.match(value.strip()))


def _is_safe_text_answer(value: str) -> bool:
    normalized = _strip_text_command(value)
    compact = normalized.replace(" ", "")
    return len(compact) >= 2 and bool(re.fullmatch(r"[a-z]+(?: [a-z]+)*", normalized))


def _text_answers_match(ground_truth: str, answer: str) -> bool:
    if not (_has_text_command(ground_truth) or _has_text_command(answer)):
        return False
    if not (_is_safe_text_answer(ground_truth) and _is_safe_text_answer(answer)):
        return False
    return _strip_text_command(ground_truth) == _strip_text_command(answer)


def _has_percent(value: str) -> bool:
    return "%" in value or "\\percent" in value


def _has_base_subscript(value: str) -> bool:
    return bool(_BASE_SUBSCRIPT_RE.search(value))


def _numeric_base_literal(value: str) -> tuple[str, str] | None:
    match = _NUMERIC_BASE_LITERAL_RE.fullmatch(value)
    if match is None:
        return None
    return match.group("digits"), match.group("braced_base") or match.group(
        "plain_base"
    )


def _score_numeric_base_literals_if_applicable(
    ground_truth: str, answer: str
) -> float | None:
    if not (_has_base_subscript(ground_truth) or _has_base_subscript(answer)):
        return None
    gold_literal = _numeric_base_literal(ground_truth)
    answer_literal = _numeric_base_literal(answer)
    if gold_literal is None or answer_literal is None:
        return 0.0
    return 1.0 if gold_literal == answer_literal else 0.0


def _clock_literal(value: str) -> tuple[int, int] | None:
    normalized = value.replace("\\!", "").replace(" ", "").strip()
    match = _CLOCK_LITERAL_RE.fullmatch(normalized)
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if minute >= 60:
        return None
    return hour, minute


def _score_clock_literals_if_applicable(ground_truth: str, answer: str) -> float | None:
    gold_clock = _clock_literal(ground_truth)
    answer_clock = _clock_literal(answer)
    if gold_clock is None and answer_clock is None:
        return None
    if gold_clock is None or answer_clock is None:
        return 0.0
    return 1.0 if gold_clock == answer_clock else 0.0


def _looks_like_or_guess(answer: str) -> bool:
    return bool(_OR_GUESS_RE.search(answer))


def _is_single_letter_symbol(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]", value.strip()))


def _top_level_comma_items(value: str) -> list[str] | None:
    stripped = value.strip()
    if _THOUSANDS_NUMBER_RE.fullmatch(stripped):
        return None
    items = []
    start = 0
    depth = 0
    escaped = False
    saw_comma = False
    for idx, char in enumerate(stripped):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char in "{[(":
            depth += 1
        elif char in "}])" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            saw_comma = True
            items.append(stripped[start:idx].strip())
            start = idx + 1
    if not saw_comma:
        return None
    items.append(stripped[start:].strip())
    if any(not item for item in items):
        return None
    return items


def _normalized_item_multiset(items: list[str]) -> Counter:
    return Counter(_strip_string(item) for item in items)


def _score_duplicate_gold_list_if_applicable(
    ground_truth: str, answer: str
) -> float | None:
    gold_items = _top_level_comma_items(ground_truth)
    if gold_items is None:
        return None
    gold_multiset = _normalized_item_multiset(gold_items)
    if all(count == 1 for count in gold_multiset.values()):
        return None
    answer_items = _top_level_comma_items(answer)
    if answer_items is None:
        return 0.0
    return 1.0 if _normalized_item_multiset(answer_items) == gold_multiset else 0.0


def _semantic_rescue_allowed(ground_truth: str, answer: str) -> bool:
    if _has_percent(ground_truth) != _has_percent(answer):
        return False
    if _has_base_subscript(ground_truth) or _has_base_subscript(answer):
        return False
    return True


def _verify_semantic_boxed_child(
    result_queue,
    ground_truth_boxed: str,
    answer_boxed: str,
    parse_timeout: int,
    verify_timeout: int,
) -> None:
    try:
        result = math_prd_worker.verify_semantic_boxed(
            ground_truth_boxed,
            answer_boxed,
            parse_timeout,
            verify_timeout,
        )
        result_queue.put(("ok", bool(result)))
    except (
        BaseException
    ) as exc:  # pragma: no cover - exercised through parent classification.
        result_queue.put(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "module": type(exc).__module__,
                    "message": str(exc),
                },
            )
        )


def _multiprocessing_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - non-Unix fallback.
        return multiprocessing.get_context()


def _semantic_verify_diagnostic(ground_truth: str, answer: str) -> str:
    ground_truth_boxed = _boxed(ground_truth)
    answer_boxed = _boxed(answer)
    return (
        f"gold_len={len(ground_truth_boxed)} "
        f"pred_len={len(answer_boxed)} "
        f"gold_sha256={hashlib.sha256(ground_truth_boxed.encode()).hexdigest()} "
        f"pred_sha256={hashlib.sha256(answer_boxed.encode()).hexdigest()} "
        f"gold_preview={ground_truth_boxed[:MATH_VERIFY_DIAGNOSTIC_PREVIEW_CHARS]!r} "
        f"pred_preview={answer_boxed[:MATH_VERIFY_DIAGNOSTIC_PREVIEW_CHARS]!r}"
    )


def _with_semantic_verify_diagnostic(
    message: str, ground_truth: str, answer: str
) -> str:
    return f"{message}; {_semantic_verify_diagnostic(ground_truth, answer)}"


def _is_math_verify_timeout_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("module") == "math_verify.errors"
        and payload.get("type") == "TimeoutException"
    )


def _run_math_verify_equiv_once(
    ground_truth: str,
    answer: str,
    *,
    parse_timeout: int,
    verify_timeout: int,
    process_timeout: float,
) -> bool:
    ctx = _multiprocessing_context()
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_verify_semantic_boxed_child,
        args=(
            result_queue,
            _boxed(ground_truth),
            _boxed(answer),
            parse_timeout,
            verify_timeout,
        ),
    )
    process.start()
    process.join(process_timeout)
    if process.is_alive():
        process.terminate()
        process.join(MATH_VERIFY_TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join(MATH_VERIFY_TERMINATE_GRACE_SECONDS)
        result_queue.close()
        result_queue.join_thread()
        raise MathPRDTimeoutError(
            f"MATH PRD semantic reward timed out after {process_timeout} seconds"
        )

    queue_closed = False
    try:
        status, payload = result_queue.get(timeout=MATH_VERIFY_TERMINATE_GRACE_SECONDS)
    except queue.Empty as exc:
        result_queue.close()
        result_queue.join_thread()
        queue_closed = True
        raise MathPRDRewardError(
            "MATH PRD semantic reward subprocess exited without a result: "
            f"exitcode={process.exitcode}"
        ) from exc
    finally:
        if not queue_closed:
            result_queue.close()
            result_queue.join_thread()

    if status == "ok":
        return bool(payload)
    if _is_math_verify_timeout_payload(payload):
        raise MathPRDTimeoutError(
            "MATH PRD semantic reward timed out inside math_verify"
        )
    raise MathPRDRewardError(
        f"MATH PRD semantic reward failed inside subprocess: {payload}"
    )


def _math_verify_equiv(ground_truth: str, answer: str) -> bool:
    if find_spec("math_verify") is None:
        raise MathPRDRewardError("math_verify is required for PRD semantic math reward")
    try:
        return _run_math_verify_equiv_once(
            ground_truth,
            answer,
            parse_timeout=MATH_VERIFY_PARSE_TIMEOUT_SECONDS,
            verify_timeout=MATH_VERIFY_VERIFY_TIMEOUT_SECONDS,
            process_timeout=MATH_VERIFY_TIMEOUT_SECONDS,
        )
    except MathPRDTimeoutError as first_timeout:
        try:
            return _run_math_verify_equiv_once(
                ground_truth,
                answer,
                parse_timeout=MATH_VERIFY_RETRY_PARSE_TIMEOUT_SECONDS,
                verify_timeout=MATH_VERIFY_RETRY_VERIFY_TIMEOUT_SECONDS,
                process_timeout=MATH_VERIFY_RETRY_TIMEOUT_SECONDS,
            )
        except MathPRDTimeoutError as retry_timeout:
            raise MathPRDTimeoutError(
                _with_semantic_verify_diagnostic(
                    "MATH PRD semantic reward timed out after retry "
                    f"(first={first_timeout}; retry={retry_timeout})",
                    ground_truth,
                    answer,
                )
            ) from retry_timeout
        except MathPRDRewardError as retry_error:
            raise MathPRDRewardError(
                _with_semantic_verify_diagnostic(
                    "MATH PRD semantic reward failed after timeout retry: "
                    f"{retry_error}",
                    ground_truth,
                    answer,
                )
            ) from retry_error
    except MathPRDRewardError as exc:
        raise MathPRDRewardError(
            _with_semantic_verify_diagnostic(str(exc), ground_truth, answer)
        ) from exc


def score_math_prd(completion: str, answer: str) -> float:
    extraction = extract_last_complete_boxed_answer(completion)
    if extraction.status is not BoxStatus.COMPLETE or extraction.answer is None:
        return 0.0
    predicted = extraction.answer
    if not str(answer).strip() or not predicted.strip():
        return 0.0
    if _looks_like_or_guess(predicted):
        return 0.0
    if _legacy_is_equiv(predicted, str(answer)):
        return 1.0
    duplicate_list_score = _score_duplicate_gold_list_if_applicable(
        str(answer), predicted
    )
    if duplicate_list_score is not None:
        return duplicate_list_score
    if _is_single_letter_symbol(str(answer)) and _is_single_letter_symbol(predicted):
        return 0.0
    if _text_answers_match(str(answer), predicted):
        return 1.0
    clock_score = _score_clock_literals_if_applicable(str(answer), predicted)
    if clock_score is not None:
        return clock_score
    base_literal_score = _score_numeric_base_literals_if_applicable(
        str(answer), predicted
    )
    if base_literal_score is not None:
        return base_literal_score
    if not _semantic_rescue_allowed(str(answer), predicted):
        return 0.0
    return 1.0 if _math_verify_equiv(str(answer), predicted) else 0.0


def math_prd_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids,
    completion_ids,
    answer: str | None = None,
    **kwargs: Any,
) -> float:
    del prompt, prompt_ids, completion_ids, kwargs
    if answer is None:
        raise MathPRDDataContractError("math_prd_reward_fn requires a non-None answer")
    return score_math_prd(str(completions), str(answer))
