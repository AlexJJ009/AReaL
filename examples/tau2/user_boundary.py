"""Classify policy boundaries from the official trajectory, before rendering."""

import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _field(message: Any, name: str, default: Any = None) -> Any:
    return (
        message.get(name, default)
        if isinstance(message, Mapping)
        else getattr(message, name, default)
    )


def classify_user_boundaries(messages: Sequence[Any]) -> list[int]:
    """Return one code per policy completion: user=1, tool=2, final user=3, other=4.

    The fixed greeting has no raw_data and is not a policy completion. Tool
    results retain their official role even when a chat template renders them
    under a user header. Final user messages need not appear in policy tensors.
    """
    policy_indices = [
        i
        for i, message in enumerate(messages)
        if _field(message, "role") == "assistant"
        and _field(message, "raw_data") is not None
    ]
    result = []
    for ordinal, index in enumerate(policy_indices):
        has_next = ordinal + 1 < len(policy_indices)
        next_index = policy_indices[ordinal + 1] if has_next else len(messages)
        following = messages[index + 1 : next_index]
        if following and _field(following[0], "role") == "user":
            result.append(1 if has_next else 3)
        elif (
            following
            and _field(following[0], "role") == "tool"
            and _field(following[0], "requestor") == "assistant"
        ):
            result.append(2)
        elif has_next:
            raise ValueError(
                "Policy boundary has no official user or assistant-tool event"
            )
        else:
            result.append(4)
    return result


@dataclass(frozen=True)
class UserBoundaryContext:
    """Official history through this boundary's user message (no future events)."""

    messages: Sequence[Any]
    boundary_type: int
    stop_reason: str | None


@dataclass(frozen=True)
class BoundaryDetection:
    """A detector's decision and stable provenance name, or None when unflagged."""

    flag: bool
    detector: str | None = None


UserBoundaryDetector = Callable[
    [UserBoundaryContext], BoundaryDetection | Awaitable[BoundaryDetection]
]


# Source: user-sim-rerun-20261018/analysis/lib.py, is_u1_rule and its
# dependencies, under _artifacts/AReaL/codex-sao-math. Keep the original regexes
# and decision logic verbatim; normalization below supplies its indexed schema.
AFFIRM = re.compile(
    r"\b(yes|yeah|yep|yup|confirm(ed|s)?|go ahead|go for it|proceed|please do|that'?s correct|that is correct|sounds good|approve[ds]?|authori[sz]e[ds]?|i agree|let'?s do it|do it|that works|please (process|cancel|exchange|return|book|update|modify|make|change|send|upgrade))\b",
    re.I,
)
ASK = re.compile(
    r"(do you confirm|please confirm|can you confirm|could you confirm|confirm (that|if|whether|you)|to confirm|your confirmation|(please|just) (say|respond|reply|type|answer)|shall i|should i|would you like (me )?to|do you want (me )?to|do you (wish|authori[sz]e)|proceed\?|go ahead\?|ready to proceed|okay to|ok to|\(yes|yes/no|yes to|reply yes|say yes|before i (proceed|process|go ahead)|once you confirm|if you confirm|if you.{0,20}(agree|approve))",
    re.I,
)
DONE = re.compile(
    r"(anything else|is there anything|has been (successfully )?(processed|completed|cancel|updated|modified|booked|exchanged|returned|changed|placed))",
    re.I,
)
STRONGASK = re.compile(
    r"(confirm|proceed|go ahead|shall i|would you like me|do you want me|yes)", re.I
)


def final_user_idx(e: dict[str, Any]) -> int | None:
    for m in reversed(e["msgs"]):
        if m["role"] == "user":
            return m["i"]


def is_u1_rule(e: dict[str, Any]) -> dict[str, int] | None:
    if e["stop"] != "user_stop":
        return None
    fi = final_user_idx(e)
    if fi is None or fi == 0:
        return None
    t = e["msgs"][fi]["content"] or ""
    if "###STOP###" not in t or "###TRANSFER###" in t or "###OUT-OF-SCOPE###" in t:
        return None
    prev = e["msgs"][fi - 1]
    if prev["role"] != "assistant" or prev["tc"]:
        return None
    if not AFFIRM.search(t.replace("###STOP###", "")):
        return None
    a = prev["content"] or ""
    tail = a[-500:]
    if not ASK.search(tail):
        return None
    if DONE.search(a[-250:]) and not STRONGASK.search(tail):
        return None
    return dict(fi=fi)


class RuleUserBoundaryDetector:
    """U1 confirmation+STOP, using the faithful offline rule port."""

    def __call__(self, context: UserBoundaryContext) -> BoundaryDetection:
        messages = [
            {
                "i": i,
                "role": _field(message, "role"),
                "content": _field(message, "content"),
                "tc": _field(message, "tool_calls") or [],
            }
            for i, message in enumerate(context.messages)
        ]
        if is_u1_rule({"stop": context.stop_reason, "msgs": messages}):
            return BoundaryDetection(True, "U1")
        return BoundaryDetection(False)


async def detect_user_boundaries(
    messages: Sequence[Any],
    stop_reason: str | None,
    detector: UserBoundaryDetector,
    *,
    final_reward: float,
) -> dict[str, Any]:
    """Call a sync/async detector once per official user boundary after an episode.

    Flags require the final training reward < 1, after fixed-variant scoring.
    Evaluate candidates even on success so suppressed hits remain observable.
    The same gate applies to every sync/async callable. Tool boundaries never
    trigger a detector call.
    """
    from areal.trainer.ppo.user_boundary import USER_BOUNDARY_DETECTOR_NAMES

    kinds = classify_user_boundaries(messages)
    policy_indices = [
        i
        for i, message in enumerate(messages)
        if _field(message, "role") == "assistant"
        and _field(message, "raw_data") is not None
    ]
    flags, names = [], []
    failed = final_reward < 1
    gate_suppressed = False
    for index, kind in zip(policy_indices, kinds, strict=True):
        detection = BoundaryDetection(False)
        if kind in (1, 3):
            context = UserBoundaryContext(
                messages=messages[: index + 2],
                boundary_type=kind,
                stop_reason=stop_reason if kind == 3 else None,
            )
            detection = detector(context)
            if inspect.isawaitable(detection):
                detection = await detection
            if (
                not isinstance(detection, BoundaryDetection)
                or not isinstance(detection.flag, bool)
                or detection.detector not in USER_BOUNDARY_DETECTOR_NAMES
                or detection.flag != (detection.detector is not None)
            ):
                raise ValueError("Invalid user boundary detector result")
        if detection.flag and not failed:
            gate_suppressed = True
            detection = BoundaryDetection(False)
        flags.append(detection.flag)
        names.append(detection.detector)
    return {
        "user_boundary_types": kinds,
        "user_boundary_flags": flags,
        "user_boundary_detectors": names,
        "user_boundary_gate_suppressed": gate_suppressed,
    }
