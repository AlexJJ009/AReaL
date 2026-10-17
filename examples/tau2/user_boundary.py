"""Classify policy boundaries from the official trajectory, before rendering."""

from collections.abc import Mapping, Sequence
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
