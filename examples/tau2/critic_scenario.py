# SPDX-License-Identifier: Apache-2.0
"""Scenario-only critic context; never render the task or its evaluator fields."""

from typing import Any

import torch
from tau2.data_model.tasks import StructuredUserInstructions, UserScenario

from areal.utils import stats_tracker


def render_critic_profile(task: Any) -> str:
    """Validate the simulator's scenario field by field before calling its str().

    Evaluator actions/assertions may legitimately mention the same order IDs or
    intent as the scenario. Isolation is structural, not a substring blacklist.
    Reject added fields (including model_copy/model_construct bypasses) rather
    than risk rendering a future schema extension without review.
    """
    scenario = getattr(task, "user_scenario", None)
    if not isinstance(scenario, UserScenario):
        raise ValueError("critic user scenario is missing or invalid")
    if set(scenario.__dict__) != {"persona", "instructions"} or scenario.model_extra:
        raise ValueError("Unexpected user scenario fields")
    if scenario.persona is not None and not isinstance(scenario.persona, str):
        raise ValueError("Invalid user scenario persona")
    instructions = scenario.instructions
    if isinstance(instructions, StructuredUserInstructions):
        allowed = {
            "domain",
            "reason_for_call",
            "known_info",
            "unknown_info",
            "task_instructions",
        }
        if set(instructions.__dict__) != allowed or instructions.model_extra:
            raise ValueError("Unexpected user scenario instruction fields")
        for name in allowed:
            value = getattr(instructions, name)
            if not isinstance(value, str) and not (
                name in {"known_info", "unknown_info"} and value is None
            ):
                raise ValueError(f"Invalid user scenario instruction field: {name}")
        if (
            not instructions.reason_for_call.strip()
            or not instructions.task_instructions.strip()
        ):
            raise ValueError("Empty user scenario instructions")
    elif not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Missing user scenario instructions")
    return (
        "<|im_start|>system\n<user_profile>\n"
        + str(scenario)
        + "\n</user_profile><|im_end|>\n"
    )


def critic_scenario_metadata(task: Any, tokenizer: Any) -> dict[str, list[int]]:
    """Tokenize using the pinned actor tokenizer; a missing scenario is fatal."""
    try:
        profile = render_critic_profile(task)
    except ValueError:
        # Tokenization runs in to_thread; use a named tracker rather than a
        # shared mutable scope stack across concurrently failing episodes.
        stats_tracker.get("critic_scenario").denominator(
            missing_scenario=torch.tensor([True])
        )
        raise
    ids = tokenizer.encode(profile, add_special_tokens=False)
    if not ids:
        raise ValueError("Empty critic scenario tokenization")
    return {"critic_prefix_input_ids": ids}
