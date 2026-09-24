# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn

_MANIFEST_VERSION = 1
_ATTENTION_MODULE_NAMES = ("self_attn", "linear_attn")


def _module_path_matches(name: str, target: str) -> bool:
    return target in name.split(".")


def _matched_parameter_names(
    model: nn.Module, module_names: Iterable[str]
) -> tuple[dict[str, list[str]], list[str]]:
    names_by_type = {module_name: [] for module_name in module_names}
    frozen_names: set[str] = set()

    for module_name, module in model.named_modules():
        for target in module_names:
            if _module_path_matches(module_name, target):
                param_names = [
                    name for name, _ in module.named_parameters(prefix=module_name)
                ]
                names_by_type[target].extend(param_names)
                frozen_names.update(param_names)

    return names_by_type, sorted(frozen_names)


def _trainable_numel(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def apply_critic_attention_freeze(
    model: nn.Module, *, enabled: bool, is_critic: bool
) -> dict[str, Any]:
    """Freeze Qwen3.5 hybrid attention modules for critic training."""
    if not enabled:
        return {
            "version": _MANIFEST_VERSION,
            "enabled": False,
            "frozen_parameters": [],
            "frozen_numel": 0,
            "trainable_numel": _trainable_numel(model),
        }

    if not is_critic:
        raise ValueError("critic attention freeze can only be enabled for critics")

    names_by_type, frozen_names = _matched_parameter_names(
        model, _ATTENTION_MODULE_NAMES
    )
    missing = [
        module_name
        for module_name, param_names in names_by_type.items()
        if not param_names
    ]
    if missing:
        raise ValueError(
            "critic attention freeze requires both self_attn and linear_attn "
            f"parameters; missing {', '.join(missing)}"
        )

    frozen_name_set = set(frozen_names)
    frozen_numel = 0
    for name, param in model.named_parameters():
        if name in frozen_name_set:
            param.requires_grad_(False)
            frozen_numel += param.numel()

    return {
        "version": _MANIFEST_VERSION,
        "enabled": True,
        "frozen_parameters": frozen_names,
        "frozen_numel": frozen_numel,
        "trainable_numel": _trainable_numel(model),
    }


def validate_critic_freeze_manifest(
    expected: dict[str, Any], saved: dict[str, Any] | None
) -> None:
    """Reject optimizer resume when critic-freeze topology has changed."""
    if saved is None:
        if not expected.get("enabled", False):
            return
        raise ValueError("checkpoint is missing critic attention freeze manifest")

    expected_policy = {
        "enabled": expected.get("enabled"),
        "frozen_parameters": expected.get("frozen_parameters"),
    }
    saved_policy = {
        "enabled": saved.get("enabled"),
        "frozen_parameters": saved.get("frozen_parameters"),
    }
    if expected_policy != saved_policy:
        raise ValueError(
            "critic attention freeze manifest mismatch: "
            f"expected enabled={expected.get('enabled')} "
            f"frozen_parameters={expected.get('frozen_parameters')}, "
            f"saved enabled={saved.get('enabled')} "
            f"frozen_parameters={saved.get('frozen_parameters')}"
        )
