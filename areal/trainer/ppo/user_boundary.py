# SPDX-License-Identifier: Apache-2.0
"""User-boundary critic eligibility and fresh-batch diagnostics.

Types are aligned to the generated segment's last token b (not the actor slot
b-1): 0=no boundary, 1=user, 2=tool, 3=terminal user, 4=other ending.
"""

from contextlib import nullcontext
from typing import Any

import torch

from areal.utils import stats_tracker
from areal.utils.stats_tracker import ReduceType

# Stable transport IDs: 2=U7a remains reserved for historical dumps.
# Training selects U1 only; gold/judge names support injected callables.
USER_BOUNDARY_DETECTOR_NAMES = (None, "U1", "U7a", "gold", "judge")


def validate_boundary_beta(beta: Any, actor_mask: torch.Tensor, name: str) -> None:
    if (
        not isinstance(beta, torch.Tensor)
        or beta.shape != (actor_mask.shape[0],)
        or beta.device != actor_mask.device
        or not torch.is_floating_point(beta)
    ):
        raise ValueError(f"{name} must be floating with shape (batch_size,)")
    torch._assert_async(
        torch.all(torch.isfinite(beta) & (beta >= 0) & (beta <= 1)),
        f"{name} must be finite and in [0, 1]",
    )


def user_boundary_mask(data: dict[str, Any], actor_mask: torch.Tensor) -> torch.Tensor:
    """Validate opt-in transport before admitting extra post-agent critic states."""
    types = data["user_boundary_types"]
    beta = data.get("user_td_beta")
    if (
        not isinstance(types, torch.Tensor)
        or types.shape != actor_mask.shape
        or types.device != actor_mask.device
        or types.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("user_boundary_types must be token-aligned integer types")
    flagged_mode = any(
        key in data
        for key in (
            "user_boundary_flags",
            "user_td_beta_flagged",
            "user_boundary_detector_ids",
        )
    )
    if flagged_mode:
        if "user_td_beta" in data:
            raise ValueError(
                "user_td_beta and flagged boundary beta are mutually exclusive"
            )
        flags = data.get("user_boundary_flags")
        ids = data.get("user_boundary_detector_ids")
        for name, tensor, dtypes in (
            ("user_boundary_flags", flags, (torch.bool,)),
            ("user_boundary_detector_ids", ids, (torch.int32, torch.int64)),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != types.shape
                or tensor.device != types.device
                or tensor.dtype not in dtypes
            ):
                raise ValueError(
                    f"{name} must be token-aligned with the correct dtype/device"
                )
        torch._assert_async(
            torch.all(~flags | ((types == 1) | (types == 3))),
            "Only user boundaries may be flagged",
        )
        torch._assert_async(
            torch.all(
                (ids >= 0)
                & (ids < len(USER_BOUNDARY_DETECTOR_NAMES))
                & ((ids != 0) == flags)
            ),
            "Invalid user boundary detector IDs/flags",
        )
        validate_boundary_beta(
            data.get("user_td_beta_flagged"), actor_mask, "user_td_beta_flagged"
        )
    else:
        validate_boundary_beta(beta, actor_mask, "user_td_beta")
    torch._assert_async(torch.all((types >= 0) & (types <= 4)), "Invalid boundary type")
    previous_action = actor_mask.bool().roll(1, -1)
    previous_action[:, 0] = False
    torch._assert_async(
        torch.all(
            (types == 0)
            | (previous_action & ~actor_mask.bool() & data["attention_mask"].bool())
        ),
        "Boundaries must mark generated segment endpoints b after active actor slots",
    )
    return (types == 1) | (types == 3)


def critic_loss_mask(data: dict[str, Any]) -> torch.Tensor:
    """Keep actor eligibility separate in loss, microbatch weights and diagnostics."""
    return data.get("critic_loss_mask", data["loss_mask"])


def log_user_boundary_stats(data: dict[str, Any]) -> None:
    """Log initial full critic values once, before any batch fitting (also in SAO)."""
    if "user_boundary_types" not in data:
        return
    types = data["user_boundary_types"]
    values, targets = data["values"].detach().float(), data["returns"].detach().float()
    active = data["loss_mask"].bool()
    width = active.shape[1]
    positions = torch.arange(width, device=active.device).expand_as(types)
    next_action = (
        torch.where(active, positions, width).flip(-1).cummin(-1).values.flip(-1)
    )
    successor = values.gather(1, next_action.clamp(max=width - 1))
    # At terminal user boundaries R already sits in the independent U target.
    successor = torch.where(types == 3, targets, successor)
    user_segment = successor - values
    residual = values - targets
    with stats_tracker.scope("user_boundary"):
        if "user_boundary_flags" in data:
            flags = data["user_boundary_flags"]
            stats_tracker.denominator(flagged_count=flags)
            suppressed = data.get("user_boundary_gate_suppressed")
            if suppressed is not None:
                stats_tracker.denominator(gate_suppressed_count=suppressed)
            ids = data["user_boundary_detector_ids"]
            for detector_id, name in enumerate(USER_BOUNDARY_DETECTOR_NAMES[1:], 1):
                stats_tracker.denominator(
                    **{f"detector_{name}_count": flags & (ids == detector_id)}
                )
            with stats_tracker.scope("flagged"):
                stats_tracker.denominator(u_points=flags)
                stats_tracker.stat(
                    user_segment_mean=user_segment,
                    denominator="u_points",
                    reduce_type=ReduceType.AVG,
                )
        for subset, mask in (
            ("", (types == 1) | (types == 3)),
            ("internal", types == 1),
            ("terminal", types == 3),
        ):
            with stats_tracker.scope(subset) if subset else nullcontext():
                stats_tracker.denominator(u_points=mask)
                stats_tracker.stat(
                    user_segment_mean=user_segment,
                    user_segment_second_moment=user_segment.square(),
                    u_target=targets,
                    u_target_second_moment=targets.square(),
                    u_residual=residual,
                    u_mse=residual.square(),
                    denominator="u_points",
                    reduce_type=ReduceType.AVG,
                )
