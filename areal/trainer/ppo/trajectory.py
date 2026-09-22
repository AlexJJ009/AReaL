# SPDX-License-Identifier: Apache-2.0

"""Explicit episode metadata for action-token GAE, independent of padding."""

from typing import Any

import torch


def action_trajectory_metadata(
    data: dict[str, Any], loss_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate one episode per row and return termination, length, last action.

    Masks here use the training convention: position t predicts token t+1.
    Observations remain in model context but are not Bellman timesteps. Multiple
    episodes must be separate rows before engine packing; reject prepacked rows.
    """
    if loss_mask.ndim != 2 or data["attention_mask"].shape != loss_mask.shape:
        raise ValueError("Explicit GAE requires padded [batch, tokens] episode rows")
    bs, width = loss_mask.shape
    for key in ("terminated", "truncated"):
        if key not in data or data[key].shape != (bs,) or data[key].dtype != torch.bool:
            raise ValueError(f"{key} must be a bool tensor with shape [{bs}]")
    terminated, truncated = data["terminated"], data["truncated"]
    torch._assert_async(
        torch.all(terminated ^ truncated), "terminated XOR truncated is required"
    )
    active = loss_mask.bool()
    torch._assert_async(torch.all(active.sum(-1) > 0), "Empty action episode")
    attention = data["attention_mask"].bool()
    lengths = attention.sum(-1).long()
    positions = torch.arange(width, device=loss_mask.device).expand(bs, -1)
    torch._assert_async(
        torch.all(attention == (positions < lengths[:, None])),
        "Explicit GAE requires right-padded episode rows",
    )
    torch._assert_async(
        torch.all(~active | (positions < lengths[:, None] - 1)),
        "Every action needs its preceding state inside its episode",
    )
    episode_ids = data.get("episode_ids")
    if episode_ids is not None:
        if episode_ids.shape != loss_mask.shape or episode_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("episode_ids must be token-aligned integer IDs")
        ids = episode_ids.roll(-1, -1)
        low = torch.where(active, ids, torch.iinfo(ids.dtype).max).amin(-1)
        high = torch.where(active, ids, -1).amax(-1)
        torch._assert_async(
            torch.all((low == high) & (low >= 0)),
            "Separate packed episodes into rows before GAE",
        )
    last_action = torch.where(active, positions, -1).amax(-1)
    return terminated, lengths, last_action


def action_token_rewards(data: dict[str, Any], loss_mask: torch.Tensor) -> torch.Tensor:
    """Read action-aligned reward increments; require producer-side tool mapping.

    token_rewards uses the original token convention, like rollout loss_mask.
    Any reward on an observation/prompt/padding position is rejected instead of
    discarded. Scalar outcome reward, if any, is added separately at last action.
    """
    raw = data.get("token_rewards")
    if raw is None:
        return torch.zeros_like(loss_mask)
    if raw.shape != loss_mask.shape or not torch.is_floating_point(raw):
        raise ValueError("token_rewards must be token-aligned floating rewards")
    rewards = raw.roll(-1, -1)
    torch._assert_async(torch.all(torch.isfinite(rewards)), "Non-finite token reward")
    torch._assert_async(
        torch.all((rewards == 0) | loss_mask.bool()),
        "Map observation rewards to their producing action before GAE",
    )
    return rewards.detach()
