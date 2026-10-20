# SPDX-License-Identifier: Apache-2.0
"""Build critic-only prefixed inputs without changing the actor-aligned batch."""

from typing import Any

import torch

from areal.utils.data import CRITIC_PREFIX_KEYS


def actor_input(data: dict[str, Any]) -> dict[str, Any]:
    """Remove privileged transport before any actor engine invocation."""
    if not any(key in data for key in CRITIC_PREFIX_KEYS):
        return data
    return {key: value for key, value in data.items() if key not in CRITIC_PREFIX_KEYS}


def prepare_critic_input(data: dict[str, Any]) -> dict[str, Any]:
    """Prepend each sample's prefix and zero-mask every added loss position.

    Engines receive ordinary right-padded text inputs and generate/pack positions
    as usual. Prefix transport is consumed here, before engine microbatching.
    """
    if not any(key in data for key in CRITIC_PREFIX_KEYS):
        return data
    if not all(key in data for key in CRITIC_PREFIX_KEYS):
        raise ValueError("Incomplete critic prefix transport")
    ids = data["input_ids"]
    prefix = data["critic_prefix_input_ids"]
    lengths = data["critic_prefix_lengths"]
    batch, width = ids.shape
    if (
        prefix.ndim != 2
        or prefix.shape[0] != batch
        or lengths.shape != (batch,)
        or prefix.dtype not in (torch.int32, torch.int64)
        or lengths.dtype not in (torch.int32, torch.int64)
        or prefix.device != ids.device
        or lengths.device != ids.device
    ):
        raise ValueError("Invalid critic prefix shape/dtype/device")
    torch._assert_async(
        torch.all((lengths > 0) & (lengths <= prefix.shape[1])),
        "Invalid critic prefix lengths",
    )
    # Padded transport width is already known on the host; no GPU length readback.
    new_width = width + prefix.shape[1]
    positions = torch.arange(width, device=ids.device)[None, :] + lengths[:, None]
    result = {}
    for key, value in data.items():
        if key in CRITIC_PREFIX_KEYS:
            continue
        if torch.is_tensor(value) and value.ndim == 2 and value.shape == ids.shape:
            shifted = value.new_zeros((batch, new_width))
            shifted.scatter_(1, positions, value)
            result[key] = shifted
        else:
            result[key] = value
    prefix_positions = torch.arange(prefix.shape[1], device=ids.device)[None, :]
    prefix_mask = prefix_positions < lengths[:, None]
    result["input_ids"][:, : prefix.shape[1]] = torch.where(
        prefix_mask, prefix.to(ids.dtype), result["input_ids"][:, : prefix.shape[1]]
    )
    result["attention_mask"][:, : prefix.shape[1]] |= prefix_mask.to(
        data["attention_mask"].dtype
    )
    result["position_ids"] = torch.arange(
        new_width, device=ids.device, dtype=torch.long
    ).expand(batch, -1)
    return result


def align_critic_values(values: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    """Drop L_i prefix values independently, returning the original padded shape."""
    if "critic_prefix_lengths" not in data:
        return values
    positions = (
        torch.arange(data["input_ids"].shape[1], device=values.device)[None, :]
        + data["critic_prefix_lengths"][:, None]
    )
    # FSDP pads outputs only to max(valid length), which can be shorter than
    # the input's padded width. Restore that padding before shifting back.
    padded_width = data["input_ids"].shape[1] + data["critic_prefix_input_ids"].shape[1]
    values = torch.nn.functional.pad(
        values, (0, max(0, padded_width - values.shape[1]))
    )
    return values.gather(1, positions).masked_fill(~data["attention_mask"].bool(), 0)
