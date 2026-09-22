# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch


def _stat_key(scope: str, name: str) -> str:
    return f"{scope}/{name}" if scope else name


def derive_critic_update_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    """Derive critic MSE and explained variance from reduced linear moments."""
    result = dict(stats)
    scopes = {
        key[: -len("/critic_target_second_moment")]
        for key in result
        if key.endswith("/critic_target_second_moment")
    }
    if "critic_target_second_moment" in result:
        scopes.add("")

    for scope in scopes:
        target_key = _stat_key(scope, "critic_target")
        target_second_key = _stat_key(scope, "critic_target_second_moment")
        residual_key = _stat_key(scope, "critic_residual")
        mse_key = _stat_key(scope, "critic_mse")
        variance_key = _stat_key(scope, "critic_target_variance")
        residual_variance_key = _stat_key(scope, "critic_residual_variance")
        ev_key = _stat_key(scope, "critic_explained_variance")
        ev_defined_key = _stat_key(scope, "critic_explained_variance_defined")

        if (
            target_key not in result
            or target_second_key not in result
            or residual_key not in result
            or mse_key not in result
        ):
            continue

        target_mean = result[target_key]
        target_second_moment = result[target_second_key]
        residual_mean = result[residual_key]
        mse = result[mse_key]
        if (
            target_mean is None
            or target_second_moment is None
            or residual_mean is None
            or mse is None
        ):
            result[variance_key] = None
            result[residual_variance_key] = None
            result[ev_key] = None
            result[ev_defined_key] = 0.0
            continue

        variance = float(target_second_moment) - float(target_mean) ** 2
        residual_variance = float(mse) - float(residual_mean) ** 2
        result[variance_key] = variance
        result[residual_variance_key] = residual_variance
        if variance <= 0.0:
            result[ev_key] = None
            result[ev_defined_key] = 0.0
        else:
            result[ev_key] = 1.0 - residual_variance / variance
            result[ev_defined_key] = 1.0

    return result


def infer_token_denominator(
    input_data: dict[str, Any],
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Infer the full token mask for stats logging.

    Context parallelism may slice intermediate tensors such as ``loss_mask`` or
    model outputs, while the original micro-batch metadata still describes the
    full logical sequence. Prefer that metadata for ``n_tokens`` so statistics
    stay consistent with and without context parallelism.
    """
    common_kwargs = {"dtype": torch.bool, "device": fallback.device}

    attention_mask = input_data.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return torch.ones_like(attention_mask, **common_kwargs)

    cu_seqlens = input_data.get("cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.numel() > 0:
        return torch.ones(int(cu_seqlens[-1].item()), **common_kwargs)

    input_ids = input_data.get("input_ids")
    # Tree-packed batches keep input_ids padded to tree size while token-level
    # stats stay at packed-token length. Only reuse input_ids when it already
    # matches the stat tensor shape.
    if isinstance(input_ids, torch.Tensor) and input_ids.shape == fallback.shape:
        return torch.ones_like(input_ids, **common_kwargs)

    return torch.ones_like(fallback, **common_kwargs)
