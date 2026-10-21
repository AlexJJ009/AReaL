# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch


def training_sequence_mask(input_data: dict[str, Any]) -> torch.Tensor:
    """Real padded rows, including real rows whose loss tokens were filtered."""
    padding = input_data.get("is_padding")
    if padding is not None:
        return ~padding.any(dim=-1)
    ids = input_data["input_ids"]
    return torch.ones(ids.shape[0], device=ids.device, dtype=torch.bool)


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

    # Retire SAO's old unqualified merged EV rather than change its meaning.
    # The explicit primary key compares the fresh-batch forward in PPO and SAO;
    # indexed scopes retain each same-batch fit separately (including domains).
    for scope in scopes:
        parts = scope.split("/")
        if parts[-1] == "critic_update_1":
            primary_scope = "/".join(parts[:-1])
        elif any(part.startswith("critic_update_") for part in parts):
            continue
        elif result.get(_stat_key(scope, "critic_single_minibatch")) == 1.0:
            primary_scope = scope
        else:
            continue
        for suffix in ("", "_defined"):
            source = _stat_key(scope, f"critic_explained_variance{suffix}")
            if source in result:
                result[
                    _stat_key(
                        primary_scope, f"critic_explained_variance_pre_update{suffix}"
                    )
                ] = result[source]

    # User-boundary moments come from the initial values snapshot, once per
    # batch, rather than pooling SAO's post-fit actor recomputation.
    for key in list(result):
        if not key.endswith("/u_points"):
            continue
        scope = key.removesuffix("/u_points")
        mean_key = _stat_key(scope, "user_segment_mean")
        mean = result.setdefault(mean_key, None)
        second = result.get(_stat_key(scope, "user_segment_second_moment"))
        result[_stat_key(scope, "user_segment_std")] = (
            None
            if mean is None or second is None
            else math.sqrt(max(0.0, second - mean * mean))
        )
        target = result.get(_stat_key(scope, "u_target"))
        target2 = result.get(_stat_key(scope, "u_target_second_moment"))
        residual = result.get(_stat_key(scope, "u_residual"))
        mse = result.get(_stat_key(scope, "u_mse"))
        variance = None if target is None or target2 is None else target2 - target**2
        defined = (
            variance is not None
            and variance > 0
            and residual is not None
            and mse is not None
        )
        result[_stat_key(scope, "critic_explained_variance_pre_update")] = (
            1 - (mse - residual**2) / variance if defined else None
        )
        result[_stat_key(scope, "critic_explained_variance_pre_update_defined")] = (
            float(defined)
        )
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
    padding = input_data.get("is_padding")
    if isinstance(padding, torch.Tensor):
        if isinstance(attention_mask, torch.Tensor):
            return (~padding.any(-1, keepdim=True)).expand_as(attention_mask)
        return ~padding.bool()
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
