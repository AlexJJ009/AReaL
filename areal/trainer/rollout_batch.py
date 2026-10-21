# SPDX-License-Identifier: Apache-2.0
"""Finite online epochs and equal-weight data-parallel tail dispatch."""

import copy
from math import gcd

import torch


class FiniteEpochBatcher:
    """Restart the native finite iterator only after its epoch is fully drained."""

    def __init__(self, rollout, epoch: int = 0):
        self.rollout = rollout
        self.prepare = rollout.prepare_batch
        self.epoch = epoch

    def __call__(self, *args, **kwargs):
        kwargs.update(finite_epoch=True, fail_on_rejection=True)
        batch = self.prepare(*args, **kwargs)
        if not batch:
            # A finite empty result means all submitted inputs/results have
            # drained. Preserve controller task IDs and staleness state.
            del self.rollout.data_generator
            self.epoch += 1
            dataloader = args[0] if args else kwargs["dataloader"]
            dataloader.sampler.set_epoch(self.epoch)
            batch = self.prepare(*args, **kwargs)
        if not batch:
            raise RuntimeError("Finite training dataset is empty")
        return batch


def group_episode_count(group: dict) -> int:
    """Trajectories in one rollout group; a group may have lost members."""
    return int(group["attention_mask"].shape[0])


def repeat_groups_for_dispatch(
    groups: list[dict], dp_size: int
) -> tuple[list[dict], int]:
    """Uniformly replicate complete tail groups; token-mean gradients stay equal.

    This is physical DP dispatch only, not new rollout samples. Never replicate
    just some groups: that would change their relative optimization weights.
    """
    if not groups:
        raise ValueError("Cannot dispatch an empty training batch")
    replicas = dp_size // gcd(len(groups), dp_size)
    return [dict(group) for _ in range(replicas) for group in groups], replicas


def pad_groups_for_dispatch(groups: list[dict], dp_size: int) -> tuple[list[dict], int]:
    """Add minimal zero-weight groups while retaining valid forward context.

    ``is_padding`` is token-aligned so it survives microbatching and packing.
    Attention stays intact: empty sequences are invalid model/GAE inputs. All
    optimization eligibility and boundary metadata on the copies are zeroed.
    """
    if not groups:
        raise ValueError("Cannot dispatch an empty training batch")
    if dp_size < 1:
        raise ValueError("dp_size must be positive")
    padding = (-len(groups)) % dp_size
    if not padding:
        return list(groups), 0
    physical = [
        {
            **group,
            "is_padding": torch.zeros_like(group["attention_mask"], dtype=torch.bool),
        }
        for group in groups
    ]
    # A short existing group minimizes the extra forward cost without changing
    # any real sequence or requiring synthetic tokenizer-specific inputs.
    template = min(groups, key=lambda group: group["attention_mask"].numel())
    for _ in range(padding):
        dummy = copy.deepcopy(template)
        dummy["is_padding"] = torch.ones_like(dummy["attention_mask"], dtype=torch.bool)
        for key in (
            "loss_mask",
            "critic_loss_mask",
            "action_origin_mask",
            "rewards",
            "token_rewards",
            "bootstrap_mask",
            "user_boundary_types",
            "user_boundary_flags",
            "user_boundary_detector_ids",
            "user_boundary_gate_suppressed",
            "user_reliability_weight",
        ):
            if key in dummy:
                dummy[key].zero_()
        physical.append(dummy)
    return physical, padding
