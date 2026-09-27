# SPDX-License-Identifier: Apache-2.0
"""Finite online epochs and equal-weight data-parallel tail dispatch."""

from math import gcd


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
