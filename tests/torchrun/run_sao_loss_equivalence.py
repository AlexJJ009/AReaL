# SPDX-License-Identifier: Apache-2.0
"""Distributed sidecar for SAO PPO per-token loss scaling.

Run later with real allocation, for example:

    torchrun --standalone --nproc-per-node=2 tests/torchrun/run_sao_loss_equivalence.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"Expected 2 ranks for this sidecar, got {world_size}")

    mb_losses = [
        torch.tensor([0.20, 0.80], dtype=torch.float32),
        torch.tensor([-0.40, 1.10], dtype=torch.float32),
    ][rank]
    mb_weights = [
        torch.tensor([2.0, 3.0], dtype=torch.float32),
        torch.tensor([1.0, 4.0], dtype=torch.float32),
    ][rank]

    total_weight = mb_weights.sum()
    dist.all_reduce(total_weight, op=dist.ReduceOp.SUM)
    local_scaled = (mb_losses * mb_weights / total_weight * world_size).sum()

    averaged = local_scaled.clone()
    dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
    averaged = averaged / world_size

    local_numerator = (mb_losses * mb_weights).sum()
    dist.all_reduce(local_numerator, op=dist.ReduceOp.SUM)
    expected = local_numerator / total_weight

    torch.testing.assert_close(averaged, expected, rtol=0.0, atol=1e-7)
    if rank == 0:
        print(
            {
                "status": "ok",
                "pid": os.getpid(),
                "world_size": world_size,
                "global_weight": float(total_weight),
                "loss": float(averaged),
            }
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
