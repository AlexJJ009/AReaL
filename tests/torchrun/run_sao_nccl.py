# SPDX-License-Identifier: Apache-2.0
"""Exercise all eight devices and both four-GPU groups without model mutation."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    groups = [dist.new_group(ranks) for ranks in (list(range(4)), list(range(4, 8)))]
    value = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(value)
    torch.testing.assert_close(value, torch.tensor(36.0, device="cuda"))
    subgroup = torch.tensor(float(rank + 1), device="cuda")
    dist.all_reduce(subgroup, group=groups[rank // 4])
    torch.testing.assert_close(
        subgroup, torch.tensor(10.0 if rank < 4 else 26.0, device="cuda")
    )
    props = torch.cuda.get_device_properties(local_rank)
    row = {
        "rank": rank,
        "pid": os.getpid(),
        "device": local_rank,
        "name": props.name,
        "uuid": str(props.uuid),
        "world_sum": value.item(),
        "subgroup_sum": subgroup.item(),
        "capability": [props.major, props.minor],
    }
    rows = [None] * 8
    dist.all_gather_object(rows, row)
    if rank == 0:
        maps = Path("/proc/self/maps").read_text().splitlines()
        libraries = sorted(
            {
                line.split()[-1]
                for line in maps
                if "libcuda" in line or "libnccl" in line
            }
        )
        Path(args.output).write_text(
            json.dumps(
                {
                    "passed": True,
                    "ranks": rows,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "loaded_libraries": libraries,
                },
                indent=2,
            )
            + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
