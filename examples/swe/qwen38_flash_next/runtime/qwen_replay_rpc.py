# SPDX-License-Identifier: Apache-2.0
"""Diagnostic RPC entrypoint with synchronized grouped GEMM observations."""

import json
import os
import runpy
import time
from pathlib import Path

import torch
import transformer_engine.pytorch.module.grouped_linear as grouped

_original = grouped.general_grouped_gemm
_gc_devices = set()


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [describe(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def observed(*args, **kwargs):
    device = torch.cuda.current_device()
    activated_now = False
    if (
        os.environ.get("QWEN_REPLAY_ENABLE_ALLOCATOR_GC") == "1"
        and device not in _gc_devices
    ):
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        _gc_devices.add(device)
        activated_now = True
    root = Path(os.environ["QWEN_GEMM_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{os.uname().nodename}-{os.getpid()}.jsonl"

    def record(phase, error=None):
        stats = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        row = {
            "allocator_gc_setter_executed": device in _gc_devices,
            "cuda_device_index": device,
            "rank": int(os.environ.get("RANK", "-1")),
            "device_free_bytes": free,
            "device_total_bytes": total,
            "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
            "allocator_backend": torch.cuda.get_allocator_backend(),
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "time": time.time(),
            "phase": phase,
            "error": error,
            "args": describe(args),
            "kwargs": {k: describe(v) for k, v in kwargs.items()},
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak": torch.cuda.max_memory_allocated(),
        }
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    if activated_now:
        record("allocator_gc_activated")
    phase = "before_sync"
    record(phase)
    try:
        torch.cuda.synchronize()
        phase = "gemm"
        result = _original(*args, **kwargs)
        phase = "after_sync"
        torch.cuda.synchronize()
        record("complete")
        return result
    except BaseException as exc:
        record(phase + "_failed", repr(exc))
        raise


grouped.general_grouped_gemm = observed
if __name__ == "__main__":
    runpy.run_module("areal.infra.rpc.rpc_server", run_name="__main__")
