# SPDX-License-Identifier: Apache-2.0
"""CPU optimizer experiment entrypoint without synchronized GEMM instrumentation."""

import json
import os
import runpy
from pathlib import Path

import qwen256_optimizer_audit as diagnostic
import torch


def cpu_config(*args, **kwargs):
    config = diagnostic.cpu_config(*args, **kwargs)
    if os.environ.get("QWEN_REPLAY_ENABLE_ALLOCATOR_GC") == "1":
        device = torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        root = Path(os.environ["QWEN_GEMM_AUDIT_DIR"])
        (root / f"allocator-gc-{os.uname().nodename}-{os.getpid()}.json").write_text(
            json.dumps({"setter_executed": True, "device": device, "fraction": 1.0})
        )
    return config


diagnostic.engine.MCoreOptimizerConfig = cpu_config
if __name__ == "__main__":
    runpy.run_module("areal.infra.rpc.rpc_server", run_name="__main__")
