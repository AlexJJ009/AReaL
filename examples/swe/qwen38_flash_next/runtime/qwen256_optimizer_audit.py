# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only native CPU Adam configuration, before optimizer post-init."""

import json
import os
import runpy
from pathlib import Path

import torch

import areal.engine.megatron_engine as engine

_original_config = engine.MCoreOptimizerConfig


def cpu_config(*args, **kwargs):
    kwargs.update(
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
        use_torch_optimizer_for_cpu_offload=True,
        overlap_cpu_optimizer_d2h_h2d=True,
        use_precision_aware_optimizer=False,
        main_grads_dtype=torch.float32,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    config = _original_config(*args, **kwargs)
    assert config.optimizer_cpu_offload and config.optimizer_offload_fraction == 1.0
    root = Path(os.environ["QWEN_GEMM_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    (
        root / f"cpu-optimizer-config-{os.uname().nodename}-{os.getpid()}.json"
    ).write_text(
        json.dumps(
            {
                k: str(getattr(config, k))
                for k in kwargs
                if k
                in (
                    "optimizer_cpu_offload",
                    "optimizer_offload_fraction",
                    "use_torch_optimizer_for_cpu_offload",
                    "overlap_cpu_optimizer_d2h_h2d",
                    "use_precision_aware_optimizer",
                    "main_grads_dtype",
                    "main_params_dtype",
                    "exp_avg_dtype",
                    "exp_avg_sq_dtype",
                )
            }
        )
    )
    return config


_original_optimizer = engine.get_megatron_optimizer


def audited_optimizer(*args, **kwargs):
    optimizer = _original_optimizer(*args, **kwargs)
    original_step = optimizer.step
    step_number = 0

    def step(*step_args, **step_kwargs):
        nonlocal step_number
        step_number += 1
        result = original_step(*step_args, **step_kwargs)
        pending, seen, rows = [optimizer], set(), []
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            attributes = vars(current)
            row = {
                "class": type(current).__name__,
                "state_devices": {},
                "moment_samples": [],
            }
            for state in attributes.get("state", {}).values():
                if not isinstance(state, dict):
                    continue
                for key, value in state.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    device = str(value.device)
                    row["state_devices"][device] = (
                        row["state_devices"].get(device, 0) + value.numel()
                    )
                    if (
                        key in ("exp_avg", "exp_avg_sq")
                        and len(row["moment_samples"]) < 4
                    ):
                        sample = value.detach().flatten()[:64].float()
                        row["moment_samples"].append(
                            {
                                "field": key,
                                "device": device,
                                "finite": bool(torch.isfinite(sample).all()),
                                "nonzero": bool(torch.count_nonzero(sample)),
                            }
                        )
            rows.append(row)
            for value in attributes.values():
                candidates = value if isinstance(value, (list, tuple)) else [value]
                for child in candidates:
                    if hasattr(child, "__dict__") and (
                        isinstance(child, torch.optim.Optimizer)
                        or "Optimizer" in type(child).__name__
                    ):
                        pending.append(child)
        root = Path(os.environ["QWEN_GEMM_AUDIT_DIR"])
        (
            root
            / f"cpu-optimizer-step-{step_number:04d}-{os.uname().nodename}-{os.getpid()}.json"
        ).write_text(
            json.dumps(
                {
                    "step": step_number,
                    "result": str(result),
                    "optimizers": rows,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        )
        return result

    optimizer.step = step
    return optimizer


engine.get_megatron_optimizer = audited_optimizer
engine.MCoreOptimizerConfig = cpu_config
if __name__ == "__main__":
    runpy.run_module("qwen_replay_rpc", run_name="__main__")
