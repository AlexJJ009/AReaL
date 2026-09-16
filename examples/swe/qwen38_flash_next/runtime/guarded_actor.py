# SPDX-License-Identifier: Apache-2.0
"""GSM8K diagnostic RPC: stop nonfinite backward/update before AWEX publication."""

import json
import math
import os
import runpy
from pathlib import Path

if os.environ.get("QWEN_GSM8K_FLASH_ATTN") == "1":
    os.environ.update(NVTE_FUSED_ATTN="0", NVTE_FLASH_ATTN="1", NVTE_UNFUSED_ATTN="0")

import qwen256_cpu_optimizer_rpc as runtime
import torch

if os.environ.get("QWEN_GSM8K_FLASH_ATTN") == "1":
    from megatron.core.transformer.enums import AttnBackend

    original_bridge = runtime.diagnostic.engine.MCoreBridgeAdapter

    class FlashBridge(original_bridge):
        def __init__(self, *args, **kwargs):
            overrides = dict(kwargs.get("transformer_config_overrides") or {})
            overrides["attention_backend"] = AttnBackend.flash
            kwargs["transformer_config_overrides"] = overrides
            super().__init__(*args, **kwargs)

    runtime.diagnostic.engine.MCoreBridgeAdapter = FlashBridge

factory = runtime.diagnostic.engine.get_megatron_optimizer


def guarded_optimizer(*args, **kwargs):
    optimizer = factory(*args, **kwargs)
    step = optimizer.step
    models = args[1] if len(args) > 1 else kwargs["model_chunks"]
    audit_step = 0

    def guarded_step(*step_args, **step_kwargs):
        nonlocal audit_step
        audit_step += 1
        if (
            os.environ.get("QWEN_GSM8K_DETECT_ANOMALY") == "1"
            or os.environ.get("QWEN_GSM8K_GRAD_AUDIT") == "1"
        ):
            checked = 0
            bad = []
            for chunk, model in enumerate(models):
                for name, parameter in model.named_parameters():
                    grad = getattr(parameter, "main_grad", None)
                    if grad is None:
                        grad = parameter.grad
                    if grad is None:
                        continue
                    checked += 1
                    if not bool(torch.isfinite(grad).all()):
                        bad.append(f"{chunk}:{name}")
            root = Path(os.environ["QWEN_GEMM_AUDIT_DIR"])
            root.mkdir(parents=True, exist_ok=True)
            (
                root
                / f"pre-optimizer-step-{audit_step:04d}-{os.uname().nodename}-{os.getpid()}.json"
            ).write_text(
                json.dumps(
                    {
                        "checked_parameters": checked,
                        "nonfinite_parameters": bad,
                        "rank": os.environ.get("RANK"),
                        "stage": "before optimizer.step",
                    }
                )
            )
            if bad:
                raise FloatingPointError(
                    f"Nonfinite gradient before optimizer: {bad[:3]}"
                )
        result = step(*step_args, **step_kwargs)
        norm = result[1]
        finite = norm is not None and math.isfinite(float(norm))
        if not finite:
            root = Path(os.environ["QWEN_GEMM_AUDIT_DIR"])
            root.mkdir(parents=True, exist_ok=True)
            (root / f"nonfinite-stop-{os.getpid()}.json").write_text(
                json.dumps(
                    {"result": str(result), "action": "raise before AWEX publication"}
                )
            )
            raise FloatingPointError(
                "Nonfinite optimizer grad norm; block AWEX publication"
            )
        return result

    optimizer.step = guarded_step
    return optimizer


runtime.diagnostic.engine.get_megatron_optimizer = guarded_optimizer
if os.environ.get("QWEN_GSM8K_DETECT_ANOMALY") == "1":
    torch.autograd.set_detect_anomaly(True, check_nan=True)
if __name__ == "__main__":
    runpy.run_module("areal.infra.rpc.rpc_server", run_name="__main__")
