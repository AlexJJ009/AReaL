# SPDX-License-Identifier: Apache-2.0

"""Utilities for torch_memory_saver (TMS) configuration and setup.

This module handles the environment variable setup required for TMS to work
properly with LD_PRELOAD hooks.
"""

import os
import re
from contextlib import nullcontext

try:
    from torch_memory_saver import torch_memory_saver
except ImportError:

    class MockTorchMemorySaver:
        def disable(self):
            return nullcontext()

        def pause(self):
            pass

        def resume(self):
            pass

    torch_memory_saver = MockTorchMemorySaver()


_TMS_PRELOAD_LIB = "torch_memory_saver_hook_mode_preload.abi3.so"
_TMS_INIT_FLAGS = ("TMS_INIT_ENABLE", "TMS_INIT_ENABLE_CPU_BACKUP")


def _get_tms_preload_path() -> str:
    import torch_memory_saver as tms_pkg

    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(tms_pkg.__file__)),
        _TMS_PRELOAD_LIB,
    )

    if not os.path.exists(dynlib_path):
        raise RuntimeError(f"LD_PRELOAD so file {dynlib_path} does not exist.")
    return dynlib_path


def get_tms_env_vars() -> dict[str, str]:
    """Get process-start environment variables for torch_memory_saver (TMS)."""
    dynlib_path = _get_tms_preload_path()

    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    preload = ":".join(path for path in (existing_preload, dynlib_path) if path)
    env_vars = {"LD_PRELOAD": preload}
    env_vars.update(dict.fromkeys(_TMS_INIT_FLAGS, "1"))
    return env_vars


def strip_tms_env_vars(env: dict[str, str]) -> None:
    """Remove the TMS hook that ``get_tms_env_vars`` adds, in place.

    Inference servers launched from a TMS-enabled worker inherit its
    environment. With the hook preloaded, the server's own CUDA allocations go
    through TMS even when its native memory saver is off, which broke NCCL
    weight updates (illegal memory access) on Qwen3.5. Other preloads are kept.
    """
    for flag in _TMS_INIT_FLAGS:
        env.pop(flag, None)
    hook_prefix = _TMS_PRELOAD_LIB.split(".", 1)[0]
    preloads = [p for p in re.split(r"[:\s]+", env.get("LD_PRELOAD", "")) if p]
    kept = [p for p in preloads if not os.path.basename(p).startswith(hook_prefix)]
    if kept == preloads:
        return
    if kept:
        env["LD_PRELOAD"] = ":".join(kept)
    else:
        env.pop("LD_PRELOAD", None)


def normalize_tms_worker_preload() -> None:
    """Expose only the TMS hook after the loader consumed all preload entries.

    ``torch_memory_saver`` passes the live ``LD_PRELOAD`` value to
    ``ctypes.CDLL`` and therefore requires it to be one library path. Other
    preload entries may still be needed when the worker process starts, so the
    launcher keeps the composed value until engine initialization.
    """
    if not is_tms_enabled():
        return

    dynlib_path = _get_tms_preload_path()
    preload_entries = [
        path for path in os.environ.get("LD_PRELOAD", "").split(":") if path
    ]
    if dynlib_path not in preload_entries:
        raise RuntimeError(
            "TMS preload hook is missing from the worker LD_PRELOAD environment."
        )
    os.environ["LD_PRELOAD"] = dynlib_path


def is_tms_enabled() -> bool:
    return os.environ.get("TMS_INIT_ENABLE", "0") == "1"
