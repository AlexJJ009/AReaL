# SPDX-License-Identifier: Apache-2.0

"""Utilities for torch_memory_saver (TMS) configuration and setup.

This module handles the environment variable setup required for TMS to work
properly with LD_PRELOAD hooks.
"""

import os
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


def _get_tms_preload_path() -> str:
    import torch_memory_saver as tms_pkg

    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(tms_pkg.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )

    if not os.path.exists(dynlib_path):
        raise RuntimeError(f"LD_PRELOAD so file {dynlib_path} does not exist.")
    return dynlib_path


def get_tms_env_vars() -> dict[str, str]:
    """Get process-start environment variables for torch_memory_saver (TMS)."""
    dynlib_path = _get_tms_preload_path()

    existing_preload = os.environ.get("LD_PRELOAD", "").strip()
    preload = ":".join(path for path in (existing_preload, dynlib_path) if path)
    env_vars = {
        "LD_PRELOAD": preload,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }
    return env_vars


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
