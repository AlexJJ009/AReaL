# SPDX-License-Identifier: Apache-2.0
"""Diagnostic AWEX entry: preserve behavior and record exceptions before SG catches them."""

import atexit
import faulthandler
import json
import os
import socket
import time
import traceback
from pathlib import Path

_fault_file = None


def record(event, **fields):
    root = Path(os.environ["QWEN_SGLANG_EXIT_AUDIT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    row = {
        "event": event,
        "wall_time": time.time(),
        "monotonic": time.monotonic(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "host": socket.gethostname(),
        **fields,
    }
    path = root / f"{socket.gethostname()}-{os.getpid()}.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(fd, "a") as output:
            output.write(json.dumps(row) + "\n")
    except BaseException:
        raise


def guard_call(call, *args, **kwargs):
    try:
        result = call(*args, **kwargs)
    except BaseException as exc:
        record(
            "exception",
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        raise
    else:
        record("normal_return")
        return result


def scheduler_entry(*args, **kwargs):
    global _fault_file
    from sglang.srt.managers.scheduler import Scheduler

    from areal.engine.awex.sglang_plugin import awex_run_scheduler_process

    status = Path("/proc/self/status").read_text()
    identity = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in status.splitlines()
        if line.startswith(("NSpid:", "NSpgid:", "NSsid:"))
    }
    record("scheduler_start", identity=identity)
    root = Path(os.environ["QWEN_SGLANG_EXIT_AUDIT_DIR"])
    fault_path = root / f"{socket.gethostname()}-{os.getpid()}.fault.txt"
    _fault_file = os.fdopen(
        os.open(fault_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a"
    )
    faulthandler.enable(_fault_file, all_threads=True)
    atexit.register(record, "atexit")
    original = Scheduler.run_event_loop

    def event_loop(self, *a, **kw):
        record(
            "event_loop_enter",
            gpu_id=getattr(self, "gpu_id", None),
            tp_rank=getattr(self, "tp_rank", None),
            port=getattr(self.server_args, "port", None),
        )
        return guard_call(original, self, *a, **kw)

    Scheduler.run_event_loop = event_loop
    return guard_call(awex_run_scheduler_process, *args, **kwargs)


def main():
    # Match the native AWEX entry's allocator handling before importing torch.
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" in conf.lower():
        tokens = [
            t.strip()
            for t in conf.split(",")
            if t.strip() and not t.strip().lower().startswith("expandable_segments")
        ]
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
            tokens + ["expandable_segments:False"]
        )
    import sys

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    from areal.engine.awex.sglang_plugin import _load_sglang_plugins_if_available

    _load_sglang_plugins_if_available()
    args = prepare_server_args(sys.argv[1:])
    args.skip_server_warmup = True
    record("server_start", port=args.port, base_gpu_id=args.base_gpu_id)
    try:
        return guard_call(
            launch_server, args, run_scheduler_process_func=scheduler_entry
        )
    finally:
        record("server_cleanup")
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
