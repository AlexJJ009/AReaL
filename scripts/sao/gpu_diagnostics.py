# SPDX-License-Identifier: Apache-2.0
"""Opt-in, task-scoped GPU diagnostics for SAO launches."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from scripts.sao.gpu_admission import (
    GPUAdmissionCancelled,
    GPUAdmissionConfig,
    GPUAdmissionError,
    GPUAdmissionQueryError,
    _parse_csv_rows,
    _query_gpu_index_by_uuid,
    _run_nvidia_smi,
    wait_for_scoped_gpus_free,
)

DEFAULT_DEVICES = tuple(str(index) for index in range(8))
PROC_ROOT = Path("/proc")


def _parse_proc_stat_starttime(stat: str) -> int | None:
    right = stat.rfind(")")
    fields = stat[right + 2 :].split() if right >= 0 else []
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _readlink(path: Path) -> str | None:
    try:
        return os.readlink(path)
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def read_process_owner(pid: int, *, proc_root: Path = PROC_ROOT) -> dict[str, object]:
    proc = proc_root / str(pid)
    stat = _read_text(proc / "stat")
    owner = {
        "cwd": _readlink(proc / "cwd"),
        "exe": _readlink(proc / "exe"),
        "comm": _read_text(proc / "comm"),
        "starttime": _parse_proc_stat_starttime(stat) if stat is not None else None,
    }
    owner["status"] = (
        "unavailable_in_current_pid_namespace_or_exited"
        if all(value is None for value in owner.values())
        else "available"
    )
    return owner


def query_compute_memory_snapshot(
    devices: tuple[str, ...],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: float = 10.0,
    proc_root: Path = PROC_ROOT,
) -> list[dict[str, object]]:
    uuid_to_index = _query_gpu_index_by_uuid(
        runner=runner, timeout_seconds=timeout_seconds
    )
    missing = set(devices) - set(uuid_to_index.values())
    if missing:
        raise GPUAdmissionQueryError(
            f"Scoped GPU devices do not exist: {sorted(missing)}"
        )
    output = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    rows: list[dict[str, object]] = []
    for row in _parse_csv_rows(output):
        if len(row) != 3:
            raise GPUAdmissionQueryError(f"Unexpected compute-app row: {row!r}")
        pid_text, gpu_uuid, used_gpu_memory = row
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            raise GPUAdmissionQueryError(
                f"Compute process {pid_text} reports unknown GPU UUID {gpu_uuid}"
            )
        if gpu_index not in devices:
            continue
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise GPUAdmissionQueryError(f"Unexpected PID value: {pid_text!r}") from exc
        rows.append(
            {
                "pid": pid,
                "gpu_uuid": gpu_uuid,
                "gpu_index": gpu_index,
                "used_gpu_memory_mib": used_gpu_memory,
                "owner": read_process_owner(pid, proc_root=proc_root),
            }
        )
    return rows


def write_snapshot(
    path: Path,
    *,
    phase: str,
    devices: tuple[str, ...],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: float = 10.0,
    proc_root: Path = PROC_ROOT,
) -> dict[str, object]:
    record: dict[str, object] = {
        "captured_ns": time.time_ns(),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": phase,
        "devices": list(devices),
    }
    try:
        record["processes"] = query_compute_memory_snapshot(
            devices,
            runner=runner,
            timeout_seconds=timeout_seconds,
            proc_root=proc_root,
        )
    except GPUAdmissionError as exc:
        record["error"] = str(exc)
        record["processes"] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _start_monitors(
    output_dir: Path,
    devices: tuple[str, ...],
    *,
    delay_seconds: int,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> tuple[list[subprocess.Popen[bytes]], list[object]]:
    processes = []
    streams = []
    for mode in ("dmon", "pmon"):
        stream = (output_dir / f"nvidia-smi-{mode}.log").open("ab")
        try:
            cmd = [
                "nvidia-smi",
                mode,
                "-i",
                ",".join(devices),
                "-d",
                str(delay_seconds),
                "-o",
                "DT",
            ]
            if mode == "pmon":
                cmd += ["-s", "um"]
            process = popen(
                cmd,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            stream.close()
            _stop_monitors(processes, streams)
            raise GPUAdmissionQueryError(
                f"failed to start nvidia-smi {mode}: {exc}"
            ) from exc
        streams.append(stream)
        processes.append(process)
    return processes, streams


def _stop_monitors(
    processes: Sequence[subprocess.Popen[bytes]],
    streams: Sequence[object],
) -> None:
    deadline = time.monotonic() + 2.0
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
    for stream in streams:
        stream.close()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _start_sampler(
    stop: threading.Event,
    sample_path: Path,
    *,
    devices: tuple[str, ...],
    interval_seconds: float,
    query_timeout_seconds: float,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    proc_root: Path,
) -> threading.Thread:
    def loop() -> None:
        while not stop.wait(interval_seconds):
            write_snapshot(
                sample_path,
                phase="periodic",
                devices=devices,
                runner=runner,
                timeout_seconds=query_timeout_seconds,
                proc_root=proc_root,
            )

    thread = threading.Thread(target=loop, name="sao-gpu-sampler", daemon=True)
    thread.start()
    return thread


def run_diagnostics(
    command: Sequence[str],
    *,
    output_dir: Path,
    admission: GPUAdmissionConfig,
    sample_seconds: float = 5.0,
    monitor_delay_seconds: int = 5,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    proc_root: Path = PROC_ROOT,
) -> int:
    if not command:
        raise ValueError("command is required")
    if sample_seconds <= 0 or monitor_delay_seconds <= 0:
        raise ValueError("sample and monitor delays must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "gpu-process-samples.jsonl"
    processes: list[subprocess.Popen[bytes]] = []
    streams: list[object] = []
    stop_sampler = threading.Event()
    sampler: threading.Thread | None = None
    received: list[int] = []
    child_pid: int | None = None
    previous = {}

    def handle_signal(signum, _frame) -> None:
        received.append(signum)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signum)
            except ProcessLookupError:
                pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_signal)

    result_path = output_dir / "gpu-diagnostics-result.json"
    started_ns = time.time_ns()

    def finish(returncode: int, **extra: object) -> int:
        payload: dict[str, object] = {
            "argv0": command[0],
            "argc": len(command),
            "devices": list(admission.devices),
            "returncode": returncode,
            "signals_received": received,
            "started_ns": started_ns,
            "completed_ns": time.time_ns(),
        }
        payload.update(extra)
        _write_json(result_path, payload)
        return returncode

    try:
        write_snapshot(
            sample_path,
            phase="initial_pre_monitor",
            devices=admission.devices,
            runner=runner,
            timeout_seconds=admission.query_timeout_seconds,
            proc_root=proc_root,
        )
        processes, streams = _start_monitors(
            output_dir,
            admission.devices,
            delay_seconds=monitor_delay_seconds,
            popen=popen,
        )
        sampler = _start_sampler(
            stop_sampler,
            sample_path,
            devices=admission.devices,
            interval_seconds=sample_seconds,
            query_timeout_seconds=admission.query_timeout_seconds,
            runner=runner,
            proc_root=proc_root,
        )
        pre_wait = write_snapshot(
            sample_path,
            phase="initial_pre_admission",
            devices=admission.devices,
            runner=runner,
            timeout_seconds=admission.query_timeout_seconds,
            proc_root=proc_root,
        )
        admission_result = wait_for_scoped_gpus_free(
            admission,
            runner=runner,
            sleep=sleep,
            monotonic=monotonic,
            cancel_check=lambda: bool(received),
        )
        if received:
            raise GPUAdmissionCancelled("GPU admission wait was cancelled")
        child = popen(list(command), start_new_session=True)
        child_pid = child.pid
        cleanup_deadline: float | None = None
        while True:
            returncode = child.poll()
            if returncode is not None:
                break
            if received:
                if cleanup_deadline is None:
                    cleanup_deadline = monotonic() + 2.0
                    try:
                        os.killpg(child.pid, received[-1])
                    except ProcessLookupError:
                        pass
                elif monotonic() >= cleanup_deadline:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            sleep(min(0.5, sample_seconds))
        write_snapshot(
            sample_path,
            phase="final",
            devices=admission.devices,
            runner=runner,
            timeout_seconds=admission.query_timeout_seconds,
            proc_root=proc_root,
        )
        return finish(
            returncode,
            initial_process_count=len(pre_wait.get("processes", [])),
            admission=admission_result,
        )
    except GPUAdmissionCancelled:
        returncode = 128 + received[-1] if received else 125
        return finish(returncode, error="GPU admission wait was cancelled")
    except GPUAdmissionError as exc:
        return finish(125, error=str(exc))
    finally:
        child_pid = None
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        stop_sampler.set()
        if sampler is not None:
            sampler.join(timeout=2)
        _stop_monitors(processes, streams)


def _admission_from_args(args: argparse.Namespace) -> GPUAdmissionConfig:
    devices = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    return GPUAdmissionConfig(
        devices=devices,
        timeout_seconds=args.admission_timeout_seconds,
        poll_seconds=args.admission_poll_seconds,
        stable_polls=args.admission_stable_polls,
        query_timeout_seconds=args.query_timeout_seconds,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "This wrapper is opt-in only: run it explicitly around one command. "
            "Disable it by launching the command directly. Remove evidence by "
            "deleting the chosen output directory after retention review. The "
            "sampler records GPU PID, memory, cwd, exe, comm, and /proc starttime; "
            "it never reads process environments or full command lines."
        ),
    )
    parser.add_argument("--output-dir", default="gpu-diagnostics")
    parser.add_argument("--devices", default=",".join(DEFAULT_DEVICES))
    parser.add_argument("--admission-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--admission-poll-seconds", type=float, default=2.0)
    parser.add_argument("--admission-stable-polls", type=int, default=2)
    parser.add_argument("--query-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--monitor-delay-seconds", type=int, default=5)
    parser.add_argument(
        "--proc-root",
        default=str(PROC_ROOT),
        help="procfs root for PID ownership readback; use a host /proc mount if needed",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    command = tuple(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if not command:
        print("gpu_diagnostics.py requires a command after --", file=sys.stderr)
        return 2
    return run_diagnostics(
        command,
        output_dir=Path(args.output_dir),
        admission=_admission_from_args(args),
        sample_seconds=args.sample_seconds,
        monitor_delay_seconds=args.monitor_delay_seconds,
        proc_root=Path(args.proc_root),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
