# SPDX-License-Identifier: Apache-2.0
"""Supervise PPO launch cleanup before handing GPUs to successor jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.sao.gpu_admission import (
    GPUAdmissionConfig,
    GPUAdmissionQueryError,
    GPUAdmissionTimeout,
    admission_config_from_env,
    scoped_busy_processes,
)

from areal.utils import logging

logger = logging.getLogger("PPOSupervisor")

STATE_FILE = "supervisor-state.json"
STOP_REQUEST_FILE = "stop-request.json"


class StopRequestError(RuntimeError):
    """Raised when a cooperative stop request cannot be bound to a run."""


class GPUHandoffError(RuntimeError):
    """Raised when GPU release cannot be confirmed within the bounded handoff."""

    def __init__(self, message: str, *, returncode: int | None = None):
        super().__init__(message)
        self.returncode = returncode


class _DeferredTerminationSignals:
    def __init__(
        self,
        *,
        forward_pgid: int | None = None,
        killpg: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self.received: list[int] = []
        self.forward_pgid = forward_pgid
        self._killpg = killpg
        self._previous: dict[int, object] = {}

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._record)
        return self

    def __exit__(self, *_exc_info) -> None:
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)

    def _record(self, signum, _frame) -> None:
        self.received.append(signum)
        if self.forward_pgid is None:
            return
        try:
            self._killpg(self.forward_pgid, signum)
        except ProcessLookupError:
            self.forward_pgid = None


@dataclass(frozen=True)
class PPOHandoffConfig:
    admission: GPUAdmissionConfig
    min_release_grace_seconds: float = 60.0
    cleanup_terminate_grace_seconds: float = 10.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.min_release_grace_seconds)
            or self.min_release_grace_seconds < 0
        ):
            raise ValueError(
                "min_release_grace_seconds must be a finite non-negative value"
            )
        if (
            not math.isfinite(self.cleanup_terminate_grace_seconds)
            or self.cleanup_terminate_grace_seconds < 0
        ):
            raise ValueError(
                "cleanup_terminate_grace_seconds must be a finite non-negative value"
            )
        if self.admission.timeout_seconds < self.min_release_grace_seconds:
            raise ValueError(
                "admission timeout must be at least min_release_grace_seconds"
            )


@dataclass(frozen=True)
class PPOHandoffReport:
    confirmed: bool
    devices: list[str]
    grace_seconds: float
    polls: int
    query_errors: int
    stable_polls: int
    returncode: int | None = None
    elapsed_seconds: float = 0.0


def handoff_config_from_env(env: Mapping[str, str] | None = None) -> PPOHandoffConfig:
    env = os.environ if env is None else env
    admission = admission_config_from_env(env)
    return PPOHandoffConfig(
        admission=admission,
        min_release_grace_seconds=float(
            env.get("SAO_PPO_HANDOFF_MIN_GRACE_SECONDS", "60")
        ),
        cleanup_terminate_grace_seconds=float(
            env.get("SAO_PPO_HANDOFF_CLEANUP_GRACE_SECONDS", "10")
        ),
    )


def _sleep_until(
    target: float,
    *,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> None:
    while True:
        remaining = target - monotonic()
        if remaining <= 0:
            return
        sleep(remaining)


def wait_for_confirmed_gpu_release(
    config: PPOHandoffConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    cancel_check: Callable[[], bool] | None = None,
) -> PPOHandoffReport:
    """Wait at least the grace period, then confirm scoped GPUs are stably free."""

    start = monotonic()
    deadline = start + config.admission.timeout_seconds
    grace_target = start + config.min_release_grace_seconds
    _sleep_until(grace_target, sleep=sleep, monotonic=monotonic)

    stable = 0
    polls = 0
    query_errors = 0
    last_busy = "none"
    while True:
        if cancel_check is not None and cancel_check():
            raise GPUHandoffError("GPU handoff confirmation was cancelled")
        now = monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise GPUHandoffError(
                "GPU handoff was not confirmed before the bounded deadline: "
                f"devices={list(config.admission.devices)} "
                f"timeout_seconds={config.admission.timeout_seconds} "
                f"stable_required={config.admission.stable_polls} "
                f"stable_observed={stable} query_errors={query_errors} "
                f"last_busy={last_busy}"
            )
        try:
            busy = scoped_busy_processes(
                config.admission.devices,
                runner=runner,
                timeout_seconds=min(config.admission.query_timeout_seconds, remaining),
            )
        except GPUAdmissionQueryError:
            query_errors += 1
            stable = 0
            sleep(min(config.admission.poll_seconds, max(0.0, deadline - monotonic())))
            continue
        polls += 1
        if busy:
            stable = 0
            last_busy = ", ".join(
                f"pid={process.pid}@gpu={process.gpu_index}" for process in busy
            )
        else:
            stable += 1
        now = monotonic()
        if now >= deadline:
            raise GPUHandoffError(
                "GPU handoff was not confirmed before the bounded deadline: "
                f"devices={list(config.admission.devices)} "
                f"timeout_seconds={config.admission.timeout_seconds} "
                f"stable_required={config.admission.stable_polls} "
                f"stable_observed={stable} query_errors={query_errors} "
                f"last_busy={last_busy}"
            )
        if stable >= config.admission.stable_polls:
            return PPOHandoffReport(
                confirmed=True,
                devices=list(config.admission.devices),
                grace_seconds=config.min_release_grace_seconds,
                elapsed_seconds=now - start,
                polls=polls,
                query_errors=query_errors,
                stable_polls=stable,
            )
        sleep(min(config.admission.poll_seconds, max(0.0, deadline - now)))


def cleanup_process_group(
    pgid: int,
    *,
    terminate_grace_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    killpg: Callable[[int, int], None] = os.killpg,
) -> bool:
    """Terminate only the supervised process group; never target unrelated PIDs."""

    try:
        killpg(pgid, 0)
    except ProcessLookupError:
        return False
    killpg(pgid, signal.SIGTERM)
    deadline = monotonic() + terminate_grace_seconds
    while monotonic() < deadline:
        try:
            killpg(pgid, 0)
        except ProcessLookupError:
            return True
        sleep(min(0.5, max(0.0, deadline - monotonic())))
    try:
        killpg(pgid, 0)
    except ProcessLookupError:
        return True
    killpg(pgid, signal.SIGKILL)
    return True


def finalize_gpu_handoff(
    *,
    config: PPOHandoffConfig,
    returncode: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> PPOHandoffReport:
    try:
        report = wait_for_confirmed_gpu_release(
            config, runner=runner, sleep=sleep, monotonic=monotonic
        )
    except GPUHandoffError as exc:
        raise GPUHandoffError(str(exc), returncode=returncode) from exc
    return PPOHandoffReport(
        confirmed=report.confirmed,
        devices=report.devices,
        grace_seconds=report.grace_seconds,
        polls=report.polls,
        query_errors=report.query_errors,
        stable_polls=report.stable_polls,
        returncode=returncode,
        elapsed_seconds=report.elapsed_seconds,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(tmp, payload)
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise StopRequestError(f"JSON payload is not an object: {path}")
    return payload


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _state_payload(
    *,
    run_id: str,
    launch_path: Path,
    command: Sequence[str],
    status: str,
    started_ns: int,
    child_pid: int | None = None,
    stop_requested_ns: int | None = None,
    completed_ns: int | None = None,
    returncode: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": run_id,
        "supervisor_pid": os.getpid(),
        "launch_dir": str(launch_path),
        "command": list(command),
        "status": status,
        "started_ns": started_ns,
    }
    if child_pid is not None:
        payload["child_pid"] = child_pid
    if stop_requested_ns is not None:
        payload["stop_requested_ns"] = stop_requested_ns
    if completed_ns is not None:
        payload["completed_ns"] = completed_ns
    if returncode is not None:
        payload["returncode"] = returncode
    return payload


def request_stop(launch_dir: str | Path) -> dict[str, object]:
    """Write a run-bound cooperative stop request for an active supervisor."""

    launch_path = Path(launch_dir)
    state_path = launch_path / STATE_FILE
    if not state_path.is_file():
        raise StopRequestError(f"missing supervisor state: {state_path}")
    state = _read_json(state_path)
    if state.get("status") not in {"running", "stop_requested"}:
        raise StopRequestError(
            f"supervisor is not running: status={state.get('status')!r}"
        )
    if state.get("launch_dir") != str(launch_path):
        raise StopRequestError("supervisor state launch_dir does not match request")
    run_id = state.get("run_id")
    supervisor_pid = state.get("supervisor_pid")
    if not isinstance(run_id, str) or not isinstance(supervisor_pid, int):
        raise StopRequestError("supervisor state is missing run_id or supervisor_pid")
    if not _process_is_alive(supervisor_pid):
        raise StopRequestError(f"supervisor pid is not alive: {supervisor_pid}")

    request = {
        "run_id": run_id,
        "supervisor_pid": supervisor_pid,
        "requested_by_pid": os.getpid(),
        "requested_ns": time.time_ns(),
        "status": "requested",
    }
    _atomic_write_json(launch_path / STOP_REQUEST_FILE, request)
    return request


def _matching_stop_request(
    launch_path: Path, *, run_id: str, supervisor_pid: int
) -> dict[str, object] | None:
    request_path = launch_path / STOP_REQUEST_FILE
    if not request_path.is_file():
        return None
    try:
        request = _read_json(request_path)
    except (OSError, json.JSONDecodeError, StopRequestError):
        return None
    if (
        request.get("run_id") != run_id
        or request.get("supervisor_pid") != supervisor_pid
        or request.get("status") != "requested"
    ):
        return None
    return request


def _poll_supervised_process(
    process: subprocess.Popen[bytes],
    signals: _DeferredTerminationSignals,
    config: PPOHandoffConfig,
    *,
    launch_path: Path,
    run_id: str,
    command: Sequence[str],
    started_ns: int,
    state_path: Path,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> tuple[int, bool, bool]:
    signal_cleanup_requested = False
    stop_request_cleanup_requested = False
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode, signal_cleanup_requested, stop_request_cleanup_requested
        if signals.received and not signal_cleanup_requested:
            cleanup_process_group(
                process.pid,
                terminate_grace_seconds=config.cleanup_terminate_grace_seconds,
                sleep=sleep,
                monotonic=monotonic,
            )
            signal_cleanup_requested = True
        request = _matching_stop_request(
            launch_path, run_id=run_id, supervisor_pid=os.getpid()
        )
        if request is not None and not stop_request_cleanup_requested:
            _atomic_write_json(
                state_path,
                _state_payload(
                    run_id=run_id,
                    launch_path=launch_path,
                    command=command,
                    status="stop_requested",
                    started_ns=started_ns,
                    child_pid=process.pid,
                    stop_requested_ns=request.get("requested_ns")
                    if isinstance(request.get("requested_ns"), int)
                    else time.time_ns(),
                ),
            )
            cleanup_process_group(
                process.pid,
                terminate_grace_seconds=config.cleanup_terminate_grace_seconds,
                sleep=sleep,
                monotonic=monotonic,
            )
            stop_request_cleanup_requested = True
        sleep(0.5)


def supervise_launch(
    launch_dir: str | Path,
    *,
    command: Sequence[str] = ("bash", "run.sh"),
    config: PPOHandoffConfig | None = None,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    launch_path = Path(launch_dir)
    launch_path.mkdir(parents=True, exist_ok=True)
    config = handoff_config_from_env(env) if config is None else config
    train_log = launch_path / "train.log"
    process_result_path = launch_path / "process-result.json"
    handoff_path = launch_path / "handoff.json"
    state_path = launch_path / STATE_FILE
    run_id = uuid.uuid4().hex
    started_ns = time.time_ns()
    _atomic_write_json(
        state_path,
        _state_payload(
            run_id=run_id,
            launch_path=launch_path,
            command=command,
            status="running",
            started_ns=started_ns,
        ),
    )
    with train_log.open("ab") as log_stream:
        with _DeferredTerminationSignals() as signals:
            process = subprocess.Popen(
                list(command),
                cwd=launch_path,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=None if env is None else dict(env),
            )
            _atomic_write_json(
                state_path,
                _state_payload(
                    run_id=run_id,
                    launch_path=launch_path,
                    command=command,
                    status="running",
                    started_ns=started_ns,
                    child_pid=process.pid,
                ),
            )
            signals.forward_pgid = process.pid
            returncode, signal_cleanup, stop_request_cleanup = _poll_supervised_process(
                process,
                signals,
                config,
                launch_path=launch_path,
                run_id=run_id,
                command=command,
                started_ns=started_ns,
                state_path=state_path,
                sleep=sleep,
                monotonic=monotonic,
            )
            signals.forward_pgid = None
            final_cleanup = cleanup_process_group(
                process.pid,
                terminate_grace_seconds=config.cleanup_terminate_grace_seconds,
                sleep=sleep,
                monotonic=monotonic,
            )
            _write_json(
                process_result_path,
                {
                    "command": list(command),
                    "returncode": returncode,
                    "signals_received": signals.received,
                    "signal_cleanup_requested": signal_cleanup,
                    "stop_request_cleanup_requested": stop_request_cleanup,
                    "final_cleanup_requested": final_cleanup,
                    "run_id": run_id,
                },
            )
            try:
                report = finalize_gpu_handoff(
                    config=config,
                    returncode=returncode,
                    runner=runner,
                    sleep=sleep,
                    monotonic=monotonic,
                )
            except GPUHandoffError as exc:
                _write_json(
                    handoff_path,
                    {
                        "confirmed": False,
                        "devices": list(config.admission.devices),
                        "returncode": returncode,
                        "error": str(exc),
                        "run_id": run_id,
                    },
                )
                _atomic_write_json(
                    state_path,
                    _state_payload(
                        run_id=run_id,
                        launch_path=launch_path,
                        command=command,
                        status="handoff_failed",
                        started_ns=started_ns,
                        child_pid=process.pid,
                        completed_ns=time.time_ns(),
                        returncode=returncode,
                    ),
                )
                raise
            _write_json(
                handoff_path,
                {
                    "confirmed": report.confirmed,
                    "devices": report.devices,
                    "grace_seconds": report.grace_seconds,
                    "elapsed_seconds": report.elapsed_seconds,
                    "polls": report.polls,
                    "query_errors": report.query_errors,
                    "stable_polls": report.stable_polls,
                    "returncode": returncode,
                    "run_id": run_id,
                },
            )
            _atomic_write_json(
                state_path,
                _state_payload(
                    run_id=run_id,
                    launch_path=launch_path,
                    command=command,
                    status="complete",
                    started_ns=started_ns,
                    child_pid=process.pid,
                    completed_ns=time.time_ns(),
                    returncode=returncode,
                ),
            )
    return returncode


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Cooperative stop: use --request-stop LAUNCH_DIR for ordinary "
            "predecessor shutdown. This writes a run-bound stop-request.json; "
            "the running supervisor then cleans up its child, waits the configured "
            "GPU handoff grace, confirms GPUs are free, and exits. `pueue kill` "
            "is a force kill path and bypasses this barrier."
        ),
    )
    parser.add_argument(
        "--request-stop",
        metavar="LAUNCH_DIR",
        help="write a cooperative stop request for an active supervisor and exit",
    )
    parser.add_argument("launch_dir", nargs="?", default=".")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after --; defaults to bash run.sh",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.request_stop is not None:
        try:
            request = request_stop(args.request_stop)
        except StopRequestError as exc:
            logger.error("PPO supervisor stop request failed: %s", exc)
            return 2
        print(json.dumps(request, sort_keys=True))
        return 0

    command = tuple(args.command[1:] if args.command[:1] == ["--"] else args.command)
    if not command:
        command = ("bash", "run.sh")
    try:
        return supervise_launch(args.launch_dir, command=command)
    except (GPUHandoffError, GPUAdmissionTimeout) as exc:
        logger.error("PPO supervisor handoff failed: %s", exc)
        return 125


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
