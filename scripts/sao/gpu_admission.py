# SPDX-License-Identifier: Apache-2.0
"""Bounded GPU admission checks for SAO launchers."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite


class GPUAdmissionError(RuntimeError):
    """Base error for GPU admission failures."""


class GPUAdmissionQueryError(GPUAdmissionError):
    """Raised when nvidia-smi cannot provide a trusted readback."""


class GPUAdmissionTimeout(GPUAdmissionError):
    """Raised when scoped GPUs do not become stably free before the deadline."""


class GPUAdmissionCancelled(GPUAdmissionError):
    """Raised when the caller cancels a pending admission wait."""


@dataclass(frozen=True)
class GPUProcess:
    pid: str
    gpu_uuid: str
    gpu_index: str


@dataclass(frozen=True)
class GPUAdmissionConfig:
    devices: tuple[str, ...]
    timeout_seconds: float = 180.0
    poll_seconds: float = 2.0
    stable_polls: int = 2
    query_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isfinite(self.timeout_seconds) or self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be a finite non-negative value")
        if not isfinite(self.poll_seconds) or self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be a finite positive value")
        if self.stable_polls <= 0:
            raise ValueError("stable_polls must be positive")
        if not isfinite(self.query_timeout_seconds) or self.query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be a finite positive value")
        if not self.devices:
            raise ValueError("at least one GPU device must be scoped")


def _split_devices(raw: str) -> tuple[str, ...]:
    devices = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not devices:
        raise ValueError("GPU device scope is empty")
    return devices


def admission_config_from_env(
    env: Mapping[str, str] | None = None,
) -> GPUAdmissionConfig:
    env = os.environ if env is None else env
    devices = _split_devices(env.get("SAO_GPU_ADMISSION_DEVICES", "0,1,2,3,4,5,6,7"))
    return GPUAdmissionConfig(
        devices=devices,
        timeout_seconds=float(env.get("SAO_GPU_ADMISSION_TIMEOUT_SECONDS", "180")),
        poll_seconds=float(env.get("SAO_GPU_ADMISSION_POLL_SECONDS", "2")),
        stable_polls=int(env.get("SAO_GPU_ADMISSION_STABLE_POLLS", "2")),
        query_timeout_seconds=float(
            env.get("SAO_GPU_ADMISSION_QUERY_TIMEOUT_SECONDS", "10")
        ),
    )


def _run_nvidia_smi(
    args: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: float,
) -> str:
    try:
        result = runner(
            ["nvidia-smi", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise GPUAdmissionQueryError(
            "nvidia-smi query timed out: "
            f"args={args!r} timeout_seconds={timeout_seconds}"
        ) from exc
    except OSError as exc:
        raise GPUAdmissionQueryError(
            f"nvidia-smi query could not start: args={args!r} error={exc}"
        ) from exc
    if result.returncode != 0:
        raise GPUAdmissionQueryError(
            "nvidia-smi query failed: "
            f"args={args!r} returncode={result.returncode} stderr={result.stderr.strip()!r}"
        )
    return result.stdout


def _parse_csv_rows(output: str) -> list[list[str]]:
    rows = []
    for line in output.splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def _query_gpu_index_by_uuid(
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: float = 10.0,
) -> dict[str, str]:
    gpu_rows = _parse_csv_rows(
        _run_nvidia_smi(
            ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            runner=runner,
            timeout_seconds=timeout_seconds,
        )
    )
    uuid_to_index = {}
    for row in gpu_rows:
        if len(row) != 2:
            raise GPUAdmissionQueryError(f"Unexpected GPU row: {row!r}")
        uuid_to_index[row[1]] = row[0]
    return uuid_to_index


def _query_compute_rows(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: float = 10.0,
) -> list[list[str]]:
    return _parse_csv_rows(
        _run_nvidia_smi(
            ["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            runner=runner,
            timeout_seconds=timeout_seconds,
        )
    )


def query_compute_processes(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: float = 10.0,
) -> list[GPUProcess]:
    uuid_to_index = _query_gpu_index_by_uuid(
        runner=runner, timeout_seconds=timeout_seconds
    )
    process_rows = _query_compute_rows(runner=runner, timeout_seconds=timeout_seconds)
    return _process_rows_to_gpu_processes(process_rows, uuid_to_index)


def _process_rows_to_gpu_processes(
    process_rows: list[list[str]], uuid_to_index: dict[str, str]
) -> list[GPUProcess]:
    processes = []
    for row in process_rows:
        if len(row) != 2:
            raise GPUAdmissionQueryError(f"Unexpected compute-app row: {row!r}")
        pid, gpu_uuid = row
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            raise GPUAdmissionQueryError(
                f"Compute process {pid} reports unknown GPU UUID {gpu_uuid}"
            )
        processes.append(GPUProcess(pid=pid, gpu_uuid=gpu_uuid, gpu_index=gpu_index))
    return processes


def scoped_busy_processes(
    devices: tuple[str, ...],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    timeout_seconds: float = 10.0,
) -> list[GPUProcess]:
    scoped = set(devices)
    uuid_to_index = _query_gpu_index_by_uuid(
        runner=runner, timeout_seconds=timeout_seconds
    )
    known_indices = set(uuid_to_index.values())
    if not scoped.issubset(known_indices):
        missing = sorted(scoped - known_indices)
        raise GPUAdmissionQueryError(f"Scoped GPU devices do not exist: {missing}")
    processes = _process_rows_to_gpu_processes(
        _query_compute_rows(runner=runner, timeout_seconds=timeout_seconds),
        uuid_to_index,
    )
    return [process for process in processes if process.gpu_index in scoped]


def _format_processes(processes: list[GPUProcess]) -> str:
    if not processes:
        return "none"
    return ", ".join(
        f"pid={process.pid}@gpu={process.gpu_index}" for process in processes
    )


def wait_for_scoped_gpus_free(
    config: GPUAdmissionConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, object]:
    deadline = monotonic() + config.timeout_seconds
    stable = 0
    polls = 0
    last_busy: list[GPUProcess] = []
    while True:
        if cancel_check is not None and cancel_check():
            raise GPUAdmissionCancelled("GPU admission wait was cancelled")
        now = monotonic()
        remaining = deadline - now
        if remaining <= 0:
            raise GPUAdmissionTimeout(
                "Timed out waiting for scoped GPUs to become free: "
                f"devices={list(config.devices)} timeout_seconds={config.timeout_seconds} "
                f"stable_required={config.stable_polls} stable_observed={stable} "
                f"last_busy={_format_processes(last_busy)}"
            )
        busy = scoped_busy_processes(
            config.devices,
            runner=runner,
            timeout_seconds=min(config.query_timeout_seconds, remaining),
        )
        polls += 1
        if busy:
            stable = 0
            last_busy = busy
        else:
            stable += 1
        now = monotonic()
        if now >= deadline:
            raise GPUAdmissionTimeout(
                "Timed out waiting for scoped GPUs to become free: "
                f"devices={list(config.devices)} timeout_seconds={config.timeout_seconds} "
                f"stable_required={config.stable_polls} stable_observed={stable} "
                f"last_busy={_format_processes(last_busy)}"
            )
        if stable >= config.stable_polls:
            return {
                "passed": True,
                "devices": list(config.devices),
                "polls": polls,
                "stable_polls": stable,
            }
        sleep(min(config.poll_seconds, max(0.0, deadline - now)))
