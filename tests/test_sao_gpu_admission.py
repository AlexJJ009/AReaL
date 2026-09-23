# SPDX-License-Identifier: Apache-2.0
"""CPU tests for bounded GPU admission waits."""

from __future__ import annotations

import math
import subprocess

import pytest

from scripts.sao.gpu_admission import (
    GPUAdmissionCancelled,
    GPUAdmissionConfig,
    GPUAdmissionQueryError,
    GPUAdmissionTimeout,
    admission_config_from_env,
    query_compute_processes,
    wait_for_scoped_gpus_free,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeNvidiaSmi:
    def __init__(self, compute_outputs, *, gpu_output=None, fail_gpu=False):
        self.compute_outputs = list(compute_outputs)
        self.gpu_output = gpu_output or "0, GPU-0\n1, GPU-1\n"
        self.fail_gpu = fail_gpu
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        assert cmd[0] == "nvidia-smi"
        if self.fail_gpu == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
        if self.fail_gpu == "oserror":
            raise OSError("nvidia-smi missing")
        if "--query-gpu=index,uuid" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                1 if self.fail_gpu else 0,
                stdout="" if self.fail_gpu else self.gpu_output,
                stderr="gpu query failed" if self.fail_gpu else "",
            )
        if "--query-compute-apps=pid,gpu_uuid" in cmd:
            if not self.compute_outputs:
                output = ""
            else:
                output = self.compute_outputs.pop(0)
                if isinstance(output, Exception):
                    return subprocess.CompletedProcess(
                        cmd, 1, stdout="", stderr=str(output)
                    )
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")


def test_wait_for_scoped_gpus_busy_then_stably_free():
    clock = FakeClock()
    runner = FakeNvidiaSmi(["123, GPU-0\n", "", ""])
    config = GPUAdmissionConfig(
        devices=("0", "1"), timeout_seconds=10, poll_seconds=2, stable_polls=2
    )

    report = wait_for_scoped_gpus_free(
        config, runner=runner, sleep=clock.sleep, monotonic=clock.monotonic
    )

    assert report["passed"] is True
    assert report["polls"] == 3
    assert clock.sleeps == [2, 2]
    assert all(call[1]["timeout"] <= 10 for call in runner.calls)


def test_wait_for_scoped_gpus_ignores_out_of_scope_processes():
    clock = FakeClock()
    runner = FakeNvidiaSmi(["123, GPU-1\n"])
    config = GPUAdmissionConfig(
        devices=("0",), timeout_seconds=1, poll_seconds=1, stable_polls=1
    )

    report = wait_for_scoped_gpus_free(
        config, runner=runner, sleep=clock.sleep, monotonic=clock.monotonic
    )

    assert report["passed"] is True
    assert report["devices"] == ["0"]


def test_wait_for_scoped_gpus_all_busy_times_out_with_diagnostics():
    clock = FakeClock()
    runner = FakeNvidiaSmi(["123, GPU-0\n"] * 10)
    config = GPUAdmissionConfig(
        devices=("0",), timeout_seconds=3, poll_seconds=1, stable_polls=2
    )

    with pytest.raises(GPUAdmissionTimeout) as excinfo:
        wait_for_scoped_gpus_free(
            config, runner=runner, sleep=clock.sleep, monotonic=clock.monotonic
        )

    message = str(excinfo.value)
    assert "devices=['0']" in message
    assert "pid=123@gpu=0" in message
    assert "stable_required=2" in message


def test_query_compute_processes_reports_query_failures():
    runner = FakeNvidiaSmi([RuntimeError("compute query failed")])

    with pytest.raises(GPUAdmissionQueryError, match="compute query failed"):
        query_compute_processes(runner=runner)


def test_wait_for_scoped_gpus_cancel_check_stops_before_query():
    runner = FakeNvidiaSmi([""])
    config = GPUAdmissionConfig(
        devices=("0",), timeout_seconds=10, poll_seconds=1, stable_polls=1
    )

    with pytest.raises(GPUAdmissionCancelled):
        wait_for_scoped_gpus_free(config, runner=runner, cancel_check=lambda: True)

    assert runner.calls == []


def test_wait_for_scoped_gpus_zero_budget_times_out_without_query():
    runner = FakeNvidiaSmi([""])
    config = GPUAdmissionConfig(
        devices=("0",), timeout_seconds=0, poll_seconds=1, stable_polls=1
    )

    with pytest.raises(GPUAdmissionTimeout):
        wait_for_scoped_gpus_free(config, runner=runner)

    assert runner.calls == []


def test_wait_for_scoped_gpus_times_out_when_query_crosses_deadline():
    clock = FakeClock()

    def slow_empty_query(cmd, **kwargs):
        assert kwargs["timeout"] <= 1
        if "--query-gpu=index,uuid" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-0\n", stderr="")
        if "--query-compute-apps=pid,gpu_uuid" in cmd:
            clock.now = 1.1
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    config = GPUAdmissionConfig(
        devices=("0",), timeout_seconds=1, poll_seconds=1, stable_polls=1
    )

    with pytest.raises(GPUAdmissionTimeout) as excinfo:
        wait_for_scoped_gpus_free(
            config,
            runner=slow_empty_query,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert "stable_observed=1" in str(excinfo.value)
    assert clock.sleeps == []


def test_admission_config_from_env_uses_scoped_devices_and_defaults():
    config = admission_config_from_env(
        {
            "SAO_GPU_ADMISSION_DEVICES": "0, 2,7",
            "SAO_GPU_ADMISSION_TIMEOUT_SECONDS": "5",
            "SAO_GPU_ADMISSION_POLL_SECONDS": "0.5",
            "SAO_GPU_ADMISSION_STABLE_POLLS": "3",
        }
    )

    assert config.devices == ("0", "2", "7")
    assert config.timeout_seconds == 5
    assert config.poll_seconds == 0.5
    assert config.stable_polls == 3
    assert config.query_timeout_seconds == 10


def test_admission_config_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        GPUAdmissionConfig(devices=("0",), timeout_seconds=math.inf)
    with pytest.raises(ValueError, match="finite"):
        GPUAdmissionConfig(devices=("0",), poll_seconds=math.nan)
    with pytest.raises(ValueError, match="finite"):
        GPUAdmissionConfig(devices=("0",), query_timeout_seconds=math.inf)


def test_query_timeout_becomes_query_error():
    runner = FakeNvidiaSmi([""], fail_gpu="timeout")

    with pytest.raises(GPUAdmissionQueryError, match="timed out"):
        query_compute_processes(runner=runner, timeout_seconds=0.25)


def test_query_oserror_becomes_query_error():
    runner = FakeNvidiaSmi([""], fail_gpu="oserror")

    with pytest.raises(GPUAdmissionQueryError, match="could not start"):
        query_compute_processes(runner=runner)


def test_wait_for_scoped_gpus_rejects_nonexistent_devices():
    runner = FakeNvidiaSmi([""], gpu_output="0, GPU-0\n")
    config = GPUAdmissionConfig(devices=("1",), timeout_seconds=1)

    with pytest.raises(GPUAdmissionQueryError, match="do not exist"):
        wait_for_scoped_gpus_free(config, runner=runner)


def test_wait_caps_query_timeout_by_remaining_deadline():
    clock = FakeClock()
    runner = FakeNvidiaSmi(["123, GPU-0\n"] * 10)
    config = GPUAdmissionConfig(
        devices=("0",),
        timeout_seconds=3,
        poll_seconds=2,
        stable_polls=2,
        query_timeout_seconds=10,
    )

    with pytest.raises(GPUAdmissionTimeout):
        wait_for_scoped_gpus_free(
            config, runner=runner, sleep=clock.sleep, monotonic=clock.monotonic
        )

    timeouts = [call[1]["timeout"] for call in runner.calls]
    assert min(timeouts) <= 1
