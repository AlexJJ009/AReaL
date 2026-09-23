# SPDX-License-Identifier: Apache-2.0
"""CPU tests for PPO predecessor GPU handoff supervision."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.sao.gpu_admission import GPUAdmissionConfig
from scripts.sao.ppo_supervisor import (
    STATE_FILE,
    STOP_REQUEST_FILE,
    GPUHandoffError,
    PPOHandoffConfig,
    StopRequestError,
    _DeferredTerminationSignals,
    _write_json,
    cleanup_process_group,
    finalize_gpu_handoff,
    request_stop,
    supervise_launch,
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
    def __init__(self, compute_outputs):
        self.compute_outputs = list(compute_outputs)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        assert cmd[0] == "nvidia-smi"
        if "--query-gpu=index,uuid" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-0\n", stderr="")
        if "--query-compute-apps=pid,gpu_uuid" in cmd:
            output = self.compute_outputs.pop(0) if self.compute_outputs else ""
            if isinstance(output, BaseException):
                raise output
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
        raise AssertionError(f"unexpected command: {cmd}")


def _config(*, timeout=180, min_grace=60, poll=2, stable=2):
    return PPOHandoffConfig(
        admission=GPUAdmissionConfig(
            devices=("0",),
            timeout_seconds=timeout,
            poll_seconds=poll,
            stable_polls=stable,
            query_timeout_seconds=10,
        ),
        min_release_grace_seconds=min_grace,
        cleanup_terminate_grace_seconds=1,
    )


def test_handoff_config_rejects_non_finite_grace_values():
    with pytest.raises(ValueError, match="finite"):
        PPOHandoffConfig(
            admission=GPUAdmissionConfig(devices=("0",), timeout_seconds=180),
            min_release_grace_seconds=math.nan,
        )
    with pytest.raises(ValueError, match="finite"):
        PPOHandoffConfig(
            admission=GPUAdmissionConfig(devices=("0",), timeout_seconds=180),
            cleanup_terminate_grace_seconds=math.inf,
        )


def test_finalize_retries_query_error_then_busy_then_stably_free():
    clock = FakeClock()
    runner = FakeNvidiaSmi(
        [
            subprocess.TimeoutExpired(["nvidia-smi"], 10),
            "123, GPU-0\n",
            "",
            "",
        ]
    )

    report = finalize_gpu_handoff(
        config=_config(),
        returncode=7,
        runner=runner,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert report.confirmed is True
    assert report.returncode == 7
    assert report.query_errors == 1
    assert report.polls == 3
    assert report.stable_polls == 2
    assert 60 in clock.sleeps
    assert clock.now < 180


def test_finalize_timeout_does_not_claim_confirmed_and_preserves_returncode():
    clock = FakeClock()
    runner = FakeNvidiaSmi(["123, GPU-0\n"] * 10)

    with pytest.raises(GPUHandoffError) as excinfo:
        finalize_gpu_handoff(
            config=_config(timeout=65, min_grace=60, poll=2),
            returncode=9,
            runner=runner,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert excinfo.value.returncode == 9
    assert "not confirmed" in str(excinfo.value)
    assert "last_busy=pid=123@gpu=0" in str(excinfo.value)


def test_deferred_signals_forward_only_while_child_is_running(monkeypatch):
    installed = {}
    forwarded = []

    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(
        signal, "signal", lambda sig, handler: installed.setdefault(sig, handler)
    )

    guard = _DeferredTerminationSignals(
        forward_pgid=9876, killpg=lambda pgid, sig: forwarded.append((pgid, sig))
    )
    with guard:
        guard._record(signal.SIGTERM, None)
        guard.forward_pgid = None
        guard._record(signal.SIGTERM, None)

    assert guard.received == [signal.SIGTERM, signal.SIGTERM]
    assert forwarded == [(9876, signal.SIGTERM)]
    assert signal.SIGTERM in installed


def test_cleanup_process_group_targets_only_supervised_group():
    clock = FakeClock()
    signals = []
    probes_before_exit = 2

    def killpg(pgid, sig):
        nonlocal probes_before_exit
        signals.append((pgid, sig))
        if sig == 0:
            if probes_before_exit <= 0:
                raise ProcessLookupError
            probes_before_exit -= 1

    cleaned = cleanup_process_group(
        4321,
        terminate_grace_seconds=5,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        killpg=killpg,
    )

    assert cleaned is True
    assert signals[0] == (4321, 0)
    assert (4321, signal.SIGTERM) in signals
    assert all(pgid == 4321 for pgid, _sig in signals)
    assert (4321, signal.SIGKILL) not in signals


def test_supervise_launch_preserves_train_exit_code_on_clean_handoff(tmp_path):
    clock = FakeClock()
    runner = FakeNvidiaSmi(["", ""])

    returncode = supervise_launch(
        tmp_path,
        command=(sys.executable, "-c", "print('train failed'); raise SystemExit(7)"),
        config=_config(),
        runner=runner,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert returncode == 7
    assert "train failed" in (tmp_path / "train.log").read_text()
    assert 60 in clock.sleeps
    assert '"returncode": 7' in (tmp_path / "process-result.json").read_text()
    handoff = (tmp_path / "handoff.json").read_text()
    assert '"confirmed": true' in handoff
    assert '"returncode": 7' in handoff


def test_supervise_launch_handoff_failure_raises_with_train_exit_code(tmp_path):
    clock = FakeClock()
    runner = FakeNvidiaSmi(["123, GPU-0\n"] * 10)

    with pytest.raises(GPUHandoffError) as excinfo:
        supervise_launch(
            tmp_path,
            command=(sys.executable, "-c", "raise SystemExit(3)"),
            config=_config(timeout=65, min_grace=60, poll=2),
            runner=runner,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert excinfo.value.returncode == 3
    assert "not confirmed" in str(excinfo.value)
    handoff = (tmp_path / "handoff.json").read_text()
    assert '"confirmed": false' in handoff
    assert '"returncode": 3' in handoff


def _read_json(path: Path):
    return json.loads(path.read_text())


def _wait_for(predicate, *, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out waiting for predicate")


def _write_fake_nvidia_smi(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "nvidia-smi"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'case "$*" in\n'
        "  *--query-gpu=index,uuid*) printf '0, GPU-0\\n' ;;\n"
        "  *--query-compute-apps=pid,gpu_uuid*) true ;;\n"
        "  *) printf 'unexpected fake nvidia-smi args: %s\\n' \"$*\" >&2; exit 2 ;;\n"
        "esac\n"
    )
    script.chmod(0o755)


def _supervisor_env(fake_bin: Path, *, min_grace: float = 0.2):
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SAO_PPO_HANDOFF_MIN_GRACE_SECONDS": str(min_grace),
            "SAO_PPO_HANDOFF_CLEANUP_GRACE_SECONDS": "1",
            "SAO_GPU_ADMISSION_DEVICES": "0",
            "SAO_GPU_ADMISSION_TIMEOUT_SECONDS": "5",
            "SAO_GPU_ADMISSION_POLL_SECONDS": "0.1",
            "SAO_GPU_ADMISSION_STABLE_POLLS": "1",
            "SAO_GPU_ADMISSION_QUERY_TIMEOUT_SECONDS": "1",
        }
    )
    return env


def test_request_stop_writes_run_bound_request(tmp_path):
    state = {
        "run_id": "run-123",
        "supervisor_pid": os.getpid(),
        "launch_dir": str(tmp_path),
        "status": "running",
    }
    _write_json(tmp_path / STATE_FILE, state)

    request = request_stop(tmp_path)

    assert request["run_id"] == "run-123"
    assert request["supervisor_pid"] == os.getpid()
    assert _read_json(tmp_path / STOP_REQUEST_FILE)["status"] == "requested"


def test_request_stop_rejects_completed_or_dead_state(tmp_path):
    _write_json(
        tmp_path / STATE_FILE,
        {
            "run_id": "run-123",
            "supervisor_pid": os.getpid(),
            "launch_dir": str(tmp_path),
            "status": "complete",
        },
    )

    with pytest.raises(StopRequestError, match="not running"):
        request_stop(tmp_path)


def test_supervise_launch_ignores_stale_stop_request(tmp_path):
    _write_json(
        tmp_path / STOP_REQUEST_FILE,
        {
            "run_id": "old-run",
            "supervisor_pid": os.getpid(),
            "requested_by_pid": os.getpid(),
            "requested_ns": time.time_ns(),
            "status": "requested",
        },
    )

    returncode = supervise_launch(
        tmp_path,
        command=(sys.executable, "-c", "raise SystemExit(5)"),
        config=_config(timeout=1, min_grace=0, poll=0.1, stable=1),
        runner=FakeNvidiaSmi([""]),
    )

    assert returncode == 5
    process_result = _read_json(tmp_path / "process-result.json")
    assert process_result["stop_request_cleanup_requested"] is False
    assert _read_json(tmp_path / "handoff.json")["confirmed"] is True
    assert _read_json(tmp_path / STATE_FILE)["status"] == "complete"


def test_cli_request_stop_keeps_supervisor_running_until_handoff(tmp_path):
    launch = tmp_path / "launch"
    launch.mkdir()
    fake_bin = tmp_path / "bin"
    _write_fake_nvidia_smi(fake_bin)
    events = launch / "events.jsonl"
    child = launch / "child.py"
    child.write_text(
        "import json, os, signal, sys, time\n"
        f"events = {str(events)!r}\n"
        "def write(event):\n"
        "    with open(events, 'a') as f:\n"
        "        f.write(json.dumps({'event': event, 'ns': time.time_ns(), 'pid': os.getpid()}) + '\\n')\n"
        "def handle(signum, _frame):\n"
        "    write('sigterm')\n"
        "    raise SystemExit(143)\n"
        "signal.signal(signal.SIGTERM, handle)\n"
        "write('start')\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    env = _supervisor_env(fake_bin, min_grace=0.2)
    supervisor = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.sao.ppo_supervisor",
            str(launch),
            "--",
            sys.executable,
            str(child),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        state = _wait_for(
            lambda: _read_json(launch / STATE_FILE)
            if (launch / STATE_FILE).is_file()
            and _read_json(launch / STATE_FILE).get("child_pid")
            else None
        )
        _wait_for(lambda: events.is_file() and "start" in events.read_text())
        start = time.monotonic()
        request = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.sao.ppo_supervisor",
                "--request-stop",
                str(launch),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert request.returncode == 0, request.stderr
        assert json.loads(request.stdout)["run_id"] == state["run_id"]
        returncode = supervisor.wait(timeout=10)
        elapsed = time.monotonic() - start
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=5)
    stdout, stderr = supervisor.communicate(timeout=1)

    assert returncode == 143, (stdout, stderr)
    assert elapsed >= 0.2
    assert "sigterm" in events.read_text()
    process_result = _read_json(launch / "process-result.json")
    assert process_result["stop_request_cleanup_requested"] is True
    assert process_result["signal_cleanup_requested"] is False
    handoff = _read_json(launch / "handoff.json")
    assert handoff["confirmed"] is True
    assert handoff["elapsed_seconds"] >= 0.2
    final_state = _read_json(launch / STATE_FILE)
    assert final_state["status"] == "complete"
    assert final_state["run_id"] == state["run_id"]
