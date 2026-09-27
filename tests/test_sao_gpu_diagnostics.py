# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the opt-in SAO GPU diagnostic wrapper."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

from scripts.sao.gpu_admission import GPUAdmissionConfig, GPUAdmissionQueryError
from scripts.sao.gpu_diagnostics import (
    _parse_proc_stat_starttime,
    _start_monitors,
    query_compute_memory_snapshot,
    read_process_owner,
    run_diagnostics,
)


class FakeNvidiaSmi:
    def __init__(self, compute_outputs):
        self.compute_outputs = list(compute_outputs)

    def __call__(self, cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        if "--query-gpu=index,uuid" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="0, GPU-0\n", stderr="")
        if "--query-compute-apps=pid,gpu_uuid,used_gpu_memory" in cmd:
            output = self.compute_outputs.pop(0) if self.compute_outputs else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")
        if "--query-compute-apps=pid,gpu_uuid" in cmd:
            output = self.compute_outputs.pop(0) if self.compute_outputs else ""
            rows = []
            for line in output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2:
                    rows.append(f"{parts[0]}, {parts[1]}")
            return subprocess.CompletedProcess(
                cmd, 0, stdout="\n".join(rows) + ("\n" if rows else ""), stderr=""
            )
        raise AssertionError(f"unexpected command: {cmd}")


class FakeMonitorProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


def _admission(*, timeout=5, stable=1):
    return GPUAdmissionConfig(
        devices=("0",),
        timeout_seconds=timeout,
        poll_seconds=0.1,
        stable_polls=stable,
        query_timeout_seconds=1,
    )


def _proc_entry(root: Path, pid: int, *, comm: str = "python") -> None:
    proc = root / str(pid)
    proc.mkdir(parents=True)
    (proc / "comm").write_text(comm)
    fields = ["S", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    fields += ["10", "11", "12", "13", "14", "15", "16", "17", "18", "98765"]
    (proc / "stat").write_text(f"{pid} ({comm}) {' '.join(fields)}\n")
    (proc / "cwd").symlink_to(root)
    exe = proc / "exe-target"
    exe.write_text("")
    (proc / "exe").symlink_to(exe)


def _popen_with_fake_monitors(commands):
    def popen(cmd, **kwargs):
        if cmd[0] == "nvidia-smi":
            commands.append((cmd, kwargs))
            return FakeMonitorProcess()
        return subprocess.Popen(cmd, **kwargs)

    return popen


def test_query_compute_memory_snapshot_records_owner_without_cmdline_or_env(tmp_path):
    proc_root = tmp_path / "proc"
    _proc_entry(proc_root, 123, comm="train")

    rows = query_compute_memory_snapshot(
        ("0",),
        runner=FakeNvidiaSmi(["123, GPU-0, 4096\n"]),
        proc_root=proc_root,
        timeout_seconds=1,
    )

    assert rows == [
        {
            "pid": 123,
            "gpu_uuid": "GPU-0",
            "gpu_index": "0",
            "used_gpu_memory_mib": "4096",
            "owner": {
                "cwd": str(proc_root),
                "exe": str(proc_root / "123" / "exe-target"),
                "comm": "train",
                "starttime": 98765,
                "status": "available",
            },
        }
    ]
    assert "cmdline" not in json.dumps(rows)
    assert "environ" not in json.dumps(rows)


def test_proc_stat_parser_handles_comm_with_spaces():
    stat = "123 (python worker) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 24680"

    assert _parse_proc_stat_starttime(stat) == 24680


def test_start_monitors_uses_standard_dmon_and_pmon_with_timestamps(tmp_path):
    commands = []

    processes, streams = _start_monitors(
        tmp_path, ("0",), delay_seconds=5, popen=_popen_with_fake_monitors(commands)
    )
    for process in processes:
        process.terminate()
    for stream in streams:
        stream.close()

    assert [cmd[1] for cmd, _kwargs in commands] == ["dmon", "pmon"]
    for cmd, kwargs in commands:
        assert cmd[:2] == ["nvidia-smi", cmd[1]]
        assert "-o" in cmd and "DT" in cmd
        assert "-d" in cmd and "5" in cmd
        if cmd[1] == "pmon":
            assert "-s" in cmd and "um" in cmd
        assert kwargs["stderr"] == subprocess.STDOUT
    assert (tmp_path / "nvidia-smi-dmon.log").is_file()
    assert (tmp_path / "nvidia-smi-pmon.log").is_file()


def test_start_monitors_cleans_up_first_process_when_second_fails(tmp_path):
    first = FakeMonitorProcess()

    def popen(cmd, **kwargs):
        if cmd[1] == "dmon":
            return first
        raise OSError("pmon unavailable")

    try:
        _start_monitors(tmp_path, ("0",), delay_seconds=5, popen=popen)
    except GPUAdmissionQueryError as exc:
        assert "pmon" in str(exc)
    else:
        raise AssertionError("pmon start failure was accepted")

    assert first.terminated is True


def test_run_diagnostics_captures_initial_before_wait_and_child_exit(tmp_path):
    proc_root = tmp_path / "proc"
    _proc_entry(proc_root, 222)
    commands = []

    returncode = run_diagnostics(
        (sys.executable, "-c", "raise SystemExit(7)"),
        output_dir=tmp_path,
        admission=_admission(),
        runner=FakeNvidiaSmi(["222, GPU-0, 1024\n", "222, GPU-0, 1024\n", "", ""]),
        popen=_popen_with_fake_monitors(commands),
        sleep=lambda _seconds: None,
        proc_root=proc_root,
    )

    assert returncode == 7
    result = json.loads((tmp_path / "gpu-diagnostics-result.json").read_text())
    assert result["returncode"] == 7
    assert result["initial_process_count"] == 1
    assert result["argv0"] == sys.executable
    assert result["argc"] == 3
    assert "raise SystemExit" not in json.dumps(result)
    samples = [
        json.loads(line)
        for line in (tmp_path / "gpu-process-samples.jsonl").read_text().splitlines()
    ]
    assert samples[0]["phase"] == "initial_pre_monitor"
    assert samples[1]["phase"] == "initial_pre_admission"
    assert samples[-1]["phase"] == "final"


def test_run_diagnostics_signal_during_admission_stops_monitor_and_returns_signal(
    tmp_path, monkeypatch
):
    installed = {}
    commands = []

    def fake_signal(sig, handler):
        installed[sig] = handler
        return signal.SIG_DFL

    def sleep(_seconds):
        if installed.get(signal.SIGTERM) is not None:
            installed[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.setattr(signal, "getsignal", lambda _sig: signal.SIG_DFL)

    returncode = run_diagnostics(
        (sys.executable, "-c", "raise SystemExit(0)"),
        output_dir=tmp_path,
        admission=_admission(timeout=1, stable=2),
        runner=FakeNvidiaSmi(["333, GPU-0, 2048\n"] * 10),
        popen=_popen_with_fake_monitors(commands),
        sleep=sleep,
    )

    assert returncode == 128 + signal.SIGTERM
    result = json.loads((tmp_path / "gpu-diagnostics-result.json").read_text())
    assert result["error"] == "GPU admission wait was cancelled"
    assert result["signals_received"] == [signal.SIGTERM]


def test_cli_help_is_cpu_only_and_documents_opt_in():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.sao.gpu_diagnostics", "--help"],
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "opt-in only" in result.stdout
    assert "never reads process environments or full command lines" in " ".join(
        result.stdout.split()
    )


def test_read_process_owner_tolerates_exited_process(tmp_path):
    owner = read_process_owner(999999, proc_root=tmp_path / "proc")

    assert owner == {
        "cwd": None,
        "exe": None,
        "comm": None,
        "starttime": None,
        "status": "unavailable_in_current_pid_namespace_or_exited",
    }
