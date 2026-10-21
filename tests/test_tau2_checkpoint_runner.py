# SPDX-License-Identifier: Apache-2.0
"""CPU tests: no model imports, credentials, GPU discovery or API calls."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from scripts.tau2 import checkpoint_eval as runner


@pytest.fixture
def recipe(tmp_path):
    """Construct an isolated run and a nonsecret runtime description."""
    run = tmp_path / "trial"
    config = run / "logs/root/tau2-ppo/trial/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "tau2-ppo",
                "trial_name": "trial",
                "cluster": {"fileroot": str(run)},
                "actor": {"path": "Qwen/Qwen3-4B-Instruct-2507@base-revision"},
                "econfig": {"benchmark_variant": "fixed"},
            }
        )
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "python": sys.executable,
                "hf_hub_cache": str(tmp_path / "hf"),
                "sao_artifact_root": str(tmp_path / "shared"),
                "tau2_data_dir": str(tmp_path / "data"),
                "ark_plan_env_file": str(tmp_path / "DO-NOT-READ.env"),
            }
        )
    )
    return runner.parse_args([str(run), "--runtime", str(runtime)])


def make_hf(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")


def fake_result(job, attempt):
    attempt.mkdir(parents=True, exist_ok=True)
    domain = {
        "episodes": 80,
        "evaluated_episodes": 78,
        "infra_excluded_episodes": 2,
        "context_limit_episodes": 3,
        "mean_at_k": 0.5,
        "pass_hat_k": {"1": 0.5, "2": 0.3, "3": 0.2, "4": 0.1},
        "pass_hat_k_tasks": {"1": 20, "2": 20, "3": 20, "4": 18},
    }
    (attempt / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": job["model_path"],
                "task_set": "ppo",
                "repeats": 4,
                "temperature": 0.0,
                "planned_episodes": 400,
            }
        )
    )
    (attempt / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "planned_episodes": 400,
                "completed_episodes": 398,
                "infra_excluded_episodes": 2,
                "repeats": 4,
                "official": {"by_domain": {"airline": domain}},
            }
        )
    )


def test_plan_default_steps_paths_gpu_ports_and_protocol(recipe):
    """Plan all fourteen checkpoints with disjoint slot ports and fixed protocol."""
    plan = runner.build_plan(recipe)
    jobs = plan["jobs"]
    assert [j["step"] for j in jobs] == [0, *range(10, 121, 10), 126]
    assert jobs[0]["model_path"].endswith(
        "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/base-revision"
    )
    assert jobs[1]["model_path"].endswith("default/epoch1epochstep0globalstep9")
    assert jobs[-1]["model_path"].endswith("default/epoch13epochstep8globalstep125")
    assert [j["gpu"] for j in jobs[:8]] == [0, 1, 2, 3, 4, 5, 6, 0]
    ports = [
        set(range(*slot["port_range"])) | {slot["policy_port"], slot["nccl_port"]}
        for slot in plan["slots"]
    ]
    assert all(
        not left & right for i, left in enumerate(ports) for right in ports[i + 1 :]
    )
    assert all(40080 not in p and 40081 not in p for p in ports)
    command = " ".join(jobs[0]["command"])
    for arg in (
        "--task-set ppo",
        "--repeats 4",
        "--temperature 0",
        "eval_episode_timeout_seconds=900",
        "evaluation_rollout.agent.tool_call_parser=qwen25",
        "evaluation_rollout.max_concurrent_rollouts=1",
        "sglang.disable_radix_cache=false",
        "cluster.n_gpus_per_node=1",
        "enable_offload=false",
        "ARK_PLAN_API_KEY",
        "deepseek-v4-1-flash-261017",
        "enable_thinking:false",
    ):
        assert arg in command
    assert f"stats_logger.fileroot={jobs[0]['output_dir']}/attempt-001" in command
    assert f"cluster.fileroot={jobs[0]['output_dir']}/attempt-001" in command
    assert (
        f"cluster.name_resolve.nfs_record_root={jobs[0]['output_dir']}/attempt-001/name-resolve"
        in command
    )
    user = " ".join(plan["user_command"])
    assert "--mem-fraction-static 0.75" in user
    assert "--reasoning-parser qwen3 --tool-call-parser qwen3_coder" in user
    assert plan["user_gpu"] == 7


def test_dry_run_does_not_spawn_source_secrets_or_write(recipe, monkeypatch, capsys):
    """Dry planning is safe even with absent credentials and checkpoints."""
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    recipe.dry_run = True
    assert runner.run(recipe) == 0
    assert "globalstep125" in capsys.readouterr().out
    assert not Path(
        recipe.output_root or Path(recipe.run) / "offline-checkpoint-eval"
    ).exists()


def test_checkpoint_shards_require_all_files(tmp_path):
    """An incomplete sharded save must not enter the queue."""
    make_hf(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
        )
    )
    (tmp_path / "model-1.safetensors").write_bytes(b"a")
    assert not runner.checkpoint_ready(tmp_path)
    (tmp_path / "model-2.safetensors").write_bytes(b"b")
    assert runner.checkpoint_ready(tmp_path)


def test_resume_requires_success_identity_and_complete_metrics(recipe):
    """A partial, stale or failed output is never treated as finished."""
    job = runner.build_plan(recipe)["jobs"][0]
    attempt = Path(job["output_dir"]) / "attempt-001"
    fake_result(job, attempt)
    assert runner.completed_attempt(job) is None
    runner.record_success(job, attempt)
    assert runner.completed_attempt(job) == attempt
    changed = dict(job, fingerprint="other-run")
    assert runner.completed_attempt(changed) is None
    summary = json.loads((attempt / "summary.json").read_text())
    summary["status"] = "partial"
    (attempt / "summary.json").write_text(json.dumps(summary))
    assert runner.completed_attempt(job) is None


def test_summary_preserves_official_domain_scores_and_exclusions(recipe):
    """Do not recompute official scores from rounded display values."""
    plan = runner.build_plan(recipe)
    job = plan["jobs"][0]
    attempt = Path(job["output_dir"]) / "attempt-001"
    fake_result(job, attempt)
    runner.record_success(job, attempt)
    runner.write_summaries(plan)
    root = Path(plan["output_root"])
    rows = json.loads((root / "summary.json").read_text())["checkpoints"]
    assert rows[0]["domains"]["airline"]["mean_at_k"] == 0.5
    assert rows[0]["domains"]["airline"]["pass_hat_k"]["4"] == 0.1
    md = (root / "summary.md").read_text()
    assert "infra-excluded" in md and "context-limit" in md and "0.100000" in md
    assert rows[1]["status"] == "missing"


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--workers", "8"),
        ("--judge-concurrency", "0"),
        ("--steps", "-1"),
        ("--port-base", "40080"),
    ],
)
def test_invalid_plan_rejected(recipe, flag, value):
    """Reject invalid slots, steps, concurrency and user port overlap."""
    with pytest.raises((ValueError, SystemExit)):
        args = runner.parse_args([recipe.run, "--runtime", recipe.runtime, flag, value])
        runner.build_plan(args)


def test_queue_retry_once_resume_and_missing_without_gpu(recipe, monkeypatch):
    """Drive the real queue with fake processes; preserve both attempts."""
    recipe.steps = "0,10,20"
    recipe.workers = 1
    plan = runner.build_plan(recipe)
    for job in plan["jobs"][:2]:
        make_hf(Path(job["model_path"]))
    calls = []

    def execute(job, attempt, *args, **kwargs):
        calls.append((job["step"], attempt.name))
        if job["step"] == 0 and attempt.name == "attempt-001":
            attempt.mkdir(parents=True, exist_ok=True)
            return 1
        fake_result(job, attempt)
        return 0

    monkeypatch.setattr(runner, "execute_checkpoint", execute)
    assert runner.run_queue(plan, recipe, object()) == 1
    assert calls == [(0, "attempt-001"), (0, "attempt-002"), (10, "attempt-001")]
    calls.clear()
    assert runner.run_queue(plan, recipe, object()) == 1
    assert calls == []


def test_cleanup_kills_owned_process_group_and_rejects_stale_pgid(tmp_path):
    """Exercise real TERM cleanup using only a CPU sleeping process tree."""
    pgid_file = tmp_path / "process.pgid"
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    try:
        runner.write_pgid(pgid_file, process.pid)
        runner.stop_process_group(process, pgid_file, grace=0.2)
        assert process.wait(timeout=2) == -signal.SIGTERM
        assert not pgid_file.exists()
        pgid_file.write_text(str(os.getpgrp()))
        with pytest.raises(ValueError):
            runner.stop_process_group(process, pgid_file, grace=0.1)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_cleanup_signals_entire_owned_process_tree(tmp_path):
    """The session leader reaps its child after both receive the group TERM."""
    import time

    child_pid = tmp_path / "child.pid"
    code = """
import signal, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
def terminate(signum, frame):
    child.wait(timeout=2)
    sys.exit(0)
signal.signal(signal.SIGTERM, terminate)
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(child_pid)],
        start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    try:
        deadline = time.monotonic() + 3
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid.exists()
        pgid_file = tmp_path / "tree.pgid"
        runner.write_pgid(pgid_file, process.pid)
        runner.stop_process_group(process, pgid_file, grace=1)
        assert process.returncode == 0
        with pytest.raises(ProcessLookupError):
            os.kill(int(child_pid.read_text()), 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_cleanup_escalates_when_owned_process_ignores_term(tmp_path):
    """A stubborn CPU process must not survive the cleanup grace period."""
    import time

    ready = tmp_path / "ready"
    code = "import signal,time,sys; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(sys.argv[1]).touch(); time.sleep(60)"
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(ready)],
        start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        pgid = tmp_path / "stubborn.pgid"
        runner.write_pgid(pgid, process.pid)
        runner.stop_process_group(process, pgid, grace=0.1)
        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def test_retry_rebinds_all_preload_write_paths(recipe):
    """load_expr_config writes before bind_eval_config; every attempt must be isolated."""
    job = runner.build_plan(recipe)["jobs"][0]
    attempt = Path(job["output_dir"]) / "attempt-002"
    process = Mock()
    process.poll.return_value = 0
    process.returncode = 0
    manager = Mock()
    manager.start.return_value = process
    assert (
        runner.execute_checkpoint(
            job, attempt, manager, runner.build_plan(recipe), recipe
        )
        == 0
    )
    command = manager.start.call_args.args[0]
    assert str(attempt) in command
    assert f"stats_logger.fileroot={attempt}" in command
    assert f"cluster.fileroot={attempt}" in command
    assert f"cluster.name_resolve.nfs_record_root={attempt}/name-resolve" in command
    assert not any("attempt-001" in arg for arg in command)


def test_wait_flag_allows_late_checkpoint(recipe, monkeypatch):
    """Only explicit waiting polls a checkpoint not present at queue entry."""
    recipe.steps = "10"
    recipe.wait_for_checkpoints = True
    recipe.wait_timeout = 0.5
    recipe.poll_interval = 0.01
    plan = runner.build_plan(recipe)
    model = Path(plan["jobs"][0]["model_path"])
    import threading

    timer = threading.Timer(0.03, lambda: make_hf(model))

    def execute(job, attempt, *args, **kwargs):
        fake_result(job, attempt)
        return 0

    monkeypatch.setattr(runner, "execute_checkpoint", execute)
    timer.start()
    try:
        assert runner.run_queue(plan, recipe, object()) == 0
    finally:
        timer.join()


def test_parallel_queue_caps_workers_and_serializes_each_gpu(recipe, monkeypatch):
    """Use a barrier to prove real queue overlap and enforce one policy per GPU."""
    import threading
    import time

    recipe.steps = "0,10,20,30"
    recipe.workers = 2
    plan = runner.build_plan(recipe)
    for job in plan["jobs"]:
        make_hf(Path(job["model_path"]))
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = set()
    peak = 0

    def execute(job, attempt, *args):
        nonlocal peak
        with lock:
            assert job["gpu"] not in active
            active.add(job["gpu"])
            peak = max(peak, len(active))
        barrier.wait(timeout=3)
        time.sleep(0.02)
        fake_result(job, attempt)
        with lock:
            active.remove(job["gpu"])
        return 0

    monkeypatch.setattr(runner, "execute_checkpoint", execute)
    assert runner.run_queue(plan, recipe, object()) == 0
    assert peak == 2 and not active


def test_failed_attempt_budget_survives_resume(recipe, monkeypatch):
    """A repeated invocation cannot silently retry a failed checkpoint forever."""
    recipe.steps = "0"
    plan = runner.build_plan(recipe)
    make_hf(Path(plan["jobs"][0]["model_path"]))
    execute = Mock(return_value=7)
    monkeypatch.setattr(runner, "execute_checkpoint", execute)
    assert runner.run_queue(plan, recipe, object()) == 1
    assert execute.call_count == 2
    assert runner.run_queue(plan, recipe, object()) == 1
    assert execute.call_count == 2


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_signal_run_cleans_user_session_without_api_or_gpu(recipe, tmp_path, sig):
    """Synthetic run uses only sleeping CPU processes; emulate pueue TERM."""
    ready = tmp_path / "ready"
    pidfile = tmp_path / "child.pid"
    driver = tmp_path / "driver.py"
    driver.write_text("""
import json, os, sys, time
from pathlib import Path
from scripts.tau2 import checkpoint_eval as r
args = r.parse_args(sys.argv[1:3] + ["--runtime", sys.argv[3]])
plan = r.build_plan(args)
plan["user_command"] = [sys.executable, "-c", "import os,time; from pathlib import Path; Path(" + repr(sys.argv[5]) + ").write_text(str(os.getpid())); time.sleep(60)"]
r.build_plan = lambda args: plan
r.parse_args = lambda: args
r.checkpoint_ready = lambda path: True
r.process_env = lambda plan, gpu: {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
r.assert_ports_free = lambda plan: None
def ready(*a, **kw):
    Path(sys.argv[4]).write_text("ready")
r.wait_http = ready
def queue(plan, args, manager):
    manager.stopped.wait(20)
    raise InterruptedError("synthetic cancellation")
r.run_queue = queue
r.main()
""")
    # args.run is followed by a valid optional flag to keep this driver tiny.
    process = subprocess.Popen(
        [
            sys.executable,
            str(driver),
            recipe.run,
            "--wait-for-checkpoints",
            recipe.runtime,
            str(ready),
            str(pidfile),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(runner.REPO)},
    )
    try:
        import time

        deadline = time.monotonic() + 5
        while not (ready.exists() and pidfile.exists()) and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        assert ready.exists() and pidfile.exists()
        os.kill(process.pid, sig)
        assert process.wait(timeout=8) == 143
        child = int(pidfile.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        assert not (
            Path(recipe.run) / "offline-checkpoint-eval/user-server.pgid"
        ).exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
