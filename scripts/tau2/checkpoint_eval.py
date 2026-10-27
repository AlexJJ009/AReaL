# SPDX-License-Identifier: Apache-2.0
"""Queue offline PPO checkpoints on GPUs 0..6, with one user server on GPU 7.

With --user-backend deepseek-official, no user server starts and GPUs 0..7 all
run policies. --judge-backend independently selects the NL judge provider.

Planning reads YAML as data, never through load_expr_config. --dry-run does not
source runtime/key files, create directories, probe ports or import model code.
See README_checkpoint_eval.md for runtime JSON and the resume protocol.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
USER_MODEL = "qwen35-4b-user"
USER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
JUDGE_MODEL = "deepseek-v4-1-flash-261017"
JUDGE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
USER_ARGS = "{temperature:0,top_p:1.0,max_tokens:4096,extra_body:{chat_template_kwargs:{enable_thinking:false}}}"
DEFAULT_STEPS = "0," + ",".join(map(str, range(10, 121, 10))) + ",126"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_URL = "https://api.deepseek.com"
RUNTIME_FIELDS = ("python", "hf_hub_cache", "sao_artifact_root", "tau2_data_dir")
# Fixed-v3 test tasks per domain; every task runs 4 trials.
TEST_TASKS = {"airline": 20, "retail": 40, "telecom": 40}
PPO_EPISODES = 4 * sum(TEST_TASKS.values())
KEY_FIELD = {
    "qwen-local": "ark_plan_env_file",
    "volcano-plan": "ark_plan_env_file",
    "deepseek-official": "deepseek_env_file",
}
KEY_VARIABLE = {
    "qwen-local": "ARK_PLAN_API_KEY",
    "volcano-plan": "ARK_PLAN_API_KEY",
    "deepseek-official": "DEEPSEEK_API_KEY",
}
QUOTA_EXIT_CODE = 86  # evaluate.py: provider breaker is fatal, never retry
PROCESS_OWNER_ENV = "TAU2_CHECKPOINT_PROCESS_OWNER"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", help="Run directory, or trial name with --artifact-root."
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=os.environ.get("TAU2_PPO_ARTIFACT_ROOT")
    )
    parser.add_argument(
        "--runtime",
        required=True,
        help="Nonsecret runtime JSON; keys are sourced only on execution.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Default: run/logs/root/tau2-ppo/<trial>/config.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--steps",
        default=DEFAULT_STEPS,
        help="Completed update counts, comma separated; 0 is base actor.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Extra absolute HF checkpoint evaluated with the same config; repeatable.",
    )
    parser.add_argument(
        "--user-backend",
        choices=("qwen-local", "deepseek-official"),
        default="qwen-local",
        help="qwen-local: Qwen user on GPU 7. "
        "deepseek-official: official API user, no user server.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("volcano-plan", "deepseek-official"),
        help="NL judge provider (default: Volcano for Qwen users, official for official users).",
    )
    parser.add_argument(
        "--task-set",
        choices=("ppo", "test"),
        default="ppo",
        help="ppo: the fixed 100-task set over all three domains. "
        "test: the fixed test split of the run's own config domains.",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=9)
    parser.add_argument("--final-step", type=int, default=126)
    parser.add_argument(
        "--workers",
        type=int,
        default=7,
        help="GPUs 0..workers-1: 1..7 with the local user on GPU 7, 1..8 otherwise.",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=1,
        help="Concurrent episodes per process (one sequential judge stream per episode).",
    )
    parser.add_argument(
        "--max-parallel-steps",
        type=int,
        help="Maximum simultaneous checkpoint attempts (default: --workers).",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=21000,
        help="Each GPU gets a disjoint 100-port block.",
    )
    parser.add_argument("--user-port", type=int, default=40080)
    parser.add_argument("--user-nccl-port", type=int, default=40081)
    parser.add_argument(
        "--user-seed",
        type=int,
        default=42,
        help="SGLang --random-seed of the local Qwen user server.",
    )
    parser.add_argument("--wait-for-checkpoints", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=86400)
    parser.add_argument("--poll-interval", type=float, default=5)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument(
        "--port-wait-timeout",
        type=float,
        default=180,
        help="Seconds to wait for prior servers to release TCP/UDP ports before launch.",
    )
    parser.add_argument(
        "--checkpoint-timeout",
        type=float,
        default=172800,
        help="Wall clock bound per attempt, including server startup.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def snapshot_path(value: str, cache: Path) -> Path:
    if "@" in value:
        repo, revision = value.split("@", 1)
        return cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "Base actor must be a pinned repo@revision or absolute HF path"
        )
    return path


def runtime_command(
    command: list[str],
    runtime: dict[str, str],
    *,
    judge: bool,
    backend: str = "qwen-local",
    judge_backend: str | None = None,
) -> list[str]:
    """Source the existing CUDA environment; never echo/capture credential bytes."""
    setup = [
        "set -euo pipefail",
        "set +x",
        'source "$1"',
        "shift",
        'export CPATH="$1"',
        "shift",
    ]
    args = [
        str(REPO / "scripts/sao/runtime_env.sh"),
        str(
            Path(runtime["python"]).parent.parent
            / "lib/python3.12/site-packages/nvidia/curand/include"
        ),
    ]
    credentials = []
    if judge:
        if backend == "deepseek-official":
            credentials.append(backend)
        credentials.append(judge_backend or backend)
    for credential in dict.fromkeys(credentials):
        setup += [
            'test "$(stat -c %a -- "$1")" = 600',
            "set -a",
            'source "$1"',
            "set +a",
            "shift",
            f': "${{{KEY_VARIABLE[credential]}:?missing API key}}"',
        ]
        args.append(runtime[KEY_FIELD[credential]])
    setup.append('exec "$@"')
    return ["bash", "-c", "; ".join(setup), "tau2-runtime", *args, *command]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    local_user = args.user_backend == "qwen-local"
    judge_backend = args.judge_backend or (
        "volcano-plan" if local_user else "deepseek-official"
    )
    max_workers = 7 if local_user else 8
    if not 1 <= args.workers <= max_workers or args.judge_concurrency < 1:
        raise ValueError(
            f"workers must be 1..{max_workers} and judge-concurrency positive"
        )
    if args.steps_per_epoch < 1 or args.final_step < 1:
        raise ValueError("steps-per-epoch and final-step must be positive")
    if args.max_parallel_steps is not None and args.max_parallel_steps < 1:
        raise ValueError("max-parallel-steps must be positive")
    if min(
        args.wait_timeout,
        args.poll_interval,
        args.startup_timeout,
        args.checkpoint_timeout,
        args.port_wait_timeout,
    ) <= 0 or not math.isfinite(args.port_wait_timeout):
        raise ValueError("timeouts and poll-interval must be positive")
    steps = list(dict.fromkeys(int(s) for s in args.steps.split(",")))
    if not steps or any(s < 0 or s > args.final_step for s in steps):
        raise ValueError("steps must be between 0 and final-step")
    run = Path(args.run).expanduser()
    if args.artifact_root and not run.is_absolute():
        run = Path(args.artifact_root) / run
    run = run.resolve()
    trial = run.name
    config_path = (
        args.config or run / "logs/root/tau2-ppo" / trial / "config.yaml"
    ).resolve()
    raw_bytes = config_path.read_bytes()
    config = yaml.safe_load(raw_bytes)
    if config["trial_name"] != trial or config["experiment_name"] != "tau2-ppo":
        raise ValueError("Run directory must match the tau2-ppo config trial_name")
    if config["econfig"]["benchmark_variant"] != "fixed":
        raise ValueError("PPO checkpoint evaluation requires fixed data")
    if args.task_set == "ppo":
        planned_episodes = PPO_EPISODES
    else:
        domains = config["domains"]
        if isinstance(domains, str):
            domains = [domains]
        if not domains or set(domains) - set(TEST_TASKS):
            raise ValueError(f"--task-set test supports {sorted(TEST_TASKS)} domains")
        planned_episodes = 4 * sum(TEST_TASKS[d] for d in dict.fromkeys(domains))
    runtime = read_json(Path(args.runtime))
    credentials = [KEY_FIELD[judge_backend]]
    if not local_user:
        credentials.append(KEY_FIELD[args.user_backend])
    required = (*RUNTIME_FIELDS, *dict.fromkeys(credentials))
    if runtime is None or any(
        not runtime.get(k) or not Path(runtime[k]).is_absolute() for k in required
    ):
        raise ValueError(
            "Runtime JSON must provide absolute paths for " + ", ".join(required)
        )
    if set(runtime) - {*RUNTIME_FIELDS, *KEY_FIELD.values()}:
        raise ValueError("Unknown runtime fields; keep secrets out of the runtime JSON")
    output = (args.output_root or run / "offline-checkpoint-eval").resolve()
    fileroot = Path(config["cluster"]["fileroot"])
    if not fileroot.is_absolute():
        raise ValueError("Resolved cluster.fileroot must be absolute")
    slots = []
    used = {args.user_port, args.user_nccl_port} if local_user else set()
    if local_user and (len(used) != 2 or not all(1024 <= p <= 65535 for p in used)):
        raise ValueError("User ports must be distinct and in 1024..65535")
    for gpu in range(args.workers):
        base = args.port_base + gpu * 100
        block = set(range(base, base + 100))
        if base < 1024 or base + 99 > 32767 or block & used:
            raise ValueError(
                "Port blocks must be disjoint, below 32768, and exclude user ports"
            )
        used.update(block)
        slots.append(
            {
                "gpu": gpu,
                "policy_port": base,
                "nccl_port": base + 1,
                "port_range": [base + 2, base + 100],
            }
        )  # half-open controller/worker pool
    targets = []
    for step in steps:
        g = step - 1
        model = (
            snapshot_path(config["actor"]["path"], Path(runtime["hf_hub_cache"]))
            if step == 0
            else (
                fileroot
                / "checkpoints/root/tau2-ppo"
                / trial
                / "default"
                / f"epoch{g // args.steps_per_epoch}epochstep{g % args.steps_per_epoch}globalstep{g}"
            )
        )
        targets.append((step, f"step-{step:03d}", model))
    labels = set()
    for item in args.checkpoint:
        label, sep, value = item.partition("=")
        path = Path(value)
        if (
            not sep
            or not label.replace("-", "").replace("_", "").isalnum()
            or label in labels
            or not path.is_absolute()
        ):
            raise ValueError(
                "--checkpoint must be a unique LABEL=/absolute/path ([A-Za-z0-9_-])"
            )
        labels.add(label)
        targets.append((label, f"ckpt-{label}", path))
    if local_user:
        user_overrides = [
            f"econfig.user_llm={USER_MODEL}",
            f"econfig.user_llm_base_url=http://127.0.0.1:{args.user_port}/v1",
            "econfig.user_llm_api_key_env=TAU2_LOCAL_USER_API_KEY",
            "++econfig.user_llm_args=" + USER_ARGS,
        ]
    else:
        # Same model, endpoint and default (thinking) mode as the pre-10-17 PPO runs.
        user_overrides = [
            f"econfig.user_llm={DEEPSEEK_MODEL}",
            f"econfig.user_llm_base_url={DEEPSEEK_URL}",
            "econfig.user_llm_api_key_env=DEEPSEEK_API_KEY",
            "++econfig.user_llm_args={temperature:0.0}",
        ]
    official_judge = judge_backend == "deepseek-official"
    user_overrides += [
        f"econfig.nl_judge_llm={DEEPSEEK_MODEL if official_judge else JUDGE_MODEL}",
        f"econfig.nl_judge_llm_base_url={DEEPSEEK_URL if official_judge else JUDGE_URL}",
        f"econfig.nl_judge_llm_api_key_env={KEY_VARIABLE[judge_backend]}",
        "++econfig.nl_judge_llm_args={temperature:0.0}",
    ]
    jobs = []
    for index, (step, name, model) in enumerate(targets):
        slot = slots[index % len(slots)]
        directory = output / name
        first_attempt = directory / "attempt-001"
        overrides = [
            f"cluster.fileroot={first_attempt}",
            "cluster.name_resolve.type=nfs",
            f"cluster.name_resolve.nfs_record_root={first_attempt / 'name-resolve'}",
            f"stats_logger.fileroot={first_attempt}",
            "stats_logger.experiment_name=tau2-ppo-offline-eval",
            f"stats_logger.trial_name={trial}-{name}",
            "cluster.n_gpus_per_node=1",
            "enable_offload=false",
            "evaluation_rollout.backend=sglang:d1p1t1",
            "evaluation_rollout._version=v1",
            "evaluation_rollout.scheduling_strategy.type=separation",
            f"evaluation_rollout.max_concurrent_rollouts={args.judge_concurrency}",
            "evaluation_rollout.agent.tool_call_parser=qwen25",
            "evaluation_rollout.agent.reasoning_parser=qwen3",
            "eval_episode_timeout_seconds=900",
            "eval_gconfig.seed=null",
            "eval_gconfig.top_p=1.0",
            "eval_gconfig.greedy=false",
            "sglang.disable_radix_cache=false",
            *user_overrides,
        ]
        command = [
            runtime["python"],
            str(REPO / "scripts/tau2/evaluate.py"),
            "--config",
            str(config_path),
            "--model-path",
            str(model),
            "--output-root",
            str(directory / "attempt-001"),
            "--task-set",
            args.task_set,
            "--judge-backend",
            judge_backend,
            "--repeats",
            "4",
            "--temperature",
            "0",
            "--policy-port",
            str(slot["policy_port"]),
            "--policy-nccl-port",
            str(slot["nccl_port"]),
            *overrides,
        ]
        # Output location does not alter scores; keep success reusable when only
        # scheduling knobs (GPU count/ports) change.
        protocol = [
            o for o in overrides if not o.startswith(("cluster.", "stats_logger."))
        ]
        identity = {
            "config_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "model_path": str(model),
            "protocol": protocol,
            "runtime": runtime,
            "code_sha256": code_digest(),
            "judge_backend": judge_backend,
        }
        if args.task_set != "ppo":  # keep fingerprints of earlier ppo runs unchanged
            identity["task_set"] = args.task_set
        if local_user:
            identity["user_seed"] = args.user_seed
        jobs.append(
            {
                "step": step,
                "task_set": args.task_set,
                "judge_backend": judge_backend,
                "planned_episodes": planned_episodes,
                **slot,
                "model_path": str(model),
                "output_dir": str(directory),
                "command": runtime_command(
                    command,
                    runtime,
                    judge=True,
                    backend=args.user_backend,
                    judge_backend=judge_backend,
                ),
                "fingerprint": hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    user_path = (
        Path(runtime["hf_hub_cache"])
        / "models--Qwen--Qwen3.5-4B/snapshots"
        / USER_REVISION
    )
    user_server = [
        runtime["python"],
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(user_path),
        "--served-model-name",
        USER_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.user_port),
        "--nccl-port",
        str(args.user_nccl_port),
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--context-length",
        "32768",
        "--mem-fraction-static",
        "0.75",
        "--attention-backend",
        "flashinfer",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--max-running-requests",
        "64",
        "--random-seed",
        str(args.user_seed),
    ]
    restart = 'echo $$ > "$USER_PGID_FILE"; trap "exit 143" TERM INT HUP; while true; do "$@"; rc=$?; echo "user server exited rc=$rc; restarting in 5 s"; sleep 5; done'
    plan = {
        "schema_version": 1,
        "run_dir": str(run),
        "config": str(config_path),
        "output_root": str(output),
        "runtime": runtime,
        "user_backend": args.user_backend,
        "judge_backend": judge_backend,
        "slots": slots,
        "jobs": jobs,
        "retry_limit": 1,
        "wait_for_checkpoints": args.wait_for_checkpoints,
        "max_judge_streams": min(args.max_parallel_steps or args.workers, args.workers)
        * args.judge_concurrency,
        "max_parallel_steps": min(
            args.max_parallel_steps or args.workers, args.workers
        ),
    }
    if local_user:
        plan.update(
            {
                "user_gpu": 7,
                "user_port": args.user_port,
                "user_nccl_port": args.user_nccl_port,
                "user_command": runtime_command(
                    ["bash", "-c", restart, "user-restart", *user_server],
                    runtime,
                    judge=False,
                ),
            }
        )
    return plan


def code_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        REPO / "scripts/tau2/evaluate.py",
        REPO / "areal/utils/network.py",
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_ready(path: Path) -> bool:
    """Check metadata/tokenizer and every referenced shard, never load weights."""
    if (
        read_json(path / "config.json") is None
        or not (path / "tokenizer.json").is_file()
    ):
        return False
    index = read_json(path / "model.safetensors.index.json")
    if index is not None:
        names = set(index.get("weight_map", {}).values())
        if not names or any(Path(n).name != n for n in names):
            return False
    else:
        names = {"model.safetensors"}
    try:
        return all((path / name).stat().st_size > 0 for name in names)
    except OSError:
        return False


def valid_result(job: dict[str, Any], attempt: Path) -> bool:
    summary, manifest = (
        read_json(attempt / "summary.json"),
        read_json(attempt / "manifest.json"),
    )
    task_set = job.get("task_set", "ppo")  # plans written before --task-set
    planned = job.get("planned_episodes", PPO_EPISODES)
    return bool(
        summary
        and manifest
        and summary.get("status") == "complete"
        and summary.get("repeats") == 4
        and summary.get("planned_episodes") == planned
        and summary.get("completed_episodes", -1000)
        + summary.get("infra_excluded_episodes", -1000)
        == planned
        and summary.get("official", {}).get("by_domain")
        and manifest.get("model_path") == job["model_path"]
        and manifest.get("task_set") == task_set
        and (
            "judge_backend" not in job
            or manifest.get("judge_backend") == job["judge_backend"]
        )
        and manifest.get("repeats") == 4
        and manifest.get("temperature") == 0
        and manifest.get("planned_episodes") == planned
    )


def record_success(job: dict[str, Any], attempt: Path) -> None:
    if not valid_result(job, attempt):
        raise ValueError(
            f"Evaluator did not produce a complete {job.get('planned_episodes', PPO_EPISODES)}"
            "-episode summary"
        )
    write_json(
        attempt / "runner-success.json",
        {"fingerprint": job["fingerprint"], "returncode": 0},
    )


def completed_attempt(job: dict[str, Any]) -> Path | None:
    for attempt in sorted(Path(job["output_dir"]).glob("attempt-*"), reverse=True):
        receipt = read_json(attempt / "runner-success.json")
        if (
            receipt
            and receipt.get("fingerprint") == job["fingerprint"]
            and receipt.get("returncode") == 0
            and valid_result(job, attempt)
        ):
            return attempt
    return None


def attempts_remaining(job: dict[str, Any]) -> int:
    attempts = Path(job["output_dir"]).glob("attempt-*")
    used = sum(
        (read_json(p / "runner-attempt.json") or {}).get("fingerprint")
        == job["fingerprint"]
        for p in attempts
    )
    return max(0, 2 - used)


def fatal_quota_attempt(job: dict[str, Any]) -> bool:
    """A fatal provider failure cannot become an automatic resume retry."""
    for attempt in Path(job["output_dir"]).glob("attempt-*"):
        receipt = read_json(attempt / "runner-attempt.json") or {}
        if receipt.get("fingerprint") != job["fingerprint"]:
            continue
        failure = read_json(attempt / "runner-failure.json") or {}
        breaker = read_json(attempt / "quota-breaker.json") or {}
        summary = read_json(attempt / "summary.json") or {}
        if (
            failure.get("returncode") == QUOTA_EXIT_CODE
            or breaker.get("fatal")
            or summary.get("quota_breaker", {}).get("fatal")
        ):
            return True
    return False


def write_pgid(path: Path, pgid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pgid) + "\n")


def stop_process_group(
    process: subprocess.Popen,
    pgid_file: Path,
    grace: float = 5,
    owner: str | None = None,
) -> None:
    """Signal only the session we created, including descendants after leader exit."""
    # The live manager owns this Popen and token; a partially written receipt
    # must not block rollback. Unowned callers still validate their receipt.
    pgid = (
        process.pid if owner or not pgid_file.exists() else int(pgid_file.read_text())
    )
    if pgid != process.pid or pgid <= 1 or pgid == os.getpgrp():
        raise ValueError("Refusing stale/unowned process group")

    def owned_groups() -> set[int]:
        groups = {pgid}
        # Some inference descendants create their own sessions. The inherited
        # token also identifies them after their original parent has exited.
        if owner:
            marker = f"{PROCESS_OWNER_ENV}={owner}".encode()
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    if marker in (entry / "environ").read_bytes().split(b"\0"):
                        group = os.getpgid(int(entry.name))
                        if group > 1 and group != os.getpgrp():
                            groups.add(group)
                except (OSError, ProcessLookupError):
                    continue
        return groups

    def send(groups: set[int], sig: int) -> None:
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass

    groups = owned_groups()
    send(groups, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()  # reap the leader before checking for remaining descendants
        found = owned_groups()
        send(found - groups, signal.SIGTERM)
        groups.update(found)
        alive = False
        for group in groups:
            try:
                os.killpg(group, 0)
                alive = True
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(0.02)
    else:
        send(groups | owned_groups(), signal.SIGKILL)
    process.wait(timeout=5)
    pgid_file.unlink(missing_ok=True)


class ProcessGroups:
    """Own process sessions; TERM/INT/HUP requests cancel all active attempts."""

    def __init__(self) -> None:
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.active: dict[int, tuple[subprocess.Popen, Path, str]] = {}

    def start(
        self, command: list[str], env: dict[str, str], log: Path, pgid: Path
    ) -> subprocess.Popen:
        log.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if self.stopped.is_set():
                raise InterruptedError("Runner canceled")
            owner = uuid.uuid4().hex
            with log.open("ab") as output:
                process = subprocess.Popen(
                    command,
                    env={**env, PROCESS_OWNER_ENV: owner},
                    cwd=log.parent,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.active[process.pid] = (process, pgid, owner)
            try:
                write_pgid(pgid, process.pid)
            except BaseException:
                stop_process_group(process, pgid, owner=owner)
                self.active.pop(process.pid)
                raise
            return process

    def finish(self, process: subprocess.Popen) -> None:
        with self.lock:
            owned = self.active.get(process.pid)
        if owned:
            child, pgid, owner = owned
            stop_process_group(child, pgid, owner=owner)
            with self.lock:
                self.active.pop(process.pid, None)

    def close(self) -> None:
        self.stopped.set()
        with self.lock:
            active = list(self.active.values())
        errors = []
        for process, _, _ in active:
            try:
                self.finish(process)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Process cleanup failed: {errors}")


def process_env(plan: dict[str, Any], gpu: int) -> dict[str, str]:
    runtime = plan["runtime"]
    env = dict(os.environ)
    # Never inherit the colleague/provider credentials or their key-file selector.
    for name in ("DEEPSEEK_API_KEY", "TAU2_DEEPSEEK_ENV_FILE", "ARK_PLAN_API_KEY"):
        env.pop(name, None)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "SAO_ARTIFACT_ROOT": runtime["sao_artifact_root"],
            "AREAL_PYTHON": runtime["python"],
            "HF_HUB_CACHE": runtime["hf_hub_cache"],
            "TAU2_DATA_DIR": runtime["tau2_data_dir"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(REPO),
            "PATH": str(Path(runtime["python"]).parent)
            + os.pathsep
            + env.get("PATH", ""),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "AREAL_ALLOW_DEFAULT_ADMIN_KEY": "1",
            "TAU2_LOCAL_USER_API_KEY": "local-user-placeholder",
        }
    )
    return env


def wait_http(
    url: str,
    process: subprocess.Popen,
    *,
    timeout: float,
    stopped: threading.Event | None = None,
    model: str | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if (stopped and stopped.is_set()) or process.poll() is not None:
            raise InterruptedError("Server exited or runner canceled during startup")
        try:
            with opener.open(url, timeout=2) as response:
                payload = response.read()
                if response.status == 200 and (
                    model is None
                    or model in {m["id"] for m in json.loads(payload)["data"]}
                ):
                    return
        except (OSError, ValueError, KeyError, urllib.error.URLError):
            pass
        if stopped:
            stopped.wait(0.2)
        else:
            time.sleep(0.2)
    raise TimeoutError("Server readiness deadline expired: " + url)


def execute_checkpoint(
    job: dict[str, Any],
    attempt: Path,
    manager: ProcessGroups,
    plan: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    wait_ports_free(
        slot_ports(job), timeout=args.port_wait_timeout, stopped=manager.stopped
    )
    env = process_env(plan, job["gpu"])
    low, end = job["port_range"]
    env["AREAL_PORT_RANGE"] = f"{low}:{end - 1}"
    first = str(Path(job["output_dir"]) / "attempt-001")
    command = [part.replace(first, str(attempt)) for part in job["command"]]
    process = manager.start(
        command, env, attempt / "process.log", attempt / "process.pgid"
    )
    deadline = time.monotonic() + args.checkpoint_timeout
    fatal_quota = None
    try:
        while True:
            returncode = process.poll()
            breaker = read_json(attempt / "quota-breaker.json") or {}
            summary = read_json(attempt / "summary.json") or {}
            if breaker.get("fatal") or summary.get("quota_breaker", {}).get("fatal"):
                manager.stopped.set()
                fatal_quota = (
                    breaker if breaker.get("fatal") else summary["quota_breaker"]
                )
                return QUOTA_EXIT_CODE
            if returncode is not None:
                return returncode
            if manager.stopped.wait(0.2):
                raise InterruptedError("Runner canceled")
            if time.monotonic() >= deadline:
                raise TimeoutError("Checkpoint attempt exceeded wall clock deadline")
    finally:
        manager.finish(process)
        if fatal_quota:
            summary = read_json(attempt / "summary.json") or {}
            write_json(
                attempt / "summary.json",
                {
                    **summary,
                    "status": "failed",
                    "error": f"FatalRolloutError: {fatal_quota['fatal']}",
                    "quota_breaker": fatal_quota,
                },
            )


def run_queue(
    plan: dict[str, Any], args: argparse.Namespace, manager: ProcessGroups
) -> int:
    """One serial queue per GPU. Retrying never creates a second policy on a GPU."""
    stopped = getattr(manager, "stopped", threading.Event())
    fatal = threading.Event()
    parallel = min(args.max_parallel_steps or args.workers, args.workers)
    permits = threading.Semaphore(parallel)
    if any(fatal_quota_attempt(job) for job in plan["jobs"]):
        stopped.set()
        write_summaries(plan)
        return QUOTA_EXIT_CODE

    def slot_queue(gpu: int) -> bool:
        failed = False
        for job in (j for j in plan["jobs"] if j["gpu"] == gpu):
            if stopped.is_set():
                return True
            if completed_attempt(job):
                continue
            deadline = time.monotonic() + args.wait_timeout
            while not checkpoint_ready(Path(job["model_path"])):
                if not args.wait_for_checkpoints or time.monotonic() >= deadline:
                    write_json(
                        Path(job["output_dir"]) / "runner-state.json",
                        {"status": "missing"},
                    )
                    failed = True
                    break
                if stopped.wait(args.poll_interval):
                    return True
            else:
                directory = Path(job["output_dir"])
                previous = sorted(directory.glob("attempt-*"))
                # Two attempts per protocol identity, persisted across invocations.
                for _ in range(attempts_remaining(job)):
                    if stopped.is_set():
                        return True
                    try:
                        while not permits.acquire(timeout=0.1):
                            if stopped.is_set():
                                return True
                        try:
                            if stopped.is_set():
                                return True
                            number = (
                                max(
                                    [int(p.name.split("-")[-1]) for p in previous] + [0]
                                )
                                + 1
                            )
                            attempt = directory / f"attempt-{number:03d}"
                            write_json(
                                attempt / "runner-attempt.json",
                                {"fingerprint": job["fingerprint"]},
                            )
                            previous.append(attempt)
                            rc = execute_checkpoint(job, attempt, manager, plan, args)
                        finally:
                            permits.release()
                        if rc == QUOTA_EXIT_CODE:
                            fatal.set()
                            stopped.set()
                            write_json(
                                attempt / "runner-failure.json",
                                {
                                    "error": "Fatal provider quota breaker; no retry",
                                    "returncode": rc,
                                },
                            )
                            return True
                        if rc == 0:
                            record_success(job, attempt)
                            break
                        error = f"evaluate.py exited {rc}"
                    except InterruptedError:
                        if stopped.is_set():
                            return True
                        raise
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    write_json(attempt / "runner-failure.json", {"error": error})
                if completed_attempt(job) is None:
                    failed = True
        return failed

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(slot_queue, gpu) for gpu in range(args.workers)]
        try:
            failed = [future.result() for future in as_completed(futures)]
        except BaseException:
            stopped.set()
            raise
    write_summaries(plan)
    return QUOTA_EXIT_CODE if fatal.is_set() else int(any(failed))


def write_summaries(plan: dict[str, Any]) -> None:
    rows = []
    markdown = [
        "# PPO offline checkpoint evaluation",
        "",
        "Scores retain the official evaluator denominators; infra errors are excluded. Incomplete/failed attempts have no final score.",
        "",
        "| Step | Status | Domain | mean@4 | pass^1 | pass^2 | pass^3 | pass^4 | Episodes | infra-excluded | context-limit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for job in plan["jobs"]:
        attempt = completed_attempt(job)
        status = (
            "complete"
            if attempt
            else (
                "failed"
                if list(Path(job["output_dir"]).glob("attempt-*"))
                else "pending"
                if checkpoint_ready(Path(job["model_path"]))
                else "missing"
            )
        )
        summary = read_json(attempt / "summary.json") if attempt else {}
        domains = summary.get("official", {}).get("by_domain", {})
        rows.append(
            {
                "step": job["step"],
                "status": status,
                "model_path": job["model_path"],
                "attempt": str(attempt) if attempt else None,
                "episodes": summary.get("planned_episodes") if attempt else None,
                "infra_excluded": summary.get("infra_excluded_episodes")
                if attempt
                else None,
                "context_limit": summary.get("context_limit_episodes")
                if attempt
                else None,
                "domains": domains,
            }
        )
        for domain, metrics in (domains or {"—": {}}).items():
            values = [
                metrics.get("mean_at_k"),
                *[metrics.get("pass_hat_k", {}).get(str(k)) for k in range(1, 5)],
            ]
            scores = " | ".join("—" if v is None else f"{v:.6f}" for v in values)
            counts = " | ".join(
                str(metrics.get(k, "—"))
                for k in (
                    "episodes",
                    "infra_excluded_episodes",
                    "context_limit_episodes",
                )
            )
            markdown.append(
                f"| {job['step']} | {status} | {domain} | {scores} | {counts} |"
            )
    root = Path(plan["output_root"])
    write_json(root / "summary.json", {"schema_version": 1, "checkpoints": rows})
    (root / "summary.md").write_text("\n".join(markdown) + "\n")


def slot_ports(slot: dict[str, Any]) -> list[int]:
    return [slot["policy_port"], slot["nccl_port"], *range(*slot["port_range"])]


def wait_ports_free(
    ports: list[int],
    *,
    timeout: float,
    stopped: threading.Event,
    poll_interval: float = 1.0,
) -> None:
    """Wait for previous servers, including retry slots, without killing owners."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("port wait timeout must be finite and positive")
    deadline = time.monotonic() + timeout
    while True:
        if stopped.is_set():
            raise InterruptedError("Runner canceled while waiting for ports")
        occupied = []
        for port in ports:
            for kind, name in ((socket.SOCK_STREAM, "TCP"), (socket.SOCK_DGRAM, "UDP")):
                with socket.socket(socket.AF_INET, kind) as sock:
                    try:
                        sock.bind(("0.0.0.0", port))
                    except OSError as exc:
                        if exc.errno != errno.EADDRINUSE:
                            raise
                        occupied.append(f"{port}/{name}")
        if not occupied:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Ports still occupied after {timeout:g}s: {', '.join(occupied)}"
            )
        stopped.wait(min(poll_interval, remaining))


def run(args: argparse.Namespace) -> int:
    plan = build_plan(args)
    if args.dry_run:
        # JSON contains exact argv arrays; shell renderings are convenient to inspect.
        display = {
            **plan,
            "checkpoint_states": [
                {
                    "step": j["step"],
                    "skip_finished": completed_attempt(j) is not None,
                    "ready": checkpoint_ready(Path(j["model_path"])),
                    "attempts_remaining": attempts_remaining(j),
                }
                for j in plan["jobs"]
            ],
            "shell_commands": [shlex.join(j["command"]) for j in plan["jobs"]],
        }
        sys.stdout.write(json.dumps(display, indent=2) + "\n")
        return 0
    root = Path(plan["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    manager = ProcessGroups()
    previous_handlers = {}

    def cancel(signum: int, frame: Any) -> None:
        manager.stopped.set()

    with (root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, cancel)
        try:
            write_json(root / "plan.json", plan)
            if any(fatal_quota_attempt(job) for job in plan["jobs"]):
                return QUOTA_EXIT_CODE
            todo = [
                j
                for j in plan["jobs"]
                if not completed_attempt(j)
                and attempts_remaining(j)
                and (
                    args.wait_for_checkpoints or checkpoint_ready(Path(j["model_path"]))
                )
            ]
            if todo and "user_command" in plan:
                wait_ports_free(
                    [plan["user_port"], plan["user_nccl_port"]]
                    + [port for slot in plan["slots"] for port in slot_ports(slot)],
                    timeout=args.port_wait_timeout,
                    stopped=manager.stopped,
                )
                env = process_env(plan, 7)
                env.pop("AREAL_PORT_RANGE", None)
                pgid = root / "user-server.pgid"
                env["USER_PGID_FILE"] = str(pgid)
                user = manager.start(
                    plan["user_command"], env, root / "user-server.log", pgid
                )
                wait_http(
                    f"http://127.0.0.1:{args.user_port}/v1/models",
                    user,
                    timeout=args.startup_timeout,
                    stopped=manager.stopped,
                    model=USER_MODEL,
                )
            code = run_queue(plan, args, manager)
            return (
                code
                if code == QUOTA_EXIT_CODE
                else 143
                if manager.stopped.is_set()
                else code
            )
        except Exception as exc:
            write_json(
                root / "runner-error.json",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "canceled": manager.stopped.is_set(),
                },
            )
            raise
        finally:
            try:
                manager.close()
            finally:
                try:
                    write_summaries(plan)
                finally:
                    for sig, handler in previous_handlers.items():
                        signal.signal(sig, handler)


def main() -> None:
    try:
        code = run(parse_args())
    except InterruptedError:
        code = 143
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
