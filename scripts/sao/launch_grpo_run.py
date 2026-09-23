"""Launch the approved GRPO run using artifacts in ``SAO_LAUNCH_DIR``.

Keep this entrypoint in source control and never name it ``queue.py``:
that name shadows the stdlib module needed by datasets/pyarrow during snapshots.
"""

import collections
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts.sao.gpu_admission import (
    admission_config_from_env,
    wait_for_scoped_gpus_free,
)

ROOT = Path(os.environ["SAO_LAUNCH_DIR"]).resolve()
REPO = Path.cwd()
MANIFEST = ROOT / "manifest.json"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen():
    m = read(MANIFEST)
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != m["candidate_sha"]
    ):
        raise RuntimeError("Candidate SHA changed after authorization")
    for name, digest in m["files"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Frozen launch input changed: {name}")
    if m["launch_authorized"] is not True:
        raise RuntimeError("Missing launch authorization")
    return m


def descendants():
    parents = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                parents[int(path.name)] = int(
                    path.joinpath("stat").read_text().split(") ")[1].split()[1]
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    found, frontier = set(), {os.getpid()}
    while frontier:
        frontier = {
            pid
            for pid, parent in parents.items()
            if parent in frontier and pid not in found
        }
        found.update(frontier)
    return found


def cleanup():
    targets = descendants()
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(20):
        if not any(Path("/proc", str(pid)).exists() for pid in targets):
            break
        time.sleep(1)
    for pid in targets:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def verify_steps(run):
    evidence = run / "evidence"
    counts = read(evidence / "step-counts.json")
    config = read(evidence / "resolved-config.json")
    count = counts["expected_optimizer_steps"]
    prompts = counts["train_prompts_per_step"]
    n_samples = config["gconfig"]["n_samples"]
    if counts["samples_per_prompt"] != n_samples:
        raise RuntimeError("Sample count disagrees with resolved configuration")
    optimizer = config["actor"]["optimizer"]
    warmup = counts["resolved_warmup_steps"]
    if not (evidence / "epoch-finished.json").exists():
        raise RuntimeError("Missing epoch completion record")
    for n in range(1, count + 1):
        step = read(evidence / "steps" / f"{n}.json")
        rows = read(evidence / "consumed" / f"{n}.json")
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row["audit_source_key"]].append(row["audit_sample_idx"])
        if len(groups) != prompts or any(
            sorted(x) != list(range(n_samples)) for x in groups.values()
        ):
            raise RuntimeError(f"Broken N{n_samples} prompt group at update {n}")
        if step["published_version"] != n:
            raise RuntimeError("Wrong published policy version")
        metrics = step["metrics"]
        if any(
            isinstance(v, (int, float)) and not math.isfinite(v)
            for v in metrics.values()
        ):
            raise RuntimeError("Non-finite training metric")
        if metrics["ppo_actor/update/update_successful"] != 1:
            raise RuntimeError("Actor update skipped")
        expected_lr = optimizer["lr"] * (min((n - 1) / warmup, 1) if warmup else 1)
        if not math.isclose(metrics["ppo_actor/update/lr"], expected_lr, rel_tol=1e-5):
            raise RuntimeError("Warmup learning-rate mismatch")
        if (
            metrics.get("timeperf/recompute_logp", 0) <= 0
            or "ppo_actor/update/behave_imp_weight/avg" not in metrics
        ):
            raise RuntimeError(
                "Missing real async proximal/importance correction evidence"
            )
    return {
        "passed": True,
        "updates": count,
        "prompts_per_update": prompts,
        "n_samples": n_samples,
    }


def evaluation_versions(run):
    """Read this run's evaluation schedule instead of the historical N4 schedule."""
    evidence = run / "evidence"
    config = read(evidence / "resolved-config.json")
    count = read(evidence / "step-counts.json")["expected_optimizer_steps"]
    interval = config["evaluator"]["freq_steps"]
    versions = list(range(interval, count + 1, interval)) if interval else []
    if config["evaluator"]["eval_before_train"]:
        versions.insert(0, 0)
    if config["evaluator"]["freq_epochs"]:
        versions.append(count)
    return tuple(sorted(set(versions)))


def wait_for_gpu_admission(stage_name):
    config = admission_config_from_env()
    report = wait_for_scoped_gpus_free(config)
    (ROOT / f"gpu-admission-{stage_name}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def run_stage(name, dataset, preflight, expected_config_hash=None):
    check_frozen()
    wait_for_gpu_admission("preflight" if preflight else "formal")
    run = (
        Path(os.environ["SAO_RUN_ROOT"])
        if not preflight
        else ROOT.parent / "runs" / name
    )
    run.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(
        SAO_RUN_ROOT=str(run),
        SAO_TRIAL_NAME=name,
        SAO_DATA_PATH=str(dataset),
        SAO_PREFLIGHT="1" if preflight else "0",
    )
    command = [
        sys.executable,
        "examples/math/sao_grpo.py",
        "--config",
        "examples/math/sao_grpo.yaml",
    ]
    if preflight:
        command += [
            "train_dataset.batch_size=4",
            "valid_dataset.batch_size=4",
            "evaluator.eval_before_train=false",
            "evaluator.freq_steps=999999",
            "saver.freq_steps=999999",
            "recover.freq_steps=999999",
        ]
    with (run / "controller.log").open("ab") as log:
        process = subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        (run / "launch.json").write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "command": command,
                    "preflight": preflight,
                    "candidate_sha": read(MANIFEST)["candidate_sha"],
                },
                indent=2,
            )
        )
        while process.poll() is None:
            config = run / "evidence/resolved-config.json"
            if (
                expected_config_hash
                and config.exists()
                and sha(config) != expected_config_hash
            ):
                raise RuntimeError(
                    "Actual formal configuration differs from approved configuration"
                )
            # The async evaluator publishes/finalizes its own snapshots.
            # Do not re-read mutable eval JSONL while its next version is running.
            time.sleep(10)
        if process.returncode != 0:
            raise RuntimeError(
                f"{name} trainer exited {process.returncode}; see {run}/controller.log"
            )
    cleanup()
    report = verify_steps(run)
    if preflight:
        records = [
            json.loads(line)
            for path in (run / "evidence/samples").glob("eval-*.jsonl")
            for line in path.read_text().splitlines()
            if line
        ]
        if (
            len(records) != 20
            or len({row["source_id"] for row in records}) != 5
            or any("error" in row for row in records)
        ):
            raise RuntimeError("Preflight evaluation incomplete")
    else:
        from scripts.sao.snapshot_eval import snapshot_eval

        versions = evaluation_versions(run)
        for version in versions:
            completion = run / "evidence" / "async-eval" / f"{version}.json"
            if not completion.exists() or read(completion).get("status") != "completed":
                raise RuntimeError(
                    f"Missing successful async evaluation for version {version}"
                )
            snapshot_eval(
                run / "evidence",
                dataset,
                version,
                allowed_versions=versions,
            )
    (run / "acceptance.json").write_text(json.dumps(report, indent=2))
    return run


def main():
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    manifest = check_frozen()
    wait_for_gpu_admission("initial")
    with (ROOT / "started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_ns": time.time_ns()}, stream)
    probe = run_stage(
        f"{os.environ['SAO_TRIAL_NAME']}-preflight",
        Path(manifest["preflight_dataset"]),
        True,
    )
    (ROOT / "native-preflight-passed.json").write_text(
        json.dumps(
            {
                "passed": True,
                "run": str(probe),
                "acceptance_sha256": sha(probe / "acceptance.json"),
            },
            indent=2,
        )
    )
    check_frozen()
    formal = run_stage(
        os.environ["SAO_TRIAL_NAME"],
        Path(os.environ["SAO_DATA_PATH"]),
        False,
        manifest["resolved_config_sha256"],
    )
    (ROOT / "completed.json").write_text(
        json.dumps({"passed": True, "run": str(formal)}, indent=2)
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
