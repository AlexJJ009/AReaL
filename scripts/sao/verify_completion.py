# SPDX-License-Identifier: Apache-2.0
"""Read back full-epoch, checkpoint reload, and resource closure evidence.

This final-only verifier allocates no GPU and never alters training state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.sao.audit_run import audit_run
from scripts.sao.verify_checkpoints import verify_run


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_reload(run_root: Path, role: str, frozen: dict, config: dict) -> dict:
    path = run_root / "evidence" / f"final-reload-{role}.json"
    payload = _read(path)
    errors = []
    ranks = payload.get("ranks", [])
    if payload.get("passed") is not True or payload.get("status") != "ok":
        errors.append("aggregate reload did not pass")
    if len(ranks) != 4 or {item.get("rank") for item in ranks} != set(range(4)):
        errors.append("reload must contain exactly four ranks")
    native_role = "default" if role == "actor" else "critic"
    role_root = (
        run_root
        / "checkpoints/root"
        / config["experiment_name"]
        / config["trial_name"]
        / native_role
    )
    expected_paths = {
        "model_path": frozen["model_path"],
        "dcp_path": str(role_root / "recover_checkpoint"),
        "hf_path": str(role_root / "epoch0epochstep134globalstep134"),
    }
    for rank in ranks:
        tag = f"{role} rank{rank.get('rank')}"
        if not (
            rank.get("passed") is True
            and rank.get("status") == "ok"
            and rank.get("role") == role
            and rank.get("world_size") == 4
        ):
            errors.append(f"{tag}: invalid role/rank status")
        if rank.get("paths") != expected_paths:
            errors.append(f"{tag}: reload paths differ from final checkpoint/Base")
        steps = rank.get("optimizer_steps", {})
        required_count = 4 if role == "critic" else 3
        if (
            rank.get("optimizer_step_status") != "ok"
            or len(steps) != required_count
            or any(value != 135 for value in steps.values())
        ):
            errors.append(f"{tag}: optimizer step135 not proven")
        comparisons = rank.get("comparisons", {})
        if len(comparisons) != required_count or not all(
            item.get("allclose") is True for item in comparisons.values()
        ):
            errors.append(f"{tag}: HF weight comparison incomplete or failed")
        if not any(
            item.get("changed_from_base") is True
            for item in rank.get("noop_checks", {}).values()
        ):
            errors.append(f"{tag}: reload could be a no-op")
        if rank.get("forward", {}).get("finite_all_ranks") is not True:
            errors.append(f"{tag}: finite forward missing")
        if rank.get("checkpoint_files_stable_during_reload") is not True:
            errors.append(f"{tag}: checkpoint stability during reload missing")
        for kind, root in (
            ("dcp", Path(expected_paths["dcp_path"])),
            ("hf", Path(expected_paths["hf_path"])),
        ):
            files = rank.get("metadata", {}).get(kind, {}).get("files", {})
            if not files:
                errors.append(f"{tag}: missing {kind} metadata binding")
            current_files = (
                {path.name for path in root.iterdir() if path.is_file()}
                if root.is_dir()
                else set()
            )
            if current_files != set(files):
                errors.append(f"{tag}: changed checkpoint inventory {kind}")
            for name, binding in files.items():
                target = root / name
                if (
                    Path(name).name != name
                    or not target.is_file()
                    or target.stat().st_size != binding.get("size")
                    or target.stat().st_mtime_ns != binding.get("mtime_ns")
                    or target.stat().st_ctime_ns != binding.get("ctime_ns")
                ):
                    errors.append(f"{tag}: changed checkpoint file {kind}/{name}")
                elif (
                    binding.get("sha256")
                    and hashlib.sha256(target.read_bytes()).hexdigest()
                    != binding["sha256"]
                ):
                    errors.append(f"{tag}: changed checkpoint metadata {kind}/{name}")
    return {
        "passed": not errors,
        "errors": errors,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def inspect_resources(run_root: Path) -> dict:
    """Require the allocated eight-GPU host to be free after this run's probes."""
    excluded = set()
    pid = os.getpid()
    while pid > 0 and pid not in excluded:
        excluded.add(pid)
        status = Path(f"/proc/{pid}/status")
        if not status.exists():
            break
        pid = int(
            next(
                line.split()[1]
                for line in status.read_text().splitlines()
                if line.startswith("PPid:")
            )
        )
    marker = f"SAO_RUN_ROOT={run_root}".encode()
    remaining = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) in excluded:
            continue
        try:
            if marker in (directory / "environ").read_bytes().split(b"\0"):
                remaining.append(
                    {
                        "namespace_pid": int(directory.name),
                        "command": (directory / "comm").read_text().strip(),
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    gpu_processes = [line for line in gpu.stdout.splitlines() if line.strip()]
    return {
        "passed": not remaining and not gpu_processes,
        "task_processes": remaining,
        "gpu_compute_processes": gpu_processes,
        "scope": "all eight GPUs were allocated to this run; any compute process requires investigation",
    }


def verify(run_root: Path, dataset: Path, item: str) -> dict:
    evidence = run_root / "evidence"
    audit = audit_run(evidence, dataset_path=dataset, mode="full")
    report = {"item": item, "audit": audit, "passed": audit["passed"]}
    if item == "C15":
        frozen = _read(evidence / "frozen-run.json")
        config = _read(evidence / "resolved-config.json")
        report["checkpoints"] = verify_run(run_root)
        report["reload"] = {
            role: verify_reload(run_root, role, frozen, config)
            for role in ("actor", "critic")
        }
        report["resources"] = inspect_resources(run_root)
        report["passed"] = all(
            [
                report["passed"],
                report["checkpoints"]["passed"],
                report["resources"]["passed"],
                *(value["passed"] for value in report["reload"].values()),
            ]
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--item", choices=("C15", "C21"), required=True)
    args = parser.parse_args()
    try:
        report = verify(args.run_root.resolve(), args.dataset.resolve(), args.item)
    except Exception as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
