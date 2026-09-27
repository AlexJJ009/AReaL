#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-shot W&B upload for tau2 critic and GRPO native metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TRAIN_DONE_RE = re.compile(r"Train step\s+(\d+)/\d+ done\.")
RECOVER_RE = re.compile(r"Recovering from StepInfo\(.*global_step=(\d+)")
NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$", re.I)
DOMAINS = ("airline", "retail", "telecom")


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _put(out: dict[str, Any], key: str, value: Any, *, omit_null: bool = False):
    if value is None and omit_null:
        return
    if value is None or not isinstance(value, (int, float, str, bool)):
        raise ValueError(f"Metric {key} has unsupported value {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Metric {key} is non-finite: {value}")
    out[key] = value


def _critic_eval(report: dict[str, Any], source: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "eval/completed_step": int(report["completed_step"]),
        "source/path": str(source),
        "source/kind": "critic_validation",
    }
    scopes = [("overall", report["summary"], report["grad_norm"]["overall"])]
    scopes += [(d, report["by_domain"][d], report["grad_norm"][d]) for d in DOMAINS]
    for name, stats, grad_norm in scopes:
        prefix = f"critic_validation/{name}"
        _put(out, f"{prefix}_mse", stats["mse"])
        _put(
            out,
            f"{prefix}_explained_variance_defined",
            float(stats["explained_variance_defined"]),
        )
        _put(
            out,
            f"{prefix}_explained_variance",
            stats.get("explained_variance"),
            omit_null=True,
        )
        _put(out, f"{prefix}_grad_norm", grad_norm)
    _put(out, "critic_validation/overall_macro_mse", report["summary"]["macro_mse"])
    return out


def load_critic_metrics(
    run_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence = run_root / "tau2-critic-fit-evidence"
    train_dir = evidence / "train-metrics"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing critic train metrics under {evidence}")
    train: list[dict[str, Any]] = []
    for path in sorted(train_dir.glob("step-*.json")):
        data = _json(path)
        record: dict[str, Any] = {
            "train/completed_step": int(data["completed_step"]),
            "source/path": str(path),
            "source/kind": "critic_train",
        }
        for key, value in data["metrics"].items():
            if not key.startswith("tau2_critic_validation/") and not key.endswith(
                "__count"
            ):
                _put(record, key, value, omit_null=True)
        train.append(record)
    evals = [
        _critic_eval(_json(path), path)
        for path in sorted(evidence.glob("validation-step-*.json"))
    ]
    if not train:
        raise ValueError(f"No critic train metrics found in {evidence}")
    if not any(row["eval/completed_step"] == 0 for row in evals):
        raise ValueError(f"Missing critic validation step 0 in {evidence}")
    return train, evals


def _table_value(raw: str) -> float | int | str | bool:
    text = raw.strip()
    if text.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity"}:
        raise ValueError(f"Non-finite GRPO table value {raw!r}")
    if NUMBER_RE.match(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite GRPO table value {raw!r}")
        return int(value) if value.is_integer() else value
    if text in {"True", "False"}:
        return text == "True"
    if text:
        return text
    raise ValueError("Empty GRPO table value")


def _table_metrics(line: str) -> dict[str, Any]:
    text = ANSI_RE.sub("", line)
    if "│" not in text or "─" in text:
        return {}
    cells = [cell.strip() for cell in text.split("│")[1:-1]]
    return {
        cells[i]: _table_value(cells[i + 1])
        for i in range(0, len(cells) - 1, 2)
        if cells[i]
    }


def _table_footer(line: str) -> bool:
    return "╘" in line


def _table_separator(line: str) -> bool:
    return any(ch in line for ch in "╒╞╪╤╧├┼┤└┴┘") or "─" in line


def _commit_grpo(
    records: dict[int, dict[str, Any]], step: int, metrics: dict[str, Any], path: Path
):
    if not metrics:
        raise ValueError(f"Train step {step} in {path} has no parsed metric table")
    record: dict[str, Any] = {
        "train/completed_step": step,
        "source/path": str(path),
        "source/kind": "grpo_train",
    }
    for key, value in metrics.items():
        if not key.endswith("__count"):
            _put(record, key, value)
    if step in records and records[step] != record:
        raise ValueError(
            f"Conflicting duplicate GRPO metrics for step {step} in {path}"
        )
    records.setdefault(step, record)


def load_grpo_train_metrics(run_root: Path) -> list[dict[str, Any]]:
    logs = sorted(run_root.glob("logs/**/main.log"))
    if len(logs) != 1:
        raise ValueError(
            f"Expected exactly one GRPO main.log under {run_root}, found {logs}"
        )
    path = logs[0]
    records: dict[int, dict[str, Any]] = {}
    active_step: int | None = None
    active_metrics: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_RE.sub("", raw)
        recovery = RECOVER_RE.search(line)
        if recovery and active_step is None:
            restored_step = int(recovery.group(1))
            discarded = [step for step in records if step > restored_step]
            for step in discarded:
                del records[step]
            if discarded:
                print(
                    f"Exclude updates rolled back by recovery: {discarded}",
                    file=sys.stderr,
                )
            continue
        match = TRAIN_DONE_RE.search(line)
        if match:
            if active_step is not None:
                raise ValueError(
                    f"Train step {active_step} in {path} has no table footer"
                )
            active_step, active_metrics = int(match.group(1)), {}
            continue
        if active_step is None:
            continue
        if _table_footer(line):
            _commit_grpo(records, active_step, active_metrics, path)
            active_step, active_metrics = None, {}
            continue
        parsed = _table_metrics(line)
        if parsed:
            active_metrics.update(parsed)
        elif "│" in line and not _table_separator(line):
            raise ValueError(f"Unparsed GRPO metric row for step {active_step}: {line}")
    if active_step is not None:
        raise ValueError(f"Train step {active_step} in {path} has no table footer")
    if not records:
        raise ValueError(f"No committed GRPO train steps found in {path}")
    return [records[step] for step in sorted(records)]


def load_grpo_eval_metrics(run_root: Path) -> list[dict[str, Any]]:
    eval_dir = run_root / "evidence" / "async-eval"
    rows: list[dict[str, Any]] = []
    for path in (
        sorted(eval_dir.glob("*.json"), key=lambda p: int(p.stem))
        if eval_dir.exists()
        else []
    ):
        data = _json(path)
        if data.get("status") != "completed":
            continue
        row: dict[str, Any] = {
            "eval/completed_step": int(data["version"]),
            "eval/status_completed": 1,
            "source/path": str(path),
            "source/kind": "grpo_eval",
        }
        for domain, item in data["domains"].items():
            _put(row, f"eval/{domain}/episodes", item["episodes"])
            _put(row, f"eval/{domain}/reward_mean", item["reward_mean"])
        rows.append(row)
    return rows


def load_offline_eval_metrics(run_root: Path) -> list[dict[str, Any]]:
    """Upload complete repeated evaluation only, retaining every trial."""
    manifest = _json(run_root / "manifest.json")
    summary = _json(run_root / "summary.json")
    repeats = int(manifest["repeats"])
    if repeats < 1 or summary.get("status") != "complete":
        raise ValueError("Offline evaluation is incomplete")
    episodes = [
        json.loads(line)
        for line in (run_root / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = 100 * repeats
    if len(episodes) != expected or manifest["planned_episodes"] != expected:
        raise ValueError("Offline evaluation coverage mismatch")
    seen = set()
    grouped: dict[tuple[int, str], list[float]] = {}
    for item in episodes:
        trial, domain = int(item["trial"]), item["domain"]
        identity = (domain, str(item["task_id"]), trial)
        if identity in seen:
            raise ValueError(f"Duplicate offline evaluation cell: {identity}")
        if (
            item["status"] != "completed"
            or not 0 <= trial < repeats
            or domain not in DOMAINS
        ):
            raise ValueError(f"Invalid offline evaluation cell: {identity}")
        score = float(item["official_score"])
        if score not in (0.0, 1.0):
            raise ValueError(f"Non-binary offline evaluation score: {score}")
        seen.add(identity)
        grouped.setdefault((trial, domain), []).append(score)
    rows = []
    for trial in range(repeats):
        row: dict[str, Any] = {
            "eval/completed_step": trial + 1,
            "source/kind": "offline_eval",
        }
        total = 0.0
        for domain, count in (("airline", 20), ("retail", 40), ("telecom", 40)):
            scores = grouped.get((trial, domain), [])
            if len(scores) != count:
                raise ValueError(f"Missing {domain} coverage in trial {trial}")
            row[f"eval/{domain}/reward_mean"] = sum(scores) / count
            row[f"eval/{domain}/episodes"] = count
            total += sum(scores)
        row["eval/reward_mean"] = total / 100
        rows.append(row)
    return rows


def _load(
    kind: str, run_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if kind == "offline_eval":
        return [], load_offline_eval_metrics(run_root)
    return (
        load_critic_metrics(run_root)
        if kind == "critic"
        else (load_grpo_train_metrics(run_root), load_grpo_eval_metrics(run_root))
    )


def upload_once(args: argparse.Namespace, run: Any) -> tuple[int, int, int, int]:
    train, evals = _load(args.kind, args.run_root)
    summary = getattr(run, "summary", {})
    last_train = int(
        summary.get(f"tau2_upload/{args.kind}/last_train_completed_step", -1)
    )
    last_eval = int(
        summary.get(f"tau2_upload/{args.kind}/last_eval_completed_step", -1)
    )
    pending_train = [r for r in train if int(r["train/completed_step"]) > last_train]
    pending_eval = [r for r in evals if int(r["eval/completed_step"]) > last_eval]
    for axis, records in (("train", pending_train), ("eval", pending_eval)):
        for record in records:
            run.log(record)
            run.summary[f"tau2_upload/{args.kind}/last_{axis}_completed_step"] = int(
                record[f"{axis}/completed_step"]
            )
    run.summary[f"tau2_upload/{args.kind}/run_root"] = str(args.run_root)
    if args.kind == "offline_eval":
        # Trial curves stay visible; the headline is the mean, never the best trial.
        for key in evals[0]:
            if key.endswith("reward_mean"):
                run.summary[f"offline_eval/{key.removeprefix('eval/')}"] = sum(
                    float(row[key]) for row in evals
                ) / len(evals)
        run.summary["offline_eval/repeats"] = len(evals)
        run.summary["offline_eval/episodes"] = len(evals) * 100
    return (
        len(pending_train) + len(pending_eval),
        max((int(r["train/completed_step"]) for r in train), default=last_train),
        max((int(r["eval/completed_step"]) for r in evals), default=last_eval),
        len(train) + len(evals),
    )


def _init_wandb(args: argparse.Namespace) -> Any:
    import wandb

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.name,
        resume="allow",
        tags=["tau2", "historical-metrics-import", args.kind],
        mode="online",
        config={
            "kind": args.kind,
            "run_root": str(args.run_root),
            "uploader": "scripts/tau2/upload_metrics.py",
        },
    )
    for axis in ("train", "eval"):
        wandb.define_metric(f"{axis}/completed_step")
        wandb.define_metric(f"{axis}/*", step_metric=f"{axis}/completed_step")
    for prefix in ("critic", "ppo_actor", "rollout", "tau2_batch", "timeperf"):
        wandb.define_metric(f"{prefix}/*", step_metric="train/completed_step")
    wandb.define_metric("critic_validation/*", step_metric="eval/completed_step")
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("critic", "grpo", "offline_eval"), required=True
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--entity", default=os.getenv("TAU2_WANDB_ENTITY"))
    parser.add_argument("--project", default=os.getenv("TAU2_WANDB_PROJECT"))
    parser.add_argument("--run-id", default=os.getenv("TAU2_WANDB_RUN_ID"))
    parser.add_argument("--name", default=os.getenv("TAU2_WANDB_NAME"))
    args = parser.parse_args(argv)
    if not args.entity or not args.project or not args.run_id:
        parser.error(
            "--entity, --project, and --run-id are required or TAU2_WANDB_* envs"
        )
    args.run_root = args.run_root.resolve()
    run = _init_wandb(args)
    try:
        uploaded, max_train, max_eval, discovered = upload_once(args, run)
        run.summary[f"tau2_upload/{args.kind}/discovered_records"] = discovered
        print(
            f"{args.kind}: uploaded={uploaded} max_train={max_train} max_eval={max_eval}",
            flush=True,
        )
        run.finish(quiet=True)
        return 0
    except Exception as exc:
        print(f"upload_metrics error: {exc}", file=sys.stderr, flush=True)
        run.finish(exit_code=1, quiet=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
