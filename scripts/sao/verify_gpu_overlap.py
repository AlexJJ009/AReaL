#!/usr/bin/env python3
"""Calculate bounded GPU kernel overlap for SAO evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _short(name: str, limit: int = 160) -> str:
    return name if len(name) <= limit else name[: limit - 3] + "..."


def _kernel_interval(
    event: dict[str, Any], *, role: str, path: Path, base_ns: int | None = None
) -> dict[str, Any] | None:
    if event.get("cat") != "kernel":
        return None
    args = event.get("args")
    if not isinstance(args, dict):
        args = {}
    dur_us = event.get("dur")
    if dur_us is None:
        raise ValueError(f"{path}: kernel event missing dur")
    if base_ns is None:
        wall_ns = args.get("wall_time_ns")
        if wall_ns is None:
            raise ValueError(f"{path}: kernel event missing args.wall_time_ns")
        start = int(wall_ns)
    else:
        ts_us = event.get("ts")
        if ts_us is None:
            raise ValueError(f"{path}: kernel event missing ts")
        start = int(base_ns + float(ts_us) * 1000)
    end = start + int(float(dur_us) * 1000)
    if end <= start:
        return None
    return {
        "start_ns": start,
        "end_ns": end,
        "name": str(event.get("name") or ""),
        "role": role,
        "path": str(path),
        "device": args.get("device"),
        "stream": args.get("stream"),
        "pid": event.get("pid"),
        "tid": event.get("tid"),
    }


def _read_native(path: Path, role: str) -> list[dict[str, Any]]:
    events = []
    with path.open() as f:
        for line in f:
            if line.strip():
                event = _kernel_interval(json.loads(line), role=role, path=path)
                if event:
                    events.append(event)
    return events


def _read_sglang(path: Path) -> tuple[list[dict[str, Any]], Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        payload = json.load(f)
    base_ns = payload.get("baseTimeNanoseconds")
    if base_ns is None:
        raise ValueError(f"{path}: missing baseTimeNanoseconds")
    events = []
    for raw in payload.get("traceEvents") or []:
        event = _kernel_interval(raw, role="rollout", path=path, base_ns=int(base_ns))
        if event:
            events.append(event)
    return events, payload.get("deviceProperties")


def _clip_to_spans(
    events: list[dict[str, Any]], spans: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    clipped = []
    for event in events:
        for span in spans:
            start = max(event["start_ns"], int(span["started_ns"]))
            end = min(event["end_ns"], int(span["completed_ns"]))
            if end > start:
                clipped.append({**event, "start_ns": start, "end_ns": end})
    return clipped


def _merge_total(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total, cur_s, cur_e = 0, intervals[0][0], intervals[0][1]
    for start, end in intervals[1:]:
        if start > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
        else:
            cur_e = max(cur_e, end)
    return total + cur_e - cur_s


def _overlap(
    updates: list[dict[str, Any]], gens: list[dict[str, Any]], limit: int = 8
) -> tuple[int, list[dict[str, Any]]]:
    updates = sorted(updates, key=lambda e: e["start_ns"])
    gens = sorted(gens, key=lambda e: e["start_ns"])
    overlaps, witnesses, j = [], [], 0
    for update in updates:
        while j < len(gens) and gens[j]["end_ns"] <= update["start_ns"]:
            j += 1
        k = j
        while k < len(gens) and gens[k]["start_ns"] < update["end_ns"]:
            gen = gens[k]
            start = max(update["start_ns"], gen["start_ns"])
            end = min(update["end_ns"], gen["end_ns"])
            if end > start:
                overlaps.append((start, end))
                if len(witnesses) < limit:
                    witnesses.append(
                        {
                            "start_ns": start,
                            "end_ns": end,
                            "duration_ns": end - start,
                            "update": _witness_event(update),
                            "generation": _witness_event(gen),
                        }
                    )
            k += 1
    return _merge_total(overlaps), witnesses


def _witness_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": event["role"],
        "kernel": _short(event["name"]),
        "start_ns": event["start_ns"],
        "end_ns": event["end_ns"],
        "device": event.get("device"),
        "stream": event.get("stream"),
        "trace_path": event["path"],
    }


def _trace_files(
    run_root: Path, step: int
) -> tuple[list[tuple[Path, str]], list[Path]]:
    native = []
    snapshot = run_root / "evidence" / "profiler" / f"fsdp-step-{step}"
    for role in ("actor", "critic"):
        if snapshot.is_dir():
            paths = (snapshot / role).glob("traces-r*.jsonl")
        else:
            paths = run_root.glob(f"logs/root/*/*/perf_tracer/{role}/traces-r*.jsonl")
        native.extend((p, role) for p in sorted(paths))
    profiler = run_root / "evidence" / "profiler"
    rollout = []
    for d in sorted(profiler.glob(f"rollout-*-step-{step}")):
        rollout.extend(sorted(d.rglob("*.json")))
        rollout.extend(sorted(d.rglob("*.json.gz")))
    return native, rollout


def snapshot_native_traces(run_root: Path, step: int) -> None:
    """Preserve evidence while the step gate pauses cumulative trace writers."""
    snapshot = run_root / "evidence" / "profiler" / f"fsdp-step-{step}"
    if snapshot.exists():
        raise FileExistsError(f"Refusing to replace frozen trace evidence: {snapshot}")
    paths, _ = _trace_files(run_root, step)
    if not paths:
        raise ValueError("No native traces to snapshot")
    for path, role in paths:
        target = snapshot / role / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        before = _sha256(path)
        shutil.copy2(path, target)
        if _sha256(target) != before or _sha256(path) != before:
            raise RuntimeError(
                "Native trace changed during snapshot; keep the step paused"
            )


def analyze(run_root: Path, step: int) -> dict[str, Any]:
    step_path = run_root / "evidence" / "steps" / f"{step}.json"
    if not step_path.exists():
        raise ValueError(f"missing step evidence: {step_path}")
    step_payload = json.loads(step_path.read_text())
    spans_by_role: dict[str, list[dict[str, Any]]] = {"actor": [], "critic": []}
    for span in step_payload.get("update_spans") or []:
        role = span.get("role")
        if role in spans_by_role and "started_ns" in span and "completed_ns" in span:
            spans_by_role[role].append(span)
    if not any(spans_by_role.values()):
        raise ValueError(f"{step_path}: missing actor/critic update_spans")

    native_paths, rollout_paths = _trace_files(run_root, step)
    update_events, gen_events, devices, traces = [], [], [], []
    for path, role in native_paths:
        traces.append({"path": str(path), "role": role, "sha256": _sha256(path)})
        update_events.extend(
            _clip_to_spans(_read_native(path, role), spans_by_role[role])
        )
    for path in rollout_paths:
        traces.append({"path": str(path), "role": "rollout", "sha256": _sha256(path)})
        events, device_props = _read_sglang(path)
        gen_events.extend(events)
        if device_props is not None:
            devices.append({"trace_path": str(path), "deviceProperties": device_props})
    if not native_paths:
        raise ValueError("missing native FSDP actor/critic perf traces")
    if not rollout_paths:
        raise ValueError(f"missing rollout profiler traces for step {step}")

    total_ns, witnesses = _overlap(update_events, gen_events)
    return {
        "passed": total_ns > 0,
        "actual_gpu_overlap": total_ns > 0,
        "status": "calculated_positive" if total_ns > 0 else "calculated_no_overlap",
        "step": step,
        "step_path": str(step_path),
        "step_sha256": _sha256(step_path),
        "overlap_duration_ns": total_ns,
        "witnesses": witnesses,
        "trace_sources": traces,
        "device_metadata": devices,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-native", action="store_true")
    args = parser.parse_args()
    try:
        if args.snapshot_native:
            snapshot_native_traces(args.run_root, args.step)
        result = analyze(args.run_root, args.step)
    except Exception as exc:
        result = {"passed": False, "status": "unknown", "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
