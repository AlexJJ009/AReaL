# SPDX-License-Identifier: Apache-2.0
"""Watch SAO step evidence and preserve required recovery/eval snapshots."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sao import snapshot_eval, snapshot_recovery

REQUIRED_POINTS = snapshot_recovery.ALLOWED_COMPLETED_STEPS
RECEIPT_NAME = "snapshot-watcher.json"


class WatcherError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _event(kind: str, **fields: Any) -> dict[str, Any]:
    return {"event": kind, "recorded_ns": time.time_ns(), **fields}


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _load_receipt(path: Path, run_root: Path, dataset: Path, points: tuple[int, ...]):
    if path.exists():
        receipt = _read_json(path)
        if not isinstance(receipt, dict):
            raise WatcherError(f"watcher receipt is not an object: {path}")
        if receipt.get("status") == "failed":
            raise WatcherError(f"watcher receipt is failed: {receipt.get('error')}")
        if receipt.get("required_points") != list(points):
            raise WatcherError(f"watcher receipt required_points mismatch: {path}")
        if receipt.get("run_root") != str(run_root) or receipt.get("dataset") != str(
            dataset
        ):
            raise WatcherError(f"watcher receipt run/dataset binding mismatch: {path}")
        return receipt
    return {
        "schema_version": 1,
        "status": "pending",
        "run_root": str(run_root),
        "dataset": str(dataset),
        "required_points": list(points),
        "points": {},
    }


def _step_payload(run_root: Path, point: int) -> tuple[str, dict[str, Any] | None]:
    path = run_root / "evidence" / "steps" / f"{point}.json"
    if not path.exists():
        return "pending", None
    try:
        payload = _read_json(path)
    except json.JSONDecodeError:
        return "pending", None
    if not isinstance(payload, dict):
        raise WatcherError(f"step evidence is not an object: {path}")
    if (
        payload.get("completed_step") != point
        or payload.get("published_version") != point
    ):
        raise WatcherError(
            f"step evidence identity mismatch for {point}: "
            f"completed_step={payload.get('completed_step')} "
            f"published_version={payload.get('published_version')}"
        )
    return "complete", payload


def _snapshot_dir(run_root: Path, point: int) -> Path:
    return run_root / "recovery-snapshots" / f"step-{point:06d}"


def _live_global_step(run_root: Path) -> int | None:
    try:
        root, _relative, _config = snapshot_recovery._derive_checkpoint_root(run_root)
        payload = _read_json(root / "recover_info" / "step_info.json")
    except (FileNotFoundError, json.JSONDecodeError, snapshot_recovery.SnapshotError):
        return None
    value = payload.get("global_step") if isinstance(payload, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _verify_or_copy_recovery(run_root: Path, point: int) -> dict[str, Any]:
    if _snapshot_dir(run_root, point).is_dir():
        return snapshot_recovery.verify_snapshot(run_root, point, verify_hashes=False)
    live = _live_global_step(run_root)
    if live is not None and live > point - 1:
        raise WatcherError(
            f"cannot catch up recovery snapshot for step {point}: "
            f"live recover_info global_step={live} already passed target {point - 1}"
        )
    return snapshot_recovery.snapshot_recovery(run_root, point)


def _preserve_point(run_root: Path, dataset: Path, point: int) -> dict[str, Any]:
    recovery = _verify_or_copy_recovery(run_root, point)
    evaluation = snapshot_eval.snapshot_eval(run_root / "evidence", dataset, point)
    return {
        "status": "preserved",
        "completed_step": point,
        "recovery_status": recovery.get("status"),
        "recovery_snapshot_root": recovery.get("snapshot_root"),
        "eval_status": evaluation.get("status"),
        "eval_snapshot_root": evaluation.get("snapshot_root"),
        "preserved_ns": time.time_ns(),
    }


def _fail(receipt: dict[str, Any], receipt_path: Path, message: str) -> None:
    receipt["status"] = "failed"
    receipt["error"] = message
    receipt["failed_ns"] = time.time_ns()
    _atomic_write_json(receipt_path, receipt)


class SnapshotWatcher:
    def __init__(
        self,
        run_root: Path,
        dataset: Path,
        *,
        points: tuple[int, ...] = REQUIRED_POINTS,
        emit: Callable[[dict[str, Any]], None] = _emit,
    ) -> None:
        self.run_root = run_root.expanduser().resolve()
        self.dataset = dataset.expanduser().resolve()
        self.points = points
        self.emit = emit
        self._verified_existing: set[int] = set()

    def tick(self) -> bool:
        receipt_path = self.run_root / "evidence" / RECEIPT_NAME
        try:
            receipt = _load_receipt(
                receipt_path, self.run_root, self.dataset, self.points
            )
        except Exception as exc:
            self.emit(_event("failure", error=str(exc)))
            raise
        receipt_points = receipt.setdefault("points", {})

        try:
            for point in self.points:
                key = str(point)
                if receipt_points.get(key, {}).get("status") == "preserved":
                    if point not in self._verified_existing:
                        receipt_points[key] = _preserve_point(
                            self.run_root, self.dataset, point
                        )
                        _atomic_write_json(receipt_path, receipt)
                        self._verified_existing.add(point)
                    continue

                status, _payload = _step_payload(self.run_root, point)
                if status == "pending":
                    if not _snapshot_dir(self.run_root, point).is_dir():
                        live = _live_global_step(self.run_root)
                        if live is not None and live > point - 1:
                            raise WatcherError(
                                f"cannot catch up recovery snapshot for step {point}: "
                                f"live recover_info global_step={live} already passed "
                                f"target {point - 1}"
                            )
                    return False

                self.emit(_event("new_completed_step", completed_step=point))
                receipt_points[key] = _preserve_point(
                    self.run_root, self.dataset, point
                )
                self._verified_existing.add(point)
                receipt["last_preserved_step"] = point
                _atomic_write_json(receipt_path, receipt)
                self.emit(_event("milestone_preserved", **receipt_points[key]))

            receipt["status"] = "preservation_complete"
            receipt["completed_ns"] = time.time_ns()
            _atomic_write_json(receipt_path, receipt)
            self.emit(
                _event("preservation_complete", required_points=list(self.points))
            )
            return True
        except Exception as exc:
            _fail(receipt, receipt_path, str(exc))
            self.emit(_event("failure", error=str(exc)))
            raise


def tick(
    run_root: Path,
    dataset: Path,
    *,
    points: tuple[int, ...] = REQUIRED_POINTS,
    emit: Callable[[dict[str, Any]], None] = _emit,
) -> bool:
    return SnapshotWatcher(run_root, dataset, points=points, emit=emit).tick()


def watch(run_root: Path, dataset: Path, poll_seconds: int) -> int:
    watcher = SnapshotWatcher(run_root, dataset)
    while True:
        if watcher.tick():
            return 0
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or args.poll_seconds > 60:
        print(
            json.dumps(
                {
                    "event": "failure",
                    "error": "--poll-seconds must be between 1 and 60",
                },
                sort_keys=True,
            )
        )
        return 1
    try:
        return watch(args.run_root, args.dataset, args.poll_seconds)
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
