# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the SAO snapshot preservation watcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sao import watch_snapshots


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _step(run: Path, step: int, *, completed_step: int | None = None) -> None:
    _write_json(
        run / "evidence" / "steps" / f"{step}.json",
        {
            "completed_step": step if completed_step is None else completed_step,
            "published_version": step,
        },
    )


def _patch_helpers(
    monkeypatch, calls: list[tuple[str, int]], *, fail_at: int | None = None
):
    def verify(run_root: Path, completed_step: int, *, verify_hashes: bool = False):
        calls.append(("verify", completed_step))
        assert verify_hashes is False
        return {
            "status": "exists",
            "snapshot_root": str(
                run_root / "recovery-snapshots" / f"step-{completed_step:06d}"
            ),
        }

    def recover(run_root: Path, completed_step: int):
        calls.append(("recover", completed_step))
        if fail_at == completed_step:
            raise RuntimeError("copy failed")
        return {
            "status": "created",
            "snapshot_root": str(
                run_root / "recovery-snapshots" / f"step-{completed_step:06d}"
            ),
        }

    def eval_snapshot(evidence_dir: Path, dataset: Path, version: int):
        calls.append(("eval", version))
        return {
            "status": "exists",
            "snapshot_root": str(evidence_dir / "eval-snapshots" / f"version{version}"),
        }

    monkeypatch.setattr(watch_snapshots.snapshot_recovery, "verify_snapshot", verify)
    monkeypatch.setattr(watch_snapshots.snapshot_recovery, "snapshot_recovery", recover)
    monkeypatch.setattr(watch_snapshots.snapshot_eval, "snapshot_eval", eval_snapshot)
    monkeypatch.setattr(watch_snapshots, "_live_global_step", lambda _run: None)


def test_already_preserved_20_uses_fast_verify_not_large_copy(tmp_path, monkeypatch):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    (run / "recovery-snapshots" / "step-000020").mkdir(parents=True)
    _step(run, 20)
    calls = []
    _patch_helpers(monkeypatch, calls)

    done = watch_snapshots.tick(run, dataset, points=(20, 40), emit=lambda _event: None)

    assert done is False
    assert calls == [("verify", 20), ("eval", 20)]
    receipt = json.loads((run / "evidence" / "snapshot-watcher.json").read_text())
    assert receipt["points"]["20"]["status"] == "preserved"


def test_next_point_triggers_after_correct_step_file(tmp_path, monkeypatch):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    _step(run, 40)
    events = []
    calls = []
    _patch_helpers(monkeypatch, calls)

    done = watch_snapshots.tick(run, dataset, points=(40,), emit=events.append)

    assert done is True
    assert calls == [("recover", 40), ("eval", 40)]
    assert [event["event"] for event in events] == [
        "new_completed_step",
        "milestone_preserved",
        "preservation_complete",
    ]


def test_no_premature_copy_when_step_file_absent_or_incomplete(tmp_path, monkeypatch):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    (run / "evidence" / "steps").mkdir(parents=True)
    (run / "evidence" / "steps" / "20.json").write_text("{", encoding="utf-8")
    calls = []
    _patch_helpers(monkeypatch, calls)

    assert (
        watch_snapshots.tick(run, dataset, points=(20,), emit=lambda _event: None)
        is False
    )
    assert calls == []


def test_wrong_step_identity_fails_closed(tmp_path, monkeypatch):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    _step(run, 20, completed_step=19)
    calls = []
    events = []
    _patch_helpers(monkeypatch, calls)

    with pytest.raises(watch_snapshots.WatcherError, match="identity mismatch"):
        watch_snapshots.tick(run, dataset, points=(20,), emit=events.append)

    receipt = json.loads((run / "evidence" / "snapshot-watcher.json").read_text())
    assert receipt["status"] == "failed"
    assert calls == []
    assert events[-1]["event"] == "failure"


def test_lagged_live_recover_info_fails_before_copy(tmp_path, monkeypatch):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    calls = []
    _patch_helpers(monkeypatch, calls)
    monkeypatch.setattr(watch_snapshots, "_live_global_step", lambda _run: 39)

    with pytest.raises(watch_snapshots.WatcherError, match="cannot catch up"):
        watch_snapshots.tick(run, dataset, points=(20,), emit=lambda _event: None)

    assert calls == []
    receipt = json.loads((run / "evidence" / "snapshot-watcher.json").read_text())
    assert receipt["status"] == "failed"


def test_helper_failure_fails_closed_without_retry(tmp_path, monkeypatch):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    _step(run, 20)
    calls = []
    events = []
    _patch_helpers(monkeypatch, calls, fail_at=20)

    with pytest.raises(RuntimeError, match="copy failed"):
        watch_snapshots.tick(run, dataset, points=(20,), emit=events.append)

    assert calls == [("recover", 20)]
    receipt = json.loads((run / "evidence" / "snapshot-watcher.json").read_text())
    assert receipt["status"] == "failed"
    assert events[-1]["event"] == "failure"


def test_done_only_after_all_required_points(tmp_path, monkeypatch):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    _step(run, 20)
    calls = []
    events = []
    _patch_helpers(monkeypatch, calls)
    watcher = watch_snapshots.SnapshotWatcher(
        run, dataset, points=(20, 40), emit=events.append
    )

    assert watcher.tick() is False
    assert not any(event["event"] == "preservation_complete" for event in events)

    _step(run, 40)
    assert watcher.tick() is True
    receipt = json.loads((run / "evidence" / "snapshot-watcher.json").read_text())
    assert receipt["status"] == "preservation_complete"
    assert set(receipt["points"]) == {"20", "40"}


@pytest.mark.parametrize("field", ["run_root", "dataset"])
def test_wrong_receipt_binding_is_reported_without_overwrite(tmp_path, field):
    run = tmp_path / "run"
    dataset = tmp_path / "manifest.json"
    receipt = {
        "status": "pending",
        "required_points": [20],
        "run_root": str(run),
        "dataset": str(dataset),
        "points": {},
    }
    receipt[field] = str(tmp_path / "other")
    path = run / "evidence" / "snapshot-watcher.json"
    _write_json(path, receipt)
    before = path.read_bytes()
    events = []
    with pytest.raises(watch_snapshots.WatcherError, match="binding mismatch"):
        watch_snapshots.tick(run, dataset, points=(20,), emit=events.append)
    assert events[-1]["event"] == "failure"
    assert path.read_bytes() == before
