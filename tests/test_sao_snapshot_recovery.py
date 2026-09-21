# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixtures for SAO recovery snapshot preservation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from scripts.sao import snapshot_recovery
from scripts.sao.snapshot_recovery import SnapshotError


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_dcp(role_root: Path) -> None:
    dcp = role_root / "recover_checkpoint"
    dcp.mkdir(parents=True, exist_ok=True)
    (dcp / ".metadata").write_bytes(
        pickle.dumps(
            [
                "dcp.optim.state.fixture.exp_avg",
                "__0_0.distcp",
                "__1_0.distcp",
            ]
        )
    )
    (dcp / "__0_0.distcp").write_bytes(b"state-0")
    (dcp / "__1_0.distcp").write_bytes(b"state-1")


def _fixture(tmp_path: Path, *, completed_step: int = 20) -> Path:
    run = tmp_path / "run"
    _write_json(
        run / "evidence/resolved-config.json",
        {
            "experiment_name": "sao-math-async-ppo",
            "trial_name": "formal",
            "cluster": {"fileroot": str(run)},
            "saver": {"fileroot": str(run)},
        },
    )
    root = run / "checkpoints/root/sao-math-async-ppo/formal"
    for role in ("default", "critic"):
        _write_dcp(root / role)
    _write_json(
        root / "recover_info/step_info.json",
        {"global_step": completed_step - 1, "steps_per_epoch": 135},
    )
    _write_json(
        root / "recover_info/saver_info.json", {"step": {"steps": completed_step}}
    )
    _write_json(
        root / "recover_info/checkpoint_info.json",
        {"step": {"steps": completed_step}},
    )
    _write_json(root / "recover_info/evaluator_info.json", {"step": completed_step})
    _write_json(root / "recover_info/stats_logger_info.json", {"step": completed_step})
    (root / "recover_info/dataloader_info.pkl").write_bytes(pickle.dumps({"rank": 0}))
    return run


def test_snapshot_success_copies_native_relative_tree(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    report = snapshot_recovery.snapshot_recovery(run, 20)
    target = run / "recovery-snapshots/step-000020"
    root = target / "checkpoints/root/sao-math-async-ppo/formal"

    assert report["status"] == "created"
    assert (root / "default/recover_checkpoint/.metadata").is_file()
    assert (root / "critic/recover_checkpoint/__1_0.distcp").read_bytes() == b"state-1"
    assert (root / "recover_info/step_info.json").is_file()
    receipt = json.loads((target / "snapshot-receipt.json").read_text())
    assert receipt["completed_step"] == 20
    assert sorted(receipt["expected_paths"]) == [
        "critic/recover_checkpoint/.metadata",
        "critic/recover_checkpoint/__0_0.distcp",
        "critic/recover_checkpoint/__1_0.distcp",
        "default/recover_checkpoint/.metadata",
        "default/recover_checkpoint/__0_0.distcp",
        "default/recover_checkpoint/__1_0.distcp",
        "recover_info/checkpoint_info.json",
        "recover_info/dataloader_info.pkl",
        "recover_info/evaluator_info.json",
        "recover_info/saver_info.json",
        "recover_info/stats_logger_info.json",
        "recover_info/step_info.json",
    ]
    assert set(receipt["destination_stat_manifest"]) == set(receipt["expected_paths"])
    first_stat = receipt["destination_stat_manifest"][
        "default/recover_checkpoint/__0_0.distcp"
    ]
    assert isinstance(first_stat["mtime_ns"], int)
    assert isinstance(first_stat["ctime_ns"], int)
    assert receipt["copied_files"]["default/recover_checkpoint/__0_0.distcp"]["sha256"]


def test_snapshot_existing_valid_is_idempotent_when_source_advanced(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    snapshot_recovery.snapshot_recovery(run, 20)
    _write_json(
        run / "checkpoints/root/sao-math-async-ppo/formal/recover_info/step_info.json",
        {"global_step": 39, "steps_per_epoch": 135},
    )

    report = snapshot_recovery.snapshot_recovery(run, 20)

    assert report["status"] == "exists"
    assert report["step_info"]["global_step"] == 19
    assert report["hash_verification"]["verified_now"] is True


def test_verify_snapshot_fast_path_binds_receipt_stats(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    snapshot_recovery.snapshot_recovery(run, 20)

    report = snapshot_recovery.verify_snapshot(run, 20)

    assert report["status"] == "exists"
    assert report["hash_verification"] == {
        "verified_now": False,
        "last_full_hash_verification": "creation-copy",
    }


def test_verify_snapshot_missing_receipt_fails(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    snapshot_recovery.snapshot_recovery(run, 20)
    (run / "recovery-snapshots/step-000020/snapshot-receipt.json").unlink()

    with pytest.raises(SnapshotError, match="missing snapshot receipt"):
        snapshot_recovery.verify_snapshot(run, 20)


def test_verify_snapshot_same_size_tamper_fails(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    snapshot_recovery.snapshot_recovery(run, 20)
    target = (
        run
        / "recovery-snapshots/step-000020/checkpoints/root/sao-math-async-ppo/formal/default/recover_checkpoint/__0_0.distcp"
    )
    assert len(b"state-0") == len(b"tamper!")
    target.write_bytes(b"tamper!")

    with pytest.raises(SnapshotError, match="stat manifest changed|hash mismatch"):
        snapshot_recovery.verify_snapshot(run, 20)


def test_wrong_step_fails(tmp_path):
    run = _fixture(tmp_path, completed_step=40)

    with pytest.raises(SnapshotError, match="global_step must be 19"):
        snapshot_recovery.snapshot_recovery(run, 20)


def test_missing_role_or_shard_fails(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    (
        run
        / "checkpoints/root/sao-math-async-ppo/formal/critic/recover_checkpoint/__1_0.distcp"
    ).unlink()

    with pytest.raises(SnapshotError, match="missing DCP shard files"):
        snapshot_recovery.snapshot_recovery(run, 20)


def test_missing_required_recover_info_file_fails(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    (
        run
        / "checkpoints/root/sao-math-async-ppo/formal/recover_info/evaluator_info.json"
    ).unlink()

    with pytest.raises(SnapshotError, match="missing or empty recover_info file"):
        snapshot_recovery.snapshot_recovery(run, 20)


def test_invalid_dataloader_pickle_fails_without_unpickle(tmp_path):
    run = _fixture(tmp_path, completed_step=20)
    (
        run
        / "checkpoints/root/sao-math-async-ppo/formal/recover_info/dataloader_info.pkl"
    ).write_bytes(b"not-a-pickle")

    with pytest.raises(SnapshotError, match="pickle is not opcode-parseable"):
        snapshot_recovery.snapshot_recovery(run, 20)


def test_source_changed_midcopy_preserves_failed_staging(tmp_path, monkeypatch):
    run = _fixture(tmp_path, completed_step=20)
    original_copy = snapshot_recovery._copy_with_hash
    changed = False

    def changing_copy(src: Path, dst: Path):
        nonlocal changed
        result = original_copy(src, dst)
        if not changed and src.name == "__0_0.distcp":
            src.write_bytes(src.read_bytes() + b"-changed")
            changed = True
        return result

    monkeypatch.setattr(snapshot_recovery, "_copy_with_hash", changing_copy)

    with pytest.raises(SnapshotError, match="source changed during copy"):
        snapshot_recovery.snapshot_recovery(run, 20)

    staging = list((run / "recovery-snapshots").glob(".staging-step-000020-*"))
    assert staging
    assert not (run / "recovery-snapshots/step-000020").exists()


def test_source_inventory_changed_midcopy_preserves_failed_staging(
    tmp_path, monkeypatch
):
    run = _fixture(tmp_path, completed_step=20)
    original_copy = snapshot_recovery._copy_with_hash
    changed = False

    def changing_copy(src: Path, dst: Path):
        nonlocal changed
        result = original_copy(src, dst)
        if not changed and src.name == "__0_0.distcp":
            (
                run
                / "checkpoints/root/sao-math-async-ppo/formal/default/recover_checkpoint/extra.distcp"
            ).write_bytes(b"new")
            changed = True
        return result

    monkeypatch.setattr(snapshot_recovery, "_copy_with_hash", changing_copy)

    with pytest.raises(SnapshotError, match="source changed during copy"):
        snapshot_recovery.snapshot_recovery(run, 20)

    staging = list((run / "recovery-snapshots").glob(".staging-step-000020-*"))
    assert staging
    assert not (run / "recovery-snapshots/step-000020").exists()
