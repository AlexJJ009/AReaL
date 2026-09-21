# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixtures for SAO checkpoint structure verification."""

from __future__ import annotations

import json
import pickle
import struct
from pathlib import Path

from scripts.sao.snapshot_recovery import snapshot_recovery
from scripts.sao.verify_checkpoints import verify_run


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_safetensors(path: Path, *, payload: bytes = b"\0\0\0\0") -> None:
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        separators=(",", ":"),
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _fixture(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    _write_json(
        run / "evidence/resolved-config.json",
        {
            "experiment_name": "sao-math-async-ppo",
            "trial_name": "formal",
            "cluster": {"fileroot": str(run)},
            "train_dataset": {"batch_size": 128},
            "saver": {"freq_steps": 20},
        },
    )
    _write_json(
        run / "evidence/epoch-order.json",
        {"dataloader_steps": 135, "preflight": False},
    )
    _write_json(
        run / "evidence/epoch-finished.json",
        {"train_dataset_rows": 17157, "expected_steps": 135, "preflight": False},
    )
    root = run / "checkpoints/root/sao-math-async-ppo/formal"
    for role in ("default", "critic"):
        role_root = root / role
        for step in (19, 39, 59, 79, 99, 119, 134):
            ckpt = role_root / f"epoch0epochstep{step}globalstep{step}"
            _write_json(ckpt / "config.json", {"model_type": "fixture"})
            _write_safetensors(ckpt / "model.safetensors")
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
        (dcp / "__0_0.distcp").write_bytes(b"state")
        (dcp / "__1_0.distcp").write_bytes(b"state")
    _write_json(
        root / "recover_info/step_info.json",
        {"global_step": 134, "steps_per_epoch": 135},
    )
    _write_json(root / "recover_info/saver_info.json", {"step": {"steps": 135}})
    _write_json(root / "recover_info/checkpoint_info.json", {"step": {"steps": 135}})
    _write_json(root / "recover_info/evaluator_info.json", {"step": {"steps": 135}})
    _write_json(root / "recover_info/stats_logger_info.json", {"step": 135})
    (root / "recover_info/dataloader_info.pkl").write_bytes(pickle.dumps({"step": 135}))
    for completed in (20, 40, 60, 80, 100, 120, 135):
        _write_json(
            root / "recover_info/step_info.json",
            {"global_step": completed - 1, "steps_per_epoch": 135},
        )
        snapshot_recovery(run, completed)
    return run


def test_complete_fixture_passes(tmp_path):
    report = verify_run(_fixture(tmp_path))
    assert report["passed"], report["errors"]
    assert report["evidence"]["derived"]["expected_global_steps"] == [
        19,
        39,
        59,
        79,
        99,
        119,
        134,
    ]


def test_missing_critic_fails(tmp_path):
    run = _fixture(tmp_path)
    critic = run / "checkpoints/root/sao-math-async-ppo/formal/critic"
    for path in sorted(critic.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    critic.rmdir()
    report = verify_run(run)
    assert not report["passed"]
    assert any("missing role checkpoint root" in error for error in report["errors"])


def test_wrong_recover_step_fails(tmp_path):
    run = _fixture(tmp_path)
    _write_json(
        run / "checkpoints/root/sao-math-async-ppo/formal/recover_info/step_info.json",
        {"global_step": 119, "steps_per_epoch": 135},
    )
    report = verify_run(run)
    assert not report["passed"]
    assert any("completed step 135" in error for error in report["errors"])


def test_empty_weight_file_fails(tmp_path):
    run = _fixture(tmp_path)
    target = (
        run
        / "checkpoints/root/sao-math-async-ppo/formal/default/epoch0epochstep19globalstep19/model.safetensors"
    )
    target.write_bytes(b"")
    report = verify_run(run)
    assert not report["passed"]
    assert any("empty or truncated safetensors" in error for error in report["errors"])


def test_truncated_safetensors_payload_fails(tmp_path):
    run = _fixture(tmp_path)
    target = (
        run
        / "checkpoints/root/sao-math-async-ppo/formal/default/epoch0epochstep19globalstep19/model.safetensors"
    )
    _write_safetensors(target, payload=b"\0\0")
    report = verify_run(run)
    assert not report["passed"]
    assert any("payload size mismatch" in error for error in report["errors"])


def test_missing_referenced_dcp_rank_shard_fails(tmp_path):
    run = _fixture(tmp_path)
    target = (
        run
        / "checkpoints/root/sao-math-async-ppo/formal/default/recover_checkpoint/__1_0.distcp"
    )
    target.unlink()
    report = verify_run(run)
    assert not report["passed"]
    assert any(
        "missing DCP shard files referenced by metadata" in error
        for error in report["errors"]
    )


def test_preflight_is_unavailable(tmp_path):
    run = _fixture(tmp_path)
    _write_json(
        run / "evidence/epoch-order.json", {"dataloader_steps": 135, "preflight": True}
    )
    report = verify_run(run)
    assert not report["passed"]
    assert any("preflight must be false" in error for error in report["errors"])


def test_missing_retained_recovery_receipt_fails(tmp_path):
    run = _fixture(tmp_path)
    (run / "recovery-snapshots/step-000020/snapshot-receipt.json").unlink()
    report = verify_run(run)
    assert not report["passed"]
    assert any("recovery snapshot step20" in error for error in report["errors"])


def test_extra_post_epoch_checkpoint_fails(tmp_path):
    run = _fixture(tmp_path)
    (
        run
        / "checkpoints/root/sao-math-async-ppo/formal/default/epoch0epochstep135globalstep135"
    ).mkdir()
    report = verify_run(run)
    assert not report["passed"]
    assert any(
        "unexpected=['epoch0epochstep135globalstep135']" in error
        for error in report["errors"]
    )
