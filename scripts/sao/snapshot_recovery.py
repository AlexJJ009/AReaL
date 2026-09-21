# SPDX-License-Identifier: Apache-2.0
"""Copy SAO native recovery checkpoints into immutable step snapshots.

The native RecoverHandler/Saver layout keeps a single ``recover_checkpoint`` per
role and overwrites it at each save point.  This CLI preserves the current
native tree under ``recovery-snapshots/step-XXXXXX`` so later saves cannot erase
the recovery state for an already completed step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickletools
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sao.verify_checkpoints import _verify_dcp

ROLES = ("default", "critic")
ALLOWED_COMPLETED_STEPS = (20, 40, 60, 80, 100, 120, 135)
EXPECTED_STEPS_PER_EPOCH = 135
RECEIPT_NAME = "snapshot-receipt.json"
BUFFER_SIZE = 16 * 1024 * 1024
SHA256_HEX_LENGTH = 64
RECOVER_INFO_JSON_FILES = (
    "step_info.json",
    "saver_info.json",
    "evaluator_info.json",
    "stats_logger_info.json",
    "checkpoint_info.json",
)
RECOVER_INFO_PICKLE_FILES = ("dataloader_info.pkl",)
RECOVER_INFO_FILES = RECOVER_INFO_JSON_FILES + RECOVER_INFO_PICKLE_FILES


class SnapshotError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _derive_checkpoint_root(run_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = run_root / "evidence" / "resolved-config.json"
    if not config_path.is_file():
        raise SnapshotError(f"missing resolved config: {config_path}")
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise SnapshotError(f"resolved config is not an object: {config_path}")

    experiment = config.get("experiment_name")
    trial = config.get("trial_name")
    fileroot = _get_nested(config, "cluster", "fileroot") or _get_nested(
        config, "saver", "fileroot"
    )
    if not all(
        isinstance(value, str) and value for value in (experiment, trial, fileroot)
    ):
        raise SnapshotError("config is missing experiment_name/trial_name/fileroot")

    resolved_run_root = run_root.resolve()
    resolved_fileroot = Path(fileroot).expanduser().resolve()
    if resolved_fileroot != resolved_run_root:
        raise SnapshotError(
            f"config fileroot does not match run root: {resolved_fileroot} != {resolved_run_root}"
        )

    relative_root = Path("checkpoints") / "root" / experiment / trial
    return resolved_run_root / relative_root, relative_root, config


def _validate_step_info(checkpoint_root: Path, completed_step: int) -> dict[str, Any]:
    step_info_path = checkpoint_root / "recover_info" / "step_info.json"
    if not step_info_path.is_file():
        raise SnapshotError(f"missing recover step info: {step_info_path}")
    step_info = _read_json(step_info_path)
    if not isinstance(step_info, dict):
        raise SnapshotError(f"recover step info is not an object: {step_info_path}")
    expected_global_step = completed_step - 1
    if step_info.get("global_step") != expected_global_step:
        raise SnapshotError(
            f"recover_info global_step must be {expected_global_step} for completed step "
            f"{completed_step}, observed {step_info.get('global_step')}"
        )
    if step_info.get("steps_per_epoch") != EXPECTED_STEPS_PER_EPOCH:
        raise SnapshotError(
            f"recover_info steps_per_epoch must be {EXPECTED_STEPS_PER_EPOCH}, "
            f"observed {step_info.get('steps_per_epoch')}"
        )
    return step_info


def _validate_recover_info(
    checkpoint_root: Path, completed_step: int
) -> dict[str, Any]:
    recover_info = checkpoint_root / "recover_info"
    if not recover_info.is_dir():
        raise SnapshotError(f"missing recover_info directory: {recover_info}")

    parsed_json: dict[str, Any] = {}
    for name in RECOVER_INFO_JSON_FILES:
        path = recover_info / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SnapshotError(f"missing or empty recover_info file: {path}")
        parsed_json[name] = _read_json(path)

    for name in RECOVER_INFO_PICKLE_FILES:
        path = recover_info / name
        if not path.is_file() or path.stat().st_size == 0:
            raise SnapshotError(f"missing or empty recover_info file: {path}")
        _validate_pickle_opcodes(path)

    step_info = _validate_step_info(checkpoint_root, completed_step)
    if parsed_json["step_info.json"] != step_info:
        raise SnapshotError(
            f"recover_info step_info changed while validating: {recover_info}"
        )
    return step_info


def _validate_pickle_opcodes(path: Path) -> None:
    try:
        with path.open("rb") as stream:
            for _opcode, _arg, _pos in pickletools.genops(stream):
                pass
    except Exception as exc:
        raise SnapshotError(
            f"recover_info pickle is not opcode-parseable: {path}"
        ) from exc


def _iter_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SnapshotError(f"missing directory: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def _stat_manifest(files: list[Path], base: Path) -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    for path in files:
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise SnapshotError(f"source file disappeared: {path}") from exc
        if not path.is_file():
            raise SnapshotError(f"not a regular file: {path}")
        manifest[path.relative_to(base).as_posix()] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
    return manifest


def _source_files(checkpoint_root: Path) -> list[Path]:
    files: list[Path] = []
    for role in ROLES:
        role_root = checkpoint_root / role
        if not role_root.is_dir():
            raise SnapshotError(f"missing role checkpoint root: {role_root}")
        _verify_role_dcp(role_root)
        files.extend(_iter_files(role_root / "recover_checkpoint"))
    recover_info = checkpoint_root / "recover_info"
    files.extend(recover_info / name for name in RECOVER_INFO_FILES)
    return sorted(files)


def _verify_role_dcp(role_root: Path) -> dict[str, Any]:
    try:
        return _verify_dcp(role_root)
    except Exception as exc:
        raise SnapshotError(str(exc)) from exc


def _copy_with_hash(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with src.open("rb") as src_stream, dst.open("xb") as dst_stream:
        while True:
            chunk = src_stream.read(BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            dst_stream.write(chunk)
            total += len(chunk)
    src_stat = src.stat()
    dst_stat = dst.stat()
    if dst_stat.st_size != src_stat.st_size or total != src_stat.st_size:
        raise SnapshotError(
            f"copied size mismatch for {src}: source={src_stat.st_size} "
            f"destination={dst_stat.st_size} streamed={total}"
        )
    source_sha256 = digest.hexdigest()
    destination_sha256 = _sha256_file(dst)
    if destination_sha256 != source_sha256:
        raise SnapshotError(f"copied hash mismatch for {src}")
    return {"bytes": dst_stat.st_size, "sha256": source_sha256}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_HEX_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _copy_tree_files(
    files: list[Path], source_base: Path, destination_base: Path
) -> dict[str, dict[str, Any]]:
    copied: dict[str, dict[str, Any]] = {}
    for src in files:
        rel = src.relative_to(source_base)
        digest = _copy_with_hash(src, destination_base / rel)
        copied[rel.as_posix()] = digest
    return copied


def verify_snapshot(
    run_root: Path, completed_step: int, *, verify_hashes: bool = False
) -> dict[str, Any]:
    """Verify an existing recovery snapshot without reading live source weights.

    Fast verification binds the snapshot to its receipt, exact file inventory,
    byte sizes, mtimes, and ctimes.  Full file hashes are recomputed only when
    ``verify_hashes`` is true; otherwise the report explicitly says the full
    hashes were last verified during the original copy.
    """
    if completed_step not in ALLOWED_COMPLETED_STEPS:
        raise SnapshotError(
            f"completed step must be one of {list(ALLOWED_COMPLETED_STEPS)}, got {completed_step}"
        )

    run_root = run_root.expanduser().resolve()
    _source_checkpoint_root, relative_checkpoint_root, _config = (
        _derive_checkpoint_root(run_root)
    )
    snapshot_root = run_root / "recovery-snapshots" / f"step-{completed_step:06d}"
    if not snapshot_root.is_dir():
        raise SnapshotError(f"missing snapshot: {snapshot_root}")

    checkpoint_root = snapshot_root / relative_checkpoint_root
    step_info = _validate_recover_info(checkpoint_root, completed_step)
    receipt_path = snapshot_root / RECEIPT_NAME
    if not receipt_path.is_file():
        raise SnapshotError(f"missing snapshot receipt: {receipt_path}")
    receipt = _read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise SnapshotError(f"snapshot receipt is not an object: {receipt_path}")
    if receipt.get("completed_step") != completed_step:
        raise SnapshotError(f"snapshot receipt completed_step mismatch: {receipt_path}")
    if receipt.get("expected_global_step") != completed_step - 1:
        raise SnapshotError(
            f"snapshot receipt expected_global_step mismatch: {receipt_path}"
        )
    if receipt.get("checkpoint_relative_root") != relative_checkpoint_root.as_posix():
        raise SnapshotError(
            f"snapshot receipt checkpoint root mismatch: {receipt_path}"
        )
    if receipt.get("snapshot_root") != str(snapshot_root):
        raise SnapshotError(f"snapshot receipt snapshot_root mismatch: {receipt_path}")
    if receipt.get("step_info") != step_info:
        raise SnapshotError(f"snapshot receipt step_info mismatch: {receipt_path}")

    copied_files = receipt.get("copied_files")
    if not isinstance(copied_files, dict) or not copied_files:
        raise SnapshotError(f"snapshot receipt missing copied_files: {receipt_path}")
    expected_paths = receipt.get("expected_paths")
    if not isinstance(expected_paths, list) or not all(
        isinstance(path, str) for path in expected_paths
    ):
        raise SnapshotError(f"snapshot receipt missing expected_paths: {receipt_path}")
    destination_stat_manifest = receipt.get("destination_stat_manifest")
    if not isinstance(destination_stat_manifest, dict) or not destination_stat_manifest:
        raise SnapshotError(
            f"snapshot receipt missing destination_stat_manifest: {receipt_path}"
        )

    expected_files = _source_files(checkpoint_root)
    actual_paths = {
        path.relative_to(checkpoint_root).as_posix() for path in expected_files
    }
    receipt_paths = set(copied_files)
    expected_path_set = set(expected_paths)
    stat_paths = set(destination_stat_manifest)
    if sorted(expected_paths) != expected_paths or len(expected_paths) != len(
        expected_path_set
    ):
        raise SnapshotError(
            f"snapshot receipt expected_paths is not canonical: {receipt_path}"
        )
    if actual_paths != expected_path_set:
        raise SnapshotError(
            "snapshot file inventory does not match receipt expected_paths: "
            f"actual={sorted(actual_paths)} receipt={expected_paths}"
        )
    if actual_paths != receipt_paths:
        raise SnapshotError(
            "snapshot file inventory does not match receipt copied_files: "
            f"actual={sorted(actual_paths)} receipt={sorted(receipt_paths)}"
        )
    if actual_paths != stat_paths:
        raise SnapshotError(
            "snapshot file inventory does not match receipt destination_stat_manifest: "
            f"actual={sorted(actual_paths)} receipt={sorted(stat_paths)}"
        )

    for rel, copied in copied_files.items():
        if not isinstance(copied, dict):
            raise SnapshotError(f"snapshot receipt copied file entry is invalid: {rel}")
        if not isinstance(copied.get("bytes"), int) or copied["bytes"] < 0:
            raise SnapshotError(f"snapshot receipt copied file bytes is invalid: {rel}")
        if not _is_sha256(copied.get("sha256")):
            raise SnapshotError(
                f"snapshot receipt copied file sha256 is invalid: {rel}"
            )
        stat_entry = destination_stat_manifest.get(rel)
        if not isinstance(stat_entry, dict):
            raise SnapshotError(f"snapshot receipt stat entry is invalid: {rel}")
        for key in ("bytes", "mtime_ns", "ctime_ns"):
            if not isinstance(stat_entry.get(key), int) or stat_entry[key] < 0:
                raise SnapshotError(f"snapshot receipt stat {key} is invalid: {rel}")
        if stat_entry["bytes"] != copied["bytes"]:
            raise SnapshotError(f"snapshot receipt size mismatch for {rel}")

    current_stat_manifest = _stat_manifest(expected_files, checkpoint_root)
    if current_stat_manifest != destination_stat_manifest:
        raise SnapshotError("snapshot destination stat manifest changed since copy")

    hash_mismatches: list[str] = []
    if verify_hashes:
        for path in expected_files:
            rel = path.relative_to(checkpoint_root).as_posix()
            if _sha256_file(path) != copied_files[rel]["sha256"]:
                hash_mismatches.append(rel)
        if hash_mismatches:
            raise SnapshotError(
                f"snapshot hash mismatch for files: {sorted(hash_mismatches)}"
            )

    dcp_reports = {role: _verify_role_dcp(checkpoint_root / role) for role in ROLES}
    return {
        "status": "exists",
        "snapshot_root": str(snapshot_root),
        "checkpoint_root": str(checkpoint_root),
        "completed_step": completed_step,
        "step_info": step_info,
        "dcp": dcp_reports,
        "receipt_path": str(receipt_path),
        "copied_file_count": len(copied_files),
        "hash_verification": {
            "verified_now": verify_hashes,
            "last_full_hash_verification": (
                "current-verify" if verify_hashes else "creation-copy"
            ),
        },
    }


def snapshot_recovery(run_root: Path, completed_step: int) -> dict[str, Any]:
    if completed_step not in ALLOWED_COMPLETED_STEPS:
        raise SnapshotError(
            f"completed step must be one of {list(ALLOWED_COMPLETED_STEPS)}, got {completed_step}"
        )

    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise SnapshotError(f"run root is not a directory: {run_root}")

    source_checkpoint_root, relative_checkpoint_root, config = _derive_checkpoint_root(
        run_root
    )
    snapshot_parent = run_root / "recovery-snapshots"
    target = snapshot_parent / f"step-{completed_step:06d}"

    if target.exists():
        try:
            report = verify_snapshot(run_root, completed_step, verify_hashes=True)
        except Exception as exc:
            raise SnapshotError(
                f"existing snapshot is invalid and will not be overwritten: {target}: {exc}"
            ) from exc
        return report

    step_info = _validate_recover_info(source_checkpoint_root, completed_step)
    source_files = _source_files(source_checkpoint_root)
    before_manifest = _stat_manifest(source_files, source_checkpoint_root)

    snapshot_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".staging-step-{completed_step:06d}-",
            dir=snapshot_parent,
        )
    )
    staging_checkpoint_root = staging / relative_checkpoint_root

    dcp_reports = {
        role: _verify_role_dcp(source_checkpoint_root / role) for role in ROLES
    }
    copied = _copy_tree_files(
        source_files,
        source_checkpoint_root,
        staging_checkpoint_root,
    )
    after_source_files = _source_files(source_checkpoint_root)
    after_manifest = _stat_manifest(after_source_files, source_checkpoint_root)
    if after_manifest != before_manifest:
        raise SnapshotError(
            f"source changed during copy; preserving failed staging for inspection: {staging}"
        )

    target_files = [staging_checkpoint_root / Path(rel) for rel in copied]
    destination_stat_manifest = _stat_manifest(target_files, staging_checkpoint_root)
    for rel, copied_file in copied.items():
        expected = before_manifest[rel]["bytes"]
        if (
            copied_file["bytes"] != expected
            or destination_stat_manifest[rel]["bytes"] != expected
        ):
            raise SnapshotError(
                f"snapshot copy size mismatch for {rel}: expected {expected}"
            )

    receipt = {
        "status": "created",
        "run_root": str(run_root),
        "source_checkpoint_root": str(source_checkpoint_root),
        "snapshot_root": str(target),
        "checkpoint_relative_root": relative_checkpoint_root.as_posix(),
        "completed_step": completed_step,
        "expected_global_step": completed_step - 1,
        "step_info": step_info,
        "config_binding": {
            "experiment_name": config.get("experiment_name"),
            "trial_name": config.get("trial_name"),
            "fileroot": str(run_root),
        },
        "dcp": dcp_reports,
        "source_stat_manifest": before_manifest,
        "destination_stat_manifest": destination_stat_manifest,
        "expected_paths": sorted(copied),
        "copied_files": copied,
        "hash_verification": {
            "verified_now": True,
            "last_full_hash_verification": "creation-copy",
        },
    }
    _write_json(staging / RECEIPT_NAME, receipt)
    try:
        os.rename(staging, target)
    except FileExistsError as exc:
        raise SnapshotError(f"snapshot target appeared during copy: {target}") from exc

    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--completed-step",
        required=True,
        type=int,
        choices=ALLOWED_COMPLETED_STEPS,
    )
    args = parser.parse_args(argv)
    try:
        report = snapshot_recovery(args.run_root, args.completed_step)
        print(json.dumps({"passed": True, **report}, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
