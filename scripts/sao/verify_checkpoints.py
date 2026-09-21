# SPDX-License-Identifier: Apache-2.0
"""Read-only SAO checkpoint cadence verifier.

This script checks structural checkpoint evidence only.  It does not load model
weights, import torch, validate eval metrics, or prove that reload succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickletools
import struct
import sys
from pathlib import Path
from typing import Any

ROLES = ("default", "critic")
EXPECTED_TRAIN_ROWS = 17157
EXPECTED_BATCH_SIZE = 128
DEFAULT_SAVE_INTERVAL = 20
HASH_LIMIT = 4 * 1024 * 1024


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_small(path: Path) -> str:
    size = path.stat().st_size
    if size > HASH_LIMIT:
        raise ValueError(f"refusing to hash large file: {path} ({size} bytes)")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_json(path: Path, evidence: dict[str, Any]) -> None:
    if path.exists():
        evidence.setdefault("hash_bindings", {})[str(path)] = {
            "sha256": _sha256_small(path),
            "bytes": path.stat().st_size,
        }


def _add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _derive_expected(
    config: dict[str, Any], epoch_order: dict[str, Any], epoch_finished: dict[str, Any]
) -> tuple[int | None, int | None, list[int]]:
    rows = epoch_finished.get("train_dataset_rows")
    batch_size = _get_nested(config, "train_dataset", "batch_size")
    steps = epoch_finished.get("expected_steps") or epoch_order.get("dataloader_steps")
    if isinstance(rows, int) and isinstance(batch_size, int) and batch_size > 0:
        steps = math.ceil(rows / batch_size)
    save_interval = _get_nested(config, "saver", "freq_steps")
    if not isinstance(save_interval, int) or save_interval <= 0:
        save_interval = DEFAULT_SAVE_INTERVAL
    if not isinstance(steps, int) or steps <= 0:
        return rows if isinstance(rows, int) else None, None, []
    completed = list(range(save_interval, steps + 1, save_interval))
    if not completed or completed[-1] != steps:
        completed.append(steps)
    return (
        rows if isinstance(rows, int) else None,
        steps,
        [step - 1 for step in completed],
    )


def _safetensors_header(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing safetensors: {path}")
    if path.stat().st_size < 9:
        raise ValueError(f"empty or truncated safetensors: {path}")
    with path.open("rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        if header_len <= 0 or header_len > 128 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = stream.read(header_len)
    if len(header) != header_len:
        raise ValueError(f"truncated safetensors header: {path}")
    try:
        parsed = json.loads(header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors header JSON: {path}") from exc
    tensor_keys = [key for key in parsed if key != "__metadata__"]
    if not tensor_keys:
        raise ValueError(f"safetensors has no tensor entries: {path}")
    max_end = 0
    for key in tensor_keys:
        tensor = parsed[key]
        if not isinstance(tensor, dict):
            raise ValueError(f"safetensors tensor entry is not an object: {path}:{key}")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in offsets
            )
        ):
            raise ValueError(
                f"safetensors tensor has invalid data_offsets: {path}:{key}"
            )
        start, end = offsets
        if start < 0 or end < start:
            raise ValueError(
                f"safetensors tensor has invalid data_offsets: {path}:{key}"
            )
        max_end = max(max_end, end)
    payload_bytes = path.stat().st_size - 8 - header_len
    if payload_bytes < 0:
        raise ValueError(f"safetensors file is smaller than its header: {path}")
    if max_end != payload_bytes:
        raise ValueError(
            f"safetensors payload size mismatch: {path} header_requires={max_end} actual_payload={payload_bytes}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "header_bytes": header_len,
        "payload_bytes": payload_bytes,
        "tensor_count": len(tensor_keys),
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def _verify_hf_checkpoint(role_root: Path, global_step: int) -> dict[str, Any]:
    directory = role_root / f"epoch0epochstep{global_step}globalstep{global_step}"
    if not directory.is_dir():
        raise ValueError(f"missing checkpoint directory: {directory}")
    config = directory / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise ValueError(f"missing or empty config.json: {config}")
    _read_json(config)
    header = _safetensors_header(directory / "model.safetensors")
    return {
        "directory": str(directory),
        "config_bytes": config.stat().st_size,
        "config_sha256": _sha256_small(config),
        "safetensors": header,
    }


def _verify_dcp(role_root: Path) -> dict[str, Any]:
    dcp_dir = role_root / "recover_checkpoint"
    metadata = dcp_dir / ".metadata"
    if not metadata.is_file() or metadata.stat().st_size == 0:
        raise ValueError(f"missing or empty DCP metadata: {metadata}")
    metadata_bytes = metadata.read_bytes()
    referenced_strings = _metadata_strings(metadata_bytes)
    referenced_shards = sorted(
        {Path(value).name for value in referenced_strings if value.endswith(".distcp")}
    )
    optimizer_refs = sorted(
        value
        for value in referenced_strings
        if "optim" in value.lower()
        and ("dcp." in value or "optimizer" in value.lower())
    )
    if not referenced_shards:
        raise ValueError(f"DCP metadata does not reference shard files: {metadata}")
    if not optimizer_refs:
        raise ValueError(f"DCP metadata does not reference optimizer state: {metadata}")
    shards = [dcp_dir / name for name in referenced_shards]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise ValueError(f"missing DCP shard files referenced by metadata: {missing}")
    empty = [str(path) for path in shards if path.stat().st_size == 0]
    if empty:
        raise ValueError(f"empty DCP shard files: {empty}")
    prefix = metadata_bytes[:HASH_LIMIT]
    return {
        "directory": str(dcp_dir),
        "metadata_bytes": metadata.stat().st_size,
        "metadata_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "metadata_prefix_bytes": len(prefix),
        "referenced_distcp_files": referenced_shards,
        "optimizer_reference_count": len(optimizer_refs),
        "distcp_files": [
            {"path": str(path), "bytes": path.stat().st_size} for path in shards
        ],
    }


def _metadata_strings(metadata: bytes) -> list[str]:
    strings: list[str] = []
    try:
        for opcode, arg, _pos in pickletools.genops(metadata):
            if opcode.name in {
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "UNICODE",
                "BINUNICODE8",
            }:
                strings.append(str(arg))
    except Exception as exc:
        raise ValueError("DCP metadata is not parseable as a pickle stream") from exc
    return strings


def verify_run(run_root: Path) -> dict[str, Any]:
    # Import here: snapshot creation also reuses this module's DCP inspector.
    from scripts.sao.snapshot_recovery import verify_snapshot

    errors: list[str] = []
    evidence: dict[str, Any] = {"run_root": str(run_root), "hash_bindings": {}}
    evidence_dir = run_root / "evidence"
    config_path = evidence_dir / "resolved-config.json"
    epoch_order_path = evidence_dir / "epoch-order.json"
    epoch_finished_path = evidence_dir / "epoch-finished.json"

    try:
        config = _read_json(config_path)
        epoch_order = _read_json(epoch_order_path)
        epoch_finished = _read_json(epoch_finished_path)
        for path in (config_path, epoch_order_path, epoch_finished_path):
            _hash_json(path, evidence)
    except Exception as exc:
        return {"passed": False, "errors": [str(exc)], "evidence": evidence}

    if epoch_order.get("preflight") is not False:
        _add_error(errors, "epoch-order preflight must be false for formal C15")
    if epoch_finished.get("preflight") is not False:
        _add_error(errors, "epoch-finished preflight must be false for formal C15")

    rows, steps, expected_global_steps = _derive_expected(
        config, epoch_order, epoch_finished
    )
    evidence["derived"] = {
        "train_rows": rows,
        "steps_per_epoch": steps,
        "expected_global_steps": expected_global_steps,
    }
    if rows != EXPECTED_TRAIN_ROWS:
        _add_error(
            errors, f"expected {EXPECTED_TRAIN_ROWS} train rows, observed {rows}"
        )
    if _get_nested(config, "train_dataset", "batch_size") != EXPECTED_BATCH_SIZE:
        _add_error(
            errors,
            f"expected train batch size {EXPECTED_BATCH_SIZE}, observed {_get_nested(config, 'train_dataset', 'batch_size')}",
        )
    if steps != math.ceil(EXPECTED_TRAIN_ROWS / EXPECTED_BATCH_SIZE):
        _add_error(errors, f"expected 135 completed steps, observed {steps}")
    if expected_global_steps != [19, 39, 59, 79, 99, 119, 134]:
        _add_error(errors, f"unexpected save cadence: {expected_global_steps}")

    experiment = config.get("experiment_name")
    trial = config.get("trial_name")
    fileroot = _get_nested(config, "cluster", "fileroot") or _get_nested(
        config, "saver", "fileroot"
    )
    if not all(
        isinstance(value, str) and value for value in (experiment, trial, fileroot)
    ):
        _add_error(errors, "config is missing experiment_name/trial_name/fileroot")
        checkpoint_root = None
    else:
        checkpoint_root = Path(fileroot) / "checkpoints" / "root" / experiment / trial
        evidence["checkpoint_root"] = str(checkpoint_root)

    if checkpoint_root is not None:
        role_evidence: dict[str, Any] = {}
        for role in ROLES:
            role_root = checkpoint_root / role
            role_report: dict[str, Any] = {"hf": [], "dcp": None}
            if not role_root.is_dir():
                _add_error(errors, f"missing role checkpoint root: {role_root}")
                role_evidence[role] = role_report
                continue
            expected_names = {
                f"epoch0epochstep{step}globalstep{step}"
                for step in expected_global_steps
            }
            observed_names = {
                path.name
                for path in role_root.glob("epoch*epochstep*globalstep*")
                if path.is_dir()
            }
            if observed_names != expected_names:
                _add_error(
                    errors,
                    f"{role} checkpoint directories differ from exact cadence: missing={sorted(expected_names - observed_names)}, unexpected={sorted(observed_names - expected_names)}",
                )
            for global_step in expected_global_steps:
                try:
                    role_report["hf"].append(
                        _verify_hf_checkpoint(role_root, global_step)
                    )
                except Exception as exc:
                    _add_error(errors, str(exc))
            try:
                role_report["dcp"] = _verify_dcp(role_root)
            except Exception as exc:
                _add_error(errors, str(exc))
            role_evidence[role] = role_report
        evidence["roles"] = role_evidence

        recover_dir = checkpoint_root / "recover_info"
        step_info_path = recover_dir / "step_info.json"
        saver_info_path = recover_dir / "saver_info.json"
        checkpoint_info_path = recover_dir / "checkpoint_info.json"
        for path in (step_info_path, saver_info_path, checkpoint_info_path):
            try:
                _hash_json(path, evidence)
            except Exception as exc:
                _add_error(errors, str(exc))
        if step_info_path.exists():
            step_info = _read_json(step_info_path)
            evidence["recover_step_info"] = step_info
            if (
                step_info.get("global_step") != 134
                or step_info.get("steps_per_epoch") != 135
            ):
                _add_error(
                    errors, f"recover_info is not at completed step 135: {step_info}"
                )
        else:
            _add_error(errors, f"missing recover step info: {step_info_path}")

    snapshot_evidence = {}
    for global_step in expected_global_steps:
        completed_step = global_step + 1
        try:
            snapshot = verify_snapshot(run_root, completed_step, verify_hashes=False)
            snapshot_evidence[str(completed_step)] = snapshot
            _hash_json(Path(snapshot["receipt_path"]), evidence)
        except Exception as exc:
            _add_error(errors, f"recovery snapshot step{completed_step}: {exc}")
    evidence["recovery_snapshots"] = snapshot_evidence

    return {"passed": not errors, "errors": errors, "evidence": evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="SAO run root to verify")
    args = parser.parse_args(argv)
    report = verify_run(args.run_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
