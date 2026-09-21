# SPDX-License-Identifier: Apache-2.0
"""Freeze one complete SAO evaluation version into immutable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sao.audit_run import (  # noqa: E402
    EXPECTED_EVAL_ROWS,
    EXPECTED_EVAL_TOTAL,
    EXPECTED_N,
    aggregate_eval,
    hash_file,
    load_expected_sources,
)

ALLOWED_EVAL_VERSIONS = (20, 40, 60, 80, 100, 120, 135)
SUMMARY_NAME = "summary.json"
RAW_NAME_TEMPLATE = "eval-version{version}.jsonl"
ERROR_FIELDS = (
    "error_type",
    "infra_error",
    "infrastructure_error",
    "generation_error",
    "scorer_error",
    "reward_error",
    "exception",
    "traceback",
)


class SnapshotError(RuntimeError):
    pass


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    if not path.is_file():
        raise SnapshotError(f"not a regular file: {path}")
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "mode": stat.st_mode,
    }


def _source_binding(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise SnapshotError(f"partial trailing line in source eval JSONL: {path}")
    stat = _file_stat(path)
    if stat["bytes"] != len(data):
        raise SnapshotError(f"source size changed while reading: {path}")
    return {"path": str(path), "sha256": _sha256_bytes(data), "stat": stat}


def _assert_sources_unchanged(bindings: dict[str, dict[str, Any]]) -> None:
    for raw_path, binding in bindings.items():
        path = Path(raw_path)
        if _file_stat(path) != binding["stat"] or hash_file(path) != binding["sha256"]:
            raise SnapshotError(f"source eval JSONL changed during snapshot: {path}")


def _single_behavior_version(
    record: dict[str, Any], *, path: Path, line_no: int
) -> int:
    versions = record.get("behavior_versions")
    if not isinstance(versions, list) or not versions:
        raise SnapshotError(f"{path}:{line_no} missing behavior_versions")
    if not all(
        isinstance(version, int) and not isinstance(version, bool)
        for version in versions
    ):
        raise SnapshotError(f"{path}:{line_no} invalid behavior_versions")
    unique = set(versions)
    if len(unique) != 1:
        raise SnapshotError(
            f"{path}:{line_no} mixed behavior_versions: {sorted(unique)}"
        )
    return unique.pop()


def _has_runtime_or_scorer_error(record: dict[str, Any]) -> bool:
    return any(
        field in record and record[field] not in (None, False, "", [])
        for field in ERROR_FIELDS
    )


def _read_eval_sources(
    evidence_dir: Path, version: int
) -> tuple[list[dict[str, Any]], list[bytes], dict[str, dict[str, Any]]]:
    sample_dir = evidence_dir / "samples"
    paths = sorted(sample_dir.glob("eval-*.jsonl"))
    if not paths:
        raise SnapshotError(f"missing eval JSONL files under {sample_dir}")

    records: list[dict[str, Any]] = []
    raw_lines: list[bytes] = []
    bindings: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    for path in paths:
        bindings[str(path)] = _source_binding(path)
        with path.open("rb") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SnapshotError(
                        f"{path}:{line_no} invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise SnapshotError(f"{path}:{line_no} is not a JSON object")
                record_version = _single_behavior_version(
                    record, path=path, line_no=line_no
                )
                if record_version == version:
                    if _has_runtime_or_scorer_error(record):
                        errors.append(
                            {
                                "path": str(path),
                                "line": line_no,
                                "request_id": record.get("request_id"),
                                "error": "runtime_or_scorer_error",
                            }
                        )
                    records.append(record)
                    raw_lines.append(line)
    if errors:
        raise SnapshotError(
            "target version contains runtime/scorer errors: "
            + json.dumps(errors[:5], sort_keys=True)
        )
    return records, raw_lines, bindings


def _dataset_sources(dataset: Path) -> set[str]:
    if dataset.is_file():
        sources = load_expected_sources(manifest_path=dataset)
    else:
        sources = load_expected_sources(dataset_path=dataset)
    return sources["test"]


def _response_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    stop_reasons: Counter[str] = Counter()
    parse_status: Counter[str] = Counter()
    total_generated_tokens = 0
    starts: list[int] = []
    ends: list[int] = []
    for record in records:
        stop_reasons[str(record.get("stop_reason", "missing"))] += 1
        parse_status[str(record.get("parse_status", "missing"))] += 1
        output_tokens = record.get("output_tokens")
        if isinstance(output_tokens, list):
            total_generated_tokens += len(output_tokens)
        started_ns = record.get("started_ns")
        completed_ns = record.get("generation_completed_ns", record.get("completed_ns"))
        if isinstance(started_ns, int) and not isinstance(started_ns, bool):
            starts.append(started_ns)
        if isinstance(completed_ns, int) and not isinstance(completed_ns, bool):
            ends.append(completed_ns)
    span_s = (max(ends) - min(starts)) / 1_000_000_000 if starts and ends else None
    return {
        "response_count": len(records),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "parse_status": dict(sorted(parse_status.items())),
        "total_generated_tokens": total_generated_tokens,
        "elapsed_response_window_s": span_s,
        "response_span_ns": {
            "started_min": min(starts) if starts else None,
            "completed_max": max(ends) if ends else None,
        },
    }


def _validate_version(
    records: list[dict[str, Any]], *, expected_sources: set[str], version: int
) -> dict[str, Any]:
    report = aggregate_eval(records, expected_source_ids=expected_sources)
    errors: list[Any] = list(report["sample_errors"])
    version_report = report["versions"].get(str(version))
    if set(report["versions"]) - {str(version)}:
        errors.append(
            {
                "error": "unexpected_selected_versions",
                "versions": sorted(report["versions"]),
            }
        )
    if version_report is None:
        errors.append({"version": version, "error": "missing_target_version"})
    else:
        if version_report["records"] != EXPECTED_EVAL_TOTAL * EXPECTED_N:
            errors.append(
                {
                    "version": version,
                    "error": "expected_2800_eval_records",
                    "records": version_report["records"],
                }
            )
        if version_report["unique_sources"] != EXPECTED_EVAL_TOTAL:
            errors.append(
                {
                    "version": version,
                    "error": "expected_700_eval_sources",
                    "unique_sources": version_report["unique_sources"],
                }
            )
        for benchmark, expected_count in EXPECTED_EVAL_ROWS.items():
            observed = version_report["per_set"].get(benchmark, {}).get("count")
            if observed != expected_count:
                errors.append(
                    {
                        "version": version,
                        "benchmark": benchmark,
                        "error": "wrong_eval_source_count",
                        "count": observed,
                        "expected_count": expected_count,
                    }
                )
        for key in (
            "missing_sources",
            "unexpected_sources",
            "duplicate_sources",
            "bad_sample_sets",
        ):
            if version_report[key]:
                errors.append(
                    {
                        "version": version,
                        "error": key,
                        "details": version_report[key][:20],
                    }
                )
        if (
            version_report["macro_mean@4"] is None
            or version_report["macro_pass@4"] is None
        ):
            errors.append({"version": version, "error": "incomplete_macro_metrics"})
    if errors:
        raise SnapshotError(
            "eval snapshot validation failed: "
            + json.dumps(errors[:10], sort_keys=True)
        )
    return version_report


def _write_snapshot_records(path: Path, raw_lines: list[bytes]) -> None:
    with path.open("xb") as stream:
        for line in raw_lines:
            stream.write(line)


def _snapshot_file_bindings(snapshot_root: Path) -> dict[str, str]:
    return {
        path.name: hash_file(path) for path in sorted(snapshot_root.glob("*.jsonl"))
    }


def _verify_existing(
    target: Path,
    *,
    source_digest: str,
    version: int,
    version_report: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    summary_path = target / SUMMARY_NAME
    if not summary_path.is_file():
        raise SnapshotError(
            f"existing snapshot has no summary and will not be overwritten: {target}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise SnapshotError(
            f"existing snapshot summary is not an object: {summary_path}"
        )
    if summary.get("version") != version or summary.get("passed") is not True:
        raise SnapshotError(
            f"existing snapshot summary conflicts with requested version: {target}"
        )
    if summary.get("matching_records_sha256") != source_digest:
        raise SnapshotError(
            f"existing snapshot raw records differ from current source selection: {target}"
        )
    if summary.get("eval") != {str(version): version_report}:
        raise SnapshotError(f"existing snapshot summary eval metrics changed: {target}")
    for key, expected in stats.items():
        if summary.get(key) != expected:
            raise SnapshotError(
                f"existing snapshot summary response stats changed: {key}: {target}"
            )
    if summary.get("snapshot_files") != _snapshot_file_bindings(target):
        raise SnapshotError(f"existing snapshot file hashes changed: {target}")
    return {**summary, "status": "exists"}


def snapshot_eval(evidence_dir: Path, dataset: Path, version: int) -> dict[str, Any]:
    if version not in ALLOWED_EVAL_VERSIONS:
        raise SnapshotError(
            f"version must be one of {list(ALLOWED_EVAL_VERSIONS)}; baseline version0 is not mutated by this tool"
        )
    evidence_dir = evidence_dir.expanduser().resolve()
    dataset = dataset.expanduser().resolve()
    if not evidence_dir.is_dir():
        raise SnapshotError(f"evidence dir is not a directory: {evidence_dir}")
    expected_sources = _dataset_sources(dataset)
    if len(expected_sources) != EXPECTED_EVAL_TOTAL:
        raise SnapshotError(
            f"expected dataset test split to contain 700 sources, got {len(expected_sources)}"
        )

    records, raw_lines, source_bindings = _read_eval_sources(evidence_dir, version)
    matching_digest = hashlib.sha256(b"".join(raw_lines)).hexdigest()
    version_report = _validate_version(
        records, expected_sources=expected_sources, version=version
    )
    stats = _response_stats(records)

    snapshot_parent = evidence_dir / "eval-snapshots"
    target = snapshot_parent / f"version{version}"
    if target.exists():
        _assert_sources_unchanged(source_bindings)
        return _verify_existing(
            target,
            source_digest=matching_digest,
            version=version,
            version_report=version_report,
            stats=stats,
        )

    snapshot_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".staging-version{version}-", dir=snapshot_parent)
    )
    raw_path = staging / RAW_NAME_TEMPLATE.format(version=version)
    _write_snapshot_records(raw_path, raw_lines)
    summary = {
        "passed": True,
        "status": "created",
        "schema_version": 1,
        "evidence_dir": str(evidence_dir),
        "dataset": str(dataset),
        "version": version,
        "expected_sources": EXPECTED_EVAL_TOTAL,
        "expected_responses": EXPECTED_EVAL_TOTAL * EXPECTED_N,
        "matching_records_sha256": matching_digest,
        "eval": {str(version): version_report},
        **stats,
        "source_files": source_bindings,
        "snapshot_files": _snapshot_file_bindings(staging),
        "snapshot_root": str(target),
    }
    _write_json(staging / SUMMARY_NAME, summary)
    _assert_sources_unchanged(source_bindings)
    try:
        os.rename(staging, target)
    except FileExistsError as exc:
        raise SnapshotError(f"snapshot target appeared during copy: {target}") from exc
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--version", required=True, type=int)
    args = parser.parse_args(argv)
    try:
        report = snapshot_eval(args.evidence_dir, args.dataset, args.version)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
