# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for immutable SAO eval snapshots."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.sao import snapshot_eval
from scripts.sao.audit_run import EXPECTED_EVAL_ROWS
from scripts.sao.snapshot_eval import SnapshotError


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records, *, final_newline: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    if not final_newline:
        text = text.rstrip("\n")
    path.write_text(text, encoding="utf-8")


def _sample(*, benchmark: str, source: int, sample_idx: int, version: int = 20):
    return {
        "request_id": f"{benchmark}-{source}-{sample_idx}-v{version}",
        "source_id": f"{benchmark}:{source}",
        "benchmark": benchmark,
        "task_id": source,
        "sample_idx": sample_idx,
        "is_eval": True,
        "started_ns": 1_000_000_000 + source * 10 + sample_idx,
        "generation_completed_ns": 2_000_000_000 + source * 10 + sample_idx,
        "input_tokens": [1, 2],
        "output_tokens": [3, 4, 5],
        "behavior_logprobs": [-0.1, -0.2, -0.3],
        "behavior_versions": [version, version, version],
        "answer": "1",
        "completion": "\\boxed{1}",
        "parsed_answer": "1",
        "parse_status": "complete",
        "truncated": False,
        "stop_reason": "stop",
        "reward": float(sample_idx == 0),
    }


def _records(version: int = 20, *, n_samples: int = 4):
    return [
        _sample(
            benchmark=benchmark, source=source, sample_idx=sample_idx, version=version
        )
        for benchmark, count in EXPECTED_EVAL_ROWS.items()
        for source in range(count)
        for sample_idx in range(n_samples)
    ]


def _manifest(path: Path) -> None:
    _write_json(
        path,
        {
            "source_ids": {
                "train": ["dapo:0"],
                "test": [
                    f"{benchmark}:{source}"
                    for benchmark, count in EXPECTED_EVAL_ROWS.items()
                    for source in range(count)
                ],
            },
            "test_counts_by_benchmark": EXPECTED_EVAL_ROWS,
            "splits": {"train": 1, "test": 700},
        },
    )


def _fixture(tmp_path: Path, records=None) -> tuple[Path, Path]:
    evidence = tmp_path / "evidence"
    dataset = tmp_path / "manifest.json"
    _manifest(dataset)
    _write_jsonl(evidence / "samples" / "eval-1.jsonl", records or _records())
    return evidence, dataset


def test_snapshot_eval_writes_raw_records_summary_and_is_idempotent(tmp_path):
    evidence, dataset = _fixture(tmp_path)
    report = snapshot_eval.snapshot_eval(evidence, dataset, 20)
    target = evidence / "eval-snapshots" / "version20"
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["response_count"] == 2800
    assert report["eval"]["20"]["macro_mean@4"] == 0.25
    assert report["eval"]["20"]["macro_pass@4"] == 1.0
    assert (target / "eval-version20.jsonl").read_text(encoding="utf-8").count(
        "\n"
    ) == 2800
    assert summary["matching_records_sha256"] == report["matching_records_sha256"]

    _write_jsonl(evidence / "samples" / "eval-2.jsonl", _records(version=40))
    second = snapshot_eval.snapshot_eval(evidence, dataset, 20)
    assert second["status"] == "exists"
    assert second["matching_records_sha256"] == report["matching_records_sha256"]


def test_snapshot_eval_accepts_n2_and_uses_n2_metric_keys(tmp_path):
    evidence, dataset = _fixture(tmp_path, _records(n_samples=2))
    report = snapshot_eval.snapshot_eval(evidence, dataset, 20, n_samples=2)
    version = report["eval"]["20"]

    assert report["passed"] is True
    assert report["expected_n"] == 2
    assert report["expected_responses"] == 1400
    assert report["response_count"] == 1400
    assert version["macro_mean@2"] == 0.5
    assert version["macro_pass@2"] == 1.0
    assert "macro_mean@4" not in version


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "expected_1400_eval_records"),
        ("duplicate", "duplicate_sources|bad_sample_sets"),
        ("wrong_count", "invalid_sample_idx|expected_1400_eval_records"),
    ],
)
def test_snapshot_eval_n2_rejects_missing_duplicate_and_wrong_sample_count(
    tmp_path, mutation, message
):
    records = _records(n_samples=2)
    if mutation == "missing":
        records.pop(0)
    elif mutation == "duplicate":
        records[1] = dict(records[0])
    else:
        records = _records(n_samples=4)
    evidence, dataset = _fixture(tmp_path, records)

    with pytest.raises(SnapshotError, match=message):
        snapshot_eval.snapshot_eval(evidence, dataset, 20, n_samples=2)


def test_snapshot_eval_existing_summary_binds_n_samples(tmp_path):
    evidence, dataset = _fixture(tmp_path, _records(n_samples=2))
    snapshot_eval.snapshot_eval(evidence, dataset, 20, n_samples=2)
    summary_path = evidence / "eval-snapshots" / "version20" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["expected_n"] = 4
    _write_json(summary_path, summary)

    with pytest.raises(SnapshotError, match="sample count conflicts"):
        snapshot_eval.snapshot_eval(evidence, dataset, 20, n_samples=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("eval_metric", "summary eval metrics changed"),
        ("response_stats", "summary response stats changed"),
    ],
)
def test_snapshot_eval_rejects_tampered_existing_summary(tmp_path, mutation, message):
    evidence, dataset = _fixture(tmp_path)
    snapshot_eval.snapshot_eval(evidence, dataset, 20)
    summary_path = evidence / "eval-snapshots" / "version20" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "eval_metric":
        summary["eval"]["20"]["macro_mean@4"] = 0.0
    else:
        summary["response_count"] = 123
    _write_json(summary_path, summary)

    with pytest.raises(SnapshotError, match=message):
        snapshot_eval.snapshot_eval(evidence, dataset, 20)


def test_snapshot_eval_rejects_missing_sample(tmp_path):
    records = _records()
    records.pop(0)
    evidence, dataset = _fixture(tmp_path, records)
    with pytest.raises(SnapshotError, match="expected_2800_eval_records"):
        snapshot_eval.snapshot_eval(evidence, dataset, 20)


def test_snapshot_eval_rejects_duplicate_sample(tmp_path):
    records = _records()
    records[1] = dict(records[0])
    evidence, dataset = _fixture(tmp_path, records)
    with pytest.raises(SnapshotError, match="duplicate_sources|bad_sample_sets"):
        snapshot_eval.snapshot_eval(evidence, dataset, 20)


def test_snapshot_eval_rejects_wrong_version(tmp_path):
    records = _records()
    records[0]["behavior_versions"] = [20, 40, 20]
    evidence, dataset = _fixture(tmp_path, records)
    with pytest.raises(SnapshotError, match="mixed behavior_versions"):
        snapshot_eval.snapshot_eval(evidence, dataset, 20)


def test_snapshot_eval_rejects_partial_tail(tmp_path):
    evidence, dataset = _fixture(tmp_path)
    _write_jsonl(evidence / "samples" / "eval-1.jsonl", _records(), final_newline=False)
    with pytest.raises(SnapshotError, match="partial trailing line"):
        snapshot_eval.snapshot_eval(evidence, dataset, 20)


def test_snapshot_eval_rejects_source_mutation_during_snapshot(tmp_path, monkeypatch):
    evidence, dataset = _fixture(tmp_path)
    original = snapshot_eval._write_snapshot_records

    def mutating_write(path: Path, raw_lines: list[bytes]) -> None:
        original(path, raw_lines)
        with (evidence / "samples" / "eval-1.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    _sample(benchmark="aime24", source=0, sample_idx=0, version=40)
                )
                + "\n"
            )

    monkeypatch.setattr(snapshot_eval, "_write_snapshot_records", mutating_write)
    with pytest.raises(SnapshotError, match="changed during snapshot"):
        snapshot_eval.snapshot_eval(evidence, dataset, 20)
    assert not (evidence / "eval-snapshots" / "version20").exists()
    assert list((evidence / "eval-snapshots").glob(".staging-version20-*"))


def test_snapshot_eval_rejects_runtime_or_scorer_error(tmp_path):
    records = _records()
    records[0]["error_type"] = "scorer"
    evidence, dataset = _fixture(tmp_path, records)
    with pytest.raises(SnapshotError, match="runtime/scorer errors"):
        snapshot_eval.snapshot_eval(evidence, dataset, 20)


def test_snapshot_eval_direct_cli(tmp_path):
    evidence, dataset = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sao/snapshot_eval.py",
            "--evidence-dir",
            str(evidence),
            "--dataset",
            str(dataset),
            "--version",
            "20",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["response_count"] == 2800


@pytest.mark.parametrize("version", [0, 134, 536])
def test_grpo_schedule_snapshots_baseline_and_last_update(tmp_path, version):
    evidence, dataset = _fixture(tmp_path, _records(version=version))
    with pytest.raises(SnapshotError, match="version must"):
        snapshot_eval.snapshot_eval(evidence, dataset, version)
    result = snapshot_eval.snapshot_eval(
        evidence,
        dataset,
        version,
        allowed_versions=(0, 20, 40, 60, 80, 100, 120, 134, 536),
    )
    assert result["passed"] is True
    assert result["response_count"] == 2800
    assert result["eval"][str(version)]["macro_mean@4"] == 0.25
