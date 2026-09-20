import gzip
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "sao" / "verify_gpu_overlap.py"
SPEC = importlib.util.spec_from_file_location("verify_gpu_overlap", SCRIPT)
overlap = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(overlap)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _write_gz_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(payload, f)


def _step(root, start=1_000_000, end=2_000_000):
    _write_json(
        root / "evidence" / "steps" / "1.json",
        {
            "completed_step": 1,
            "update_spans": [
                {"role": "actor", "started_ns": start, "completed_ns": end},
                {"role": "critic", "started_ns": start, "completed_ns": end},
            ],
        },
    )


def _native(root, *, role="actor", start=1_100_000, dur_us=300):
    _write_jsonl(
        root
        / "logs"
        / "root"
        / "exp"
        / "trial"
        / "perf_tracer"
        / role
        / "traces-r0.jsonl",
        [
            {
                "ph": "X",
                "cat": "kernel",
                "name": f"{role}_kernel",
                "dur": dur_us,
                "args": {"wall_time_ns": start, "role": role, "device": 0},
            }
        ],
    )


def _rollout(root, *, base=1_000_000, ts_us=200, dur_us=200, cat="kernel"):
    _write_gz_json(
        root / "evidence" / "profiler" / "rollout-0-step-1" / "trace.json.gz",
        {
            "baseTimeNanoseconds": base,
            "deviceProperties": [{"id": 0, "name": "test-gpu"}],
            "traceEvents": [
                {
                    "ph": "X",
                    "cat": cat,
                    "name": "gen_kernel",
                    "ts": ts_us,
                    "dur": dur_us,
                    "args": {"device": 0, "stream": 7},
                }
            ],
        },
    )


def test_real_overlap_positive(tmp_path):
    _step(tmp_path)
    _native(tmp_path, role="actor", start=1_100_000, dur_us=500)
    _rollout(tmp_path, base=1_000_000, ts_us=200, dur_us=500)

    result = overlap.analyze(tmp_path, 1)

    assert result["passed"] is True
    assert result["overlap_duration_ns"] == 400_000
    assert result["witnesses"][0]["update"]["role"] == "actor"
    assert result["witnesses"][0]["generation"]["kernel"] == "gen_kernel"
    assert "externally_inspected" not in result


def test_disjoint_negative(tmp_path):
    _step(tmp_path)
    _native(tmp_path, role="critic", start=1_100_000, dur_us=100)
    _rollout(tmp_path, base=1_000_000, ts_us=500, dur_us=100)

    result = overlap.analyze(tmp_path, 1)

    assert result["passed"] is False
    assert result["overlap_duration_ns"] == 0
    assert result["witnesses"] == []


def test_snapshot_survives_later_cumulative_trace_writes(tmp_path):
    _step(tmp_path)
    _native(tmp_path)
    _rollout(tmp_path)
    overlap.snapshot_native_traces(tmp_path, 1)
    original = overlap.analyze(tmp_path, 1)
    _native(tmp_path, start=10_000_000)
    assert overlap.analyze(tmp_path, 1) == original
    with pytest.raises(FileExistsError):
        overlap.snapshot_native_traces(tmp_path, 1)


def test_cpu_event_only_reject(tmp_path):
    _step(tmp_path)
    _native(tmp_path, role="actor", start=1_100_000, dur_us=500)
    _rollout(tmp_path, base=1_000_000, ts_us=200, dur_us=500, cat="cpu_op")

    result = overlap.analyze(tmp_path, 1)

    assert result["passed"] is False
    assert result["overlap_duration_ns"] == 0
    assert result["witnesses"] == []


def test_malformed_missing_clock_reject(tmp_path):
    _step(tmp_path)
    _write_jsonl(
        tmp_path
        / "logs"
        / "root"
        / "exp"
        / "trial"
        / "perf_tracer"
        / "actor"
        / "traces-r0.jsonl",
        [{"ph": "X", "cat": "kernel", "name": "bad", "dur": 1, "args": {}}],
    )
    _rollout(tmp_path)

    try:
        overlap.analyze(tmp_path, 1)
    except ValueError as exc:
        assert "wall_time_ns" in str(exc)
    else:
        raise AssertionError("missing wall clock was accepted")
