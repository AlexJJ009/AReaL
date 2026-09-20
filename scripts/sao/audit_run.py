# SPDX-License-Identifier: Apache-2.0
"""Audit SAO/PPO run evidence without touching the training runtime.

The auditor reads only JSON/JSONL artifacts emitted by the run wrapper and
workflow.  It intentionally separates host-side request/update overlap from
externally inspected accelerator overlap, because C09 is not proven by HTTP/RPC
spans alone.

Use ``audit DIR --mode partial`` during first-five supervision and ``--mode full``
at epoch end. Full mode expects versions 0, 20, 40, ... and the final update,
computed from source IDs (or manifest splits.train) and resolved-config.json's
train_dataset.batch_size (default 128). Sample counts come from consumed records'
parsed_answer/truncated fields, not metric-name guesses. Missing fields stay unknown.

Lag > 2 is diagnostic, not a join failure. Supply an explanation before approval
in supervision/lag-explanations.json (or --lag-explanations), keyed by step with
``step_sha256`` and nonempty ``explanation`` fields. This preserves the immutable
step record already hashed by the runtime gate. Negative lag always fails.
An external GPU review JSON must contain externally_inspected=true,
actual_gpu_overlap=true, inspected_by, trace_path, trace_sha256 and step_sha256.
The trace is supplied and inspected externally; this tool only validates its
existence and hashes. Relative trace paths resolve beside the review JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_EVAL_ROWS = {
    "aime24": 30,
    "aime25": 30,
    "amc23": 40,
    "beyond_aime": 100,
    "math500": 500,
}
EXPECTED_EVAL_TOTAL = sum(EXPECTED_EVAL_ROWS.values())
EXPECTED_N = 4


def audit_source_key(source_id: str) -> int:
    """Mirror ``AuditedMathWorkflow``'s collision-sensitive source hash."""
    return int.from_bytes(
        hashlib.sha256(str(source_id).encode()).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_no} is not a JSON object")
                records.append(record)
    return records


def read_numbered_json(directory: Path) -> dict[int, Any]:
    if not directory.exists():
        return {}
    result = {}
    for path in sorted(directory.glob("*.json"), key=lambda p: int(p.stem)):
        result[int(path.stem)] = read_json(path)
    return result


def read_sample_records(
    samples_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    if not samples_dir.exists():
        return train, evals
    for path in sorted(samples_dir.glob("train-*.jsonl")):
        train.extend(read_jsonl(path))
    for path in sorted(samples_dir.glob("eval-*.jsonl")):
        evals.extend(read_jsonl(path))
    return train, evals


def load_manifest_sources(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {"train": set(), "test": set()}
    manifest = read_json(path)
    sources = {"train": set(), "test": set()}
    for split in sources:
        values = manifest.get(f"{split}_source_ids") or manifest.get(
            "source_ids", {}
        ).get(split)
        if values is not None:
            sources[split] = _unique_source_ids(values, split)
    return sources


def load_dataset_sources(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {"train": set(), "test": set()}
    try:
        from datasets import load_from_disk  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise RuntimeError(
            "Install datasets or pass a manifest with source IDs"
        ) from exc

    dataset = load_from_disk(str(path))
    return {
        split: _unique_source_ids(dataset[split]["source_id"], split)
        for split in ("train", "test")
    }


def _unique_source_ids(values: Any, split: str) -> set[str]:
    ids = [str(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate expected source IDs in {split}")
    return set(ids)


def load_expected_sources(
    *, manifest_path: Path | None = None, dataset_path: Path | None = None
) -> dict[str, set[str]]:
    sources = load_manifest_sources(manifest_path)
    dataset_sources = load_dataset_sources(dataset_path)
    for split, values in dataset_sources.items():
        if values:
            if sources[split] and sources[split] != values:
                raise ValueError(f"manifest and dataset disagree on {split} source IDs")
            sources[split] = values
    return sources


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, **details: Any
) -> None:
    checks.append({"name": name, "passed": bool(passed), "required": True, **details})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _single_behavior_version(record: dict[str, Any]) -> int:
    versions = record.get("behavior_versions")
    if not isinstance(versions, list) or not versions:
        raise ValueError(
            f"missing behavior_versions: request_id={record.get('request_id')}"
        )
    if not all(_is_int(version) and version >= 0 for version in versions):
        raise ValueError("behavior_versions must contain nonnegative integers")
    unique = set(versions)
    if len(unique) != 1:
        raise ValueError(
            f"mixed behavior_versions for request_id={record.get('request_id')}: {sorted(unique)}"
        )
    return unique.pop()


def _validate_sample_record(record: dict[str, Any], *, is_eval: bool) -> list[str]:
    errors: list[str] = []
    output_tokens = record.get("output_tokens")
    logprobs = record.get("behavior_logprobs")
    versions = record.get("behavior_versions")
    reward = record.get("reward")
    if not isinstance(output_tokens, list) or not output_tokens:
        errors.append("missing_or_empty_output_tokens")
    token_count = len(output_tokens) if isinstance(output_tokens, list) else 0
    for key in ("input_tokens", "output_tokens"):
        tokens = record.get(key)
        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(_is_int(token) and token >= 0 for token in tokens)
        ):
            errors.append(f"invalid_{key}")
    if not isinstance(logprobs, list) or len(logprobs) != token_count:
        errors.append("behavior_logprobs_length_mismatch")
    elif not all(_finite_number(value) for value in logprobs):
        errors.append("non_finite_behavior_logprobs")
    if not isinstance(versions, list) or len(versions) != token_count:
        errors.append("behavior_versions_length_mismatch")
    elif not all(_is_int(version) and version >= 0 for version in versions):
        errors.append("invalid_behavior_versions")
    elif is_eval:
        try:
            _single_behavior_version(record)
        except ValueError as exc:
            errors.append(str(exc))
    if not _finite_number(reward) or reward not in (0, 1):
        errors.append("non_binary_reward")
    for key in ("source_id", "task_id", "sample_idx"):
        if key not in record:
            errors.append(f"missing_{key}")
    if not isinstance(record.get("source_id"), str) or not record["source_id"]:
        errors.append("invalid_source_id")
    if not _is_int(record.get("task_id")) or record["task_id"] < 0:
        errors.append("invalid_task_id")
    if not _is_int(record.get("sample_idx")) or record["sample_idx"] not in range(
        EXPECTED_N
    ):
        errors.append("invalid_sample_idx")
    if record.get("is_eval") is not is_eval:
        errors.append("is_eval_mismatch")
    if record.get("stop_reason") == "abort":
        errors.append("aborted_generation")
    if "truncated" in record and not isinstance(record["truncated"], bool):
        errors.append("invalid_truncated")
    if (
        "truncated" in record
        and "stop_reason" in record
        and (record["truncated"] != (record["stop_reason"] == "length"))
    ):
        errors.append("truncated_stop_reason_mismatch")
    if "parse_status" in record:
        status = record["parse_status"]
        if status not in ("complete", "absent", "malformed"):
            errors.append("invalid_parse_status")
        if "parsed_answer" in record:
            answer = record["parsed_answer"]
            if (status == "complete" and not isinstance(answer, str)) or (
                status in ("absent", "malformed") and answer is not None
            ):
                errors.append("parsed_answer_status_mismatch")
            if reward == 1 and (not isinstance(answer, str) or not answer.strip()):
                errors.append("positive_reward_without_parsed_answer")
    # Native RLVR masks are prompt zeros followed by response ones. Check actual
    # masks when exported; absence must not be reported as observed tensor proof.
    if isinstance(record.get("input_tokens"), list):
        expected_mask = [0] * len(record["input_tokens"]) + [1] * token_count
        if "loss_mask" in record and record["loss_mask"] != expected_mask:
            errors.append("loss_mask_mismatch")
    if "response_mask" in record and record["response_mask"] != [1] * token_count:
        errors.append("response_mask_mismatch")
    if is_eval and record.get("benchmark") not in EXPECTED_EVAL_ROWS:
        errors.append("unknown_eval_benchmark")
    if is_eval and str(record.get("source_id", "")).split(":", 1)[0] != record.get(
        "benchmark"
    ):
        errors.append("eval_source_benchmark_mismatch")
    return errors


def aggregate_eval(
    eval_records: list[dict[str, Any]],
    *,
    expected_source_ids: set[str] | None = None,
    expected_n: int = EXPECTED_N,
) -> dict[str, Any]:
    """Return per-version N=4 mean/pass metrics with strict grouping."""
    if expected_n != EXPECTED_N:
        raise ValueError("this contract requires N=4")
    by_version: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sample_errors = []
    for index, record in enumerate(eval_records):
        errors = _validate_sample_record(record, is_eval=True)
        if errors:
            sample_errors.append({"index": index, "errors": errors})
            continue
        try:
            by_version[_single_behavior_version(record)].append(record)
        except ValueError as exc:
            sample_errors.append({"index": index, "errors": [str(exc)]})

    versions: dict[str, Any] = {}
    expected_sources = expected_source_ids or set()
    for version, records in sorted(by_version.items()):
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_source[str(record["source_id"])].append(record)

        missing_sources = (
            sorted(expected_sources - set(by_source)) if expected_sources else []
        )
        unexpected_sources = (
            sorted(set(by_source) - expected_sources) if expected_sources else []
        )
        duplicate_sources = []
        bad_sample_sets = []
        rewards_by_set: dict[str, list[float]] = defaultdict(list)
        pass_by_set: dict[str, list[float]] = defaultdict(list)
        per_source: dict[str, dict[str, Any]] = {}

        for source_id, source_records in sorted(by_source.items()):
            sample_indices = [int(record["sample_idx"]) for record in source_records]
            benchmark_values = {
                str(record.get("benchmark")) for record in source_records
            }
            rewards = [
                float(record["reward"])
                for record in sorted(source_records, key=lambda r: int(r["sample_idx"]))
            ]
            if len(benchmark_values) != 1:
                bad_sample_sets.append(
                    {"source_id": source_id, "reason": "mixed_benchmark"}
                )
                benchmark = next(iter(benchmark_values))
            else:
                benchmark = next(iter(benchmark_values))
            has_expected_samples = len(source_records) == expected_n and sorted(
                sample_indices
            ) == list(range(expected_n))
            if not has_expected_samples:
                bad_sample_sets.append(
                    {
                        "source_id": source_id,
                        "reason": "expected_exact_sample_idx_0_to_3",
                        "sample_idx": sorted(sample_indices),
                    }
                )
            if len(sample_indices) != len(set(sample_indices)):
                duplicate_sources.append(source_id)
            if has_expected_samples and len(benchmark_values) == 1:
                mean_score = sum(rewards) / expected_n
                pass_score = 1.0 if any(reward == 1.0 for reward in rewards) else 0.0
                rewards_by_set[benchmark].append(mean_score)
                pass_by_set[benchmark].append(pass_score)
                per_source[source_id] = {
                    "benchmark": benchmark,
                    "mean@4": mean_score,
                    "pass@4": pass_score,
                }

        per_set = {}
        for benchmark in sorted(set(rewards_by_set) | set(EXPECTED_EVAL_ROWS)):
            means = rewards_by_set.get(benchmark, [])
            passes = pass_by_set.get(benchmark, [])
            per_set[benchmark] = {
                "count": len(means),
                "expected_count": EXPECTED_EVAL_ROWS.get(benchmark),
                "mean@4": sum(means) / len(means) if means else None,
                "pass@4": sum(passes) / len(passes) if passes else None,
            }
        complete_sets = [
            item
            for name, item in per_set.items()
            if name in EXPECTED_EVAL_ROWS and item["count"] == item["expected_count"]
        ]
        versions[str(version)] = {
            "records": len(records),
            "unique_sources": len(by_source),
            "missing_sources": missing_sources,
            "unexpected_sources": unexpected_sources,
            "duplicate_sources": duplicate_sources,
            "bad_sample_sets": bad_sample_sets,
            "per_set": per_set,
            "macro_mean@4": (
                sum(item["mean@4"] for item in complete_sets) / len(EXPECTED_EVAL_ROWS)
                if len(complete_sets) == len(EXPECTED_EVAL_ROWS)
                else None
            ),
            "macro_pass@4": (
                sum(item["pass@4"] for item in complete_sets) / len(EXPECTED_EVAL_ROWS)
                if len(complete_sets) == len(EXPECTED_EVAL_ROWS)
                else None
            ),
            "weighted_overall_mean@4": (
                sum(
                    item["mean@4"] * item["count"]
                    for item in per_set.values()
                    if item["mean@4"] is not None
                )
                / sum(
                    item["count"]
                    for item in per_set.values()
                    if item["mean@4"] is not None
                )
                if any(item["mean@4"] is not None for item in per_set.values())
                else None
            ),
            "weighted_overall_pass@4": (
                sum(
                    item["pass@4"] * item["count"]
                    for item in per_set.values()
                    if item["pass@4"] is not None
                )
                / sum(
                    item["count"]
                    for item in per_set.values()
                    if item["pass@4"] is not None
                )
                if any(item["pass@4"] is not None for item in per_set.values())
                else None
            ),
            "per_source": per_source,
        }
    return {"sample_errors": sample_errors, "versions": versions}


def _sample_lookup(
    train_records: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    lookup = {}
    for record in train_records:
        key = (record["task_id"], record["sample_idx"])
        if key in lookup:
            raise ValueError(f"duplicate train audit record for task/sample {key}")
        lookup[key] = record
    return lookup


def _lag_summary(histogram: Counter[int]) -> dict[str, Any]:
    return {
        "count": sum(histogram.values()),
        "min": min(histogram, default=None),
        "max": max(histogram, default=None),
        "histogram": dict(sorted(histogram.items())),
    }


def _completion_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only successfully joined, actually consumed completions."""
    counts = Counter(
        dict.fromkeys(
            (
                "correct",
                "incorrect",
                "unparseable",
                "parser_unknown",
                "truncated",
                "truncation_unknown",
                "prompt_tokens",
                "response_tokens",
                "observed_masks",
                "parse_complete",
                "parse_absent",
                "parse_malformed",
            ),
            0,
        )
    )
    for record in records:
        counts["correct" if record["reward"] == 1 else "incorrect"] += 1
        if "parsed_answer" not in record or "parse_status" not in record:
            counts["parser_unknown"] += 1
        elif (
            record["parsed_answer"] is None or not str(record["parsed_answer"]).strip()
        ):
            counts["unparseable"] += 1
        if record.get("parse_status") in ("complete", "absent", "malformed"):
            counts["parse_" + record["parse_status"]] += 1
        if not isinstance(record.get("truncated"), bool):
            counts["truncation_unknown"] += 1
        else:
            counts["truncated"] += int(record["truncated"])
        counts["prompt_tokens"] += len(record["input_tokens"])
        counts["response_tokens"] += len(record["output_tokens"])
        counts["observed_masks"] += int(
            "loss_mask" in record or "response_mask" in record
        )
    return {
        "samples": len(records),
        **counts,
        "expected_response_mask_tokens": counts["response_tokens"],
        "mask_evidence": (
            "exported_masks"
            if records and counts["observed_masks"] == len(records)
            else "sample_lengths_and_native_single_turn_contract_only"
        ),
    }


def _epoch_schedule(
    evidence_dir: Path, sources: set[str], manifest_path: Path | None
) -> dict[str, Any]:
    config_path = evidence_dir / "resolved-config.json"
    config = read_json(config_path) if config_path.exists() else {}
    manifest = read_json(manifest_path) if manifest_path else {}
    declared_count = manifest.get("splits", {}).get("train")
    declared_sets = manifest.get("test_counts_by_benchmark")
    if declared_sets is not None and declared_sets != EXPECTED_EVAL_ROWS:
        raise ValueError(
            "manifest evaluation benchmarks disagree with testset700 contract"
        )
    count = len(sources) if sources else declared_count
    batch_size = config.get("train_dataset", {}).get("batch_size", 128)
    if not _is_int(batch_size) or batch_size <= 0:
        raise ValueError("train_dataset.batch_size must be a positive integer")
    if count is not None and (not _is_int(count) or count <= 0):
        raise ValueError("train source count must be a positive integer")
    if sources and declared_count is not None and declared_count != len(sources):
        raise ValueError("manifest train count disagrees with source IDs")
    expected_steps = math.ceil(count / batch_size) if count else None
    versions = (
        sorted({0, *range(20, expected_steps + 1, 20), expected_steps})
        if expected_steps
        else None
    )
    return {
        "train_sources": count,
        "prompt_batch_size": batch_size,
        "expected_steps": expected_steps,
        "expected_eval_versions": versions,
    }


def _step_update_errors(payload: dict[str, Any], step: int) -> list[str]:
    errors = []
    if payload.get("completed_step") != step:
        errors.append("completed_step_mismatch")
    if payload.get("published_version") != step:
        errors.append("published_version_mismatch")
    if payload.get("raw_global_step", step - 1) != step - 1:
        errors.append("raw_global_step_mismatch")
    for role in ("actor", "critic"):
        update = payload.get("role_updates", {}).get(role, {})
        for field in ("grad_norm", "update_successful", "optimizer_steps"):
            values = update.get(field)
            if not isinstance(values, dict) or not values:
                errors.append(f"missing_{role}_{field}")
                continue
            for value in values.values():
                expected = 1 if field == "update_successful" else step
                if not _finite_number(value) or (
                    value <= 0 if field == "grad_norm" else value != expected
                ):
                    errors.append(f"invalid_{role}_{field}")
    for name, value in (payload.get("metrics") or {}).items():
        if "loss" in name and not _finite_number(value):
            errors.append(f"nonfinite_loss:{name}")
    return errors


def _gpu_overlap_review(path: Path, step_paths: list[Path]) -> tuple[bool, list[str]]:
    """Validate external attestation; this function does not inspect GPU traces."""
    if not path.exists():
        return False, ["pending externally inspected profiler evidence"]
    review = read_json(path)
    errors = []
    for flag in ("externally_inspected", "actual_gpu_overlap"):
        if review.get(flag) is not True:
            errors.append(f"requires {flag}=true")
    if (
        not isinstance(review.get("inspected_by"), str)
        or not review["inspected_by"].strip()
    ):
        errors.append("missing inspected_by")
    trace_path = Path(review.get("trace_path") or "")
    if not trace_path.is_absolute():
        trace_path = path.parent / trace_path
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        errors.append("missing or empty trace_path")
    elif review.get("trace_sha256") != hash_file(trace_path):
        errors.append("trace_sha256 mismatch")
    if review.get("step_sha256") not in {
        hash_file(step_path) for step_path in step_paths
    }:
        errors.append("step_sha256 mismatch")
    return not errors, errors


def audit_run(
    evidence_dir: Path,
    *,
    manifest_path: Path | None = None,
    dataset_path: Path | None = None,
    max_allowed_lag: int = 2,
    mode: str = "full",
    through_step: int | None = None,
    gpu_overlap_path: Path | None = None,
    lag_explanations_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in ("partial", "full"):
        raise ValueError("mode must be partial or full")
    if through_step is not None and (
        mode != "partial" or not _is_int(through_step) or through_step < 1
    ):
        raise ValueError("through_step requires partial mode and a positive step")
    if not _is_int(max_allowed_lag) or max_allowed_lag < 0:
        raise ValueError("max_allowed_lag must be a nonnegative integer")
    expected_sources = load_expected_sources(
        manifest_path=manifest_path, dataset_path=dataset_path
    )
    steps = read_numbered_json(evidence_dir / "steps")
    consumed = read_numbered_json(evidence_dir / "consumed")
    if through_step is not None:
        steps = {step: value for step, value in steps.items() if step <= through_step}
        consumed = {
            step: value for step, value in consumed.items() if step <= through_step
        }
    train_records, eval_records = read_sample_records(evidence_dir / "samples")
    checks: list[dict[str, Any]] = []
    lag_explanations_path = (
        lag_explanations_path or evidence_dir / "supervision" / "lag-explanations.json"
    )
    lag_reviews = (
        read_json(lag_explanations_path) if lag_explanations_path.exists() else {}
    )
    schedule = _epoch_schedule(evidence_dir, expected_sources["train"], manifest_path)
    final_step = (
        schedule["expected_steps"]
        if mode == "full"
        else (through_step if through_step is not None else max(steps, default=0))
    )
    expected_steps = set(range(1, final_step + 1)) if final_step else set()
    _add_check(
        checks,
        "completed_steps_and_consumed_files_match",
        bool(steps) and set(steps) == set(consumed) == expected_steps,
        expected_steps=final_step,
        missing_steps=sorted(expected_steps - set(steps)),
        unexpected_steps=sorted(set(steps) - expected_steps),
        missing_consumed=sorted(set(steps) - set(consumed)),
        orphan_consumed=sorted(set(consumed) - set(steps)),
    )

    all_source_ids = (
        {
            str(record["source_id"])
            for record in train_records + eval_records
            if "source_id" in record
        }
        | expected_sources["train"]
        | expected_sources["test"]
    )
    source_key_map: dict[int, str] = {}
    collisions = []
    for source_id in sorted(all_source_ids):
        key = audit_source_key(source_id)
        if key in source_key_map and source_key_map[key] != source_id:
            collisions.append(
                {"key": key, "source_ids": [source_key_map[key], source_id]}
            )
        source_key_map[key] = source_id
    _add_check(
        checks,
        "source_hash_mapping_collision_free",
        not collisions,
        collisions=collisions,
    )

    train_errors, valid_records = [], []
    for index, record in enumerate(train_records):
        errors = _validate_sample_record(record, is_eval=False)
        if errors:
            train_errors.append({"index": index, "errors": errors})
        else:
            valid_records.append(record)
    _add_check(
        checks, "train_sample_records_valid", not train_errors, errors=train_errors[:20]
    )
    lookup_error = None
    try:
        train_lookup = _sample_lookup(valid_records)
    except ValueError as exc:
        train_lookup = {}
        lookup_error = str(exc)
    _add_check(
        checks, "train_task_sample_ids_unique", lookup_error is None, error=lookup_error
    )

    consumed_errors = []
    observed_epoch_pairs: dict[str, Counter[int]] = defaultdict(Counter)
    consumed_tasks: Counter[tuple[int, int]] = Counter()
    lag_histogram: Counter[int] = Counter()
    negative_lags, lag_diagnostics = [], []
    step_reports, step_errors = {}, []
    for step, rows in sorted(consumed.items()):
        if not isinstance(rows, list) or not rows:
            consumed_errors.append(
                {"step": step, "error": "missing_or_empty_consumed_list"}
            )
            continue
        payload = steps.get(step, {})
        completed_step = payload.get("completed_step", step)
        if not _is_int(completed_step):
            completed_step = step
        joined = []
        step_lags: Counter[int] = Counter()
        review = lag_reviews.get(str(step), {})
        explanation = review.get("explanation")
        explained = (
            isinstance(explanation, str)
            and bool(explanation.strip())
            and step in steps
            and review.get("step_sha256")
            == hash_file(evidence_dir / "steps" / f"{step}.json")
        )
        for row in rows:
            if not isinstance(row, dict) or not all(
                _is_int(row.get(key))
                for key in ("audit_task_id", "audit_sample_idx", "audit_source_key")
            ):
                consumed_errors.append(
                    {"step": step, "error": "invalid_consumed_metadata"}
                )
                continue
            task_id, sample_idx = row["audit_task_id"], row["audit_sample_idx"]
            identity = {"step": step, "task_id": task_id, "sample_idx": sample_idx}
            consumed_tasks[(task_id, sample_idx)] += 1
            record = train_lookup.get((task_id, sample_idx))
            if record is None:
                consumed_errors.append({**identity, "error": "missing_train_record"})
                continue
            source_id = record["source_id"]
            identity["source_id"] = source_id
            if row["audit_source_key"] != audit_source_key(source_id):
                consumed_errors.append({**identity, "error": "source_key_mismatch"})
                continue
            joined.append(record)
            observed_epoch_pairs[source_id][sample_idx] += 1
            token_lags = Counter(
                completed_step - 1 - v for v in record["behavior_versions"]
            )
            step_lags.update(token_lags)
            if min(token_lags) < 0:
                negative_lags.append({**identity, **_lag_summary(token_lags)})
            if max(token_lags) > max_allowed_lag:
                lag_diagnostics.append(
                    {
                        **identity,
                        **_lag_summary(token_lags),
                        "explained": explained,
                        "explanation": explanation if explained else None,
                    }
                )
        lag_histogram.update(step_lags)
        counts = _completion_counts(joined)
        step_reports[str(step)] = {
            "consumed_records": len(rows),
            "joined_records": len(joined),
            "counts": counts,
            "behavior_lag": _lag_summary(step_lags),
        }
        if counts["parser_unknown"] or counts["truncation_unknown"]:
            step_errors.append(
                {
                    "step": step,
                    "error": "unknown_completion_counts",
                    "parser_unknown": counts["parser_unknown"],
                    "truncation_unknown": counts["truncation_unknown"],
                }
            )
        if not joined or not counts["response_tokens"]:
            step_errors.append(
                {"step": step, "error": "missing_consumed_response_tokens"}
            )
    _add_check(
        checks,
        "consumed_records_join_train_audit_records",
        not consumed_errors,
        errors=consumed_errors[:50],
    )

    duplicates = [
        {"source_id": source, "sample_idx": index, "count": count}
        for source, indices in sorted(observed_epoch_pairs.items())
        for index, count in sorted(indices.items())
        if count != 1
    ]
    duplicate_tasks = [
        {"task_id": task, "sample_idx": index, "count": count}
        for (task, index), count in sorted(consumed_tasks.items())
        if count != 1
    ]
    unexpected_sources = sorted(set(observed_epoch_pairs) - expected_sources["train"])
    _add_check(
        checks,
        "consumed_source_sample_pairs_unique",
        not duplicates and not duplicate_tasks,
        duplicate_pairs=duplicates[:50],
        duplicate_tasks=duplicate_tasks[:50],
    )
    _add_check(
        checks,
        "consumed_sources_belong_to_dataset",
        bool(expected_sources["train"]) and not unexpected_sources,
        required=bool(expected_sources["train"]),
        expected_ids_available=bool(expected_sources["train"]),
        unexpected_sources=unexpected_sources[:50] if expected_sources["train"] else [],
    )
    _add_check(
        checks,
        "consumed_behavior_versions_not_future",
        not negative_lags,
        errors=negative_lags[:50],
    )
    unexplained = [item for item in lag_diagnostics if not item["explained"]]
    _add_check(
        checks,
        "behavior_lag_outliers_explained",
        not unexplained,
        status="pending_explanation" if unexplained else "passed",
        threshold=max_allowed_lag,
        unexplained_samples=len(unexplained),
        diagnostics=lag_diagnostics[:50],
    )
    if mode == "full":
        missing_sources = sorted(expected_sources["train"] - set(observed_epoch_pairs))
        bad_pairs = [
            {
                "source_id": source,
                "sample_idx_counts": dict(sorted(indices.items())),
                "count": sum(indices.values()),
            }
            for source, indices in sorted(observed_epoch_pairs.items())
            if indices != Counter(range(EXPECTED_N))
        ]
        _add_check(
            checks,
            "full_epoch_consumes_each_train_source_exactly_4",
            bool(expected_sources["train"])
            and not missing_sources
            and not unexpected_sources
            and not bad_pairs,
            missing_sources=missing_sources[:50],
            unexpected_sources=unexpected_sources[:50],
            bad_pairs=bad_pairs[:50],
            expected_ids_available=bool(expected_sources["train"]),
            reason=None
            if expected_sources["train"]
            else "pass dataset or manifest source IDs",
        )

    host_overlap_steps, host_spans_including_reward, update_errors = set(), set(), []
    for step, payload in sorted(steps.items()):
        errors = _step_update_errors(payload, step)
        if errors:
            update_errors.append({"step": step, "errors": errors})
        # Compare individual RPC spans: the gap between actor and critic is not overlap.
        for record in train_records:
            start = record.get("started_ns")
            end = record.get("generation_completed_ns")
            generation_endpoint = _is_int(end)
            if not generation_endpoint:
                end = record.get("completed_ns")
            for span in payload.get("update_spans") or []:
                train_start, train_end = (
                    span.get("started_ns"),
                    span.get("completed_ns"),
                )
                if all(_is_int(t) for t in (start, end, train_start, train_end)) and (
                    max(start, train_start) < min(end, train_end)
                ):
                    target = (
                        host_overlap_steps
                        if generation_endpoint
                        else host_spans_including_reward
                    )
                    target.add(step)
                    break
    _add_check(
        checks,
        "completed_joint_updates_and_publication",
        not update_errors,
        errors=update_errors[:50],
    )
    _add_check(
        checks,
        "consumed_samples_supply_tokens_truncation_parser_counts",
        not step_errors,
        errors=step_errors[:50],
    )
    _add_check(
        checks,
        "host_rpc_request_update_overlap_observed",
        bool(host_overlap_steps),
        evidence_level="host_rpc",
        steps=sorted(host_overlap_steps),
        required=False,
        request_spans_including_reward_steps=sorted(host_spans_including_reward),
    )

    gpu_overlap_path = (
        gpu_overlap_path or evidence_dir / "profiler" / "gpu-overlap.json"
    )
    gpu_overlap_passed, gpu_errors = _gpu_overlap_review(
        gpu_overlap_path, [evidence_dir / "steps" / f"{step}.json" for step in steps]
    )
    _add_check(
        checks,
        "actual_gpu_overlap_c09",
        gpu_overlap_passed,
        evidence_level="externally_inspected_profiler"
        if gpu_overlap_passed
        else "unknown",
        errors=gpu_errors,
        path=str(gpu_overlap_path) if gpu_overlap_path.exists() else None,
    )

    eval_report = aggregate_eval(
        eval_records, expected_source_ids=expected_sources["test"]
    )
    eval_errors = list(eval_report["sample_errors"])
    for version, report in eval_report["versions"].items():
        if report["records"] != EXPECTED_EVAL_TOTAL * EXPECTED_N:
            eval_errors.append(
                {
                    "version": version,
                    "error": "expected_2800_eval_records",
                    "records": report["records"],
                }
            )
        for name, count in EXPECTED_EVAL_ROWS.items():
            observed = report["per_set"].get(name, {}).get("count")
            if observed != count:
                eval_errors.append(
                    {
                        "version": version,
                        "benchmark": name,
                        "error": "wrong_eval_source_count",
                        "count": observed,
                    }
                )
        if any(
            report[key]
            for key in (
                "missing_sources",
                "unexpected_sources",
                "duplicate_sources",
                "bad_sample_sets",
            )
        ):
            eval_errors.append(
                {"version": version, "error": "bad_eval_source_or_sample_grouping"}
            )
    _add_check(
        checks, "eval_n4_grouping_and_metrics", not eval_errors, errors=eval_errors[:50]
    )
    if mode == "full":
        expected_versions = set(schedule["expected_eval_versions"] or [])
        actual_versions = {int(version) for version in eval_report["versions"]}
        _add_check(
            checks,
            "full_epoch_eval_versions_exact",
            bool(expected_versions) and actual_versions == expected_versions,
            expected_versions=sorted(expected_versions),
            missing_versions=sorted(expected_versions - actual_versions),
            unexpected_versions=sorted(actual_versions - expected_versions),
        )
        finished_path = evidence_dir / "epoch-finished.json"
        if finished_path.exists():
            finished = read_json(finished_path)
            _add_check(
                checks,
                "epoch_finished_matches_dataset_and_steps",
                finished.get("train_dataset_rows") == schedule["train_sources"]
                and finished.get("expected_steps") == schedule["expected_steps"],
                observed=finished,
            )

    passed = all(
        check["passed"]
        for check in checks
        if check["required"] and check["name"] != "actual_gpu_overlap_c09"
    )
    return {
        "schema_version": 2,
        "evidence_dir": str(evidence_dir),
        "mode": mode,
        "through_step": through_step,
        "schedule": schedule,
        "passed": passed and gpu_overlap_passed,
        "passed_without_accelerator_timing": passed,
        "c09_status": (
            "verified"
            if gpu_overlap_passed
            else "unknown_accelerator_timing_pending_profiler"
        ),
        "checks": checks,
        "steps": step_reports,
        "behavior_lag": {
            **_lag_summary(lag_histogram),
            "unit": "response_token",
            "threshold": max_allowed_lag,
            "negative_tokens": sum(n for lag, n in lag_histogram.items() if lag < 0),
            "over_threshold_tokens": sum(
                n for lag, n in lag_histogram.items() if lag > max_allowed_lag
            ),
            "unexplained_samples": len(unexplained),
        },
        "infrastructure_failures": {
            split: sum("error_type" in record for record in records)
            for split, records in (("train", train_records), ("eval", eval_records))
        },
        "eval": eval_report,
    }


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_step5_supervision_record(
    evidence_dir: Path,
    *,
    gpu_overlap_path: Path | None = None,
    step: int = 5,
    manifest_path: Path | None = None,
    dataset_path: Path | None = None,
    lag_explanations_path: Path | None = None,
) -> dict[str, Any]:
    """Approve one of the first five updates; step 5 also needs external GPU proof."""
    if not _is_int(step) or step not in range(1, 6):
        raise ValueError("supervision step must be between 1 and 5")
    step_path = evidence_dir / "steps" / f"{step}.json"
    if not step_path.exists():
        raise FileNotFoundError(f"missing step record: {step_path}")
    step_digest = hash_file(step_path)
    gpu_overlap_path = (
        gpu_overlap_path or evidence_dir / "profiler" / "gpu-overlap.json"
    )
    if step == 5:
        passed, errors = _gpu_overlap_review(gpu_overlap_path, [step_path])
        if not passed:
            raise ValueError(
                "step5 approval requires external GPU overlap: " + "; ".join(errors)
            )
        gpu_digest = hash_file(gpu_overlap_path)
    report = audit_run(
        evidence_dir,
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        mode="partial",
        through_step=step,
        gpu_overlap_path=gpu_overlap_path,
        lag_explanations_path=lag_explanations_path,
    )
    if not report["passed_without_accelerator_timing"]:
        failed = [
            check["name"]
            for check in report["checks"]
            if check["required"]
            and not check["passed"]
            and check["name"] != "actual_gpu_overlap_c09"
        ]
        raise ValueError("partial supervision audit failed: " + ", ".join(failed))
    if hash_file(step_path) != step_digest:
        raise ValueError("step record changed during supervision audit")
    if step == 5 and (
        report["c09_status"] != "verified" or hash_file(gpu_overlap_path) != gpu_digest
    ):
        raise ValueError("GPU overlap evidence changed during supervision audit")
    record = {
        "passed": True,
        "step": step,
        "step_sha256": step_digest,
        "audit_mode": "partial",
        "evidence_level": "real",
        "passed_checks": [
            check["name"] for check in report["checks"] if check["passed"]
        ],
        "c09_status": report["c09_status"],
    }
    if step == 5:
        record.update(
            {
                "gpu_overlap_sha256": gpu_digest,
                "gpu_overlap_path": str(gpu_overlap_path),
            }
        )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit", help="audit JSON evidence; full epoch by default"
    )
    audit.add_argument("evidence_dir", type=Path)
    audit.add_argument("--manifest", type=Path)
    audit.add_argument("--dataset", type=Path)
    audit.add_argument("--mode", choices=("partial", "full"), default="full")
    audit.add_argument("--through-step", type=int)
    audit.add_argument(
        "--max-allowed-lag",
        type=int,
        default=2,
        help="diagnostic threshold, not an unconditional token lag limit",
    )
    audit.add_argument("--gpu-overlap", type=Path)
    audit.add_argument("--lag-explanations", type=Path)
    audit.add_argument("--output", type=Path)

    approve = subparsers.add_parser(
        "approve-step5",
        aliases=["approve-step"],
        help="audit and write a first-five supervision approval",
    )
    approve.add_argument("evidence_dir", type=Path)
    approve.add_argument("--gpu-overlap", type=Path)
    approve.add_argument("--step", type=int, choices=range(1, 6), default=5)
    approve.add_argument("--manifest", type=Path)
    approve.add_argument("--dataset", type=Path)
    approve.add_argument("--lag-explanations", type=Path)
    approve.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        report = audit_run(
            args.evidence_dir,
            manifest_path=args.manifest,
            dataset_path=args.dataset,
            max_allowed_lag=args.max_allowed_lag,
            mode=args.mode,
            through_step=args.through_step,
            gpu_overlap_path=args.gpu_overlap,
            lag_explanations_path=args.lag_explanations,
        )
    else:
        report = make_step5_supervision_record(
            args.evidence_dir,
            gpu_overlap_path=args.gpu_overlap,
            step=args.step,
            manifest_path=args.manifest,
            dataset_path=args.dataset,
            lag_explanations_path=args.lag_explanations,
        )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.command == "audit" and not report["passed_without_accelerator_timing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
