# SPDX-License-Identifier: Apache-2.0
"""Offline evidence tests: no trainer imports, GPU calls, or external services."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.sao import audit_run as auditor
from scripts.sao.audit_run import (
    EXPECTED_EVAL_ROWS,
    aggregate_eval,
    audit_run,
    audit_source_key,
    hash_file,
    load_expected_sources,
    make_step5_supervision_record,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _sample(
    *,
    source_id="dapo:0",
    benchmark="dapo_math",
    task_id=1,
    sample_idx=0,
    version=0,
    reward=1.0,
    is_eval=False,
):
    return {
        "request_id": f"{source_id}-{task_id}-{sample_idx}-{version}",
        "source_id": source_id,
        "benchmark": benchmark,
        "task_id": task_id,
        "sample_idx": sample_idx,
        "is_eval": is_eval,
        "started_ns": 100,
        "generation_completed_ns": 180,
        "completed_ns": 200,
        "input_tokens": [1, 2],
        "output_tokens": [3, 4],
        "behavior_logprobs": [-0.1, -0.2],
        "behavior_versions": [version, version],
        "answer": "1",
        "completion": "\\boxed{1}",
        "parsed_answer": "1",
        "parse_status": "complete",
        "truncated": False,
        "stop_reason": "stop",
        "reward": reward,
    }


def _eval_records(versions=(0,), *, n_samples: int = 4):
    return [
        _sample(
            source_id=f"{benchmark}:{source}",
            benchmark=benchmark,
            task_id=source,
            sample_idx=index,
            version=version,
            is_eval=True,
        )
        for version in versions
        for benchmark, count in EXPECTED_EVAL_ROWS.items()
        for source in range(count)
        for index in range(n_samples)
    ]


def _step(step):
    return {
        "completed_step": step,
        "raw_global_step": step - 1,
        "published_version": step,
        "metrics": {
            "ppo_loss": -0.1,
            "critic/value_loss": 0.2,
            "ppo_actor/explicit_termination": 1,
            "ppo_actor/truncated_ratios/avg": 0.0,
            "ppo_actor/update/n_valid_tokens_in_loss": 2,
            "ppo_actor/update/n_valid_tokens_in_loss__count": 4,
        },
        "role_updates": {
            role: {
                "grad_norm": {"grad_norm": 0.5},
                "update_successful": {"update_successful": 1},
                "optimizer_steps": {"optimizer_steps_since_init": step},
            }
            for role in ("actor", "critic")
        },
        "update_spans": [
            {"role": "actor", "started_ns": 120, "completed_ns": 160},
            {"role": "critic", "started_ns": 170, "completed_ns": 190},
        ],
    }


def _metadata(record):
    return {
        "audit_source_key": audit_source_key(record["source_id"]),
        "audit_task_id": record["task_id"],
        "audit_sample_idx": record["sample_idx"],
        "terminated": not record["truncated"],
        "truncated": record["truncated"],
    }


def _training(evidence, count=5):
    """One prompt per step in a small CPU fixture; frozen config records that size."""
    groups = {
        step: [
            _sample(
                source_id=f"dapo:{step}",
                task_id=step,
                sample_idx=index,
                version=step - 1,
            )
            for index in range(4)
        ]
        for step in range(1, count + 1)
    }
    _write_json(
        evidence / "resolved-config.json",
        {"train_dataset": {"batch_size": 1}, "seed": 42},
    )
    order = [f"dapo:{step}" for step in range(1, count + 1)]
    _write_json(
        evidence / "epoch-order.json",
        {
            "source_ids": order,
            "seed": 42,
            "preflight": False,
            "dataloader_steps": count,
            "source_id_order_sha256": hashlib.sha256(
                ("\n".join(order) + "\n").encode()
            ).hexdigest(),
        },
    )
    _write_json(
        evidence / "epoch-finished.json",
        {
            "train_dataset_rows": count,
            "expected_steps": count,
            "preflight": False,
            "finished_ns": 300,
        },
    )
    for step, records in groups.items():
        _write_json(evidence / "steps" / f"{step}.json", _step(step))
        _write_json(
            evidence / "consumed" / f"{step}.json", [_metadata(r) for r in records]
        )
        return_path = evidence / "returns" / f"{step}.json"
        _write_json(
            return_path,
            {
                "passed": True,
                "step": step,
                "consumed_sha256": hash_file(evidence / "consumed" / f"{step}.json"),
            },
        )
        _write_json(
            evidence / "steps" / f"{step}.json",
            {**_step(step), "returns_audit_sha256": hash_file(return_path)},
        )
    _write_train(evidence, groups)
    return groups


def _write_train(evidence, groups):
    _write_jsonl(
        evidence / "samples" / "train-123.jsonl",
        [record for records in groups.values() for record in records],
    )


def _manifest(path, count):
    _write_json(
        path,
        {
            "source_ids": {
                "train": [f"dapo:{step}" for step in range(1, count + 1)],
                "test": [
                    f"{name}:{index}"
                    for name, n in EXPECTED_EVAL_ROWS.items()
                    for index in range(n)
                ],
            },
            "test_counts_by_benchmark": EXPECTED_EVAL_ROWS,
            "splits": {"train": count, "test": 700},
        },
    )


def _checks(report):
    return {check["name"]: check for check in report["checks"]}


def _external_review(evidence, step=5):
    """Synthetic attestation fixture; tests never claim to produce real GPU evidence."""
    path = evidence / "profiler" / "gpu-overlap.json"
    trace_path = path.parent / "test-trace.json"
    _write_json(trace_path, {"synthetic_fixture": True, "traceEvents": []})
    _write_json(
        path,
        {
            "externally_inspected": True,
            "actual_gpu_overlap": True,
            "inspected_by": "synthetic-test-reviewer",
            "trace_path": trace_path.name,
            "trace_sha256": hash_file(trace_path),
            "step_sha256": hash_file(evidence / "steps" / f"{step}.json"),
        },
    )
    return path


def test_eval_macro_distinct_from_weighted_and_version_groups():
    """All 700 sources use canonical manifest identifiers at two policy versions."""
    records = _eval_records((0, 20))
    for record in records:
        record["reward"] = float(record["benchmark"] in ("aime24", "amc23", "math500"))
        if record["behavior_versions"] == [0, 0]:
            record["reward"] = float(record["sample_idx"] == 0)
    expected = {record["source_id"] for record in records}
    report = aggregate_eval(records, expected_source_ids=expected)
    assert not report["sample_errors"]
    assert set(report["versions"]) == {"0", "20"}
    baseline, version = report["versions"]["0"], report["versions"]["20"]
    assert baseline["macro_mean@4"] == 0.25
    assert baseline["macro_pass@4"] == 1
    assert version["records"] == 2800
    assert version["per_set"]["beyond_aime"]["count"] == 100
    assert version["per_set"]["beyond_aime"]["pass@4"] == 0
    assert version["macro_mean@4"] == pytest.approx(0.6)
    assert version["macro_pass@4"] == pytest.approx(0.6)
    assert version["weighted_overall_mean@4"] == pytest.approx(570 / 700)
    assert version["weighted_overall_pass@4"] == pytest.approx(570 / 700)


def test_eval_aggregates_n2_with_dynamic_metric_keys():
    records = _eval_records((0, 20), n_samples=2)
    for record in records:
        record["reward"] = float(record["sample_idx"] == 0)
    expected = {record["source_id"] for record in records}
    report = aggregate_eval(records, expected_source_ids=expected, expected_n=2)

    assert not report["sample_errors"]
    version = report["versions"]["20"]
    assert version["records"] == 1400
    assert version["macro_mean@2"] == 0.5
    assert version["macro_pass@2"] == 1
    assert version["weighted_overall_mean@2"] == 0.5
    assert version["weighted_overall_pass@2"] == 1
    assert version["per_set"]["aime24"]["mean@2"] == 0.5
    assert "macro_mean@4" not in version


@pytest.mark.parametrize("expected_n", [0, -1, True])
def test_eval_rejects_non_positive_expected_n(expected_n):
    with pytest.raises(ValueError, match="expected_n must be a positive integer"):
        aggregate_eval(_eval_records(n_samples=2), expected_n=expected_n)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_count"])
def test_eval_n2_rejects_missing_duplicate_and_wrong_sample_count(mutation):
    records = _eval_records(n_samples=2)
    if mutation == "missing":
        records.pop(0)
    elif mutation == "duplicate":
        records[1] = dict(records[0])
    else:
        records = _eval_records(n_samples=4)

    report = aggregate_eval(records, expected_n=2)
    if mutation == "wrong_count":
        assert report["sample_errors"]
    else:
        assert report["versions"]["0"]["bad_sample_sets"]
        assert report["versions"]["0"]["macro_mean@2"] is None


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "cross_version", "mixed_tokens"]
)
def test_eval_rejects_missing_duplicate_and_mixed_samples(mutation):
    """A four-response budget cannot be satisfied by missing, repeated, or mixed versions."""
    records = _eval_records()
    if mutation == "missing":
        records.pop(0)
    elif mutation == "duplicate":
        records[1] = dict(records[0])
    elif mutation == "cross_version":
        records[0]["behavior_versions"] = [20, 20]
    else:
        records[0]["behavior_versions"] = [0, 20]
    report = aggregate_eval(records)
    assert report["versions"]["0"]["bad_sample_sets"]
    assert report["versions"]["0"]["macro_mean@4"] is None
    if mutation == "mixed_tokens":
        assert "mixed behavior_versions" in str(report["sample_errors"])


def test_eval_rejects_wrong_benchmark_spelling():
    """The manifest name is beyond_aime; misspellings must not become a sixth set."""
    record = _sample(source_id="beyondaime:0", benchmark="beyondaime", is_eval=True)
    assert aggregate_eval([record])["sample_errors"]


def test_eval_source_membership_cannot_be_replaced_by_matching_counts():
    """A different problem cannot replace an expected source while preserving 700 rows."""
    records = _eval_records()
    expected = {record["source_id"] for record in records}
    for record in records[:4]:
        record["source_id"] = "aime24:outside"
    version = aggregate_eval(records, expected_source_ids=expected)["versions"]["0"]
    assert version["missing_sources"] == ["aime24:0"]
    assert version["unexpected_sources"] == ["aime24:outside"]


def test_full_epoch_joins_all_samples_and_keeps_c09_unknown(tmp_path):
    """A complete tiny epoch passes JSON checks without pretending host spans prove C09."""
    _training(tmp_path, count=2)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, 2)
    _write_jsonl(tmp_path / "samples" / "eval-456.jsonl", _eval_records((0, 2)))
    report = audit_run(tmp_path, manifest_path=manifest)
    checks = _checks(report)
    assert report["passed_without_accelerator_timing"]
    assert checks["full_epoch_consumes_each_train_source_exactly_4"]["passed"]
    assert checks["full_epoch_eval_versions_exact"]["passed"]
    assert checks["host_rpc_request_update_overlap_observed"]["passed"]
    assert not checks["actual_gpu_overlap_c09"]["passed"]
    assert report["c09_status"] == "unknown_accelerator_timing_pending_profiler"
    assert not report["passed"]
    assert report["steps"]["1"]["counts"]["response_tokens"] == 8
    assert report["steps"]["1"]["counts"]["mask_evidence"].endswith("contract_only")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_finish",
        "preflight",
        "bad_order_digest",
        "wrong_seed",
        "duplicate_order",
    ],
)
def test_full_epoch_requires_finish_and_frozen_order(tmp_path, mutation):
    _training(tmp_path, count=2)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, 2)
    _write_jsonl(tmp_path / "samples" / "eval-456.jsonl", _eval_records((0, 2)))
    path = tmp_path / (
        "epoch-finished.json"
        if mutation in ("missing_finish", "preflight")
        else "epoch-order.json"
    )
    data = json.loads(path.read_text())
    if mutation == "missing_finish":
        path.unlink()
    else:
        if mutation == "preflight":
            data["preflight"] = True
        elif mutation == "bad_order_digest":
            data["source_id_order_sha256"] = "0" * 64
        elif mutation == "wrong_seed":
            data["seed"] = 0
        else:
            data["source_ids"] = ["dapo:1", "dapo:1"]
        _write_json(path, data)
    report = audit_run(tmp_path, manifest_path=manifest)
    assert not report["passed_without_accelerator_timing"]
    key = (
        "epoch_finished_matches_dataset_and_steps"
        if mutation in ("missing_finish", "preflight")
        else "epoch_order_is_seeded_dataset_permutation"
    )
    assert not _checks(report)[key]["passed"]


@pytest.mark.parametrize(
    "mutation", ["extra_task", "repeat_task", "duplicate_replaces_missing", "missing"]
)
def test_epoch_consumption_uses_exact_counters(tmp_path, mutation):
    """Sets of sample indices cannot hide repeated consumption, even across tasks."""
    groups = _training(tmp_path, count=1)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, 1)
    rows = [_metadata(record) for record in groups[1]]
    if mutation == "extra_task":
        extra = {**groups[1][0], "task_id": 99, "request_id": "another-task"}
        groups[1].append(extra)
        _write_train(tmp_path, groups)
        rows.append(_metadata(extra))
    elif mutation == "repeat_task":
        rows.append(dict(rows[0]))
    elif mutation == "duplicate_replaces_missing":
        rows[-1] = dict(rows[0])
    else:
        rows.pop()
    _write_json(tmp_path / "consumed" / "1.json", rows)
    checks = _checks(audit_run(tmp_path, manifest_path=manifest))
    coverage = checks["full_epoch_consumes_each_train_source_exactly_4"]
    assert not coverage["passed"]
    assert coverage["bad_pairs"][0]["count"] == len(rows)
    if mutation != "missing":
        assert not checks["consumed_source_sample_pairs_unique"]["passed"]
        assert coverage["bad_pairs"][0]["sample_idx_counts"][0] == 2
    assert checks["consumed_records_join_train_audit_records"]["passed"]


@pytest.mark.parametrize(
    "mutation", ["audit_record", "source_key", "consumed_file", "step_file"]
)
def test_missing_join_and_unpaired_step_files_fail(tmp_path, mutation):
    """The auditor fails closed on missing records and detached metadata files."""
    groups = _training(tmp_path, count=2)
    if mutation == "audit_record":
        groups[1].pop()
        _write_train(tmp_path, groups)
    elif mutation == "source_key":
        rows = [_metadata(record) for record in groups[1]]
        rows[0]["audit_source_key"] += 1
        _write_json(tmp_path / "consumed" / "1.json", rows)
    elif mutation == "consumed_file":
        (tmp_path / "consumed" / "1.json").unlink()
    else:
        (tmp_path / "steps" / "1.json").unlink()
    report = audit_run(tmp_path, mode="partial")
    assert not report["passed_without_accelerator_timing"]
    key = (
        "consumed_records_join_train_audit_records"
        if mutation in ("audit_record", "source_key")
        else "completed_steps_and_consumed_files_match"
    )
    assert not _checks(report)[key]["passed"]


def test_partial_prefix_allows_future_sources_and_eval_snapshots(tmp_path):
    """First-five auditing does not require full epoch coverage or future scheduled evals."""
    _training(tmp_path)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, 135)
    _write_jsonl(tmp_path / "samples" / "eval-456.jsonl", _eval_records())
    report = audit_run(tmp_path, manifest_path=manifest, mode="partial", through_step=5)
    assert report["passed_without_accelerator_timing"]
    assert "full_epoch_consumes_each_train_source_exactly_4" not in _checks(report)
    assert "full_epoch_eval_versions_exact" not in _checks(report)
    full = audit_run(tmp_path, manifest_path=manifest)
    assert not full["passed_without_accelerator_timing"]
    assert _checks(full)["full_epoch_eval_versions_exact"]["missing_versions"] == [
        20,
        40,
        60,
        80,
        100,
        120,
        135,
    ]


@pytest.mark.parametrize(
    "versions",
    [
        (0, 20, 40, 60, 80, 100, 120, 135),
        (0, 20, 40, 60, 80, 100, 120),
        (20, 40, 60, 80, 100, 120, 135),
        (0, 20, 40, 60, 80, 100, 120, 134, 135),
    ],
)
def test_full_eval_schedule_from_actual_manifest_count(tmp_path, versions):
    """17,157/128 rounds up to 135; baseline and the tail are independently required."""
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "splits": {"train": 17157, "test": 700},
            "test_counts_by_benchmark": EXPECTED_EVAL_ROWS,
        },
    )
    _write_jsonl(tmp_path / "samples" / "eval-456.jsonl", _eval_records(versions))
    report = audit_run(tmp_path, manifest_path=manifest)
    expected = [0, 20, 40, 60, 80, 100, 120, 135]
    assert report["schedule"]["expected_steps"] == 135
    assert report["schedule"]["expected_eval_versions"] == expected
    assert _checks(report)["full_epoch_eval_versions_exact"]["passed"] == (
        list(versions) == expected
    )
    # A count-only manifest proves the cadence but cannot prove train membership.
    assert not _checks(report)["full_epoch_consumes_each_train_source_exactly_4"][
        "passed"
    ]


def test_training_mixed_versions_counts_every_token(tmp_path):
    """Pause/resume can mix token policy versions within one training response."""
    groups = _training(tmp_path, count=2)
    for record in groups[2]:
        record["behavior_versions"] = [0, 1]
    _write_train(tmp_path, groups)
    report = audit_run(tmp_path, mode="partial")
    assert report["passed_without_accelerator_timing"]
    assert report["behavior_lag"]["unit"] == "response_token"
    assert report["behavior_lag"]["histogram"] == {0: 12, 1: 4}
    assert report["steps"]["2"]["behavior_lag"]["histogram"] == {0: 4, 1: 4}


def test_lag_above_admission_threshold_requires_explanation_not_join_failure(tmp_path):
    """Head admission limits do not impose a hard maximum on every response token."""
    groups = _training(tmp_path)
    groups[5][0]["behavior_versions"] = [0, 4]
    _write_train(tmp_path, groups)
    report = audit_run(tmp_path, mode="partial")
    checks = _checks(report)
    assert checks["train_sample_records_valid"]["passed"]
    assert checks["consumed_records_join_train_audit_records"]["passed"]
    assert checks["consumed_behavior_versions_not_future"]["passed"]
    assert checks["behavior_lag_outliers_explained"]["status"] == "pending_explanation"
    assert report["behavior_lag"]["over_threshold_tokens"] == 1
    assert not report["passed_without_accelerator_timing"]
    step_digest = hash_file(tmp_path / "steps" / "5.json")
    _write_json(
        tmp_path / "supervision" / "lag-explanations.json",
        {
            "5": {
                "step_sha256": step_digest,
                "explanation": "Inspected pause/resume: admitted before v1, resumed at v4.",
            },
        },
    )
    explained = audit_run(tmp_path, mode="partial")
    assert explained["passed_without_accelerator_timing"]
    assert explained["behavior_lag"]["over_threshold_tokens"] == 1
    assert hash_file(tmp_path / "steps" / "5.json") == step_digest
    _write_json(tmp_path / "steps" / "5.json", {**_step(5), "changed": True})
    stale = audit_run(tmp_path, mode="partial")
    assert not _checks(stale)["behavior_lag_outliers_explained"]["passed"]


def test_future_token_version_always_fails_even_with_explanation(tmp_path):
    """Training lag is relative to completed_step - 1, not published_version."""
    groups = _training(tmp_path, count=1)
    groups[1][0]["behavior_versions"] = [0, 1]
    _write_train(tmp_path, groups)
    _write_json(
        tmp_path / "steps" / "1.json",
        {**_step(1), "behavior_lag_explanation": "Cannot excuse a future version."},
    )
    report = audit_run(tmp_path, mode="partial")
    assert not _checks(report)["consumed_behavior_versions_not_future"]["passed"]
    assert report["behavior_lag"]["negative_tokens"] == 1


def test_counts_come_from_consumed_samples_not_metric_substrings_or_prefetch(tmp_path):
    """Parser, truncation and tokens are computed from joined completion records."""
    groups = _training(tmp_path, count=1)
    groups[1][1].update(
        reward=0, parsed_answer=None, parse_status="absent", completion="none"
    )
    groups[1][2].update(
        reward=0,
        parsed_answer=None,
        parse_status="malformed",
        completion="\\boxed{",
        truncated=True,
        stop_reason="length",
    )
    groups[1][3].update(reward=0, parsed_answer="2", completion="\\boxed{2}")
    prefetch = _sample(source_id="prefetch:0", task_id=99)
    prefetch.update(truncated=True, stop_reason="length")
    _write_train(tmp_path, {**groups, 99: [prefetch]})
    _write_json(
        tmp_path / "steps" / "1.json",
        {
            **_step(1),
            "metrics": {
                **_step(1)["metrics"],
                "ppo_actor/truncated_ratios/avg": 0.25,
                "some_parser_count": 999,
                "token_noise": -10,
            },
        },
    )
    _write_json(
        tmp_path / "consumed" / "1.json", [_metadata(record) for record in groups[1]]
    )
    report = audit_run(tmp_path, mode="partial")
    counts = report["steps"]["1"]["counts"]
    assert report["passed_without_accelerator_timing"]
    assert counts["samples"] == 4
    assert (counts["correct"], counts["incorrect"], counts["unparseable"]) == (1, 3, 2)
    assert (
        counts["parse_complete"],
        counts["parse_absent"],
        counts["parse_malformed"],
    ) == (2, 1, 1)
    assert counts["truncated"] == 1
    assert counts["response_tokens"] == 8
    assert counts["parser_unknown"] == counts["truncation_unknown"] == 0


def test_missing_parser_and_truncation_fields_remain_unknown(tmp_path):
    """Generic token/parser metric names cannot conceal missing per-sample evidence."""
    groups = _training(tmp_path, count=1)
    groups[1][0].pop("parsed_answer")
    groups[1][0].pop("truncated")
    _write_train(tmp_path, groups)
    report = audit_run(tmp_path, mode="partial")
    counts = report["steps"]["1"]["counts"]
    assert counts["parser_unknown"] == counts["truncation_unknown"] == 1
    assert not _checks(report)[
        "consumed_samples_supply_tokens_truncation_parser_counts"
    ]["passed"]


@pytest.mark.parametrize(
    "change",
    [
        {"reward": 0.5},
        {"output_tokens": []},
        {"behavior_logprobs": [0.1]},
        {"behavior_logprobs": [float("nan"), 0.1]},
        {"behavior_versions": [0, 0.9]},
        {"loss_mask": [1, 0, 1, 1]},
        {"response_mask": [1, 0]},
        {"stop_reason": "abort"},
        {"truncated": True, "stop_reason": "stop"},
        {"parsed_answer": None, "parse_status": "absent", "reward": 1},
    ],
)
def test_invalid_sample_contract_fails(tmp_path, change):
    """Masks, response arrays, reward, parser and generation-stop contradictions fail."""
    groups = _training(tmp_path, count=1)
    groups[1][0].update(change)
    _write_train(tmp_path, groups)
    report = audit_run(tmp_path, mode="partial")
    assert not _checks(report)["train_sample_records_valid"]["passed"]
    assert not report["passed_without_accelerator_timing"]


@pytest.mark.parametrize(
    "kind", ["reward_only", "rpc_gap", "generation", "no_generation_endpoint"]
)
def test_host_overlap_uses_generation_endpoint_and_individual_rpc_spans(tmp_path, kind):
    """Reward scoring time and gaps between updates are not generation overlap."""
    groups = _training(tmp_path, count=1)
    step = _step(1)
    for record in groups[1]:
        record.update(started_ns=100, generation_completed_ns=150, completed_ns=500)
        if kind == "rpc_gap":
            record.update(started_ns=210, generation_completed_ns=290)
        if kind == "no_generation_endpoint":
            record.pop("generation_completed_ns")
    step["update_spans"] = [
        {"role": "actor", "started_ns": 160, "completed_ns": 200},
        {"role": "critic", "started_ns": 300, "completed_ns": 350},
    ]
    if kind == "generation":
        step["update_spans"][0]["started_ns"] = 120
    _write_train(tmp_path, groups)
    _write_json(tmp_path / "steps" / "1.json", step)
    report = audit_run(tmp_path, mode="partial")
    checks = _checks(report)
    assert checks["host_rpc_request_update_overlap_observed"]["passed"] == (
        kind == "generation"
    )
    assert report["passed_without_accelerator_timing"]
    assert not checks["actual_gpu_overlap_c09"]["passed"]


def test_source_hash_matches_workflow_and_collision_is_reported(tmp_path, monkeypatch):
    """The source identity map uses the workflow's SHA256-derived signed-int64 key."""
    _training(tmp_path, count=2)
    source = "dapo:1"
    expected = int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    assert audit_source_key(source) == expected
    monkeypatch.setattr(auditor, "audit_source_key", lambda source_id: 7)
    report = audit_run(tmp_path, mode="partial")
    assert not _checks(report)["source_hash_mapping_collision_free"]["passed"]


def test_duplicate_sample_audit_record_across_workers_fails(tmp_path):
    """Two workers cannot silently overwrite the same task/sample audit identity."""
    groups = _training(tmp_path, count=1)
    _write_jsonl(tmp_path / "samples" / "train-999.jsonl", [groups[1][0]])
    report = audit_run(tmp_path, mode="partial")
    assert not _checks(report)["train_task_sample_ids_unique"]["passed"]
    assert not _checks(report)["consumed_records_join_train_audit_records"]["passed"]


def test_manifest_and_optional_hf_source_ids_must_agree(tmp_path, monkeypatch):
    """Optional HF import reads source IDs without importing the workflow runtime."""
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"source_ids": {"train": ["a"], "test": ["b"]}})
    fake = {"train": {"source_id": ["a"]}, "test": {"source_id": ["b"]}}
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_from_disk=lambda path: fake)
    )
    assert load_expected_sources(manifest_path=manifest, dataset_path=tmp_path) == {
        "train": {"a"},
        "test": {"b"},
    }
    fake["train"]["source_id"] = ["c"]
    with pytest.raises(ValueError, match="disagree"):
        load_expected_sources(manifest_path=manifest, dataset_path=tmp_path)
    fake["train"]["source_id"] = ["a", "a"]
    with pytest.raises(ValueError, match="duplicate"):
        load_expected_sources(dataset_path=tmp_path)


def test_step5_supervision_partial_audit_and_external_hashes(tmp_path):
    """Approval covers first-five evidence without demanding epoch coverage or future evals."""
    _training(tmp_path)
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, 135)
    _write_jsonl(tmp_path / "samples" / "eval-456.jsonl", _eval_records())
    gpu = _external_review(tmp_path)
    record = make_step5_supervision_record(
        tmp_path, gpu_overlap_path=gpu, manifest_path=manifest
    )
    assert record["passed"]
    assert record["step"] == 5
    assert record["audit_mode"] == "partial"
    assert record["step_sha256"] == hash_file(tmp_path / "steps" / "5.json")
    assert record["gpu_overlap_sha256"] == hash_file(gpu)
    assert record["c09_status"] == "verified"


def test_step5_accepts_gpu_witness_from_first_supervised_step(tmp_path):
    _training(tmp_path)
    gpu = _external_review(tmp_path, step=1)
    record = make_step5_supervision_record(tmp_path, gpu_overlap_path=gpu)
    assert record["passed"] and record["step"] == 5


@pytest.mark.parametrize("mutation", ["missing", "stale"])
def test_step5_requires_return_evidence_for_prior_steps(tmp_path, mutation):
    _training(tmp_path)
    gpu = _external_review(tmp_path)
    path = tmp_path / "returns" / "1.json"
    if mutation == "missing":
        path.unlink()
    else:
        _write_json(path, {"passed": True, "step": 1, "consumed_sha256": "wrong"})
    with pytest.raises(ValueError, match="return-oracle"):
        make_step5_supervision_record(tmp_path, gpu_overlap_path=gpu)


@pytest.mark.parametrize("mean", [None, 3])
def test_native_loss_mask_count_must_match_consumed_responses(tmp_path, mean):
    _training(tmp_path, count=1)
    step = _step(1)
    step["metrics"]["ppo_actor/update/n_valid_tokens_in_loss"] = mean
    _write_json(tmp_path / "steps/1.json", step)
    report = audit_run(tmp_path, mode="partial")
    assert not report["passed_without_accelerator_timing"]


@pytest.mark.parametrize(
    "mutation", ["flags_only", "not_inspected", "trace_changed", "step_changed"]
)
def test_step5_rejects_uninspected_unbound_or_changed_gpu_evidence(tmp_path, mutation):
    """Two booleans alone are insufficient; external inspection is bound to trace and step."""
    _training(tmp_path)
    gpu = _external_review(tmp_path)
    if mutation == "flags_only":
        _write_json(gpu, {"externally_inspected": True, "actual_gpu_overlap": True})
    elif mutation == "not_inspected":
        payload = json.loads(gpu.read_text())
        payload["externally_inspected"] = False
        _write_json(gpu, payload)
    elif mutation == "trace_changed":
        _write_json(gpu.parent / "test-trace.json", {"changed": True})
    else:
        _write_json(tmp_path / "steps" / "5.json", {**_step(5), "changed": True})
    with pytest.raises(ValueError, match="external GPU overlap"):
        make_step5_supervision_record(tmp_path, gpu_overlap_path=gpu)


def test_step5_rejects_broken_partial_evidence_even_with_gpu_review(tmp_path):
    """An externally inspected trace cannot excuse a duplicate consumed sample."""
    groups = _training(tmp_path)
    gpu = _external_review(tmp_path)
    rows = [_metadata(record) for record in groups[5]]
    rows.append(rows[0])
    _write_json(tmp_path / "consumed" / "5.json", rows)
    with pytest.raises(ValueError, match="consumed_source_sample_pairs_unique"):
        make_step5_supervision_record(tmp_path, gpu_overlap_path=gpu)


@pytest.mark.parametrize("target", ["step", "gpu"])
def test_supervision_rejects_artifacts_changed_during_audit(
    tmp_path, monkeypatch, target
):
    """Approval must not hash a later record than the one it actually audited."""
    _training(tmp_path)
    gpu = _external_review(tmp_path)
    original_audit = auditor.audit_run

    def changed_after_read(*args, **kwargs):
        report = original_audit(*args, **kwargs)
        path = tmp_path / "steps" / "5.json" if target == "step" else gpu
        _write_json(path, {**json.loads(path.read_text()), "changed": True})
        return report

    monkeypatch.setattr(auditor, "audit_run", changed_after_read)
    with pytest.raises(ValueError, match="changed during supervision audit"):
        make_step5_supervision_record(tmp_path, gpu_overlap_path=gpu)


def test_step1_supervision_hashes_prefix_without_gpu_or_future_steps(tmp_path):
    """Steps 1-4 can be reviewed before profiler evidence is available at step 5."""
    _training(tmp_path, count=1)
    record = make_step5_supervision_record(tmp_path, step=1)
    assert record["step_sha256"] == hash_file(tmp_path / "steps" / "1.json")
    assert record["c09_status"] == "unknown_accelerator_timing_pending_profiler"
    assert "gpu_overlap_sha256" not in record


def test_audit_cli_is_stdlib_only_and_returns_machine_report(tmp_path):
    """Direct script execution supports partial mode without resolving runtime dependencies."""
    _training(tmp_path, count=1)
    script = Path(auditor.__file__)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(script),
            "audit",
            str(tmp_path),
            "--mode",
            "partial",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed_without_accelerator_timing"]
    assert report["c09_status"] == "unknown_accelerator_timing_pending_profiler"
