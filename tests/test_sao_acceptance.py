# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.sao import verify_acceptance


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _checklist(record: Path) -> None:
    checklist = []
    for idx in range(1, 22):
        checklist.append(
            {
                "id": f"C{idx:02d}",
                "requirement": f"requirement C{idx:02d}",
                "evidence": {"level": "none", "paths": [], "anchors": []},
                "agent_status": "unverified",
                "human_status": "not_requested",
            }
        )
    _write_json(
        record / "checklist.yaml",
        {
            "schema_version": 1,
            "checklist": checklist,
            "formal_run": {
                "status": "awaiting_human",
                "candidate_sha": None,
                "command_digest": None,
                "config_digest": None,
                "human_confirmation": {"required": True, "confirmed_at": None},
            },
        },
    )


def _profile(role: str = "actor", *, score: bool = False) -> dict:
    gradients = {
        "model.language_model.embed_tokens.weight": {
            "present": True,
            "finite": True,
            "norm": 1.0,
        }
    }
    fingerprints = {
        "model.language_model.embed_tokens.weight": {
            "changed": True,
        }
    }
    if score:
        gradients["score.weight"] = {"present": True, "finite": True, "norm": 2.0}
        fingerprints["score.weight"] = {"changed": True}
    return {
        "status": "profiled_measured_update",
        "role": role,
        "elapsed_step_s": 1.25,
        "stats": {"grad_norm": 3.0, "update_successful": 1.0},
        "gradients": gradients,
        "parameter_fingerprints": fingerprints,
        "qwen35_evidence_terms": {
            "causalconv": True,
            "fla": True,
            "fullattnbackward": True,
        },
    }


def _kernel() -> dict:
    return {
        "passed": True,
        "checks": [
            {"name": "fa2_vs_sdpa_fp32", "passed": True},
            {"name": "causal_conv1d_vs_torch_fp32", "passed": True},
            {"name": "fla_chunk_vs_transformers_reference", "passed": True},
        ],
    }


def _populate_pre_artifacts(root: Path) -> None:
    model = "/models/Qwen3.5-4B-Base/snapshots/test"
    _write_text(
        root / "preflight/critic-v2/controller.log",
        "probe log\n"
        + json.dumps(
            {
                "status": "sao_fsdp_preflight_ok",
                "checkpoint_probe": {"enabled": True, "fingerprints_match": True},
            }
        )
        + "\n",
    )
    _write_text(root / "env/logs/uv-sync-final.log")
    _write_text(root / "env/logs/uv-pip-check.log", "Found 4 incompatibilities\n")
    _write_text(
        root / "env/logs/candidate-unit-tests-v2.xml",
        '<testsuites><testsuite tests="124" errors="0" failures="0" '
        'skipped="0" /></testsuites>\n',
    )
    _write_json(
        root / "env/logs/dependency-overrides.json",
        {
            "metadata_clean": False,
            "only_documented_release_overrides": True,
            "runtime_compatibility": "requires_separate_real_preflights",
        },
    )
    _write_json(root / "env/logs/kernel-parity-final-runtime.json", _kernel())
    _write_json(
        root / "docker-cleanup/plan.json",
        {"authorized_by": "user", "targets": [{"tag": "image:test"}]},
    )
    _write_json(
        root / "docker-cleanup/archive-verification.json",
        {
            "archive_sha256": "abc123",
            "all_target_config_ids_match": True,
        },
    )
    _write_json(
        root / "docker-cleanup/receipt.json",
        {
            "status": "completed",
            "removed_images": 1,
            "removed_tags": 1,
            "all_selected_tags_absent": True,
            "archive_sha256": "abc123",
            "recoverable": True,
        },
    )
    _write_json(
        root / "env/logs/resolved-config-check-v2.json",
        {
            "total_train_epochs": 1,
            "tokenizer_path": model,
            "cluster": {"n_gpus_per_node": 8},
            "train_dataset": {"batch_size": 128, "drop_last": False},
            "gconfig": {"n_samples": 4, "max_new_tokens": 8192},
            "eval_gconfig": {"n_samples": 4, "max_new_tokens": 8192},
            "rollout": {"backend": "sglang:d4p1t1"},
            "sglang": {"model_path": model},
            "actor": {
                "discount": 1.0,
                "gae_lambda": 1.0,
                "gae_timestep_unit": "token",
                "path": model,
                "backend": "fsdp:d4p1t1",
                "recompute_logprob": False,
                "reward_norm": None,
                "eps_clip": 0.2,
                "kl_ctl": 0.0,
            },
            "critic": {"path": model, "backend": "fsdp:d4p1t1", "is_critic": True},
        },
    )
    _write_json(
        root / "env/logs/nccl-eight-gpu.json",
        {
            "passed": True,
            "ranks": [
                {"rank": rank, "world_sum": 36.0, "subgroup_sum": 10.0}
                for rank in range(8)
            ],
        },
    )
    _write_json(
        root / "preflight/logprob-parity.json",
        {"passed": True, "world_size": 4},
    )
    _write_json(root / "preflight/critic-v2/rank0_summary.json", _profile("critic"))
    _write_json(
        root / "preflight/critic-head/rank0_summary.json",
        _profile("critic", score=True),
    )
    _write_json(root / "preflight/actor/rank0_summary.json", _profile("actor"))
    _write_text(root / "preflight/actor/test.trace.json", "{}\n")
    _write_json(
        root / "sglang-preflight/runtime-v4.json",
        {"ok": True, "generation": {"elapsed_s": 1.0, "completion_tokens": 8}},
    )
    trace = root / "sglang-preflight/profile.trace.json.gz"
    _write_text(trace, "trace\n")
    _write_json(
        root / "sglang-preflight/kernel-paths-v4.json",
        {
            "ok": True,
            "files": [str(trace)],
            "matched_paths": {
                "gdn_or_linear_attention": [str(trace)],
                "causal_convolution": [str(trace)],
                "full_attention": [str(trace)],
            },
        },
    )
    _write_text(root / "env/logs/loss-step-tests.xml", "<testsuite />\n")
    _write_text(root / "env/logs/loss-two-rank.log")
    _write_text(
        root / "env/logs/termination-contract-tests.xml",
        '<testsuite tests="21" errors="0" failures="0" skipped="0" />\n',
    )
    native = root / "runs/native-preflight-06-termination/evidence"
    for step in range(1, 4):
        consumed = native / "consumed" / f"{step}.json"
        _write_json(
            consumed,
            [
                {"terminated": True, "truncated": False},
                {"terminated": False, "truncated": True},
            ],
        )
        probe = native / "returns" / f"{step}.json"
        _write_json(
            probe,
            {
                "passed": True,
                "step": step,
                "samples": 2,
                "tokens": 8,
                "terminated": 1,
                "truncated": 1,
                "consumed_sha256": verify_acceptance.sha256(consumed),
            },
        )
        _write_json(
            native / "steps" / f"{step}.json",
            {
                "completed_step": step,
                "returns_audit_sha256": verify_acceptance.sha256(probe),
                "metrics": {"ppo_actor/explicit_termination": 1},
            },
        )

    data_root = root / "data/ppo-math-v1"
    _write_text(data_root / "train/data-00000-of-00001.arrow", "train")
    _write_text(data_root / "test/data-00000-of-00001.arrow", "test")
    _write_text(data_root / "dataset_dict.json", "{}\n")
    _write_text(data_root / "manifest.json", "{}\n")
    final_data_check = {
        "passed": True,
        "train_rows": 17157,
        "eval_counts": {
            "aime24": 30,
            "aime25": 30,
            "amc23": 40,
            "beyond_aime": 100,
            "math500": 500,
        },
        "manifest_sha256": verify_acceptance.sha256(data_root / "manifest.json"),
        **{
            key: verify_acceptance.sha256(
                Path(verify_acceptance.__file__).resolve().parents[2] / name
            )
            for key, name in (
                ("scorer_sha256", "areal/reward/math_prd.py"),
                ("worker_sha256", "areal/reward/math_prd_worker.py"),
                ("lock_sha256", "uv.lock"),
            )
        },
        "output_hashes": {
            "train/data-00000-of-00001.arrow": verify_acceptance.sha256(
                data_root / "train/data-00000-of-00001.arrow"
            ),
            "test/data-00000-of-00001.arrow": verify_acceptance.sha256(
                data_root / "test/data-00000-of-00001.arrow"
            ),
            "dataset_dict.json": verify_acceptance.sha256(
                data_root / "dataset_dict.json"
            ),
            "manifest.json": verify_acceptance.sha256(data_root / "manifest.json"),
        },
    }
    _write_json(root / "data/final-data-check.json", final_data_check)
    _write_json(
        data_root / "scorer-differential-v2.json",
        {"passed": True, "checked_cases": 24},
    )


def test_pre_phase_passes_with_bound_artifacts(tmp_path):
    record = tmp_path / "record"
    artifacts = tmp_path / "artifacts"
    _checklist(record)
    _populate_pre_artifacts(artifacts)

    report = verify_acceptance.verify(
        record,
        artifacts,
        list(verify_acceptance.PRELAUNCH_ITEMS),
    )

    assert report["passed"]
    assert report["status_counts"] == {"passed": 12}
    assert all(item["evidence"] for item in report["items"].values())
    assert report["items"]["C02"]["observed"]["candidate_unit_tests"]["tests"] == 124


def test_unsupported_postrun_item_is_explicitly_unavailable(tmp_path):
    record = tmp_path / "record"
    artifacts = tmp_path / "artifacts"
    _checklist(record)
    _populate_pre_artifacts(artifacts)

    report = verify_acceptance.verify(record, artifacts, ["C10"])

    assert not report["passed"]
    assert report["items"]["C10"]["status"] == "unavailable"


def test_hash_mismatch_fails_c06(tmp_path):
    record = tmp_path / "record"
    artifacts = tmp_path / "artifacts"
    _checklist(record)
    _populate_pre_artifacts(artifacts)
    _write_text(
        artifacts / "data/ppo-math-v1/train/data-00000-of-00001.arrow",
        "mutated",
    )

    report = verify_acceptance.verify(record, artifacts, ["C06"])
    c06 = report["items"]["C06"]

    assert c06["status"] == "failed"
    assert any("hash mismatch" in reason for reason in c06["reasons"])


def test_cli_emits_pure_json(tmp_path):
    record = tmp_path / "record"
    artifacts = tmp_path / "artifacts"
    _checklist(record)
    _populate_pre_artifacts(artifacts)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/sao/verify_acceptance.py",
            "--record",
            str(record),
            "--artifact-root",
            str(artifacts),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["items"]["C02"]["passed"] is True


def test_cli_requires_record_and_artifact_root():
    result = subprocess.run(
        [sys.executable, "scripts/sao/verify_acceptance.py"],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--record" in result.stderr
    assert "--artifact-root" in result.stderr
