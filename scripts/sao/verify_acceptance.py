# SPDX-License-Identifier: Apache-2.0
"""Read-only SAO acceptance evidence verifier.

The verifier binds checklist items to real artifacts and emits JSON only.  It
does not update the workflow checklist and it does not allocate GPUs.  It is
intentionally a narrow item-evidence reader for the current prelaunch path, not
a replacement for the canonical workflow gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

PRELAUNCH_ITEMS = tuple(
    [*(f"C{i:02d}" for i in range(2, 9)), *(f"C{i:02d}" for i in range(16, 21))]
)


class Evidence:
    def __init__(self, artifact_root: Path):
        self.artifact_root = artifact_root
        self.items: dict[str, dict[str, Any]] = {}

    def path(self, rel: str | Path) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else self.artifact_root / path

    def bind(self, rel: str | Path, *, required: bool = True) -> dict[str, Any]:
        path = self.path(rel)
        key = str(path)
        if key in self.items:
            return self.items[key]
        item: dict[str, Any] = {
            "path": key,
            "exists": path.exists(),
            "required": required,
        }
        if path.exists() and path.is_file():
            item["size"] = path.stat().st_size
            item["sha256"] = sha256(path)
        self.items[key] = item
        return item

    def json(
        self, rel: str | Path, *, required: bool = True
    ) -> tuple[Any | None, dict[str, Any]]:
        item = self.bind(rel, required=required)
        if not item["exists"]:
            item["json_ok"] = False
            return None, item
        try:
            data = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - reported in JSON, not swallowed
            item["json_ok"] = False
            item["json_error"] = str(exc)
            return None, item
        item["json_ok"] = True
        return data, item


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def item_result(
    cid: str,
    requirement: str,
    status: str,
    *,
    reasons: list[str] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    observed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": cid,
        "requirement": requirement,
        "status": status,
        "passed": status == "passed",
        "reasons": reasons or [],
        "evidence": evidence or [],
        "observed": observed or {},
    }


def status_from_reasons(reasons: list[str], *, unavailable: bool = False) -> str:
    if unavailable:
        return "unavailable"
    return "failed" if reasons else "passed"


def evidence_subset(ev: Evidence, paths: list[str | Path]) -> list[dict[str, Any]]:
    return [ev.bind(path) for path in paths]


def checklist_by_id(record_dir: Path) -> dict[str, dict[str, Any]]:
    checklist_path = record_dir / "checklist.yaml"
    record = load_json(checklist_path)
    return {item["id"]: item for item in record.get("checklist", [])}


def check_c02(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "env/logs/uv-sync-final.log",
        "env/logs/uv-pip-check.log",
        "env/logs/dependency-overrides.json",
        "env/logs/kernel-parity-final-runtime.json",
        "env/logs/candidate-unit-tests-v2.xml",
    ]
    overrides, _ = ev.json(paths[2])
    kernel, _ = ev.json(paths[3])
    unit_item = ev.bind(paths[4])
    reasons = []
    for path in paths[:2]:
        if not ev.bind(path)["exists"]:
            reasons.append(f"missing {path}")
    if not isinstance(overrides, dict):
        reasons.append("dependency-overrides.json is missing or invalid")
    else:
        if not overrides.get("only_documented_release_overrides"):
            reasons.append("dependency overrides are not documented release overrides")
        if (
            overrides.get("runtime_compatibility")
            != "requires_separate_real_preflights"
        ):
            reasons.append("dependency runtime compatibility contract changed")
    if not _kernel_passed(
        kernel,
        {
            "fa2_vs_sdpa_fp32",
            "causal_conv1d_vs_torch_fp32",
            "fla_chunk_vs_transformers_reference",
        },
    ):
        reasons.append("kernel forward/backward parity evidence did not pass")
    junit = _junit_summary(Path(unit_item["path"])) if unit_item["exists"] else None
    if not junit:
        reasons.append("candidate-unit-tests-v2.xml is missing or invalid")
    elif junit["errors"] != 0 or junit["failures"] != 0 or junit["tests"] < 124:
        reasons.append("candidate unit tests did not record 124 passing CPU tests")
    return item_result(
        "C02",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
        observed={"candidate_unit_tests": junit},
    )


def _junit_summary(path: Path) -> dict[str, int] | None:
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError:
        return None
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        return None
    summary = {"tests": 0, "errors": 0, "failures": 0, "skipped": 0}
    for suite in suites:
        for key in summary:
            summary[key] += int(suite.attrib.get(key, 0))
    return summary


def check_c03(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "docker-cleanup/plan.json",
        "docker-cleanup/archive-verification.json",
        "docker-cleanup/receipt.json",
    ]
    plan, _ = ev.json(paths[0])
    archive, _ = ev.json(paths[1])
    receipt, _ = ev.json(paths[2])
    reasons = []
    observed: dict[str, Any] = {}
    if (
        not isinstance(plan, dict)
        or not plan.get("authorized_by")
        or not plan.get("targets")
    ):
        reasons.append("docker cleanup plan lacks authorization or exact targets")
    if not isinstance(archive, dict):
        reasons.append("archive verification is missing or invalid")
    elif archive.get("all_target_config_ids_match") is not True:
        reasons.append("archive verification did not bind all target config IDs")
    if not isinstance(receipt, dict):
        reasons.append("cleanup receipt is missing or invalid")
    else:
        observed = {
            "status": receipt.get("status"),
            "removed_images": receipt.get("removed_images"),
            "removed_tags": receipt.get("removed_tags"),
            "recoverable": receipt.get("recoverable"),
            "all_selected_tags_absent": receipt.get("all_selected_tags_absent"),
        }
        if receipt.get("status") != "completed":
            reasons.append("cleanup receipt status is not completed")
        if receipt.get("recoverable") is not True:
            reasons.append("cleanup receipt is not recoverable")
        if receipt.get("all_selected_tags_absent") is not True:
            reasons.append("selected tags are not absent after cleanup")
        if isinstance(archive, dict) and receipt.get("archive_sha256") != archive.get(
            "archive_sha256"
        ):
            reasons.append("receipt archive_sha256 does not match archive verification")
    return item_result(
        "C03",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
        observed=observed,
    )


def _kernel_passed(data: Any, expected_names: set[str]) -> bool:
    if not isinstance(data, dict) or data.get("passed") is not True:
        return False
    checks = data.get("checks")
    if not isinstance(checks, list):
        return False
    seen = {check.get("name") for check in checks if check.get("passed") is True}
    return expected_names <= seen


def check_c04(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "env/logs/resolved-config-check-v2.json",
        "preflight/logprob-parity.json",
        "sglang-preflight/runtime-v4.json",
    ]
    config, _ = ev.json(paths[0])
    parity, _ = ev.json(paths[1])
    runtime, _ = ev.json(paths[2])
    reasons = []
    observed: dict[str, Any] = {}
    if not isinstance(config, dict):
        reasons.append("resolved config is missing or invalid")
    else:
        actor_path = (config.get("actor") or {}).get("path")
        rollout_path = (config.get("sglang") or {}).get("model_path")
        tokenizer_path = config.get("tokenizer_path")
        observed.update(
            actor_path=actor_path,
            rollout_model_path=rollout_path,
            tokenizer_path=tokenizer_path,
        )
        if not actor_path or actor_path != rollout_path or actor_path != tokenizer_path:
            reasons.append(
                "actor, rollout, and tokenizer paths are not the same local Base"
            )
    if (
        not isinstance(parity, dict)
        or parity.get("passed") is not True
        or parity.get("world_size") != 4
    ):
        reasons.append("4-rank actor/rollout logprob parity evidence did not pass")
    if not isinstance(runtime, dict) or runtime.get("ok") is not True:
        reasons.append("SGLang runtime-v4 model load/generation evidence did not pass")
    return item_result(
        "C04",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
        observed=observed,
    )


def check_c05(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "preflight/critic-v2/rank0_summary.json",
        "preflight/critic-head/rank0_summary.json",
        "preflight/critic-v2/controller.log",
    ]
    critic, _ = ev.json(paths[0])
    head, _ = ev.json(paths[1])
    reasons = []
    for label, data in (("critic", critic), ("critic-head", head)):
        if not _profiled_update(data):
            reasons.append(f"{label} profiled update evidence missing or invalid")
    score = ((head or {}).get("gradients") or {}).get("score.weight") or {}
    score_fp = ((head or {}).get("parameter_fingerprints") or {}).get(
        "score.weight"
    ) or {}
    if not (
        score.get("present")
        and score.get("finite")
        and is_finite_positive(score.get("norm"))
    ):
        reasons.append("critic score.weight finite gradient evidence missing")
    if score_fp.get("changed") is not True:
        reasons.append("critic score.weight update evidence missing")
    log = ev.path(paths[2])
    try:
        text = log.read_text()
        result = json.loads(text[text.rindex("\n{") + 1 :])
        checkpoint = result["checkpoint_probe"]
        if result["status"] != "sao_fsdp_preflight_ok" or not (
            checkpoint.get("enabled") is True
            and checkpoint.get("fingerprints_match") is True
        ):
            reasons.append("critic DCP reload fingerprint check did not pass")
    except (OSError, ValueError, KeyError):
        reasons.append("critic DCP reload evidence missing or invalid")
    return item_result(
        "C05",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
    )


def _profiled_update(data: Any) -> bool:
    if not isinstance(data, dict) or data.get("status") != "profiled_measured_update":
        return False
    stats = data.get("stats") or {}
    if stats.get("update_successful") != 1.0:
        return False
    if not is_finite_positive(stats.get("grad_norm")):
        return False
    gradients = data.get("gradients") or {}
    return bool(gradients) and all(
        g.get("present") and g.get("finite") and is_finite_positive(g.get("norm"))
        for g in gradients.values()
    )


def check_c06(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "data/final-data-check.json",
        "data/ppo-math-v1/manifest.json",
        "data/ppo-math-v1/train/data-00000-of-00001.arrow",
        "data/ppo-math-v1/test/data-00000-of-00001.arrow",
    ]
    final_check, _ = ev.json(paths[0])
    manifest_item = ev.bind(paths[1])
    reasons = []
    observed: dict[str, Any] = {}
    if not isinstance(final_check, dict) or final_check.get("passed") is not True:
        reasons.append("final-data-check did not pass")
    else:
        observed = {
            "train_rows": final_check.get("train_rows"),
            "eval_counts": final_check.get("eval_counts"),
            "manifest_sha256": final_check.get("manifest_sha256"),
        }
        if final_check.get("train_rows") != 17157:
            reasons.append("train_rows is not 17157")
        if sum((final_check.get("eval_counts") or {}).values()) != 700:
            reasons.append("eval_counts do not sum to 700")
        expected_hash = final_check.get("manifest_sha256")
        if manifest_item.get("sha256") != expected_hash:
            reasons.append("manifest sha256 does not match final-data-check")
        for rel, expected in (final_check.get("output_hashes") or {}).items():
            output_path = Path("data/ppo-math-v1") / Path(rel)
            paths.append(output_path)
            bound = ev.bind(output_path)
            if bound.get("sha256") != expected:
                reasons.append(f"hash mismatch for data/ppo-math-v1/{rel}")
    return item_result(
        "C06",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
        observed=observed,
    )


def check_c07(ev: Evidence, req: str) -> dict[str, Any]:
    path = "env/logs/resolved-config-check-v2.json"
    config, _ = ev.json(path)
    reasons = []
    observed: dict[str, Any] = {}
    if not isinstance(config, dict):
        reasons.append("resolved config is missing or invalid")
    else:
        actor = config.get("actor") or {}
        critic = config.get("critic") or {}
        rollout = config.get("rollout") or {}
        gconfig = config.get("gconfig") or {}
        eval_gconfig = config.get("eval_gconfig") or {}
        observed = {
            "total_train_epochs": config.get("total_train_epochs"),
            "train_batch_size": (config.get("train_dataset") or {}).get("batch_size"),
            "rollout_backend": rollout.get("backend"),
            "actor_backend": actor.get("backend"),
            "critic_backend": critic.get("backend"),
            "gconfig": {
                "n_samples": gconfig.get("n_samples"),
                "max_new_tokens": gconfig.get("max_new_tokens"),
            },
        }
        expectations = [
            (actor.get("discount") == 1.0, "actor.discount is not 1"),
            (actor.get("gae_lambda") == 1.0, "actor.gae_lambda is not 1"),
            (
                actor.get("gae_timestep_unit") == "token",
                "actor.gae_timestep_unit is not token",
            ),
            (config.get("total_train_epochs") == 1, "total_train_epochs is not 1"),
            (
                (config.get("train_dataset") or {}).get("batch_size") == 128,
                "train batch_size is not 128",
            ),
            (
                (config.get("train_dataset") or {}).get("drop_last") is False,
                "train_dataset.drop_last is not false",
            ),
            (
                gconfig.get("n_samples") == 4 and eval_gconfig.get("n_samples") == 4,
                "n_samples is not 4 for train/eval",
            ),
            (
                gconfig.get("max_new_tokens") == 8192
                and eval_gconfig.get("max_new_tokens") == 8192,
                "max_new_tokens is not 8192 for train/eval",
            ),
            (
                actor.get("recompute_logprob") is False,
                "actor.recompute_logprob is not false",
            ),
            (actor.get("reward_norm") is None, "actor.reward_norm is not null"),
            (actor.get("eps_clip") == 0.2, "actor.eps_clip is not 0.2"),
            (actor.get("kl_ctl") == 0.0, "actor.kl_ctl is not 0"),
            (critic.get("is_critic") is True, "critic is not configured as critic"),
            (
                rollout.get("backend") == "sglang:d4p1t1",
                "rollout backend is not sglang:d4p1t1",
            ),
            (actor.get("backend") == "fsdp:d4p1t1", "actor backend is not fsdp:d4p1t1"),
        ]
        reasons.extend(reason for ok, reason in expectations if not ok)
    return item_result(
        "C07",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, [path]),
        observed=observed,
    )


def check_c08(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "env/logs/nccl-eight-gpu.json",
        "env/logs/resolved-config-check-v2.json",
    ]
    nccl, _ = ev.json(paths[0])
    config, _ = ev.json(paths[1])
    reasons = []
    observed: dict[str, Any] = {}
    if not isinstance(nccl, dict) or nccl.get("passed") is not True:
        reasons.append("8-rank NCCL evidence did not pass")
    else:
        ranks = nccl.get("ranks") or []
        observed["nccl_ranks"] = len(ranks)
        if len(ranks) != 8:
            reasons.append("NCCL rank count is not 8")
        if sorted(rank.get("rank") for rank in ranks) != list(range(8)):
            reasons.append("NCCL ranks are not 0..7")
    if isinstance(config, dict):
        observed["n_gpus_per_node"] = (config.get("cluster") or {}).get(
            "n_gpus_per_node"
        )
        if ((config.get("cluster") or {}).get("n_gpus_per_node")) != 8:
            reasons.append("resolved config n_gpus_per_node is not 8")
        if ((config.get("actor") or {}).get("backend")) != "fsdp:d4p1t1":
            reasons.append("actor backend does not bind 4-way data parallel FSDP")
        if ((config.get("rollout") or {}).get("backend")) != "sglang:d4p1t1":
            reasons.append("rollout backend does not bind 4-way data parallel SGLang")
    else:
        reasons.append("resolved config is missing or invalid")
    return item_result(
        "C08",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
        observed=observed,
    )


def check_c16(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "data/ppo-math-v1/scorer-differential-v2.json",
        "data/final-data-check.json",
    ]
    diff, _ = ev.json(paths[0])
    final_check, _ = ev.json(paths[1])
    reasons = []
    if not isinstance(diff, dict) or diff.get("passed") is not True:
        reasons.append("scorer differential evidence did not pass")
    elif diff.get("checked_cases", 0) < 20:
        reasons.append("scorer differential checked too few cases")
    if not isinstance(final_check, dict) or not final_check.get("scorer_sha256"):
        reasons.append("final data check lacks scorer hash binding")
    else:
        repo = Path(__file__).resolve().parents[2]
        for key, name in (
            ("scorer_sha256", "areal/reward/math_prd.py"),
            ("worker_sha256", "areal/reward/math_prd_worker.py"),
            ("lock_sha256", "uv.lock"),
        ):
            if final_check.get(key) != sha256(repo / name):
                reasons.append(f"final data check is stale for {name}")
    return item_result(
        "C16",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
    )


def check_c17(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "env/logs/loss-step-tests.xml",
        "env/logs/loss-two-rank.log",
        "preflight/actor/rank0_summary.json",
        "preflight/critic-v2/rank0_summary.json",
    ]
    reasons = []
    for path in paths[:2]:
        item = ev.bind(path)
        if not item["exists"] or item.get("size", 0) <= 0:
            reasons.append(f"missing {path}")
    for label, path in (("actor", paths[2]), ("critic", paths[3])):
        data, _ = ev.json(path)
        if not _profiled_update(data):
            reasons.append(f"{label} measured update evidence missing or invalid")
    termination_tests = "env/logs/termination-contract-tests.xml"
    paths.append(termination_tests)
    junit = (
        _junit_summary(ev.path(termination_tests))
        if ev.path(termination_tests).is_file()
        else None
    )
    if (
        not junit
        or junit["tests"] < 1
        or any(junit[key] for key in ("errors", "failures", "skipped"))
    ):
        reasons.append(
            "explicit termination and padding-invariance tests missing or failed"
        )
    native_root = Path("runs/native-preflight-06-termination/evidence")
    total_terminal = total_truncated = 0
    for step in range(1, 4):
        return_path = native_root / "returns" / f"{step}.json"
        step_path = native_root / "steps" / f"{step}.json"
        consumed_path = native_root / "consumed" / f"{step}.json"
        paths.extend([return_path, step_path, consumed_path])
        probe, probe_binding = ev.json(return_path)
        update, _ = ev.json(step_path)
        consumed, consumed_binding = ev.json(consumed_path)
        if (
            not isinstance(probe, dict)
            or not isinstance(update, dict)
            or not isinstance(consumed, list)
        ):
            reasons.append(f"native step{step} termination/return evidence missing")
            continue
        if (
            probe.get("passed") is not True
            or probe.get("step") != step
            or probe.get("samples") != len(consumed)
            or not is_finite_positive(probe.get("tokens"))
            or probe.get("consumed_sha256") != consumed_binding.get("sha256")
            or update.get("returns_audit_sha256") != probe_binding.get("sha256")
            or update.get("completed_step") != step
            or update.get("metrics", {}).get("ppo_actor/explicit_termination") != 1
        ):
            reasons.append(
                f"native step{step} termination/return evidence invalid or stale"
            )
        total_terminal += probe.get("terminated", 0)
        total_truncated += probe.get("truncated", 0)
    if not total_terminal or not total_truncated:
        reasons.append(
            "native return oracle must cover both real terminal and length-truncated responses"
        )
    return item_result(
        "C17",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
    )


def check_c18(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "sglang-preflight/runtime-v4.json",
        "sglang-preflight/kernel-paths-v4.json",
    ]
    runtime, _ = ev.json(paths[0])
    kernels, _ = ev.json(paths[1])
    reasons = []
    if not isinstance(runtime, dict) or runtime.get("ok") is not True:
        reasons.append("SGLang runtime-v4 did not pass")
    else:
        gen = runtime.get("generation") or {}
        if (
            not is_finite_positive(gen.get("elapsed_s"))
            or gen.get("completion_tokens", 0) <= 0
        ):
            reasons.append("SGLang measured generation timing/tokens missing")
    if not isinstance(kernels, dict) or not kernels.get("files"):
        reasons.append("SGLang profile trace paths missing")
    elif kernels.get("ok") is not True:
        reasons.append("SGLang kernel path scan did not pass")
    elif any(not Path(path).exists() for path in kernels.get("files", [])):
        reasons.append("one or more SGLang profile traces do not exist")
    return item_result(
        "C18",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
    )


def check_c19(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "preflight/actor/rank0_summary.json",
        "preflight/actor",
    ]
    actor, _ = ev.json(paths[0])
    reasons = []
    if not _profiled_update(actor):
        reasons.append("actor profiled measured update evidence missing or invalid")
    elif not is_finite_positive(actor.get("elapsed_step_s")):
        reasons.append("actor elapsed_step_s is missing or non-positive")
    if not ev.path(paths[1]).exists() or not list(
        ev.path(paths[1]).glob("*.trace.json")
    ):
        reasons.append("actor trace files missing")
    return item_result(
        "C19",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=[ev.bind(paths[0]), ev.bind(paths[1], required=True)],
    )


def check_c20(ev: Evidence, req: str) -> dict[str, Any]:
    paths = [
        "env/logs/kernel-parity-final-runtime.json",
        "preflight/actor/rank0_summary.json",
        "preflight/critic-v2/rank0_summary.json",
        "sglang-preflight/kernel-paths-v4.json",
    ]
    kernel, _ = ev.json(paths[0])
    actor, _ = ev.json(paths[1])
    critic, _ = ev.json(paths[2])
    sglang, _ = ev.json(paths[3])
    reasons = []
    if not _kernel_passed(
        kernel,
        {
            "fa2_vs_sdpa_fp32",
            "causal_conv1d_vs_torch_fp32",
            "fla_chunk_vs_transformers_reference",
        },
    ):
        reasons.append("kernel parity final runtime did not pass")
    for label, data in (("actor", actor), ("critic", critic)):
        terms = (data or {}).get("qwen35_evidence_terms") or {}
        if not (
            terms.get("causalconv")
            and terms.get("fla")
            and terms.get("fullattnbackward")
        ):
            reasons.append(
                f"{label} trace did not bind causalconv/fla/full attention terms"
            )
    matched = (sglang or {}).get("matched_paths") or {}
    for key in ("gdn_or_linear_attention", "causal_convolution", "full_attention"):
        if not matched.get(key):
            reasons.append(f"SGLang kernel trace lacks {key}")
    if isinstance(sglang, dict) and sglang.get("ok") is not True:
        reasons.append("SGLang kernel path scan did not pass")
    return item_result(
        "C20",
        req,
        status_from_reasons(reasons),
        reasons=reasons,
        evidence=evidence_subset(ev, paths),
    )


def verify(
    record_dir: Path, artifact_root: Path, item_ids: list[str]
) -> dict[str, Any]:
    by_id = checklist_by_id(record_dir)
    ev = Evidence(artifact_root)
    selected = item_ids
    req = {cid: by_id.get(cid, {}).get("requirement", "") for cid in selected}
    checks: list[dict[str, Any]] = []
    for cid in selected:
        if cid == "C02":
            checks.append(check_c02(ev, req[cid]))
        elif cid == "C03":
            checks.append(check_c03(ev, req[cid]))
        elif cid == "C04":
            checks.append(check_c04(ev, req[cid]))
        elif cid == "C05":
            checks.append(check_c05(ev, req[cid]))
        elif cid == "C06":
            checks.append(check_c06(ev, req[cid]))
        elif cid == "C07":
            checks.append(check_c07(ev, req[cid]))
        elif cid == "C08":
            checks.append(check_c08(ev, req[cid]))
        elif cid == "C16":
            checks.append(check_c16(ev, req[cid]))
        elif cid == "C17":
            checks.append(check_c17(ev, req[cid]))
        elif cid == "C18":
            checks.append(check_c18(ev, req[cid]))
        elif cid == "C19":
            checks.append(check_c19(ev, req[cid]))
        elif cid == "C20":
            checks.append(check_c20(ev, req[cid]))
        else:
            checks.append(
                item_result(
                    cid,
                    req.get(cid, ""),
                    "unavailable",
                    reasons=[f"{cid} has no verifier in this narrow prelaunch script"],
                )
            )
    for check in checks:
        state = by_id.get(check["id"], {})
        check["checklist_state"] = {
            "agent_status": state.get("agent_status"),
            "human_status": state.get("human_status"),
            "evidence_level": (state.get("evidence") or {}).get("level"),
        }
    status_counts: dict[str, int] = {}
    for check in checks:
        status_counts[check["status"]] = status_counts.get(check["status"], 0) + 1
    passed = all(check["status"] == "passed" for check in checks)
    items = {check["id"]: check for check in checks}
    return {
        "schema_version": 1,
        "passed": passed,
        "record_dir": str(record_dir),
        "artifact_root": str(artifact_root),
        "selected_items": selected,
        "limits": {
            "read_only": True,
            "mutates_checklist": False,
            "gpu_allocation": False,
            "dependency_install": False,
            "canonical_gate_replaced": False,
        },
        "status_counts": status_counts,
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--items",
        default=",".join(PRELAUNCH_ITEMS),
        help="Comma-separated checklist IDs. Default: C02-C08,C16-C20.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)

    item_ids = [item.strip() for item in args.items.split(",") if item.strip()]
    report = verify(args.record, args.artifact_root, item_ids)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
