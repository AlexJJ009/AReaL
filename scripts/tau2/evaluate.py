# SPDX-License-Identifier: Apache-2.0

"""Standalone official τ² repeated evaluation for one policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from examples.tau2.contracts import (
    OFFICIAL_TAU2_REVISION,
    SUPPORTED_DOMAINS,
    resolve_pinned_hf_snapshot,
)
from examples.tau2.train import get_tau2_dataset, validate_tau2_recipe
from examples.tau2.utils import Tau2PPOConfig

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import SGLangConfig, load_expr_config
from areal.engine import RemoteSGLangEngine
from areal.infra import LocalScheduler
from areal.infra.rpc.rtensor import RTensor
from areal.utils import logging
from areal.utils.printing import tabulate_stats

logger = logging.getLogger("Tau2Train")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Resolved training YAML.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF snapshot/local model path loaded by the single-GPU eval server.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--check-config", action="store_true")
    return parser.parse_args(argv)


def load_eval_config(args: argparse.Namespace) -> tuple[Tau2PPOConfig, str]:
    config, config_file = load_expr_config(["--config", args.config], Tau2PPOConfig)
    if config.eval_gconfig is not None and config.eval_gconfig.seed is not None:
        raise ValueError(
            "τ² repeated eval requires eval_gconfig.seed=null so trials do not "
            "collapse to the same fixed request seed"
        )
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    validate_tau2_recipe(config)
    return config, config_file


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        return str(path)
    if "@" not in model_path:
        return model_path
    return resolve_pinned_hf_snapshot(
        model_path,
        snapshot_resolver=lambda repo_id, revision: __import__(
            "huggingface_hub"
        ).snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        ),
    )


def bind_eval_config(
    config: Tau2PPOConfig, *, model_path: str, output_root: Path
) -> Tau2PPOConfig:
    config = deepcopy(config)
    output_root.mkdir(parents=True, exist_ok=True)
    config.cluster.fileroot = str(output_root)
    config.cluster.name_resolve.nfs_record_root = str(output_root / "name-resolve")
    config.experiment_name = f"{config.experiment_name}-offline-eval"
    config.trial_name = f"{config.trial_name}-offline-eval"
    config.actor.path = model_path
    config.tokenizer_path = model_path
    config.sglang.model_path = model_path
    config.vllm.model = model_path

    rollout = deepcopy(config.evaluation_rollout)
    rollout.experiment_name = config.experiment_name
    rollout.trial_name = config.trial_name
    rollout.fileroot = str(output_root)
    rollout.tokenizer_path = model_path
    rollout.max_head_offpolicyness = int(1e12)
    rollout.consumer_batch_size = 1
    config.rollout = rollout
    config.evaluation_rollout = rollout

    # Repeated offline eval is pass@1 over independent stochastic trials. Preserve
    # the resolved validation sampling config; trials provide the repetition axis.
    config.eval_gconfig.n_samples = 1
    config.eval_gconfig.reward_normalization = False
    config.eval_gconfig.drop_incomplete_group = False
    return config


def build_eval_rows(config: Tau2PPOConfig, repeats: int) -> list[dict[str, Any]]:
    rows = []
    for row in get_tau2_dataset(
        domain=SUPPORTED_DOMAINS,
        split="test",
        seed=config.seed,
        experiment_mode=config.experiment_mode,
    ):
        base = dict(row)
        for trial in range(repeats):
            item = dict(base)
            item["trial"] = trial
            item["source_id"] = f"{item['domain']}:{item['task_id']}:trial{trial}"
            item["attempt_id"] = item["source_id"]
            rows.append(item)
    expected = 100 * repeats
    if len(rows) != expected:
        raise ValueError(f"Official test coverage mismatch: {len(rows)} != {expected}")
    return rows


def build_workflow_kwargs(config: Tau2PPOConfig) -> dict[str, Any]:
    return {
        "econfig": asdict(config.econfig),
        "gen_args": {
            "top_p": config.eval_gconfig.top_p,
            "frequency_penalty": config.eval_gconfig.frequency_penalty,
            "seed": config.eval_gconfig.seed,
            "max_total_tokens": config.econfig.context_window_tokens,
            "temperature": config.eval_gconfig.temperature,
            "max_completion_tokens": config.eval_gconfig.max_new_tokens,
        },
        "timeout": config.episode_timeout_seconds,
        "infra_retries": config.infra_retries,
    }


def record_from_trajectory(
    row: dict[str, Any], trajectory: dict[str, Any]
) -> dict[str, Any]:
    if trajectory is None:
        raise RuntimeError(f"Missing trajectory for {row['source_id']}")
    localized = RTensor.localize(trajectory)
    rewards = localized.get("rewards")
    if rewards is None:
        raise RuntimeError(f"Missing rewards for {row['source_id']}")
    score = float(rewards.reshape(-1)[-1].item())
    official_scores = localized.get("official_scores")
    official_score = (
        float(official_scores.reshape(-1)[-1].item())
        if official_scores is not None
        else score
    )
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite reward for {row['source_id']}: {score}")
    if not math.isfinite(official_score) or official_score not in (0.0, 1.0):
        raise RuntimeError(
            f"Official τ² score must be binary for {row['source_id']}: {official_score}"
        )
    task_budget_failure = localized.get("task_budget_failure")
    return {
        "status": "completed",
        "domain": row["domain"],
        "task_id": row["task_id"],
        "split": row.get("split", "test"),
        "trial": int(row["trial"]),
        "source_id": row["source_id"],
        "attempt_id": row["attempt_id"],
        "reward": score,
        "official_score": official_score,
        "task_budget_failure": bool(task_budget_failure.reshape(-1)[-1].item())
        if task_budget_failure is not None
        else False,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def aggregate(
    records: list[dict[str, Any]], *, planned: int, repeats: int
) -> dict[str, Any]:
    completed = [r for r in records if r.get("status") == "completed"]
    failed = [r for r in records if r.get("status") != "completed"]
    scores = [float(r["official_score"]) for r in completed]
    metrics: dict[str, float | int] = {
        "eval/planned_episodes": planned,
        "eval/completed_episodes": len(completed),
        "eval/infra_failed_episodes": len(failed),
        "eval/pass1_mean": sum(scores) / len(scores) if scores else 0.0,
    }
    by_domain: dict[str, dict[str, float | int]] = {}
    by_trial: dict[str, dict[str, float | int]] = {}
    by_domain_trial: dict[str, dict[str, float | int]] = {}
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in completed:
        groups[("domain", record["domain"])].append(float(record["official_score"]))
        groups[("trial", str(record["trial"]))].append(float(record["official_score"]))
        groups[(f"domain_trial:{record['domain']}", str(record["trial"]))].append(
            float(record["official_score"])
        )
    for (kind, key), values in groups.items():
        target = (
            by_domain
            if kind == "domain"
            else by_trial
            if kind == "trial"
            else by_domain_trial
        )
        target_key = (
            key if not kind.startswith("domain_trial:") else f"{kind[13:]}:{key}"
        )
        target[target_key] = {
            "episodes": len(values),
            "pass1_mean": sum(values) / len(values),
        }
        metrics[f"eval/{kind}/{key}/pass1_mean"] = sum(values) / len(values)
        metrics[f"eval/{kind}/{key}/episodes"] = len(values)
    return {
        "status": "complete" if len(completed) == planned and not failed else "partial",
        "repeats": repeats,
        "planned_episodes": planned,
        "completed_episodes": len(completed),
        "infra_failed_episodes": len(failed),
        "metrics": metrics,
        "by_domain": by_domain,
        "by_trial": by_trial,
        "by_domain_trial": by_domain_trial,
    }


def assert_fresh_output(output_root: Path, *, check_config: bool) -> None:
    protected = ["episodes.jsonl", "summary.json"]
    if not check_config:
        protected.extend(["manifest.json", "resolved_config.json"])
    existing = [
        str(output_root / name) for name in protected if (output_root / name).exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing τ² eval output: " + ", ".join(existing)
        )


def wait_for_submitted(
    controller: Any,
    submitted: list[tuple[int, dict[str, Any]]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    expected = {task_id: row for task_id, row in submitted}
    if len(expected) != len(submitted):
        raise RuntimeError("Duplicate rollout task IDs returned by controller.submit")
    raw_results = controller.dispatcher.wait_results(len(submitted), timeout=None)
    if len(raw_results) != len(submitted):
        raise RuntimeError(
            f"Evaluation coverage mismatch: {len(raw_results)} != {len(submitted)}"
        )
    mapped: dict[int, dict[str, Any] | None] = {}
    for result in raw_results:
        if result is None:
            raise RuntimeError("Rollout dispatcher returned an unbound failed result")
        task_id = result.task_id
        if task_id not in expected:
            raise RuntimeError(f"Unexpected rollout task ID: {task_id}")
        if task_id in mapped:
            raise RuntimeError(f"Duplicate rollout task result: {task_id}")
        mapped[task_id] = result.trajectory
    missing = sorted(set(expected) - set(mapped))
    if missing:
        raise RuntimeError(f"Missing rollout task results: {missing}")
    return [(row, mapped[task_id]) for task_id, row in submitted]


def run(argv: list[str]) -> dict[str, Any]:
    args = parse_args(argv)
    raw_config, config_file = load_eval_config(args)
    output_root = Path(args.output_root).absolute()
    assert_fresh_output(output_root, check_config=args.check_config)
    model_path = resolve_model_path(args.model_path)
    config = bind_eval_config(
        raw_config, model_path=model_path, output_root=output_root
    )
    rows = build_eval_rows(config, args.repeats)
    workflow_kwargs = build_workflow_kwargs(config)
    manifest = {
        "schema_version": 1,
        "config_file": config_file,
        "tau2_revision": OFFICIAL_TAU2_REVISION,
        "model_path": model_path,
        "output_root": str(output_root),
        "repeats": args.repeats,
        "planned_episodes": len(rows),
        "global_seed": config.seed,
        "request_seed": config.eval_gconfig.seed,
        "temperature": config.eval_gconfig.temperature,
        "top_p": config.eval_gconfig.top_p,
        "max_new_tokens": config.eval_gconfig.max_new_tokens,
        "max_total_tokens": config.econfig.context_window_tokens,
        "enable_thinking": config.econfig.enable_thinking,
        "add_thinking_tool": config.econfig.add_thinking_tool,
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(
        output_root / "resolved_config.json",
        OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
    )
    if args.check_config:
        summary = aggregate([], planned=len(rows), repeats=args.repeats)
        summary["status"] = "check_config"
        write_json(output_root / "summary.json", summary)
        return {"manifest": manifest, "summary": summary}

    logging.setup_file_logging(str(output_root / "eval.log"))
    scheduler = LocalScheduler(exp_config=config)
    alloc = ModelAllocation.from_str(
        config.evaluation_rollout.backend, name="eval-rollout"
    )
    server_args = SGLangConfig.build_args(
        sglang_config=config.sglang,
        tp_size=alloc.parallel.tp_size,
        pp_size=alloc.parallel.pp_size,
        base_gpu_id=0,
    )
    controller = RemoteSGLangEngine.as_controller(config.evaluation_rollout, scheduler)
    records_path = output_root / "episodes.jsonl"
    records: list[dict[str, Any]] = []
    try:
        controller.initialize(role="eval-rollout", server_args=server_args)
        controller.start_proxy()
        for trial in range(args.repeats):
            submitted = []
            for row in [item for item in rows if item["trial"] == trial]:
                task_id = controller.submit(
                    row,
                    workflow="examples.tau2.agent.Tau2AgentWorkflow",
                    workflow_kwargs=workflow_kwargs,
                    group_size=1,
                    is_eval=True,
                    reward_normalization=False,
                    drop_incomplete_group=False,
                )
                submitted.append((task_id, row))
            for row, trajectory in wait_for_submitted(controller, submitted):
                try:
                    record = record_from_trajectory(row, trajectory)
                except Exception as exc:
                    record = {
                        "status": "infra_failed",
                        "domain": row["domain"],
                        "task_id": row["task_id"],
                        "trial": int(row["trial"]),
                        "source_id": row["source_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    write_record(records_path, record)
                    records.append(record)
                    write_json(
                        output_root / "summary.json",
                        aggregate(records, planned=len(rows), repeats=args.repeats),
                    )
                    raise
                write_record(records_path, record)
                records.append(record)
    finally:
        controller.destroy()
    summary = aggregate(records, planned=len(rows), repeats=args.repeats)
    write_json(output_root / "summary.json", summary)
    logger.info("Evaluation Results: %s", tabulate_stats(summary["metrics"]))
    if summary["status"] != "complete":
        raise RuntimeError(
            "Incomplete τ² evaluation; infra failures are not reward zero"
        )
    return {"manifest": manifest, "summary": summary}


def main(argv: list[str]) -> None:
    print(json.dumps(run(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
