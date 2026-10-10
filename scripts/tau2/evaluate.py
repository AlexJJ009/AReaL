# SPDX-License-Identifier: Apache-2.0

"""Standalone official τ² repeated evaluation for one policy checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from examples.tau2.contracts import (
    OFFICIAL_SPLIT_SNAPSHOTS,
    OFFICIAL_TAU2_REVISION,
    normalize_domains,
    resolve_pinned_hf_snapshot,
)
from examples.tau2.official_runner import (
    INFRASTRUCTURE_ERROR,
    collect_episodes,
    expand_trials,
    official_metrics,
)
from examples.tau2.train import get_tau2_dataset, validate_tau2_recipe
from examples.tau2.utils import Tau2PPOConfig

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import SGLangConfig, load_expr_config
from areal.engine import RemoteSGLangEngine
from areal.infra import LocalScheduler
from areal.utils import logging
from areal.utils.printing import tabulate_stats

logger = logging.getLogger("Tau2Train")

TASK_SETS = ("test", "base")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Resolved training YAML.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="HF snapshot/local model path loaded by the single-GPU eval server.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--task-set",
        choices=TASK_SETS,
        default="test",
        help="test: official test split only. base: the official leaderboard set "
        "(train+test), with per-split scores reported separately.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Evaluation temperature (default: 0). Use 1 for the historical protocol.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated domain:task_id subset of --task-set, for smoke runs.",
    )
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides applied to --config, e.g. econfig.user_llm=<model>.",
    )
    return parser.parse_args(argv)


def load_eval_config(args: argparse.Namespace) -> tuple[Tau2PPOConfig, str]:
    config, config_file = load_expr_config(
        ["--config", args.config, *args.overrides], Tau2PPOConfig
    )
    if config.eval_gconfig is not None and config.eval_gconfig.seed is not None:
        raise ValueError(
            "τ² repeated eval requires eval_gconfig.seed=null so trials do not "
            "collapse to the same fixed request seed"
        )
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    config.eval_gconfig.temperature = args.temperature
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
    (output_root / "name-resolve").mkdir(parents=True, exist_ok=True)
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

    # Average per-trial success; task rows provide the repetition axis. Preserve
    # the resolved evaluation settings apart from the group size below.
    config.eval_gconfig.n_samples = 1
    config.eval_gconfig.reward_normalization = False
    config.eval_gconfig.drop_incomplete_group = False
    return config


def official_base_task_ids(domain: str) -> set[str]:
    from tau2.registry import registry

    return {
        str(task_id) for task_id in registry.get_task_splits_loader(domain)()["base"]
    }


def build_eval_rows(
    config: Tau2PPOConfig, repeats: int, task_set: str = "test"
) -> list[dict[str, Any]]:
    if task_set not in TASK_SETS:
        raise ValueError(f"Unsupported task set: {task_set}")
    domains = normalize_domains(config.domains)
    splits = ("test",) if task_set == "test" else ("test", "train")
    tasks = []
    for split in splits:
        # Base needs the full official train pool; tune mode would carve out dev.
        tasks.extend(
            get_tau2_dataset(
                domain=domains,
                split=split,
                seed=config.seed,
                experiment_mode=config.experiment_mode if split == "test" else "formal",
            )
        )
    if task_set == "base":
        for domain in domains:
            selected = {str(t["task_id"]) for t in tasks if t["domain"] == domain}
            if selected != official_base_task_ids(domain):
                raise ValueError(f"{domain} train+test does not equal official base")
    rows = expand_trials(tasks, repeats)
    expected = (
        sum(
            int(OFFICIAL_SPLIT_SNAPSHOTS[domain][split])
            for domain in domains
            for split in splits
        )
        * repeats
    )
    if len(rows) != expected:
        raise ValueError(
            f"Official {task_set} coverage mismatch: {len(rows)} != {expected}"
        )
    return rows


def select_tasks(rows: list[dict[str, Any]], tasks: str | None) -> list[dict[str, Any]]:
    """Keep only the named domain:task_id pairs of a validated task set."""

    if tasks is None:
        return rows
    wanted = {item.strip() for item in tasks.split(",") if item.strip()}
    selected = [r for r in rows if f"{r['domain']}:{r['task_id']}" in wanted]
    found = {f"{r['domain']}:{r['task_id']}" for r in selected}
    if found != wanted:
        raise ValueError(f"Unknown tasks for this task set: {sorted(wanted - found)}")
    return selected


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
        "timeout": config.eval_episode_timeout_seconds,
        "infra_retries": config.infra_retries,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize_user_usage(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Join per-episode user-simulator tokens to the completed evaluation cells."""

    completed = {r["source_id"] for r in records if r.get("status") == "completed"}
    usage: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                # Retried episodes append again; the last line matches the kept record.
                usage[str(item["source_id"])] = item
    matched = [usage[source_id] for source_id in completed if source_id in usage]
    totals = {
        key: sum(int(item[key]) for item in matched)
        for key in ("user_llm_calls", "user_prompt_tokens", "user_completion_tokens")
    }
    return {
        "episodes_with_usage": len(matched),
        "episodes_missing_usage": len(completed) - len(matched),
        **totals,
        "mean_prompt_tokens_per_episode": totals["user_prompt_tokens"] / len(matched)
        if matched
        else 0.0,
        "mean_completion_tokens_per_episode": totals["user_completion_tokens"]
        / len(matched)
        if matched
        else 0.0,
        "excludes": "NL-judge calls and discarded user retry attempts",
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def aggregate(
    records: list[dict[str, Any]], *, planned: int, repeats: int
) -> dict[str, Any]:
    """Official metrics plus per-split and per-trial pass@1 views.

    Infrastructure errors are excluded from every score, as in
    ``tau2.metrics.agent_metrics``, and always counted.
    """
    official = official_metrics(records, repeats=repeats)
    overall = official["overall"]
    completed = [r for r in records if r.get("status") == "completed"]
    metrics: dict[str, float | int] = {
        "eval/planned_episodes": planned,
        "eval/completed_episodes": len(completed),
        "eval/infra_excluded_episodes": overall["infra_excluded_episodes"],
        "eval/policy_free_episodes": overall["policy_free_episodes"],
        "eval/avg_reward": overall["avg_reward"],
        "eval/pass1_mean": overall["avg_reward"],
    }
    metrics[f"eval/mean@{repeats}"] = overall["mean_at_k"]
    for k, value in overall["pass_hat_k"].items():
        metrics[f"eval/pass^{k}"] = value
    for k, value in overall["pass_at_k"].items():
        metrics[f"eval/pass@{k}"] = value
    for domain, result in official["by_domain"].items():
        metrics[f"eval/domain/{domain}/avg_reward"] = result["avg_reward"]
        metrics[f"eval/domain/{domain}/mean@{repeats}"] = result["mean_at_k"]
        for k, value in result["pass_at_k"].items():
            metrics[f"eval/domain/{domain}/pass@{k}"] = value
        metrics[f"eval/domain/{domain}/infra_excluded_episodes"] = result[
            "infra_excluded_episodes"
        ]
        for k, value in result["pass_hat_k"].items():
            metrics[f"eval/domain/{domain}/pass^{k}"] = value
    by_domain: dict[str, dict[str, float | int]] = {}
    by_trial: dict[str, dict[str, float | int]] = {}
    by_domain_trial: dict[str, dict[str, float | int]] = {}
    # Base runs mix official train and test tasks; keep each part's score.
    by_domain_split: dict[str, dict[str, float | int]] = {}
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in completed:
        groups[("domain", record["domain"])].append(float(record["official_score"]))
        groups[("trial", str(record["trial"]))].append(float(record["official_score"]))
        groups[
            (f"domain_split:{record['domain']}", str(record.get("split", "test")))
        ].append(float(record["official_score"]))
        groups[(f"domain_trial:{record['domain']}", str(record["trial"]))].append(
            float(record["official_score"])
        )
    for (kind, key), values in groups.items():
        if kind == "domain":
            target, target_key = by_domain, key
        elif kind == "trial":
            target, target_key = by_trial, key
        elif kind.startswith("domain_split:"):
            target, target_key = by_domain_split, f"{kind[13:]}:{key}"
        else:
            target, target_key = by_domain_trial, f"{kind[13:]}:{key}"
        target[target_key] = {
            "episodes": len(values),
            "pass1_mean": sum(values) / len(values),
        }
        metrics[f"eval/{kind}/{key}/pass1_mean"] = sum(values) / len(values)
        metrics[f"eval/{kind}/{key}/episodes"] = len(values)
    return {
        "status": "complete" if len(records) == planned else "partial",
        "repeats": repeats,
        "planned_episodes": planned,
        "completed_episodes": len(completed),
        "infra_excluded_episodes": overall["infra_excluded_episodes"],
        "infra_excluded": [
            {k: r.get(k) for k in ("source_id", "attempts", "error_type", "error")}
            for r in records
            if r.get("status") == INFRASTRUCTURE_ERROR
        ],
        "official": official,
        "metrics": metrics,
        "by_domain": by_domain,
        "by_trial": by_trial,
        "by_domain_trial": by_domain_trial,
        "by_domain_split": by_domain_split,
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


def resolve_eval_concurrency(config: Tau2PPOConfig) -> int:
    """Use the same optional-cap fallback as the native rollout controller."""
    rollout = config.evaluation_rollout
    cap = rollout.max_concurrent_rollouts or rollout.consumer_batch_size
    if cap <= 0:
        raise ValueError("Evaluation max_concurrent_rollouts must be positive")
    return cap


def collect_eval_records(
    controller: Any,
    rows: list[dict[str, Any]],
    *,
    workflow_kwargs: dict[str, Any],
    max_inflight: int,
    output_root: Path,
    repeats: int,
) -> list[dict[str, Any]]:
    """Persist each official episode outcome before the next one is consumed."""
    records: list[dict[str, Any]] = []

    def persist(record: dict[str, Any]) -> None:
        write_record(output_root / "episodes.jsonl", record)
        records.append(record)
        write_json(
            output_root / "summary.json",
            aggregate(records, planned=len(rows), repeats=repeats),
        )

    try:
        collect_episodes(
            controller,
            rows,
            workflow_kwargs=workflow_kwargs,
            max_inflight=max_inflight,
            trajectory_root=output_root / "trajectories",
            on_record=persist,
        )
    except Exception as exc:
        summary = aggregate(records, planned=len(rows), repeats=repeats)
        summary["status"] = "partial"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        write_json(output_root / "summary.json", summary)
        raise
    return records


def run(argv: list[str]) -> dict[str, Any]:
    args = parse_args(argv)
    raw_config, config_file = load_eval_config(args)
    output_root = Path(args.output_root).absolute()
    assert_fresh_output(output_root, check_config=args.check_config)
    model_path = resolve_model_path(args.model_path)
    config = bind_eval_config(
        raw_config, model_path=model_path, output_root=output_root
    )
    rows = select_tasks(
        build_eval_rows(config, args.repeats, args.task_set), args.tasks
    )
    workflow_kwargs = build_workflow_kwargs(config)
    max_inflight = resolve_eval_concurrency(config)
    manifest = {
        "schema_version": 1,
        "config_file": config_file,
        "tau2_revision": OFFICIAL_TAU2_REVISION,
        "model_path": model_path,
        "output_root": str(output_root),
        "domains": list(normalize_domains(config.domains)),
        "task_set": args.task_set,
        "tasks": args.tasks,
        "episode_timeout_seconds": config.eval_episode_timeout_seconds,
        "infra_retries": config.infra_retries,
        "repeats": args.repeats,
        "planned_episodes": len(rows),
        "max_concurrent_episodes": max_inflight,
        "dispatch_mode": "rolling",
        "global_seed": config.seed,
        "request_seed": config.eval_gconfig.seed,
        "temperature": config.eval_gconfig.temperature,
        "top_p": config.eval_gconfig.top_p,
        # Evaluation has no per-reply cap, as official τ²; the context bounds replies.
        "max_new_tokens": None,
        "max_total_tokens": config.econfig.context_window_tokens,
        "enable_thinking": config.econfig.enable_thinking,
        "add_thinking_tool": config.econfig.add_thinking_tool,
        "overrides": list(args.overrides),
        "user_llm": config.econfig.user_llm,
        "user_llm_base_url": config.econfig.user_llm_base_url,
        "user_llm_args": dict(config.econfig.user_llm_args or {}),
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
    # Workers inherit these: keep simulator failure and usage records with this run.
    os.environ["TAU2_RUN_ROOT"] = str(output_root)
    os.environ["TAU2_USAGE_LOG"] = str(output_root / "user-usage.jsonl")
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
    try:
        controller.initialize(role="eval-rollout", server_args=server_args)
        controller.start_proxy()
        records = collect_eval_records(
            controller,
            rows,
            workflow_kwargs=workflow_kwargs,
            max_inflight=max_inflight,
            output_root=output_root,
            repeats=args.repeats,
        )
    finally:
        controller.destroy()
    summary = aggregate(records, planned=len(rows), repeats=args.repeats)
    summary["user_llm_usage"] = summarize_user_usage(
        output_root / "user-usage.jsonl", records
    )
    write_json(output_root / "summary.json", summary)
    logger.info("Evaluation Results: %s", tabulate_stats(summary["metrics"]))
    if summary["infra_excluded_episodes"]:
        logger.warning(
            "Excluded %d infrastructure-error episode(s) from metrics, as tau2 run "
            "does; see summary.json infra_excluded",
            summary["infra_excluded_episodes"],
        )
    if summary["status"] != "complete":
        raise RuntimeError("Incomplete τ² evaluation: not every episode resolved")
    return {"manifest": manifest, "summary": summary}


def main(argv: list[str]) -> None:
    print(json.dumps(run(argv), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
