"""Re-score pre-fix τ² evaluations with context overflow counted as failure.

Before commit 5df3356f, an evaluation episode that exceeded the policy context
limit was replayed (up to 4 attempts) and scored by the first attempt that
finished. The fix scores the first overflow as a completed zero-score episode.
This script applies the new rule to existing attempt records and aggregates
with the repository's ``official_metrics``.

Usage (from the worktree root):
    .venv/bin/python docs/research/tau2/scripts/rescore/rescore.py <trajectory_dir> ...
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from examples.tau2.official_runner import (  # noqa: E402
    COMPLETED,
    INFRASTRUCTURE_ERROR,
    official_metrics,
)


def load_attempts(episode_dir):
    paths = glob.glob(os.path.join(episode_dir, "*.json"))
    attempts = [json.load(open(p)) for p in paths]
    return sorted(attempts, key=lambda r: r["recorded_at"])


def resolve(attempts):
    """Score of the first attempt that finished or overflowed; None if all failed."""
    for attempt in attempts:
        if attempt.get("status") == COMPLETED:
            return COMPLETED, float(attempt["official_score"]), None
        if "context_length_exceeded" in str(attempt.get("error", "")):
            return COMPLETED, 0.0, "context_limit"
    return INFRASTRUCTURE_ERROR, None, None


def rescore(trajectory_dir, repeats=4):
    records = []
    for episode_dir in sorted(glob.glob(os.path.join(trajectory_dir, "*:*:trial*"))):
        attempts = load_attempts(episode_dir)
        if not attempts:
            continue
        first = attempts[0]
        status, score, stop_reason = resolve(attempts)
        record = {
            "status": status,
            "domain": first["domain"],
            "task_id": first["task_id"],
            "trial": first["trial"],
            "source_id": first["source_id"],
        }
        if status == COMPLETED:
            record.update(official_score=score, stop_reason=stop_reason)
        records.append(record)
    return official_metrics(records, repeats=repeats)


def summarize(stats):
    return (
        f"n={stats['episodes']} ex={stats['infra_excluded_episodes']} "
        f"ctx={stats['context_limit_episodes']} mean@4={stats['mean_at_k']:.3f} "
        f"pass^4={stats['pass_hat_k'].get(4, float('nan')):.3f}"
    )


if __name__ == "__main__":
    for root in sys.argv[1:]:
        metrics = rescore(root)
        domains = " | ".join(
            f"{name} {summarize(stats)}" for name, stats in metrics["by_domain"].items()
        )
        print(root, "| all", summarize(metrics["overall"]), "|", domains)
