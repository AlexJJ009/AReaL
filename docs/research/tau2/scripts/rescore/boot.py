"""Paired task-level bootstrap of the mean@k difference between two evaluations.

Scores follow rescore.py: the first context overflow counts as a failure.

Usage:
    python docs/research/tau2/scripts/rescore/boot.py <traj_dir_a> <traj_dir_b> <domain>
"""

import glob
import json
import os
import random
import sys


def resolve(attempts):
    for attempt in attempts:
        if attempt.get("status") == "completed":
            return float(attempt["official_score"])
        if "context_length_exceeded" in str(attempt.get("error", "")):
            return 0.0
    return None


def per_task_success(trajectory_dir, domain):
    trials = {}
    for episode_dir in glob.glob(os.path.join(trajectory_dir, f"{domain}:*:trial*")):
        paths = glob.glob(os.path.join(episode_dir, "*.json"))
        attempts = sorted(
            (json.load(open(p)) for p in paths), key=lambda r: r["recorded_at"]
        )
        score = resolve(attempts)
        if score is not None:
            trials.setdefault(episode_dir.split(":")[1], []).append(score)
    return {task: sum(s) / len(s) for task, s in trials.items()}


def main(dir_a, dir_b, domain, samples=10_000, seed=0):
    a, b = per_task_success(dir_a, domain), per_task_success(dir_b, domain)
    tasks = sorted(set(a) & set(b))
    diffs = [b[t] - a[t] for t in tasks]
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(samples)
    )
    lo, hi = boot[int(0.025 * samples)], boot[int(0.975 * samples) - 1]
    delta = sum(diffs) / len(diffs)
    print(f"{domain} tasks={len(tasks)} delta={delta:.3f} CI=[{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main(*sys.argv[1:4])
