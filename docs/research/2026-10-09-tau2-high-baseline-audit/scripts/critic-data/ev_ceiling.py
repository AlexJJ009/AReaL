"""Task-level EV ceiling of a critic from repeated τ² evals (README 16.14).

Each task is run k times with the same sampling as the critic's validation rollouts. A critic
that sees only the task can at best predict the task's success rate p_i, so the share of the
single-episode reward variance that lies between tasks bounds its EV at the first agent token:

    ceiling = (T - W) / T,  T = pbar (1 - pbar),  W = mean_i k/(k-1) p_i (1 - p_i),

with W the unbiased within-task variance. Later prefixes can reveal more, so this bounds
token-0 EV, not token-mean EV. 95% intervals bootstrap over tasks.

Usage: python ev_ceiling.py EPISODES_JSONL [CRITIC_VALIDATION_REPORT_JSON]
"""

import collections
import json
import random
import statistics
import sys


def decompose(groups):
    rewards = [r for g in groups for r in g]
    pbar = sum(rewards) / len(rewards)
    total = pbar * (1 - pbar)
    within = statistics.mean(
        len(g) / (len(g) - 1) * (sum(g) / len(g)) * (1 - sum(g) / len(g))
        for g in groups
    )
    return pbar, total, within, (total - within) / total if total > 0 else float("nan")


def bootstrap(groups, n=2000):
    rng = random.Random(0)
    draws = sorted(decompose([rng.choice(groups) for _ in groups])[3] for _ in range(n))
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


def main():
    by_task = collections.defaultdict(list)
    for line in open(sys.argv[1]):
        e = json.loads(line)
        by_task[(e["domain"], str(e["task_id"]))].append(e["reward"])
    for name in ("all", "airline", "retail"):
        groups = [g for (d, _), g in sorted(by_task.items()) if name in ("all", d)]
        pbar, total, within, ceiling = decompose(groups)
        lo, hi = bootstrap(groups)
        hist = dict(sorted(collections.Counter(int(sum(g)) for g in groups).items()))
        print(
            f"{name:8s} tasks={len(groups):2d} pbar={pbar:.3f} var={total:.3f} "
            f"within={within:.3f} ceiling={ceiling:.3f} [{lo:.3f}, {hi:.3f}] successes={hist}"
        )
    if len(sys.argv) > 2:
        # Episode-level check of ranking, ignoring the critic's offset and scale.
        rows = json.load(open(sys.argv[2]))["rows"]
        for name in ("all", "airline", "retail"):
            sel = [r for r in rows if name in ("all", r["domain"])]
            pred = [r["prediction_mean"] for r in sel]
            p_task = [
                statistics.mean(by_task[(r["domain"], r["episode_id"].split(":")[-1])])
                for r in sel
            ]
            target = [r["target"] for r in sel]
            print(
                f"{name:8s} corr(pred_mean, task p)={statistics.correlation(pred, p_task):+.3f} "
                f"corr(pred_mean, R)^2={statistics.correlation(pred, target) ** 2:.3f}"
            )


if __name__ == "__main__":
    main()
