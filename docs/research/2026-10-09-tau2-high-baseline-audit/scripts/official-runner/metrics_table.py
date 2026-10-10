# SPDX-License-Identifier: Apache-2.0
"""Recompute mean@4 / pass@k / pass^k from an evaluate.py episodes.jsonl.

Run from the repository root: PYTHONPATH=. python <this file> <episodes.jsonl>
"""

import json
import sys
from collections import defaultdict

from tau2.metrics.agent_metrics import is_successful, pass_hat_k

from examples.tau2.official_runner import official_metrics, pass_at_k

recs = [json.loads(line) for line in open(sys.argv[1])]
m = official_metrics(recs, repeats=4)


def full4(subset):
    by = defaultdict(list)
    for r in subset:
        if r["status"] == "completed":
            by[(r["domain"], r["task_id"])].append(
                is_successful(float(r["official_score"]))
            )
    t = [v for v in by.values() if len(v) == 4]
    return (
        len(t),
        sum(pass_at_k(4, sum(v), 4) for v in t) / len(t),
        sum(pass_hat_k(4, sum(v), 4) for v in t) / len(t),
    )


out = {}
for name, res, subset in [("overall", m["overall"], recs)] + [
    (d, m["by_domain"][d], [r for r in recs if r["domain"] == d])
    for d in sorted(m["by_domain"])
]:
    n4, p4, h4 = full4(subset)
    out[name] = {
        "episodes": res["episodes"],
        "excluded": res["infra_excluded_episodes"],
        "policy_free": res["policy_free_episodes"],
        "avg_reward": res["avg_reward"],
        "mean@4": res["mean_at_k"],
        **{f"pass@{k}": v for k, v in res["pass_at_k"].items()},
        **{f"pass^{k}": v for k, v in res["pass_hat_k"].items()},
        "tasks_with_4_trials": n4,
        "pass@4(4-trial tasks)": p4,
        "pass^4(4-trial tasks)": h4,
    }
print(json.dumps(out, indent=1))
