# SPDX-License-Identifier: Apache-2.0
"""Recompute mean@4 and pass^1/2/4 from an evaluate.py episodes.jsonl.

Run from the repository root: PYTHONPATH=. python <this file> <episodes.jsonl>
"""

import json
import sys

from examples.tau2.official_runner import official_metrics

recs = [json.loads(line) for line in open(sys.argv[1])]
m = official_metrics(recs, repeats=4)

out = {}
for name, res in [("overall", m["overall"])] + sorted(m["by_domain"].items()):
    out[name] = {
        "episodes": res["episodes"],
        "excluded": res["infra_excluded_episodes"],
        "policy_free": res["policy_free_episodes"],
        "tasks": res["tasks"],
        "avg_reward": res["avg_reward"],
        "mean@4": res["mean_at_k"],
        **{f"pass^{k}": v for k, v in res["pass_hat_k"].items()},
        **{f"pass^{k}_tasks": v for k, v in res["pass_hat_k_tasks"].items()},
    }
print(json.dumps(out, indent=1))
