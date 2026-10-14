# SPDX-License-Identifier: Apache-2.0
"""Compare official and fixed step-0 runs: mean@4 and pass^1/2/4 by domain and split.

``retail/train_pool`` is the fixed variant's Retail training pool: the train
split without the test twins left out of training.

Run from the repository root:
  PYTHONPATH=. python <this file> OFFICIAL_RUN_DIR FIXED_RUN_DIR
"""

import glob
import json
import os
import sys
from collections import Counter

from examples.tau2.benchmark_variant import train_excluded_task_ids
from examples.tau2.official_runner import official_metrics


def load(run):
    return [json.loads(line) for line in open(os.path.join(run, "episodes.jsonl"))]


def last_attempt(run, source_id):
    files = glob.glob(os.path.join(run, "trajectories", source_id, "*.json"))
    return json.load(open(max(files, key=os.path.getmtime))) if files else {}


def table(records):
    out = {}
    views = [("overall", records)]
    for domain in ("airline", "retail"):
        for split in (None, "train", "test"):
            subset = [
                r
                for r in records
                if r["domain"] == domain and (split is None or r["split"] == split)
            ]
            views.append((f"{domain}{'/' + split if split else ''}", subset))
    excluded = train_excluded_task_ids("fixed", "retail")
    pool = [
        r
        for r in records
        if r["domain"] == "retail"
        and r["split"] == "train"
        and r["task_id"] not in excluded
    ]
    views.append(("retail/train_pool", pool))
    for name, subset in views:
        m = official_metrics(subset, repeats=4)["overall"]
        out[name] = {
            "tasks": m["tasks"],
            "excluded": m["infra_excluded_episodes"],
            "mean@4": round(m["mean_at_k"], 3),
            **{f"pass^{k}": round(v, 3) for k, v in m["pass_hat_k"].items()},
        }
    return out


official_run, fixed_run = sys.argv[1], sys.argv[2]
official, fixed = load(official_run), load(fixed_run)
transfer_zeroed = Counter()
for record in fixed:
    info = (last_attempt(fixed_run, record["source_id"]).get("reward_info") or {}).get(
        "info"
    ) or {}
    for flag in ("unexpected_transfer", "missing_transfer"):
        if info.get(flag) and info.get("official_reward") == 1.0:
            transfer_zeroed[(record["domain"], record["split"], flag)] += 1
print(
    json.dumps(
        {
            "official": table(official),
            "fixed": table(fixed),
            "fixed_transfer_rule_zeroed": {
                f"{d}/{s}/{flag}": n
                for (d, s, flag), n in sorted(transfer_zeroed.items())
            },
        },
        indent=1,
    )
)
