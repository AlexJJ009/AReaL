"""Compare the agent's write calls with the gold writes on failed episodes.

Usage: python write_diff.py RUN_DIR [--domain retail] [--show N]
For each unmatched gold write:
  missing     no agent call of that tool at all (subtask never done)
  args:<keys> same tool called, but these arguments differ from every such call
Also counts agent writes that errored, and episodes where all gold writes were
missing although the agent wrote something else.
"""

import argparse
import glob
import json
import os
from collections import Counter

READ_PREFIX = ("get_", "find_", "list_", "calculate", "search", "check_", "think")

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--domain")
p.add_argument("--show", type=int, default=0)
a = p.parse_args()


def norm(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)


def kept(files):
    recs = [json.load(open(f)) for f in files]
    done = [r for r in recs if r.get("status") == "completed"]
    return max(done, key=lambda r: r["recorded_at"]) if done else None


kinds, arg_keys, per_ep, shown = Counter(), Counter(), Counter(), 0
for d in sorted(glob.glob(os.path.join(a.run, "trajectories", "*"))):
    rec = kept(glob.glob(os.path.join(glob.escape(d), "*.json")))
    if (
        not rec
        or rec.get("official_score") == 1.0
        or (a.domain and rec["domain"] != a.domain)
    ):
        continue
    ri = rec.get("reward_info") or {}
    gold = [c for c in ri.get("action_checks") or [] if c.get("tool_type") == "write"]
    if not gold or (ri.get("db_check") or {}).get("db_match"):
        continue
    msgs = rec["messages"]
    results = {m.get("id"): m for m in msgs if m["role"] == "tool"}
    calls = [
        c
        for m in msgs
        if m["role"] == "assistant"
        for c in (m.get("tool_calls") or [])
        if not c["name"].startswith(READ_PREFIX)
        and c["name"] != "transfer_to_human_agents"
    ]
    errored = sum(bool((results.get(c.get("id")) or {}).get("error")) for c in calls)
    labels = []
    for g in gold:
        if g.get("action_match"):
            continue
        name, gargs = g["action"]["name"], g["action"]["arguments"]
        same = [c for c in calls if c["name"] == name]
        if not same:
            labels.append(f"missing:{name}")
            continue
        best = min(
            same,
            key=lambda c: sum(
                norm(c["arguments"].get(k)) != norm(v) for k, v in gargs.items()
            ),
        )
        diff = sorted(
            k for k, v in gargs.items() if norm(best["arguments"].get(k)) != norm(v)
        )
        labels.append(f"args:{name}:{','.join(diff)}")
        arg_keys.update(f"{name}.{k}" for k in diff)
    matched = sum(bool(g.get("action_match")) for g in gold)
    if not calls:
        ep = "no_write_at_all"
    elif matched and any(lab.startswith("missing") for lab in labels):
        ep = "partial:some_gold_done_rest_missing"
    elif all(lab.startswith("missing") for lab in labels):
        ep = "wrote_only_non_gold_tools"
    else:
        ep = "wrong_arguments"
    per_ep[ep] += 1
    kinds.update(lab.split(":")[0] + ":" + lab.split(":")[1] for lab in labels)
    if errored:
        per_ep["(had_errored_write)"] += 1
    if shown < a.show and ep != "no_write_at_all":
        shown += 1
        print(rec["source_id"], ep, labels, f"errored={errored}")

print("episodes:", dict(per_ep.most_common()))
print("unmatched gold writes:", dict(kinds.most_common(15)))
print("differing args:", dict(arg_keys.most_common(15)))
