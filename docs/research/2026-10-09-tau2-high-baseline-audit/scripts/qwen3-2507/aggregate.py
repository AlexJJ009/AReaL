"""Scores, failure classes and per-task gaps from extract_episodes.py output.

Usage:
  python aggregate.py FEATURES.jsonl scores
  python aggregate.py FEATURES.jsonl classes
  python aggregate.py FEATURES.jsonl gap REF_RUN TARGET_RUN   # tasks REF solves, TARGET fails
"""

import json
import re
import sys
from collections import Counter, defaultdict

CONFIRM = re.compile(
    r"\b(yes|yeah|proceed|go ahead|confirm|please do|that'?s correct|sounds good)\b.*###STOP###",
    re.I | re.S,
)

rows = [json.loads(line) for line in open(sys.argv[1])]
mode = sys.argv[2]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def failure_class(r):
    if r["status"] != "completed":
        return "excluded:" + (
            "context" if "context" in (r.get("error") or "").lower() else "other"
        )
    if r.get("stop_reason") not in ("user_stop", "agent_stop"):
        return f"terminated:{r['stop_reason']}"
    if r["domain"] == "telecom":
        return (
            "telecom:env_assertion_failed" if r.get("env_failed") else "telecom:other"
        )
    if r.get("transfer_flag"):
        return f"transfer_rule:{r['transfer_flag']}"
    if r.get("db_match"):
        if r.get("communicate_ok") is False:
            return "db_ok:communicate_failed"
        if r.get("nl_ok") is False:
            return "db_ok:nl_failed"
        return "db_ok:other"
    if r["n_gold_writes"] and not r["agent_writes"]:
        if r["transferred"]:
            return "db_fail:no_write:transferred"
        if r["stop_reason"] == "user_stop" and CONFIRM.search(r.get("last_user") or ""):
            # The user agreed and ended the call in one message, before the write.
            return "db_fail:no_write:user_stopped_at_confirmation"
        return "db_fail:no_write"
    if not r["n_gold_writes"] and r["agent_writes"]:
        return "db_fail:write_when_gold_has_none"
    if r["gold_writes_matched"] < r["n_gold_writes"]:
        return "db_fail:wrong_or_partial_write"
    return "db_fail:other"


if mode == "scores":
    groups = defaultdict(list)
    for r in rows:
        for key in (
            (r["run"], r["domain"], "all"),
            (r["run"], r["domain"], r["split"]),
            (r["run"], "ALL", "all"),
        ):
            groups[key].append(r)
    print("run\tdomain\tsplit\tepisodes\texcluded\tmean")
    for key in sorted(groups):
        rs = groups[key]
        done = [r["score"] for r in rs if r["status"] == "completed"]
        print(
            "\t".join(map(str, key)),
            len(rs),
            len(rs) - len(done),
            f"{mean(done):.3f}",
            sep="\t",
        )
elif mode == "classes":
    by = defaultdict(Counter)
    for r in rows:
        if r["status"] == "completed" and r["score"] == 1.0:
            continue
        by[(r["run"], r["domain"])][failure_class(r)] += 1
    for key in sorted(by):
        total = sum(by[key].values())
        print(*key, f"failures={total}")
        for cls, n in by[key].most_common():
            print(f"    {cls:45s} {n:4d}  {n / total:.0%}")
elif mode == "gap":
    ref, target = sys.argv[3], sys.argv[4]
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] == "completed":
            per[(r["domain"], r["task_id"])][r["run"]].append(r["score"])
    for (domain, task), runs in sorted(per.items()):
        a, b = runs.get(ref), runs.get(target)
        if a and b and mean(a) >= 0.75 and mean(b) <= 0.25:
            fails = [
                failure_class(r)
                for r in rows
                if r["run"] == target
                and r["domain"] == domain
                and r["task_id"] == task
                and r["status"] == "completed"
                and r["score"] < 1
            ]
            print(
                f"{domain}:{task}\tref={mean(a):.2f}\ttarget={mean(b):.2f}\t{Counter(fails).most_common(2)}"
            )
