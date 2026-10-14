"""Telecom: success by issue type and number of injected faults; failing assertions.

Usage: python telecom.py FEATURES.jsonl [FEATURES.jsonl ...]
Task ids look like "[mms_issue]fault_a|fault_b[PERSONA:Hard]".
"""

import json
import re
import sys
from collections import Counter, defaultdict

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["domain"] == "telecom" and r["status"] == "completed"]
    if not rows:
        continue
    print(
        f"#### {rows[0]['run']}  episodes={len(rows)}  mean={sum(r['score'] for r in rows) / len(rows):.3f}"
    )
    by_n, by_issue, by_persona = defaultdict(list), defaultdict(list), defaultdict(list)
    failed_asserts, transfers, stops = Counter(), Counter(), Counter()
    turns_ok, turns_fail = [], []
    for r in rows:
        m = re.match(r"\[(\w+)\](.*?)\[PERSONA:(\w+)\]", r["task_id"])
        issue, faults, persona = m.group(1), m.group(2).split("|"), m.group(3)
        by_n[min(len(faults), 9)].append(r["score"])
        by_issue[issue].append(r["score"])
        by_persona[persona].append(r["score"])
        (turns_ok if r["score"] == 1 else turns_fail).append(r["policy_turns"])
        if r["score"] < 1:
            failed_asserts.update(r.get("env_failed") or [])
            transfers[r["transferred"]] += 1
            stops[r.get("stop_reason")] += 1
    f = lambda d: {  # noqa: E731
        k: f"{sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(d.items())
    }
    print("  by #faults:", f(by_n))
    print("  by issue:", f(by_issue))
    print("  by persona:", f(by_persona))
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    print(f"  policy turns median: success={med(turns_ok)} failure={med(turns_fail)}")
    print("  failures: transferred", dict(transfers), "stop", dict(stops))
    print("  failing assertions:", dict(failed_asserts.most_common(10)))
