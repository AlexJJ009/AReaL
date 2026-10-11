"""Cross-model comparison on the same tasks.

Usage: python compare.py FEATURES.jsonl [FEATURES.jsonl ...]
Prints, per run and domain: mean score, mean after dropping episodes where the
user simulator ended the call in its confirmation turn before any write
(treated as invalid, not as failures), and how many tasks are solved 4/4,
0/4, or mixed (the GRPO-contrast view at temperature 0).
"""

import json
import sys
from collections import defaultdict

files = sys.argv[1:]
sys.argv = sys.argv[:1] + [files[0], "none"]
exec(open(__file__.replace("compare.py", "aggregate.py")).read().split("if mode ==")[0])

rows = [json.loads(line) for f in files for line in open(f)]
print(
    "run\tdomain\tepisodes\tmean\tuser_stop_at_confirm\tmean_without_them\tall4/none/mixed tasks"
)
groups = defaultdict(list)
for r in rows:
    groups[(r["run"], r["domain"])].append(r)
for (run, domain), rs in sorted(groups.items()):
    done = [r for r in rs if r["status"] == "completed"]
    bad = [
        r
        for r in done
        if r["score"] < 1 and failure_class(r).endswith("user_stopped_at_confirmation")  # noqa: F821 (defined by the exec of aggregate.py above)
    ]
    keep = [r for r in done if r not in bad]
    per_task = defaultdict(list)
    for r in done:
        per_task[r["task_id"]].append(r["score"])
    full = sum(all(s == 1 for s in v) for v in per_task.values())
    none = sum(all(s == 0 for s in v) for v in per_task.values())
    print(
        run,
        domain,
        len(rs),
        f"{sum(r['score'] for r in done) / len(done):.3f}",
        len(bad),
        f"{sum(r['score'] for r in keep) / len(keep):.3f}",
        f"{full}/{none}/{len(per_task) - full - none}",
        sep="\t",
    )
