"""Score runs on the official test 60 (fixed-v2 split): mean@4 and pass^4 per domain.
Usage: python test60_table.py LABEL=episodes.jsonl ..."""

import collections
import json
import math
import sys

SPLIT = "/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v2/data/tau2/domains/{}/split_tasks.json"
test = {
    (d, str(i))
    for d in ("airline", "retail")
    for i in json.load(open(SPLIT.format(d)))["test"]
}


def passk(c, n, k):
    return math.comb(c, k) / math.comb(n, k)


for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    by = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        k = (r["domain"], str(r["task_id"]))
        if k in test and r.get("status", "completed") == "completed":
            by[k].append(float(r["official_score"]) >= 1.0)
    row = [label]
    for dom in ("airline", "retail", None):
        ks = [k for k in by if dom is None or k[0] == dom]
        eps = sum(len(by[k]) for k in ks)
        m = sum(sum(by[k]) for k in ks) / eps
        p4 = sum(passk(sum(by[k]), len(by[k]), 4) for k in ks if len(by[k]) == 4) / len(
            ks
        )
        row += [f"{len(ks)}t/{eps}e", f"{m:.3f}", f"{p4:.3f}"]
    print(" | ".join(row))
