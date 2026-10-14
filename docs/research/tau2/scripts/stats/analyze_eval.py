import collections
import json

R = json.load(
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/eval_records.json"
    )
)
cells = {}
for key, v in R.items():
    recs = v["records"]
    c = collections.defaultdict(list)
    for r in recs:
        c[(r["domain"], r["task_id"], r["idx"])].append(r)
    dup = sum(len(x) - 1 for x in c.values())
    disagree = sum(1 for x in c.values() if len({y["score"] for y in x}) > 1)
    hv = collections.Counter(r.get("hv") for r in recs)
    st_ = collections.Counter(r.get("status") for r in recs)
    last = {
        k: x[-1] for k, x in c.items()
    }  # last in file order (files sorted, lines ordered)
    first = {k: x[0] for k, x in c.items()}
    cells[key] = last
    out = [
        key,
        f"raw={len(recs)} uniq={len(c)} dupExtra={dup} dupScoreDisagree={disagree} hv={dict(hv)} status={dict(st_)}",
    ]
    for dom in ["retail", "airline", "telecom"]:
        for name, dd in [("last", last), ("first", first)]:
            vals = [r["score"] for k, r in dd.items() if k[0] == dom]
            nt = len({k[1] for k in dd if k[0] == dom})
            out.append(
                f"{dom}[{name}] {sum(vals):.0f}/{len(vals)}={sum(vals) / max(1, len(vals)):.5f} tasks={nt}"
            )
            if dup == 0:
                break
    print(" | ".join(out))
json.dump(
    {
        k: {"|".join(map(str, kk)): r["score"] for kk, r in v.items()}
        for k, v in cells.items()
    },
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/eval_cells_dedup.json",
        "w",
    ),
)
