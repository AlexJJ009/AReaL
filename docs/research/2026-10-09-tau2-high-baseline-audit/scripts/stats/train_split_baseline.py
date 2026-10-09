import collections
import json

R = json.load(
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/train_records.json"
    )
)
for run in ["grpo", "sao"]:
    recs = R[run]["records"]
    print(
        run,
        "dir==hv",
        sum(str(r["hv"]) == r["dir"] for r in recs),
        "/",
        len(recs),
        "hv-tv spread",
        collections.Counter((r["tv"] or 0) - (r["hv"] or 0) for r in recs).most_common(
            4
        ),
    )
    for label, lo, hi in [
        ("v0 only", 0, 0),
        ("v0-4", 0, 4),
        ("v0-22 (epoch1)", 0, 22),
        ("all", 0, 99),
    ]:
        sel = [r for r in recs if lo <= r["hv"] <= hi]
        out = [label]
        for d, n in [("retail", 74), ("airline", 30), ("telecom", 74)]:
            s = [r for r in sel if r["domain"] == d]
            if not s:
                out.append(f"{d}: -")
                continue
            t = collections.defaultdict(list)
            for r in s:
                t[r["task_id"]].append(r["reward"])
            macro = sum(sum(v) / len(v) for v in t.values()) / len(t)
            out.append(
                f"{d}: {sum(r['reward'] for r in s):.0f}/{len(s)}={sum(r['reward'] for r in s) / len(s):.3f} taskmacro={macro:.3f} tasks={len(t)}/{n}"
            )
        print("  ", " | ".join(out))
