import collections
import json

R = json.load(
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/train_records.json"
    )
)["grpo"]["records"]
g = collections.defaultdict(list)
for r in R:
    g[(r["dir"], r["file"])].append(r)
print("groups", len(g), "sizes", collections.Counter(len(v) for v in g.values()))
print(
    "per dir groups",
    {
        d: c
        for d, c in sorted(
            collections.Counter(k[0] for k in g).items(), key=lambda x: int(x[0])
        )
    },
)
print(
    "splits",
    collections.Counter(r["split"] for r in R),
    "tbf",
    collections.Counter(r["tbf"] for r in R),
)
print("reward values", collections.Counter(r["reward"] for r in R).most_common(6))
# same task within group?
bad = [k for k, v in g.items() if len({(x["domain"], x["task_id"]) for x in v}) > 1]
print("mixed-task groups", len(bad))


def summarize(groups, label):
    dom = collections.defaultdict(lambda: [0, 0, 0, 0])  # groups, identical, all1, all0
    eps = collections.Counter()
    rsum = collections.Counter()
    for v in groups:
        d = v[0]["domain"]
        rs = [x["reward"] for x in v]
        dom[d][0] += 1
        same = len(set(rs)) == 1
        dom[d][1] += same
        dom[d][2] += same and rs[0] == 1
        dom[d][3] += same and rs[0] == 0
        eps[d] += len(v)
        rsum[d] += sum(rs)
    tot = sum(x[0] for x in dom.values())
    ident = sum(x[1] for x in dom.values())
    print(label, "total groups", tot, "identical", ident, f"{ident / tot:.4f}")
    for d, x in sorted(dom.items()):
        print(
            f"  {d}: groups={x[0]} identical={x[1]} (all1={x[2]} all0={x[3]}) mixed={x[0] - x[1]} eps={eps[d]} share={eps[d] / sum(eps.values()):.4f} mean_reward={rsum[d] / eps[d]:.4f}"
        )


summarize(list(g.values()), "ALL DUMPED")
# distinct tasks
print(
    "distinct train tasks",
    len({(r["domain"], r["task_id"]) for r in R}),
    collections.Counter(d for d, t in {(r["domain"], r["task_id"]) for r in R}),
)
tc = collections.Counter((v[0]["domain"], v[0]["task_id"]) for v in g.values())
print("group count per task hist", collections.Counter(tc.values()))
json.dump(
    {
        f"{k[0]}/{k[1]}": [
            v[0]["domain"],
            v[0]["task_id"],
            v[0]["hv"],
            [x["reward"] for x in v],
        ]
        for k, v in g.items()
    },
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/grpo_groups.json",
        "w",
    ),
)
