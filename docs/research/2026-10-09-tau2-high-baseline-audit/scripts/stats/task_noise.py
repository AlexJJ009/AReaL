import collections
import json
import math
import random

C = json.load(
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/eval_cells_dedup.json"
    )
)


def per_task(key, doms=("retail", "airline")):
    t = collections.defaultdict(list)
    for k, s in C[key].items():
        d, rest = k.split("|", 1)
        tid, i = rest.rsplit("|", 1)
        if d in doms:
            t[(d, tid)].append(s)
    return t


for key in ["GRPO:0", "SAO:0"]:
    for doms in [("retail",), ("airline",), ("retail", "airline")]:
        t = per_task(key, doms)
        n = collections.Counter(sum(v) for v in t.values())
        assert all(len(v) == 8 for v in t.values())
        print(
            key,
            doms,
            "tasks",
            len(t),
            "8/8",
            n[8],
            "0/8",
            n[0],
            "mixed",
            len(t) - n[8] - n[0],
            "hist",
            dict(sorted(n.items())),
        )
g = per_task("GRPO:0")
s = per_task("SAO:0")
assert g.keys() == s.keys()
keys = sorted(g)
N = len(keys)
both8 = sum(sum(g[k]) == 8 and sum(s[k]) == 8 for k in keys)
both0 = sum(sum(g[k]) == 0 and sum(s[k]) == 0 for k in keys)
print("tasks 8/8 in both evals", both8, "0/8 in both", both0)


# combined mean over 60 tasks with retail+airline pooled per-episode weight (480 eps)
def m(t):
    return sum(sum(t[k]) for k in keys) / (8 * N)


diffs = [(sum(g[k]) - sum(s[k])) / 8 for k in keys]
print("GRPO0 pooled", m(g), "SAO0 pooled", m(s), "diff", m(g) - m(s))
sd = (sum((x - sum(diffs) / N) ** 2 for x in diffs) / (N - 1)) ** 0.5
print("paired SE of diff (task-level)", sd / math.sqrt(N))
rng = random.Random(0)
B = 20000
boot = sorted(sum(diffs[rng.randrange(N)] for _ in range(N)) / N for _ in range(B))
print("bootstrap 95% CI of observed diff", boot[int(0.025 * B)], boot[int(0.975 * B)])
# Theoretical SE for a step-to-step difference under H0 (same model) using per-task pass rates p_t from pooled 16 samples
p = {k: (sum(g[k]) + sum(s[k])) / 16 for k in keys}
var_diff = (
    sum(2 * p[k] * (1 - p[k]) / 8 for k in keys) / N**2
)  # within-task binomial noise only, paired by task
print(
    "model SE of diff under H0 (binomial within task, paired, pooled 60 tasks)",
    math.sqrt(var_diff),
    "-> 95% half-width",
    1.96 * math.sqrt(var_diff),
)
for dom, nt in [("retail", 40), ("airline", 20)]:
    ks = [k for k in keys if k[0] == dom]
    v = sum(2 * p[k] * (1 - p[k]) / 8 for k in ks) / nt**2
    dd = [(sum(g[k]) - sum(s[k])) / 8 for k in ks]
    sdd = (sum((x - sum(dd) / nt) ** 2 for x in dd) / (nt - 1)) ** 0.5
    print(
        dom,
        "H0 SE",
        math.sqrt(v),
        "95% hw",
        1.96 * math.sqrt(v),
        "| observed GRPO0-SAO0",
        sum(dd) / nt,
        "empirical paired SE",
        sdd / math.sqrt(nt),
    )
# Simulation of null: resample 8 Bernoulli(p_t) twice
rng = random.Random(1)
sims = []
for _ in range(20000):
    a = sum(sum(rng.random() < p[k] for _ in range(8)) for k in keys)
    b = sum(sum(rng.random() < p[k] for _ in range(8)) for k in keys)
    sims.append((a - b) / (8 * N))
sims.sort()
print("sim null 95% interval pooled", sims[500], sims[19500])
# Unpaired / naive iid SE for comparison
pp = m(g)
print(
    "naive iid SE (480 vs 480, ignoring task pairing)",
    math.sqrt(2 * pp * (1 - pp) / 480),
)
# All steps relative to pooled step-0 mean
for key in [
    "GRPO:5",
    "GRPO:10",
    "GRPO:15",
    "GRPO:20",
    "GRPO:25",
    "SAO:5",
    "SAO:10",
    "SAO:15",
    "SAO:20",
    "SAO:25",
]:
    t = per_task(key)
    if set(t) != set(keys) or any(len(v) != 8 for v in t.values()):
        print(key, "incomplete")
        continue
    print(
        key,
        "pooled R+A",
        round(m(t), 4),
        "vs own step0",
        round(m(t) - (m(g) if key.startswith("GRPO") else m(s)), 4),
    )
