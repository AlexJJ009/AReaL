# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import json
import statistics as st

D = json.load(
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/mainlog_metrics.json"
    )
)


def dedup(steps):
    m = {}
    for s in steps:
        m[s["__step"]] = s  # keep last occurrence (post-resume)
    return [m[k] for k in sorted(m)]


g = dedup(D["grpo"])
s = dedup(D["sao"])
f = lambda x: "%.3g" % x if x is not None else "-"
print("GRPO steps", len(g))
rw = [x.get("ppo_actor/task_reward/avg") for x in g]
gn = [x.get("ppo_actor/update/grad_norm") for x in g]
print(
    "GRPO all-one steps",
    [x["__step"] for x in g if x.get("ppo_actor/task_reward/avg") == 1.0],
)
print(
    "GRPO zero grad steps",
    [x["__step"] for x in g if x.get("ppo_actor/update/grad_norm") == 0.0],
)
print("GRPO reward mean", st.mean(rw), "min", min(rw), "max", max(rw))
print("GRPO epoch1 reward", st.mean(rw[:23]), "epoch2", st.mean(rw[23:]))
print("GRPO lr", set(x.get("ppo_actor/update/lr") for x in g))
keys = [
    "ppo_actor/task_reward/avg",
    "ppo_actor/update/grad_norm",
    "ppo_actor/update/approx_kl/avg",
    "ppo_actor/update/behave_approx_kl/avg",
    "ppo_actor/update/importance_weight/avg",
    "ppo_actor/update/clip_ratio/avg",
    "ppo_actor/update/entropy/avg",
    "ppo_actor/update/compute_logp/kl_div_dual/avg",
]
for name, S in [("GRPO", g), ("SAO", s)]:
    print(name)
    for k in keys:
        v = [x.get(k) for x in S]
        v = [a for a in v if a is not None]
        if not v:
            print(" ", k, "missing")
            continue
        print(
            f"  {k}: n={len(v)} first5={st.mean(v[:5]):.4g} last5={st.mean(v[-5:]):.4g} mean={st.mean(v):.4g} min={min(v):.4g} max={max(v):.4g}"
        )
print(
    "SAO steps",
    len(s),
    "optimizer_steps_since_init last",
    s[-1].get("ppo_actor/update/optimizer_steps_since_init"),
    "critic opt steps",
    s[-1].get("critic/ppo_critic/optimizer_steps_since_init"),
)
ev = [
    (
        x["__step"],
        x.get("critic/ppo_critic/critic_explained_variance"),
        x.get("critic/ppo_critic/critic_explained_variance_defined"),
    )
    for x in s
]
vals = [e for _, e, _ in ev if e is not None]
print(
    "SAO EV n",
    len(vals),
    "negative",
    sum(e < 0 for e in vals),
    "median",
    st.median(vals),
    "mean",
    st.mean(vals),
    "defined flags",
    set(d for *_, d in ev),
)
print("EV by step", [(a, round(b, 3)) for a, b, _ in ev])
# with resume duplicates included
allev = [x.get("critic/ppo_critic/critic_explained_variance") for x in D["sao"]]
allev = [e for e in allev if e is not None]
print(
    "EV incl. pre-resume dup steps: n",
    len(allev),
    "neg",
    sum(e < 0 for e in allev),
    "median",
    st.median(allev),
)
print(
    "SAO lr actor",
    set(x.get("ppo_actor/update/lr") for x in s),
    "critic lr",
    set(x.get("critic/ppo_critic/lr") for x in s),
)
print("SAO task reward mean", st.mean([x["ppo_actor/task_reward/avg"] for x in s]))
print(
    "SAO actor grad_norm by step",
    [round(x.get("ppo_actor/update/grad_norm", float("nan")), 3) for x in s],
)
print(
    "SAO zero grad steps",
    [x["__step"] for x in s if x.get("ppo_actor/update/grad_norm") == 0.0],
)
