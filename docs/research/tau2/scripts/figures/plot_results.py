"""Figures for the tau2 user-reliability study (one script, all figures).

Figures (PNG + PDF in --out):
  baseline_curves   PPO vs SAO vs PPO + beta=0.5: eval mean@4 on Airline / Retail /
                    Telecom with task-level bootstrap 95% CI, plus critic EV per
                    training step; the per-step numbers go to baseline_table.csv.
  user_sim          Primary cause of each failed episode, per user simulator; the
                    companion table (mean@4, pass^4, user-caused share) is written
                    to user_sim_table.csv.
  user_sim_patch    Same, before / after the one-line U1 guideline patch.

Style and palette follow the paper figure (overleaf "online RL",
figures/user_sim_retail.tex). Run on phai:
  .venv/bin/python docs/research/tau2/scripts/figures/plot_results.py --out docs/research/tau2/figures
"""

import argparse
import collections
import csv
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ART = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math"
SAO_CS = f"{ART}/tau2-sao-case-study-20261017"
BETA_RUN = f"{ART}/tau2-ppo-beta-20261018/ppo-formal-qwen3-beta05"  # pueue 631
RERUN = f"{ART}/user-sim-rerun-20261018"
LABELS = [
    f"{RERUN}/case-study-v4/labels.jsonl",
    f"{RERUN}/case-study-v4-volc260910/labels.jsonl",
]

# Palette from figures/user_sim_retail.tex.
C_AGENT, C_STOP, C_DEV, C_EXIT, C_BENCH = (
    "#2A78D6",
    "#EB6834",
    "#1BAF7A",
    "#E87BA4",
    "#B4B2A9",
)
BETA = "PPO + β=0.5"
ALGO_COLOR = {"PPO": C_AGENT, "SAO": C_STOP, BETA: C_DEV}

# Failure-cause buckets of the paper figure; codes as in case study v4.
CAUSES = [
    ("Agent error", C_AGENT),
    ("User: stops at confirmation", C_STOP),
    ("User: deviates from goal", C_DEV),
    ("User: leaves early", C_EXIT),
    ("Benchmark defect", C_BENCH),
]


def cause_of(code):
    if code.startswith(("M", "T")):
        return "Agent error"
    if code.startswith("B"):
        return "Benchmark defect"
    if code == "U1":
        return "User: stops at confirmation"
    if code in ("U4b", "U7a"):
        return "User: leaves early"
    if code in ("U3", "U4", "U5"):
        return "User: deviates from goal"
    raise ValueError(f"unmapped primary code {code}")


SIM_ROWS = [  # (run_key, label), top to bottom
    ("gl", "GLM-5.3"),
    ("dsv", "DeepSeek-V4.1 (Sep)"),
    ("ds", "DeepSeek-V4.1 (Oct)"),
    ("dsn", "DeepSeek-V4.1 (Oct, no think)"),
    ("lt", "Doubao-2.1-Lite"),
    ("tb", "Doubao-2.1-Turbo"),
    ("qw", "Qwen3.5-4B"),
]
PATCH_ROWS = [
    ("ds", "DeepSeek-V4.1 (Oct)"),
    ("dsp", "  + patch"),
    ("lt", "Doubao-2.1-Lite"),
    ("ltp", "  + patch"),
    ("tb", "Doubao-2.1-Turbo"),
    ("tbp", "  + patch"),
]
N_TASKS, N_TRIALS = 60, 4


def style():
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ebebeb",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(out, name))


# ---------------------------------------------------------------- baselines
def task_means(path):
    """(run, step, domain) -> {task: mean score over trials}."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(path)):
        run = "PPO" if r["run"].startswith("PPO") else r["run"]
        if run not in ALGO_COLOR:
            continue
        acc[(run, int(r["step"]), r["domain"])][r["task_id"]].append(
            float(r["official_score"])
        )
    return {k: {t: sum(v) / len(v) for t, v in d.items()} for k, d in acc.items()}


def beta_task_means(run_dir):
    """Same shape as task_means, read from the run's async-eval trajectories.

    Uses the last completed attempt per episode and drops episodes the eval
    listed as infra-excluded.
    """
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    ev_dir = f"{run_dir}/evidence/async-eval"
    for name in os.listdir(ev_dir):
        meta = json.load(open(f"{ev_dir}/{name}"))
        step = int(name.removesuffix(".json"))
        excluded = {x["source_id"] for x in meta["infra_excluded"]}
        traj = meta["trajectory_dir"]
        for src in os.listdir(traj):
            if src in excluded:
                continue
            done = [
                json.load(open(f"{traj}/{src}/{f}"))
                for f in os.listdir(f"{traj}/{src}")
            ]
            done = [x for x in done if x["status"] == "completed"]
            if not done:
                continue
            last = max(done, key=lambda x: x["recorded_at"])
            acc[(BETA, step, last["domain"])][last["task_id"]].append(
                float(last["official_score"])
            )
    return {k: {t: sum(v) / len(v) for t, v in d.items()} for k, d in acc.items()}


def beta_ev(run_dir):
    """Critic EV per training step from the run's offline wandb history."""
    import glob

    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore

    key = "critic/ppo_critic/critic_explained_variance"
    (path,) = glob.glob(f"{run_dir}/logs/root/*/*/wandb/offline-run-*/*.wandb")
    ds = DataStore()
    ds.open_for_scan(path)
    ev = []
    while (blob := ds.scan_data()) is not None:
        r = Record()
        r.ParseFromString(blob)
        if r.HasField("history"):
            for i in r.history.item:
                if (i.key or "/".join(i.nested_key)) == key:
                    ev.append(float(json.loads(i.value_json)))
    return list(enumerate(ev))  # one EV record per training step


def write_table(tm, out):
    """mean@4 per step: training domains (Airline + Retail) and Telecom."""
    path = os.path.join(out, "baseline_table.csv")
    runs = list(ALGO_COLOR)
    steps = sorted({s for (_, s, _) in tm})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["step"] + [f"{r} train" for r in runs] + [f"{r} telecom" for r in runs]
        )
        for s in steps:
            row = [s]
            for doms in (("airline", "retail"), ("telecom",)):
                for r in runs:
                    v = [x for d in doms for x in tm.get((r, s, d), {}).values()]
                    ok = all((r, s, d) in tm for d in doms)
                    row.append(f"{sum(v) / len(v):.3f}" if ok else "")
            w.writerow(row)
    print("wrote", path)


def boot_ci(values, n=2000, seed=0):
    rng = random.Random(seed)
    m = len(values)
    bs = sorted(sum(rng.choice(values) for _ in range(m)) / m for _ in range(n))
    return bs[int(0.025 * n)], bs[int(0.975 * n)]


def ev_series(cs):
    def read(name, col):
        rows = csv.DictReader(open(f"{cs}/metrics_{name}.csv"))
        return [(int(r["step"]), float(r[col])) for r in rows if r.get(col)]

    ppo_col = "critic/ppo_critic/critic_explained_variance"
    sao_col = "critic/ppo_critic/critic_update_1/critic_explained_variance"
    return {
        "PPO": read("PPO_v3", ppo_col) + read("PPO_ext", ppo_col),
        "SAO": read("SAO", sao_col),
    }


def rolling(xs, w):
    return [
        sum(xs[max(0, i - w + 1) : i + 1]) / len(xs[max(0, i - w + 1) : i + 1])
        for i in range(len(xs))
    ]


def plot_baselines(cs, beta_run, out):
    tm = task_means(f"{cs}/eval_episodes.csv")
    tm.update(beta_task_means(beta_run))
    write_table(tm, out)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.6))
    panels = [
        ("airline", "Airline (20 tasks)"),
        ("retail", "Retail (40 tasks)"),
        ("telecom", "Telecom (40 tasks, not in training)"),
    ]
    for ax, (dom, title) in zip(axs.flat, panels):
        for run, color in ALGO_COLOR.items():
            steps = sorted(s for (r, s, d) in tm if r == run and d == dom)
            mean, lo, hi = [], [], []
            for s in steps:
                v = list(tm[(run, s, dom)].values())
                a, b = boot_ci(v)
                mean.append(100 * sum(v) / len(v))
                lo.append(100 * a)
                hi.append(100 * b)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.13, linewidth=0)
            ax.plot(
                steps,
                mean,
                color=color,
                marker="o",
                markersize=2.8,
                linewidth=1.3,
                label=run,
            )
        ax.axvline(65, color="#9a9a9a", linestyle=":", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("mean@4 (%)")
        ax.set_xlim(-3, 123)
    axs[0, 0].text(
        67,
        axs[0, 0].get_ylim()[1],
        "SAO eval: new\nDeepSeek user",
        fontsize=6,
        color="#7a7a7a",
        va="top",
    )
    axs[0, 0].legend(frameon=False, loc="lower right", fontsize=6.5)

    ax = axs[1, 1]
    evs = ev_series(cs)
    evs[BETA] = beta_ev(beta_run)
    for run, series in evs.items():
        series = [(s, v) for s, v in sorted(series)]
        xs, ys = [s for s, _ in series], [v for _, v in series]
        ax.plot(xs, ys, color=ALGO_COLOR[run], alpha=0.25, linewidth=0.7)
        ax.plot(
            xs,
            rolling(ys, 10),
            color=ALGO_COLOR[run],
            linewidth=1.4,
            label=f"{run} (10-step mean)",
        )
    ax.axhline(0, color="#9a9a9a", linewidth=0.7)
    ax.set_title("Critic explained variance (training batch, pre-update)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("EV")
    ax.set_xlim(-3, 123)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, out, "baseline_curves")


# ---------------------------------------------------------- user simulators
def load_labels(paths):
    """run_key -> list of primary codes of failed episodes, and failed task set."""
    fails = collections.defaultdict(list)
    failed_tasks = collections.defaultdict(set)
    for p in paths:
        if not os.path.exists(p):
            print("missing labels, skipped:", p)
            continue
        for line in open(p):
            r = json.loads(line)
            if float(r["score"]) != 0.0:  # success anomalies (U6, early TRANSFER)
                continue
            fails[r["run_key"]].append(r["primary"])
            failed_tasks[r["run_key"]].add(f"{r['domain']}:{r['task_id']}")
    return fails, failed_tasks


def plot_cause_figure(rows, fails, failed_tasks, out, name):
    """Bar chart of failure causes; the companion table goes to {name}_table.csv.

    As in the paper, the table and the chart are separate pieces placed side by side
    (LaTeX tabular + figure there, a native table + image in Feishu).
    """
    rows = [(k, lab) for k, lab in rows if k in fails]
    n_ep = N_TASKS * N_TRIALS
    stats = []
    for k, lab in rows:
        codes = fails[k]
        counts = collections.Counter(cause_of(c) for c in codes)
        user = sum(v for c, v in counts.items() if c.startswith("User"))
        stats.append(
            dict(
                label=lab.strip(),
                mean4=100 * (1 - len(codes) / n_ep),
                pass4=100 * (N_TASKS - len(failed_tasks[k])) / N_TASKS,
                user_share=100 * user / len(codes),
                counts=counts,
            )
        )
    with open(os.path.join(out, f"{name}_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "user_simulator",
                "mean@4",
                "pass^4",
                "user_caused_share",
                *[c for c, _ in CAUSES],
            ]
        )
        for s in stats:
            w.writerow(
                [
                    s["label"],
                    f"{s['mean4']:.1f}",
                    f"{s['pass4']:.1f}",
                    f"{s['user_share']:.0f}%",
                ]
                + [s["counts"].get(c, 0) for c, _ in CAUSES]
            )
    print("wrote", os.path.join(out, f"{name}_table.csv"))

    n = len(stats)
    fig, ax = plt.subplots(figsize=(4.4, 0.32 * n + 0.9))
    ys = list(range(n))[::-1]  # first row on top
    left = [0] * n
    for cause, color in CAUSES:
        vals = [s["counts"].get(cause, 0) for s in stats]
        ax.barh(
            ys,
            vals,
            left=left,
            height=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=cause,
        )
        for y, x0, v in zip(ys, left, vals):
            if v >= 6:
                ax.text(
                    x0 + v / 2,
                    y,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
        left = [x0 + v for x0, v in zip(left, vals)]
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, max(left) * 1.08)
    ax.set_xlabel(f"Failed episodes (of {n_ep})")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.42, 1.0),
        columnspacing=0.9,
        handlelength=1.1,
        fontsize=6.5,
    )
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sao-case-study", default=SAO_CS)
    ap.add_argument("--beta-run", default=BETA_RUN)
    ap.add_argument("--labels", nargs="+", default=LABELS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    plot_baselines(a.sao_case_study, a.beta_run, a.out)
    fails, failed_tasks = load_labels(a.labels)
    plot_cause_figure(SIM_ROWS, fails, failed_tasks, a.out, "user_sim")
    plot_cause_figure(PATCH_ROWS, fails, failed_tasks, a.out, "user_sim_patch")


if __name__ == "__main__":
    main()
