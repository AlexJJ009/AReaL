"""Figures for the tau2 user-reliability study (one script, all figures).

Figures (PNG + PDF in --out):
  baseline_curves   PPO vs SAO: eval mean@4 on Airline / Retail / Telecom with
                    task-level bootstrap 95% CI, plus critic EV per training step.
  user_sim          Left: table (mean@4, pass^4, user-caused share).
                    Right: primary cause of each failed episode, per user simulator.
  user_sim_patch    Same layout, before / after the one-line U1 guideline patch.

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
ALGO_COLOR = {"PPO": C_AGENT, "SAO": C_STOP}

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
    return [sum(xs[max(0, i - w + 1) : i + 1]) / len(xs[max(0, i - w + 1) : i + 1]) for i in range(len(xs))]


def plot_baselines(cs, out):
    tm = task_means(f"{cs}/eval_episodes.csv")
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
            ax.plot(steps, mean, color=color, marker="o", markersize=2.8, linewidth=1.3, label=run)
        ax.axvline(65, color="#9a9a9a", linestyle=":", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("mean@4 (%)")
        ax.set_xlim(-3, 123)
    axs[0, 0].text(
        67, axs[0, 0].get_ylim()[1], "SAO eval: new\nDeepSeek user", fontsize=6,
        color="#7a7a7a", va="top",
    )
    axs[0, 0].legend(frameon=False, loc="lower right")

    ax = axs[1, 1]
    for run, series in ev_series(cs).items():
        series = [(s, v) for s, v in sorted(series)]
        xs, ys = [s for s, _ in series], [v for _, v in series]
        ax.plot(xs, ys, color=ALGO_COLOR[run], alpha=0.25, linewidth=0.7)
        ax.plot(xs, rolling(ys, 10), color=ALGO_COLOR[run], linewidth=1.4, label=f"{run} (10-step mean)")
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
            failed_tasks[r["run_key"]].add(f'{r["domain"]}:{r["task_id"]}')
    return fails, failed_tasks


def plot_cause_figure(rows, fails, failed_tasks, out, name, bold_best=True):
    rows = [(k, lab) for k, lab in rows if k in fails]
    n_ep = N_TASKS * N_TRIALS
    stats = []
    for k, lab in rows:
        codes = fails[k]
        counts = collections.Counter(cause_of(c) for c in codes)
        user = sum(v for c, v in counts.items() if c.startswith("User"))
        stats.append(
            dict(
                label=lab,
                mean4=100 * (1 - len(codes) / n_ep),
                pass4=100 * (N_TASKS - len(failed_tasks[k])) / N_TASKS,
                user_share=100 * user / len(codes),
                counts=counts,
                n_fail=len(codes),
            )
        )
    n = len(stats)
    fig = plt.figure(figsize=(7.2, 0.30 * n + 1.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.2], wspace=0.04)
    tab = fig.add_subplot(gs[0])
    bar = fig.add_subplot(gs[1], sharey=tab)
    ylim = (-0.7, n - 0.3)

    # Left: booktabs-style table; row i sits at y = n - 1 - i, same as its bar.
    tab.axis("off")
    tab.set_xlim(0, 1)
    tab.set_ylim(*ylim)
    cols = [(0.0, "User simulator", "left"), (0.62, "mean@4", "center"), (0.77, r"pass$^4$", "center"), (0.92, "User-caused", "center")]
    y_head = n - 0.05
    for x, h, ha in cols:
        tab.text(x, y_head, h, ha=ha, va="center", fontsize=7.5)
    for y, w in ((y_head + 0.42, 1.0), (y_head - 0.4, 0.5), (ylim[0], 1.0)):
        tab.plot([0, 1], [y, y], color="black", linewidth=w, clip_on=False)
    best_m = max(s["mean4"] for s in stats)
    best_p = max(s["pass4"] for s in stats)
    for i, s in enumerate(stats):
        y = n - 1 - i
        tab.text(0.0, y, s["label"], ha="left", va="center", fontsize=7.5)
        for x, v, best in ((0.62, s["mean4"], best_m), (0.77, s["pass4"], best_p)):
            bold = bold_best and abs(v - best) < 1e-9
            tab.text(x, y, f"{v:.1f}", ha="center", va="center", fontsize=7.5, fontweight="bold" if bold else "normal")
        tab.text(0.92, y, f"{s['user_share']:.0f}%", ha="center", va="center", fontsize=7.5)

    # Right: stacked horizontal bars, first row on top.
    ys = list(range(n))[::-1]
    left = [0] * n
    for cause, color in CAUSES:
        vals = [s["counts"].get(cause, 0) for s in stats]
        bar.barh(ys, vals, left=left, height=0.55, color=color, edgecolor="white", linewidth=0.6, label=cause)
        for y, l, v in zip(ys, left, vals):
            if v >= 6:
                bar.text(l + v / 2, y, str(v), ha="center", va="center", fontsize=6, color="white")
        left = [l + v for l, v in zip(left, vals)]
    bar.set_yticks([])
    bar.spines["left"].set_visible(False)
    bar.grid(axis="y", visible=False)
    bar.set_xlim(0, max(left) * 1.08)
    bar.set_xlabel(f"Failed episodes (of {n_ep})")
    handles, labels = bar.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.97), columnspacing=1.0, handlelength=1.1)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sao-case-study", default=SAO_CS)
    ap.add_argument("--labels", nargs="+", default=LABELS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    plot_baselines(a.sao_case_study, a.out)
    fails, failed_tasks = load_labels(a.labels)
    plot_cause_figure(SIM_ROWS, fails, failed_tasks, a.out, "user_sim")
    plot_cause_figure(PATCH_ROWS, fails, failed_tasks, a.out, "user_sim_patch", bold_best=False)


if __name__ == "__main__":
    main()
