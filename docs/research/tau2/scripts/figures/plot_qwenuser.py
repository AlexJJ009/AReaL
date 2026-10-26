"""Runs trained with the self-hosted Qwen3.5-4B user: offline eval curves and critic EV.

One figure, qwenuser_runs (PNG + PDF in --out):
  row 1  offline eval with the DeepSeek official user + judge: Airline / Retail / Telecom
  row 2  offline eval with the Qwen3.5-4B user + Volcano judge: Airline / Retail / Telecom
  row 3  training success rate and critic EV (pre-update) per step, 10-step means

mean@4 is the per-task success rate over trials (4, or 8 when two attempts of the same
checkpoint are pooled), averaged over tasks; shading is a task-level bootstrap 95% CI.
Only episodes with status "completed" count. Encoding: colour = critic (standard /
privileged), line style = actor lambda (solid 1, dashed SAO length-adaptive), grey = run 3
(trained with Telecom). Missing eval directories are skipped, so the figure fills in as the
run 4 / run 5 evals land. Per-point numbers go to qwenuser_eval.csv, per-step training
statistics to qwenuser_train.csv.

Style and palette are those of plot_results.py. Run on phai from an artifact directory:
  .venv/bin/python <worktree>/docs/research/tau2/scripts/figures/plot_qwenuser.py \
      --out <worktree>/docs/research/tau2/figures
"""

import argparse
import collections
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from plot_results import (  # noqa: E402
    C_AGENT,
    C_BENCH,
    C_DEV,
    boot_ci,
    rolling,
    save,
    style,
)

ART = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math"
TRAIN = f"{ART}/tau2-ppo-qwenuser-20261021"

# key -> (legend label, colour, line style, trial directory under TRAIN)
RUNS = {
    "run3": (
        "run 3: standard critic, $\\lambda=1$ (trained with Telecom)",
        C_BENCH,
        "-",
        "run3-std-beta1-v4",
    ),
    "run3p": (
        "run 3′: standard critic, $\\lambda=1$",
        C_AGENT,
        "-",
        "run3p-std-beta1-notel-c128",
    ),
    "run2p": (
        "run 2′: privileged critic, $\\lambda=1$",
        C_DEV,
        "-",
        "run2p-priv-beta1-notel-c128",
    ),
    "run5": (
        "run 5: standard critic, SAO $\\lambda$",
        C_AGENT,
        "--",
        "run5-std-saolam-notel-c128",
    ),
    "run4": (
        "run 4: privileged critic, SAO $\\lambda$",
        C_DEV,
        "--",
        "run4-priv-saolam-notel-c128",
    ),
}
DS, QW = "deepseek", "qwen"
# (evaluator, run key, directory holding step-NNN/ckpt-*/ result dirs)
EVAL_DIRS = [
    (DS, "run3", f"{ART}/tau2-deepseek-eval-20261024/eval"),
    (DS, "run3p", f"{ART}/tau2-deepseek-eval-20261025/eval"),
    (
        DS,
        "run4",
        f"{ART}/tau2-deepseek-eval-run45-20261025/eval-run4-priv-saolam-notel-c128",
    ),
    (
        DS,
        "run5",
        f"{ART}/tau2-deepseek-eval-run45-20261025/eval-run5-std-saolam-notel-c128",
    ),
    (QW, "run3", f"{TRAIN}/run3-std-beta1-v4/offline-checkpoint-eval-k1023"),
    (QW, "run3p", f"{ART}/tau2-qwenuser-eval-20261025/eval"),
    (
        QW,
        "run4",
        f"{ART}/tau2-qwenuser-eval-run45-20261025/eval-run4-priv-saolam-notel-c128",
    ),
    (
        QW,
        "run5",
        f"{ART}/tau2-qwenuser-eval-run45-20261025/eval-run5-std-saolam-notel-c128",
    ),
]
# Extra checkpoints evaluated inside another run's directory: label prefix -> run key.
CKPT_RUN = {"run2p": "run2p"}
# Step 0 is the shared base actor; runs without their own step-0 point borrow this one.
STEP0_FROM = {"run2p": "run3p", "run4": "run3p", "run5": "run3p"}
REFERENCE = ("ckpt-v3ext-120", "PPO trained with the DeepSeek user (v3ext, step 120)")
DOMAINS = ("airline", "retail", "telecom")
EV_KEY = "critic/ppo_critic/critic_explained_variance_pre_update"
REWARD_KEY = "ppo_actor/task_reward/avg"


def read_result_dir(path):
    """{(domain, task): [scores]} over all attempts of one checkpoint, completed only."""
    scores = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{path}/attempt-*/episodes.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if r.get("status") == "completed":
                scores[(r["domain"], r["task_id"])].append(float(r["official_score"]))
    return scores


def load_evals():
    """{(evaluator, run, step): {(domain, task): [scores]}} and the reference point."""
    out, ref = {}, {}
    for ev, run, root in EVAL_DIRS:
        if not os.path.isdir(root):
            print("missing, skipped:", root)
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if m := re.fullmatch(r"step-(\d+)", name):
                key = (ev, run, int(m.group(1)))
            elif m := re.fullmatch(r"ckpt-(\w+?)-(\d+)", name):
                if name == REFERENCE[0]:
                    ref[ev] = read_result_dir(path)
                    continue
                key = (ev, CKPT_RUN[m.group(1)], int(m.group(2)))
            else:
                continue
            scores = read_result_dir(path)
            if scores:
                out[key] = scores
    for (ev, run, step), v in list(out.items()):
        if step == 0 and run in STEP0_FROM.values():
            for r, src in STEP0_FROM.items():
                if (
                    src == run
                    and (ev, r, 0) not in out
                    and any(k[:2] == (ev, r) for k in out)
                ):
                    out[(ev, r, 0)] = v
    return out, ref


def point(scores, domain):
    """mean@k (%), CI (%), number of tasks and episodes for one domain."""
    per_task = [sum(v) / len(v) for (d, _), v in scores.items() if d == domain]
    if not per_task:
        return None
    lo, hi = boot_ci(per_task, n=2000)
    n_ep = sum(len(v) for (d, _), v in scores.items() if d == domain)
    return 100 * sum(per_task) / len(per_task), 100 * lo, 100 * hi, len(per_task), n_ep


ANSI = re.compile(r"\x1b\[[0-9;]*m")
DONE = re.compile(r"StatsLogger INFO: .*Train step (\d+)/\d+ done")


def train_stats(trial):
    """Per-step stats tables printed after each "Train step N done" line of main.log."""
    (path,) = glob.glob(f"{TRAIN}/{trial}/logs/root/tau2-ppo/*/main.log")
    steps, cur = {}, None
    with open(path, errors="replace") as f:
        for line in f:
            line = ANSI.sub("", line)
            if m := DONE.search(line):
                cur = steps.setdefault(int(m.group(1)), {})
                continue
            if cur is not None and "│" in line:
                cells = [c.strip() for c in line.strip().strip("│").split("│")]
                for k, v in zip(cells[::2], cells[1::2]):
                    try:
                        cur.setdefault(k, float(v))
                    except ValueError:
                        pass
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    evals, ref = load_evals()

    with open(os.path.join(a.out, "qwenuser_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "evaluator",
                "run",
                "step",
                "domain",
                "mean@k",
                "ci_lo",
                "ci_hi",
                "tasks",
                "episodes",
            ]
        )
        for (ev, run, step), s in sorted(evals.items()):
            for d in DOMAINS:
                if p := point(s, d):
                    w.writerow([ev, run, step, d, *(f"{x:.1f}" for x in p[:3]), *p[3:]])
        for ev, s in ref.items():
            for d in DOMAINS:
                if p := point(s, d):
                    w.writerow(
                        [ev, "v3ext", 120, d, *(f"{x:.1f}" for x in p[:3]), *p[3:]]
                    )

    train = {}
    for key, (_, _, _, trial) in RUNS.items():
        if glob.glob(f"{TRAIN}/{trial}/logs/root/tau2-ppo/*/main.log"):
            train[key] = train_stats(trial)
    with open(os.path.join(a.out, "qwenuser_train.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "train_success", "ev_pre_update"])
        for key, steps in train.items():
            for s, row in sorted(steps.items()):
                w.writerow([key, s, row.get(REWARD_KEY, ""), row.get(EV_KEY, "")])

    fig, axs = plt.subplots(3, 3, figsize=(7.4, 7.0))
    titles = {DS: "DeepSeek user", QW: "Qwen3.5-4B user"}
    sizes = {"airline": 20, "retail": 40, "telecom": 40}
    for row, ev in enumerate((DS, QW)):
        for col, dom in enumerate(DOMAINS):
            ax = axs[row, col]
            for run, (label, color, ls, _) in RUNS.items():
                pts = sorted(
                    (step, point(s, dom))
                    for (e, r, step), s in evals.items()
                    if e == ev and r == run
                )
                pts = [(x, p) for x, p in pts if p]
                if not pts:
                    continue
                xs = [x for x, _ in pts]
                ax.fill_between(
                    xs,
                    [p[1] for _, p in pts],
                    [p[2] for _, p in pts],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
                ax.plot(
                    xs,
                    [p[0] for _, p in pts],
                    color=color,
                    linestyle=ls,
                    marker="o",
                    markersize=2.6,
                    linewidth=1.3,
                    label=label,
                )
            if ev in ref and (p := point(ref[ev], dom)):
                ax.errorbar(
                    [123],
                    [p[0]],
                    yerr=[[p[0] - p[1]], [p[2] - p[0]]],
                    fmt="*",
                    color="black",
                    markersize=5,
                    elinewidth=0.8,
                    capsize=1.5,
                    label=REFERENCE[1],
                )
            ax.set_title(f"{dom.capitalize()} ({sizes[dom]}) · eval with {titles[ev]}")
            ax.set_xlabel("Training step")
            ax.set_ylabel("mean@4 (%)")
            ax.set_xlim(-4, 127)
    axs[1, 2].text(
        0.03,
        0.97,
        "Qwen-user eval of run 3′/2′/4/5 skips Telecom;\n"
        "run 3 steps 70–126: 330–378 of 400 episodes completed",
        transform=axs[1, 2].transAxes,
        fontsize=5.8,
        color="#7a7a7a",
        va="top",
    )

    for ax, key, title, ylabel in (
        (axs[2, 0], REWARD_KEY, "Training success rate (T=1)", "success rate"),
        (axs[2, 1], EV_KEY, "Critic EV (training batch, pre-update)", "EV"),
    ):
        for run, (label, color, ls, _) in RUNS.items():
            if run not in train:
                continue
            series = [(s, r[key]) for s, r in sorted(train[run].items()) if key in r]
            xs, ys = [s for s, _ in series], [v for _, v in series]
            ax.plot(xs, ys, color=color, linestyle=ls, alpha=0.22, linewidth=0.6)
            ax.plot(
                xs,
                rolling(ys, 10),
                color=color,
                linestyle=ls,
                linewidth=1.4,
                label=label,
            )
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-4, 127)
    axs[2, 1].axhline(0, color="#9a9a9a", linewidth=0.7)

    handles = {}
    for ax in axs.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    axs[2, 2].axis("off")
    axs[2, 2].legend(
        handles.values(),
        handles.keys(),
        frameon=False,
        loc="center",
        fontsize=6.3,
        title="thin: per step · thick: 10-step mean\nshading: task bootstrap 95% CI",
        title_fontsize=6,
    )
    fig.text(
        0.995,
        0.005,
        "Judges: DeepSeek official for row 1, Volcano plan DeepSeek V4.1 Flash for row 2. "
        "Step 0 (base actor) is shared by run 3′/2′/4/5.",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#7a7a7a",
    )
    fig.tight_layout(h_pad=1.0, w_pad=1.2, rect=(0, 0.015, 1, 1))
    save(fig, a.out, "qwenuser_runs")


if __name__ == "__main__":
    main()
