"""Collect per-checkpoint eval scores and per-step training stats of the Qwen-user runs into two CSVs.

Reads only the per-attempt `summary.json` files and `main.log` (episode-level results were not backed
up, so no confidence intervals). Usage:

    python3 -I collect_brief_tables.py ARTIFACTS_ROOT OUT_DIR

ARTIFACTS_ROOT is the `_artifacts/AReaL/codex-sao-math` directory (e.g. unpacked from the backup).
Writes OUT_DIR/eval_mean4.csv and OUT_DIR/train_per_step.csv.
"""

import csv
import glob
import json
import os
import subprocess
import sys

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

# (run label, evaluator, glob of checkpoint dirs relative to root, step parser)
EVALS = [
    ("step0", "deepseek", "tau2-deepseek-eval-20261025/eval/step-000"),
    ("run3p", "deepseek", "tau2-deepseek-eval-20261025/eval/step-[0-9]*"),
    ("run2p_s42", "deepseek", "tau2-deepseek-eval-20261025/eval/ckpt-run2p-*"),
    ("run2p_s43", "deepseek", "tau2-deepseek-eval-run45-20261025/eval-run2p-priv-beta1-notel-c128-s43/step-*"),
    ("run4", "deepseek", "tau2-deepseek-eval-run45-20261025/eval-run4-priv-saolam-notel-c128/step-*"),
    ("run5", "deepseek", "tau2-deepseek-eval-run45-20261025/eval-run5-std-saolam-notel-c128/step-*"),
    ("step0", "qwen", "tau2-qwenuser-eval-20261025/eval/step-000"),
    ("run3p", "qwen", "tau2-qwenuser-eval-20261025/eval/step-[0-9]*"),
    ("run2p_s42", "qwen", "tau2-qwenuser-eval-20261025/eval/ckpt-run2p-*"),
    ("run2p_s43", "qwen", "tau2-qwenuser-eval-run45-20261025/eval-run2p-priv-beta1-notel-c128-s43/step-*"),
    ("run4", "qwen", "tau2-qwenuser-eval-run45-20261025/eval-run4-priv-saolam-notel-c128/step-*"),
    ("run5", "qwen", "tau2-qwenuser-eval-run45-20261025/eval-run5-std-saolam-notel-c128/step-*"),
]

rows = []
for run, evaluator, pattern in EVALS:
    for d in sorted(glob.glob(os.path.join(root, pattern))):
        step = int(d.rsplit("-", 1)[1])
        if run == "run3p" and step == 0:
            continue  # step 0 is reported once, as "step0"
        attempts = sorted(glob.glob(os.path.join(d, "attempt-*", "summary.json")))
        per_domain = {}
        for a in attempts:
            for dom, v in json.load(open(a))["by_domain"].items():
                per_domain.setdefault(dom, []).append((v["pass1_mean"], v["episodes"]))
        for dom, vals in sorted(per_domain.items()):
            mean = sum(p for p, _ in vals) / len(vals)
            rows.append([evaluator, run, step, dom, round(100 * mean, 1), len(vals), vals[0][1]])

with open(os.path.join(out, "eval_mean4.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["evaluator", "run", "step", "domain", "mean4_pct", "attempts_averaged", "episodes_per_attempt"])
    w.writerows(rows)

RUNS = {
    "run3p": "run3p-std-beta1-notel-c128",
    "run2p_s42": "run2p-priv-beta1-notel-c128",
    "run2p_s43": "run2p-priv-beta1-notel-c128-s43",
    "run4": "run4-priv-saolam-notel-c128",
    "run5": "run5-std-saolam-notel-c128",
}
KEYS = ["rollout/reward", "critic/ppo_critic/critic_explained_variance_pre_update"]
step_metrics = os.path.join(root, "tau2-ppo-qwenuser-20261021", "step_metrics.py")
with open(os.path.join(out, "train_per_step.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "step", "train_reward", "critic_ev_pre_update"])
    for label, trial in RUNS.items():
        log = os.path.join(root, "tau2-ppo-qwenuser-20261021", trial, "logs/root/tau2-ppo", trial, "main.log")
        tsv = subprocess.run(
            [sys.executable, "-I", step_metrics, log, *KEYS], capture_output=True, text=True, check=True
        ).stdout.splitlines()
        for line in tsv[1:]:
            step, _time, reward, ev = line.split("\t")
            w.writerow([label, step, reward, ev])
