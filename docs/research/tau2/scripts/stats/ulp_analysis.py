# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
# CPU-only: changed-weight |delta| in units of bf16 ulp; what lr=1e-6 Adam implies vs ulp.
import glob
import sys

import numpy as np

sys.path.insert(
    0,
    "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats",
)
from weight_diff import load, f32  # noqa (weight_diff main loop guarded below)

snap = sorted(
    glob.glob(
        "/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e8*/"
    )
)[0]
base = load(sorted(glob.glob(snap + "*.safetensors")))
norm = lambda k: k.replace("model.language_model.", "model.").replace(
    "language_model.model.", "model."
)
bmap = {norm(k): k for k in base}
B = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"


def ulp(w):  # bf16 spacing at |w|
    w = np.abs(w).astype(np.float32)
    e = np.floor(np.log2(np.maximum(w, 1e-30)))
    return np.exp2(e - 7)


rng = np.random.default_rng(0)
for run, sub in [
    (
        "grpo",
        "grpo-formal-v2/checkpoints/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2/default/",
    ),
    (
        "sao",
        "sao-mixed-v4-disk/checkpoints/root/tau2-sao/tau2-sao-nonthinking-mixed-v4-disk/default/",
    ),
]:
    c = load([B + sub + "epoch1epochstep22globalstep45/model.safetensors"])
    print(
        run,
        "ckpt dtypes",
        {t[0] for t in c.values()},
        "base dtypes",
        {t[0] for t in base.values()},
    )
    W = []
    D = []
    for k, t in c.items():
        bk = bmap.get(norm(k))
        if bk is None or base[bk][1] != t[1] or len(t[1]) < 2:
            continue  # 2-D matrices only
        y = f32(base[bk])
        x = f32(t)
        idx = rng.choice(y.size, size=min(y.size, 20000), replace=False)
        W.append(y[idx])
        D.append(x[idx] - y[idx])
    W = np.concatenate(W)
    D = np.concatenate(D)
    u = ulp(W)
    ch = D != 0
    r = np.abs(D[ch]) / u[ch]
    print(f"  sampled {W.size} matrix elements; changed {ch.mean():.4f}")
    print("  |w| percentiles 10/50/90:", np.percentile(np.abs(W), [10, 50, 90]))
    print("  bf16 ulp at median |w|:", np.median(u))
    print(
        "  changed: |dw|/ulp percentiles 50/90/99:",
        np.percentile(r, [50, 90, 99]),
        " frac exactly 1 ulp",
        np.mean(np.isclose(r, 1)),
    )
    print(
        "  changed: median |w| %.3g vs unchanged median |w| %.3g"
        % (np.median(np.abs(W[ch])), np.median(np.abs(W[~ch])))
    )
    for steps_bound, lab in [
        (1e-6, "one Adam step (~lr)"),
        (46e-6, "46 steps same sign (max)"),
        (np.sqrt(46) * 1e-6, "46 steps random-walk"),
    ]:
        print(
            f"  frac elements with half-ulp < {lab}={steps_bound:.2e}: {(u / 2 < steps_bound).mean():.4f}"
        )
