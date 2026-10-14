# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import json
import re

B = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
paths = {
    "grpo": B
    + "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2/main.log",
    "sao": B
    + "sao-mixed-v4-disk/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v4-disk/main.log",
}
ansi = re.compile(r"\x1b\[[0-9;]*m")
cell = re.compile(
    r"│\s*([A-Za-z0-9_/\.\-:]+)\s*│\s*(-?[0-9.]+(?:e[+-]?\d+)?|nan|inf|-inf)\s*(?=│)"
)
done = re.compile(r"Train step (\d+)/(\d+) done")
out = {}
for k, p in paths.items():
    steps = []
    cur = {}
    for i, l in enumerate(open(p, errors="replace")):
        l = ansi.sub("", l)
        m = done.search(l)
        if m:
            if cur:
                steps.append(cur)  # table printed before or after? handle below
            cur = {"__step": int(m.group(1)), "__line": i + 1}
            continue
        for name, val in cell.findall(l):
            cur[name] = float(val)
    if cur:
        steps.append(cur)
    out[k] = steps
    print(k, len(steps), [s["__step"] for s in steps][:60])
json.dump(
    out,
    open(
        "/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/stats/mainlog_metrics.json",
        "w",
    ),
)
