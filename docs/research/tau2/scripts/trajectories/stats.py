# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import collections
import glob
import json

ROOT = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
srcs = {
    "grpo": ROOT
    + "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/",
    "sao": ROOT
    + "sao-mixed-v1/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v1-dedicated-eval/eval-rollout/0/",
}
st = collections.defaultdict(list)
for s, p in srcs.items():
    for f in glob.glob(p + "*.jsonl"):
        for i, l in enumerate(open(f), 1):
            r = json.loads(l)
            st[(r["domain"], r["task_id"])].append(
                (s, f.split("/")[-1], i, r["reward"])
            )
for k in sorted(st, key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else 0, k[1])):
    v = st[k]
    print(k, len(v), sum(x[3] for x in v) / len(v))
