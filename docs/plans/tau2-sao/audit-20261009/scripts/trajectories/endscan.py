# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import collections
import glob
import json
import re

ROOT = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
srcs = {
    "grpo": ROOT
    + "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/",
    "sao": ROOT
    + "sao-mixed-v1/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v1-dedicated-eval/eval-rollout/0/",
}
c = collections.Counter()
ex = collections.defaultdict(list)
for s, p in srcs.items():
    for f in glob.glob(p + "*.jsonl"):
        for i, l in enumerate(open(f), 1):
            r = json.loads(l)
            if r["domain"] == "telecom":
                continue
            last = r["segments"][-1]
            q = (
                last["role"] == "gen"
                and "<tool_call>" not in last["text"]
                and re.search(
                    r'confirm|proceed|\(yes\)|"yes"', last["text"][-400:], re.I
                )
                is not None
            )
            k = (
                r["domain"],
                r["reward"],
                "ends_on_agent_confirm_request" if q else "other",
            )
            c[k] += 1
            if q and r["reward"] == 0:
                ex[r["domain"]].append(f"{s}:{f.split('/')[-1]}:{i}:t{r['task_id']}")
for k in sorted(c):
    print(k, c[k])
for d, v in ex.items():
    print(d, len(v), v[:40])
