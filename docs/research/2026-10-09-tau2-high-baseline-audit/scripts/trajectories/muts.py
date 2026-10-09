# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import json
import re
import sys

P = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
S = {
    "grpo": "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/",
    "sao": "sao-mixed-v1/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v1-dedicated-eval/eval-rollout/0/",
}
f = sys.argv[1]
for s, src in S.items():
    for i, l in enumerate(open(P + src + f), 1):
        r = json.loads(l)
        t = "".join(x["text"] for x in r["segments"] if x["role"] == "gen")
        calls = re.findall(r"<function=(\w+)>(.*?)</function>", t, re.S)
        mut = [
            (n, re.sub(r"\s+", "", re.sub(r"</?parameter=?\w*>", " ", a))[:110])
            for n, a in calls
            if not n.startswith(("get", "find", "list", "calc", "search", "think"))
        ]
        print(s, i, r["task_id"], r["reward"], len(r["segments"]), mut)
