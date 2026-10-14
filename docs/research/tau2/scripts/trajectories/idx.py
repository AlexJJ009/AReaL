# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
import glob
import json

ROOT = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
srcs = {
    "grpo": ROOT
    + "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/",
    "sao": ROOT
    + "sao-mixed-v1/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v1-dedicated-eval/eval-rollout/0/",
}
out = open("index.tsv", "w")
for s, p in srcs.items():
    for f in sorted(glob.glob(p + "*.jsonl")):
        for i, l in enumerate(open(f), 1):
            r = json.loads(l)
            out.write(
                f"{r['domain']}\t{r['task_id']}\t{r['reward']}\t{s}\t{f.split('/')[-1]}\t{i}\t{len(r['segments'])}\n"
            )
