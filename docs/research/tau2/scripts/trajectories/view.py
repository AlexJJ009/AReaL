# ruff: noqa: E731, E741, UP031  # archived audit script, kept as run
# usage: view.py <src grpo|sao> <file.jsonl> <line>   -> prints task spec + dialogue (tool responses truncated)
import json
import re
import sys

ROOT = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
srcs = {
    "grpo": ROOT
    + "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/",
    "sao": ROOT
    + "sao-mixed-v1/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v1-dedicated-eval/eval-rollout/0/",
}
DATA = "/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619/data/tau2/domains/"
s, f, ln = sys.argv[1], sys.argv[2], int(sys.argv[3])
TL = int(sys.argv[4]) if len(sys.argv) > 4 else 400
AL = int(sys.argv[5]) if len(sys.argv) > 5 else 900
r = json.loads(open(srcs[s] + f).readlines()[ln - 1])
t = {x["id"]: x for x in json.load(open(DATA + r["domain"] + "/tasks.json"))}[
    r["task_id"]
]
print("==", s, f, ln, r["domain"], "task", r["task_id"], "reward", r["reward"])
print("USER SCENARIO:", json.dumps(t["user_scenario"], indent=1))
ec = t["evaluation_criteria"]
print("GOLD ACTIONS:")
[print("  ", a["name"], json.dumps(a["arguments"])) for a in (ec.get("actions") or [])]
print(
    "COMM:",
    ec.get("communicate_info"),
    "\nNL:",
    ec.get("nl_assertions"),
    "\nBASIS:",
    ec.get("reward_basis"),
)
print("-----DIALOGUE")
p = r["segments"][0]["text"]
i = p.rfind("</policy>")
print(re.sub(r"<think>\s*</think>\s*", "", p[i:]))
for k, seg in enumerate(r["segments"][1:], 1):
    txt = (
        re.sub(r"<think>\s*</think>\s*", "", seg["text"])
        .replace("<|im_end|>", "")
        .replace("<|im_start|>assistant", "")
    )
    if seg["role"] == "gen":
        a = txt.strip()
        print(
            f"[{k}] AGENT:",
            a if len(a) < AL else a[:AL] + " ...[+%d] " % (len(a) - AL) + a[-300:],
        )
    else:
        parts = re.split(r"<\|im_start\|>", txt)
        for pp in parts:
            pp = pp.strip()
            if not pp:
                continue
            if pp.startswith("user\n<tool_response>"):
                print(f"[{k}] TOOL:", pp[5 : 5 + TL].replace("\n", " "))
            else:
                print(f"[{k}]", pp.upper()[:4], ":", pp[5:].strip()[:1500])
