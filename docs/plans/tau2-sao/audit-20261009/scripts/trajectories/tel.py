import json
import re
import sys

P = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/"
DATA = "/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619/data/tau2/domains/telecom/tasks.json"
f, ln, AL = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = json.loads(open(P + f).readlines()[ln - 1])
t = [x for x in json.load(open(DATA)) if x["id"] == r["task_id"]][0]
us = t["user_scenario"]["instructions"]
print("TASK", r["task_id"], "reward", r["reward"])
print("REASON", us["reason_for_call"][:400])
print("TI", us["task_instructions"][:600])
print("KNOWN", us.get("known_info"), "UNK", us.get("unknown_info"))
p = r["segments"][0]["text"]
print("OPEN USER:", p.split("<|im_start|>user\n")[-1][:400])
for k, s in enumerate(r["segments"][1:], 1):
    x = re.sub(r"<think>\s*</think>\s*", "", s["text"]).replace("<|im_end|>", "")
    if s["role"] == "gen":
        calls = re.findall(r"<function=(\w+)>", x)
        txt = re.sub(r"<tool_call>.*?</tool_call>", "", x, flags=re.S).strip()
        print(
            f"[{k}] AGENT:",
            txt[:AL].replace("\n", " "),
            ("CALLS:" + str(calls)) if calls else "",
        )
    else:
        for pp in x.split("<|im_start|>"):
            pp = pp.strip()
            if pp.startswith("user\n<tool_response>"):
                print(f"[{k}] TOOL:", pp[19:140].replace("\n", " "))
            elif pp.startswith("user"):
                print(f"[{k}] USER:", pp[5:900].replace("\n", " "))
