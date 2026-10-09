import json
import sys

ROOT = "/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/"
srcs = {
    "grpo": ROOT
    + "grpo-formal-v2/logs/root/tau2-grpo/tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/",
    "sao": ROOT
    + "sao-mixed-v1/logs/root/tau2-sao/tau2-sao-nonthinking-mixed-v1-dedicated-eval/eval-rollout/0/",
}
DATA = "/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619/data/tau2/domains/"
for spec in sys.argv[1:]:
    s, f, ln = spec.split(":")
    r = json.loads(open(srcs[s] + f).readlines()[int(ln) - 1])
    t = {x["id"]: x for x in json.load(open(DATA + r["domain"] + "/tasks.json"))}[
        r["task_id"]
    ]
    print("####", spec, r["domain"], r["task_id"], r["reward"])
    print(
        "REASON:",
        t["user_scenario"]["instructions"]["reason_for_call"][:500].replace("\n", " "),
    )
    print(
        "TI:",
        t["user_scenario"]["instructions"]["task_instructions"][:400].replace(
            "\n", " "
        ),
    )
    print(
        "GOLD:",
        [
            (a["name"], a["arguments"])
            for a in t["evaluation_criteria"]["actions"]
            if not a["name"].startswith(("get", "find", "list", "calc"))
        ],
    )
    S = r["segments"]
    users = [
        x
        for x in S
        if x["role"] != "gen"
        and "<|im_start|>user\n" in x["text"]
        and "<tool_response>" not in x["text"]
    ]
    if users:
        print(
            "LAST USER:",
            users[-1]["text"].split("<|im_start|>user\n")[-1][:600].replace("\n", " "),
        )
    print("LAST AGENT:", S[-1]["text"][-500:].replace("\n", " "))
