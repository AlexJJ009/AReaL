"""Print one episode compactly with its task spec.

Usage: python view.py RUN_DIR SOURCE_ID [--think N] [--data official|fixed]
--think N prints the first N characters of each turn's reasoning (from raw_data).
"""

import argparse
import glob
import json
import os

DATA = {
    "official": "/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619/data",
    "fixed": "/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v2/data",
}

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("source_id")
p.add_argument("--think", type=int, default=0)
p.add_argument("--data", default="fixed")
p.add_argument("--width", type=int, default=400)
a = p.parse_args()

files = glob.glob(
    os.path.join(
        glob.escape(os.path.join(a.run, "trajectories", a.source_id)), "*.json"
    )
)
recs = [json.load(open(f)) for f in files]
rec = max(
    [r for r in recs if r.get("status") == "completed"] or recs,
    key=lambda r: r["recorded_at"],
)
domain, task_id = rec["domain"], rec["task_id"]
tasks = json.load(open(f"{DATA[a.data]}/tau2/domains/{domain}/tasks.json"))
task = next(t for t in tasks if str(t["id"]) == str(task_id))
W = a.width

print(
    f"=== {a.source_id} score={rec.get('official_score')} stop={rec.get('stop_reason')} turns={rec.get('policy_turns')}"
)
us = task.get("user_scenario", {}).get("instructions", {})
print("TASK:", json.dumps(us, ensure_ascii=False)[:1200])
crit = task.get("evaluation_criteria") or {}
print("GOLD:", [(x["name"], x.get("arguments")) for x in crit.get("actions") or []])
print(
    "BASIS:",
    crit.get("reward_basis"),
    "| NL:",
    crit.get("nl_assertions"),
    "| COMM:",
    crit.get("communicate_info"),
)
ri = rec.get("reward_info") or {}
print(
    "RESULT:",
    ri.get("reward_breakdown"),
    "info:",
    {k: v for k, v in (ri.get("info") or {}).items() if v},
)
print("-" * 80)
for i, m in enumerate(rec["messages"]):
    role = m["role"][:4].upper()
    if m["role"] == "assistant" and a.think and m.get("raw_data"):
        raw = ((m["raw_data"].get("choices") or [{}])[0].get("message") or {}).get(
            "content"
        ) or ""
        if "</think>" in raw:
            print(f"[{i}] THINK: {raw.rsplit('</think>', 1)[0][: a.think]!r}")
    if m.get("content"):
        flag = " (ERROR)" if m.get("error") else ""
        print(f"[{i}] {role}{flag}: {m['content'][:W]}")
    for c in m.get("tool_calls") or []:
        print(
            f"[{i}] {role} CALL {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)[:W]})"
        )
