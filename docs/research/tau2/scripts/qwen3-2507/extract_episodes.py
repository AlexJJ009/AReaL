"""Per-episode features from an offline tau2 eval run, for failure analysis.

Usage: python extract_episodes.py RUN_DIR [RUN_DIR ...] > features.jsonl
Each run directory holds episodes.jsonl and trajectories/<source_id>/<attempt>.json.
The kept attempt is the newest completed one; excluded episodes get status "excluded".
"""

import glob
import json
import os
import sys

TRANSFER = "transfer_to_human_agents"


def kept_attempt(run, source_id):
    files = glob.glob(
        os.path.join(
            glob.escape(os.path.join(run, "trajectories", source_id.replace("/", "_"))),
            "*.json",
        )
    )
    records = [json.load(open(f)) for f in files]
    done = [r for r in records if r.get("status") == "completed"]
    pool = done or records
    return max(pool, key=lambda r: r.get("recorded_at", 0)) if pool else None, len(
        records
    )


def features(run, episode):
    rec, attempts = kept_attempt(run, episode["source_id"])
    row = {
        "run": os.path.basename(run.rstrip("/")),
        "source_id": episode["source_id"],
        "domain": episode["domain"],
        "split": episode.get("split"),
        "task_id": episode["task_id"],
        "trial": episode.get("trial"),
        "status": "completed" if episode.get("status") == "completed" else "excluded",
        "score": episode.get("official_score"),
        "attempts": attempts,
    }
    if rec is None:
        return row
    if row["status"] == "excluded":
        row["error"] = (rec.get("error") or "")[:300]
    msgs = rec.get("messages") or []
    asst = [m for m in msgs if m["role"] == "assistant" and m.get("raw_data")]
    tools = [m for m in msgs if m["role"] == "tool"]
    calls = [
        c for m in msgs if m["role"] == "assistant" for c in (m.get("tool_calls") or [])
    ]
    user_calls = [
        c for m in msgs if m["role"] == "user" for c in (m.get("tool_calls") or [])
    ]
    completion = [
        int((m.get("usage") or {}).get("completion_tokens") or 0) for m in asst
    ]
    finish = [
        ((m["raw_data"].get("choices") or [{}])[0].get("finish_reason")) for m in asst
    ]
    reasoning_chars = []
    for m in asst:
        raw = ((m["raw_data"].get("choices") or [{}])[0].get("message") or {}).get(
            "content"
        ) or ""
        if "</think>" in raw:
            reasoning_chars.append(len(raw.rsplit("</think>", 1)[0]))
    ri = rec.get("reward_info") or {}
    actions = ri.get("action_checks") or []
    gold_writes = [a for a in actions if a.get("tool_type") == "write"]
    missed = [a for a in actions if not a.get("action_match")]
    gold_write_names = {a["action"]["name"] for a in gold_writes}
    agent_write_names = [
        c["name"]
        for c in calls
        if not c["name"].startswith(
            ("get_", "find_", "list_", "calculate", "search", "check_")
        )
        and c["name"] != TRANSFER
    ]
    env = ri.get("env_assertions") or []
    info = ri.get("info") or {}
    last_user = next(
        (m.get("content") for m in reversed(msgs) if m["role"] == "user"), None
    )
    row.update(
        stop_reason=rec.get("stop_reason"),
        n_messages=len(msgs),
        policy_turns=rec.get("policy_turns"),
        user_turns=sum(m["role"] == "user" for m in msgs),
        tool_calls=len(calls),
        user_tool_calls=len(user_calls),
        tool_errors=sum(bool(m.get("error")) for m in tools),
        empty_agent_msgs=sum(
            1
            for m in msgs
            if m["role"] == "assistant"
            and not m.get("content")
            and not m.get("tool_calls")
        ),
        completion_tokens_sum=sum(completion),
        completion_tokens_max=max(completion, default=0),
        finish_length=sum(f == "length" for f in finish),
        reasoning_chars_mean=(sum(reasoning_chars) / len(reasoning_chars))
        if reasoning_chars
        else None,
        reward_basis=ri.get("reward_basis"),
        breakdown=ri.get("reward_breakdown"),
        db_match=(ri.get("db_check") or {}).get("db_match"),
        n_gold_actions=len(actions),
        n_gold_writes=len(gold_writes),
        gold_writes_matched=sum(bool(a.get("action_match")) for a in gold_writes),
        first_missed_action=(
            {"name": missed[0]["action"]["name"], "type": missed[0].get("tool_type")}
            if missed
            else None
        ),
        agent_writes=agent_write_names,
        extra_write_names=sorted(set(agent_write_names) - gold_write_names),
        transferred=any(c["name"] == TRANSFER for c in calls),
        gold_transfer=any(a["action"]["name"] == TRANSFER for a in actions),
        transfer_flag=(
            "missing_transfer"
            if info.get("missing_transfer")
            else "unexpected_transfer"
            if info.get("unexpected_transfer")
            else None
        ),
        official_reward_before_rule=info.get("official_reward"),
        env_failed=[
            e.get("env_assertion", {}).get("func_name") for e in env if not e.get("met")
        ],
        n_env=len(env),
        communicate_ok=(
            all(c.get("met") for c in ri["communicate_checks"])
            if ri.get("communicate_checks")
            else None
        ),
        nl_ok=(
            all(c.get("met") for c in ri["nl_assertions"])
            if ri.get("nl_assertions")
            else None
        ),
        last_user=(last_user or "")[-200:],
    )
    return row


for run in sys.argv[1:]:
    for line in open(os.path.join(run, "episodes.jsonl")):
        ep = json.loads(line)
        print(json.dumps(features(run, ep), ensure_ascii=False))
