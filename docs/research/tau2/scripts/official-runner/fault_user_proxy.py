# SPDX-License-Identifier: Apache-2.0
"""Fault-injection proxy in front of the user-simulator provider (smoke test only).

Forwards every /chat/completions request to UPSTREAM unchanged, except the user
simulator's first turn (system prompt + the agent greeting) of scripted tasks:

  empty_once   first-turn reply is empty on the first attempt only
  stop         first-turn reply is "###STOP###"
  empty_always first-turn reply is empty on every attempt

Every decision is appended to LOG as one JSON line.

Usage: python fault_user_proxy.py PORT LOG_JSONL UPSTREAM_BASE_URL
"""

import json
import sys
import time

import aiohttp
from aiohttp import web

SCENARIOS = {
    "return the digital camera": "empty_once",  # retail:51 -> S1
    "card has only $950": "stop",  # retail:38 -> S2
    "return an item you just received: a laptop": "empty_always",  # retail:69 -> S3
}
LOG = sys.argv[2]
UPSTREAM = sys.argv[3].rstrip("/")
first_turns: dict[str, int] = {}


def _log(**fields):
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.time(), **fields}) + "\n")


def _fake(body, content):
    return {
        "id": f"fault-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "fault"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _scenario(body):
    messages = body.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        return None, None
    system = str(messages[0].get("content"))
    if "<scenario>" not in system or len(messages) != 2:
        return None, None
    for needle, scenario in SCENARIOS.items():
        if needle in system:
            return needle, scenario
    return None, None


async def handle(request):
    raw = await request.read()
    body = json.loads(raw)
    needle, scenario = _scenario(body)
    if scenario is not None:
        first_turns[needle] = first_turns.get(needle, 0) + 1
        n = first_turns[needle]
        inject = scenario in ("stop", "empty_always") or (
            scenario == "empty_once" and n == 1
        )
        _log(scenario=scenario, first_turn=n, injected=inject)
        if inject:
            content = "###STOP###" if scenario == "stop" else ""
            return web.json_response(_fake(body, content))
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in ("authorization", "content-type")
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{UPSTREAM}/chat/completions", data=raw, headers=headers
        ) as resp:
            data = await resp.read()
            return web.Response(
                body=data, status=resp.status, content_type="application/json"
            )


app = web.Application(client_max_size=64 * 1024**2)
app.router.add_post("/chat/completions", handle)
app.router.add_post("/v1/chat/completions", handle)
web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]))
