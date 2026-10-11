# SPDX-License-Identifier: Apache-2.0
"""Reproduce the proxy capacity leak that hung two fixed-v2 eval episodes.

Runs the real ``proxy_rollout_server`` app and the real client helpers. Two
episodes each grant one unit of capacity and start a session, as
``OpenAIProxyWorkflow._arun_episode`` does. Episode A's ``start_session``
request is already being handled when the handler has to wait longer than the
client's 10 s timeout; here the wait is injected by holding the server's
session lock. The client gives up and retries, but the server still finishes
the original request: it creates session ``A-0`` that nobody uses, and A's
retry takes a second unit of capacity. ``start_session`` retries HTTP 429
without limit, so when episode B arrives, one of the two retries forever.

Usage, from the repository root:
  PYTHONPATH=. python <this file> [--block-seconds 12] [--wait-seconds 40]
Exit code 0 means the leak reproduced.
"""

import argparse
import asyncio
import socket
import threading
import time

import aiohttp
import uvicorn

from areal.experimental.openai.proxy import proxy_rollout_server as server
from areal.experimental.openai.proxy.client_session import _start_session
from areal.experimental.openai.proxy.server import (
    GRANT_CAPACITY_PATHNAME,
    RL_START_SESSION_PATHNAME,
    StartSessionRequest,
)

ADMIN = "repro-admin-key"


def _serve(port: int) -> uvicorn.Server:
    server._admin_api_key = ADMIN
    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error")
    uv = uvicorn.Server(config)
    threading.Thread(target=uv.run, daemon=True).start()
    while not uv.started:
        time.sleep(0.05)
    return uv


async def main(block_seconds: float, wait_seconds: float) -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    uv = _serve(port)
    base = f"http://127.0.0.1:{port}/"
    headers = {"Authorization": f"Bearer {ADMIN}"}

    async with aiohttp.ClientSession() as http:

        async def grant():
            async with http.post(base + GRANT_CAPACITY_PATHNAME, headers=headers) as r:
                r.raise_for_status()

        async def start(task_id: str):
            t0 = time.monotonic()
            data = await _start_session(
                http,
                url=base + RL_START_SESSION_PATHNAME,
                payload=StartSessionRequest(task_id=task_id),
                headers=headers,
            )
            return data["session_id"], time.monotonic() - t0

        # Episode A: grant, then start while the session lock is held.
        await grant()
        server._lock.acquire()
        start_a = asyncio.create_task(start("A"))
        await asyncio.sleep(block_seconds)
        server._lock.release()
        await asyncio.sleep(1)
        # Episode B arrives after the blocker, as later episodes do.
        await grant()
        start_b = asyncio.create_task(start("B"))
        done, pending = await asyncio.wait({start_a, start_b}, timeout=wait_seconds)

        sessions = sorted(server._session_cache)
        print(f"capacity after both starts: {server._capacity}")
        print(f"sessions on the server: {sessions}")
        for name, task in (("A", start_a), ("B", start_b)):
            if task in done:
                sid, took = task.result()
                print(f"episode {name}: started session {sid} after {took:.1f} s")
            else:
                print(
                    f"episode {name}: still retrying on HTTP 429 after {wait_seconds:.0f} s"
                )
        for task in pending:
            task.cancel()
    uv.should_exit = True

    leaked = len(pending) == 1 and "A-0" in sessions and len(sessions) == 2
    print("REPRODUCED" if leaked else "NOT REPRODUCED")
    return 0 if leaked else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-seconds", type=float, default=12.0)
    parser.add_argument("--wait-seconds", type=float, default=40.0)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.block_seconds, args.wait_seconds)))
