# SPDX-License-Identifier: Apache-2.0
"""Metadata-only ASGI and engine audit for the owned frozen Arena experiment."""

import asyncio
import contextvars
import json
import os
import time
import uuid
from pathlib import Path

request_context = contextvars.ContextVar("qwen_request_audit", default=None)


class AuditWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / ("request-events-" + str(os.getpid()) + ".jsonl")

    def write(self, event, **fields):
        row = {"event": event, "utc_seconds": time.time(), "pid": os.getpid(), **fields}
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as stream:
            stream.write(json.dumps(row) + "\n")

    async def emit(self, event, **fields):
        await asyncio.to_thread(self.write, event, **fields)


class RequestAudit:
    def __init__(self, app, writer):
        self.app = app
        self.writer = writer

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        audit_id = uuid.uuid4().hex
        token = request_context.set(audit_id)
        started = time.monotonic()
        # Never record headers, query strings, body content or exception messages.
        await self.writer.emit(
            "http_received",
            audit_id=audit_id,
            method=scope.get("method"),
            path=scope.get("path"),
        )
        first_body = False

        async def audited_receive():
            event = await receive()
            if event["type"] == "http.disconnect":
                await self.writer.emit("http_disconnect", audit_id=audit_id)
            return event

        async def audited_send(event):
            nonlocal first_body
            await send(event)
            if event["type"] == "http.response.start":
                await self.writer.emit(
                    "http_response_start", audit_id=audit_id, status=event["status"]
                )
            elif event["type"] == "http.response.body" and not first_body:
                first_body = True
                await self.writer.emit(
                    "http_first_body",
                    audit_id=audit_id,
                    bytes=len(event.get("body", b"")),
                )

        try:
            return await self.app(scope, audited_receive, audited_send)
        except BaseException as exc:
            await self.writer.emit(
                "http_exception", audit_id=audit_id, exception_type=type(exc).__name__
            )
            raise
        finally:
            try:
                await self.writer.emit(
                    "http_finished",
                    audit_id=audit_id,
                    elapsed_seconds=time.monotonic() - started,
                )
            finally:
                request_context.reset(token)


def wrap_generate(original, writer):
    async def generate(self, req):
        audit_id = request_context.get()
        generation_id = uuid.uuid4().hex
        started = time.monotonic()
        await writer.emit(
            "engine_generate_enter",
            audit_id=audit_id,
            generation_id=generation_id,
            input_tokens=len(req.input_ids),
        )
        try:
            result = await original(self, req)
            await writer.emit(
                "engine_generate_return",
                audit_id=audit_id,
                generation_id=generation_id,
                elapsed_seconds=time.monotonic() - started,
            )
            return result
        except BaseException as exc:
            await writer.emit(
                "engine_generate_exception",
                audit_id=audit_id,
                generation_id=generation_id,
                exception_type=type(exc).__name__,
            )
            raise

    return generate
