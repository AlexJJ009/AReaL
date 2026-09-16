# SPDX-License-Identifier: Apache-2.0
"""Experiment proxy with per-call token snapshots for live RL inspection."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from request_audit import AuditWriter, RequestAudit, request_context, wrap_generate


def write_snapshot(root, name, payload):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = root / (name + ".tmp")
    destination = root / (name + ".json")
    with temporary.open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(payload, stream)
        stream.write("\n")
    temporary.replace(destination)


def wrap_token_capture(original, root):
    async def generate(self, req):
        name = f"{os.getpid()}-{uuid.uuid4().hex}"
        common = {"audit_id": request_context.get(), "request_id": req.rid}
        # ModelRequest uses CPU lists. Do not copy metadata, headers, or keys.
        await asyncio.to_thread(
            write_snapshot,
            root,
            name + "-input",
            {**common, "input_ids": list(req.input_ids)},
        )
        result = await original(self, req)
        await asyncio.to_thread(
            write_snapshot,
            root,
            name + "-output",
            {
                **common,
                "output_tokens": list(result.output_tokens),
                "output_logprobs": list(result.output_logprobs),
                "output_versions": list(result.output_versions),
                "stop_reason": result.stop_reason,
            },
        )
        return result

    return generate


def main():
    from qwen_template_defaults import wrap_create

    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.experimental.openai.client import AsyncCompletionsWithReward
    from areal.experimental.openai.proxy import proxy_rollout_server as native

    AsyncCompletionsWithReward.create = wrap_create(AsyncCompletionsWithReward.create)
    root = Path(os.environ["QWEN_ARENA_RAW_AUDIT_DIR"]).parent
    from qwen_wire_audit import install_wire_audit

    install_wire_audit(root / "wire-anomalies", write_snapshot)
    writer = AuditWriter(root / "request-audit")
    native.app.add_middleware(RequestAudit, writer=writer)
    captured = wrap_token_capture(
        RemoteSGLangEngine.agenerate, root / "live-generations"
    )
    RemoteSGLangEngine.agenerate = wrap_generate(captured, writer)
    writer.write("proxy_audit_installed", per_call_tokens=True, raw_wire_capture=True)
    native.main()


if __name__ == "__main__":
    main()
