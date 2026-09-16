# SPDX-License-Identifier: Apache-2.0
"""Bounded, asynchronous capture of anomalous SGLang wire responses."""

import copy
import hashlib
import json
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from areal.utils.logging import getLogger


def wrap_wire_capture(original, root, write_snapshot):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wire-audit")
    slots = threading.BoundedSemaphore(16)
    logger = getLogger("QwenWireAudit")
    calls = 0

    def persist(response, values, reason):
        ids = [row[1] for row in response["meta_info"]["output_token_logprobs"]]
        write_snapshot(
            root,
            uuid.uuid4().hex,
            {
                "scope": "SGLang wire response; first-call coverage or anomaly screen, not acceptance",
                "capture_reason": reason,
                "output_ids_sha256": hashlib.sha256(
                    json.dumps(ids).encode()
                ).hexdigest(),
                "mean_output_logp": sum(values) / len(values) if values else None,
                "response": response,
            },
        )

    def finished(future):
        slots.release()
        if future.exception() is not None:
            logger.error(
                "Failed to persist anomalous wire response: %s", future.exception()
            )

    def parse(self, response):
        nonlocal calls
        # Preserve the native parser and its exception semantics.
        result = original(self, response)
        calls += 1
        values = [
            row[0]
            for row in response.get("meta_info", {}).get("output_token_logprobs", [])
            if row[0] is not None
        ]
        anomalous = values and (
            not all(math.isfinite(v) for v in values)
            or (len(values) >= 32 and sum(values) / len(values) < -3)
        )
        if anomalous or calls == 1:
            if slots.acquire(blocking=False):
                try:
                    future = executor.submit(
                        persist,
                        copy.deepcopy(response),
                        values,
                        "anomalous_logp" if anomalous else "first_response",
                    )
                except Exception as exc:
                    slots.release()
                    logger.error("Could not queue anomaly evidence: %s", exc)
                else:
                    future.add_done_callback(finished)
            else:
                logger.error("Wire audit queue full; anomaly evidence dropped")
        return result

    return parse, executor


def install_wire_audit(root, write_snapshot):
    from areal.engine.sglang_remote import SGLangBackend

    original = SGLangBackend.parse_generation_response
    if getattr(original, "_qwen_wire_audit", False):
        return
    wrapped, _ = wrap_wire_capture(original, root, write_snapshot)
    wrapped._qwen_wire_audit = True
    SGLangBackend.parse_generation_response = wrapped
