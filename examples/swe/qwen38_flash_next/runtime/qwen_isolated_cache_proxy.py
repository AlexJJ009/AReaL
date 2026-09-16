# SPDX-License-Identifier: Apache-2.0
"""Diagnostic proxy: isolate prefix cache between generation requests."""

import copy
import os
import uuid
from pathlib import Path


def wrap_isolated_cache(original):
    def build(self, *args, **kwargs):
        native = original(self, *args, **kwargs)
        request = copy.copy(native)
        request.payload = dict(native.payload)
        request.payload["cache_salt"] = "qwen-cache-isolation-" + uuid.uuid4().hex
        return request

    return build


def main():
    import qwen_live_proxy

    from areal.engine.sglang_remote import SGLangBackend

    SGLangBackend.build_generation_request = wrap_isolated_cache(
        SGLangBackend.build_generation_request
    )
    root = Path(os.environ["QWEN_ARENA_RAW_AUDIT_DIR"]).parent
    qwen_live_proxy.write_snapshot(
        root / "cache-isolation",
        str(os.getpid()),
        {
            "policy": "unique cache_salt per generation request",
            "installed": True,
            "runtime_effect_verified": False,
        },
    )
    qwen_live_proxy.main()


if __name__ == "__main__":
    main()
