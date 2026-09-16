# SPDX-License-Identifier: Apache-2.0
"""New-platform Claude Code routing with raw AReaL trajectory audit."""

import os

from qwen_arena_audit import AuditedArenaWorkflow


class ClaudeArenaWorkflow(AuditedArenaWorkflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.llm_route_mode == "gateway":
            self.client.llm_api_key = self.client.api_token
            original_target = self.client._registered_llm_target

            def registered_target(registration, model_name):
                _, alias = original_target(registration, model_name)
                return os.environ["QWEN_ARENA_LLM_BASE"], alias

            self.client._registered_llm_target = registered_target
