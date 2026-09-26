"""Recover an empty simulator response without replaying policy or tool actions."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from tau2.data_model.message import UserMessage
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import UserState, ValidUserInputMessage
from tau2.utils.llm_utils import to_litellm_messages

from areal.utils import logging

logger = logging.getLogger("Tau2Agent")


class EmptyUserResponseError(RuntimeError):
    """The simulator returned no text or tool calls after one retry."""


class RetryingUserSimulator(UserSimulator):
    def __init__(self, *, debug_dir: Path, task_id: str, domain: str, **kwargs):
        super().__init__(**kwargs)
        self.debug_dir = debug_dir
        self.task_id = task_id
        self.domain = domain

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> UserMessage:
        history_size = len(state.messages)
        incident_id = uuid4().hex
        for attempt in range(2):
            # Official generation appends the incoming message before calling the
            # API. Roll it back on failure so the retry has exactly the same input.
            try:
                response = super()._generate_next_message(message, state)
            except Exception as exc:
                self._record(state, incident_id, attempt, error=exc)
                del state.messages[history_size:]
                raise
            valid = response.has_content() or response.is_tool_call()
            if not valid or attempt:
                self._record(state, incident_id, attempt, response=response)
            if valid:
                return response
            del state.messages[history_size:]
            logger.warning(
                "Empty user response for %s/%s, attempt %s/2; evidence %s/%s-%s.json",
                self.domain,
                self.task_id,
                attempt + 1,
                self.debug_dir,
                incident_id,
                attempt,
            )
        raise EmptyUserResponseError(
            f"Empty user response after 2 attempts for {self.domain}/{self.task_id}; "
            f"evidence: {self.debug_dir}/{incident_id}-*.json"
        )

    def _record(
        self,
        state: UserState,
        incident_id: str,
        attempt: int,
        *,
        response: UserMessage | None = None,
        error: Exception | None = None,
    ) -> None:
        try:
            # Keep full conversation and raw response, including reasoning and IDs.
            # Authentication is transport configuration, not part of the request body.
            args = {
                k: v
                for k, v in self.llm_args.items()
                if k not in {"api_key", "extra_headers", "headers"}
            }
            record = {
                "recorded_at": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "domain": self.domain,
                "task_id": self.task_id,
                "attempt": attempt,
                "request": {
                    "model": self.llm,
                    "messages": to_litellm_messages(
                        state.system_messages + state.flip_roles()
                    ),
                    "tools": [tool.openai_schema for tool in self.tools]
                    if self.tools
                    else None,
                    "tool_choice": "auto" if self.tools else None,
                    "args": args,
                },
                "response": response.raw_data if response is not None else None,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "body": getattr(error, "body", None),
                }
                if error
                else None,
            }
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            path = self.debug_dir / f"{incident_id}-{attempt}.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, default=str)
        except Exception:
            logger.warning(
                "Could not write user-simulator incident %s/%s",
                self.debug_dir,
                incident_id,
                exc_info=True,
            )
