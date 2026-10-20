"""Record failed user-simulator calls before the episode attempt fails."""

import json
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import tau2.user.user_simulator as simulator_module
from tau2.data_model.message import UserMessage
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import UserState, ValidUserInputMessage
from tau2.utils.llm_utils import to_litellm_messages

from examples.tau2.api_reliability import bounded_call

from areal.utils import logging

logger = logging.getLogger("Tau2Agent")
_RETRY_CONFIG = ContextVar("tau2_user_retry_config", default=None)
_OFFICIAL_GENERATE = simulator_module.generate


def _bounded_user_generate(*args, **kwargs):
    config = _RETRY_CONFIG.get()
    if config is None:
        return _OFFICIAL_GENERATE(*args, **kwargs)
    return bounded_call(_OFFICIAL_GENERATE, *args, config=config, **kwargs)


# Only the simulator binding is replaced; policy generation stays untouched.
simulator_module.generate = _bounded_user_generate


class EmptyUserResponseError(ValueError):
    """The simulator returned neither text nor tool calls."""


class RecordingUserSimulator(UserSimulator):
    """Official user simulator that keeps evidence of failed calls.

    Provider retries wrap generate, after the official simulator updates state
    once. Exhausted failures end the attempt without duplicating history.
    """

    def __init__(
        self, *, debug_dir: Path, task_id: str, domain: str, retry_config=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.retry_config = retry_config
        self.debug_dir = debug_dir
        self.task_id = task_id
        self.domain = domain
        self.calls: list[dict] = []

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> UserMessage:
        call = {
            "model": self.llm,
            "base_url_host": urlsplit(self.llm_args.get("api_base") or "").hostname,
            "status": "started",
        }
        self.calls.append(call)
        token = _RETRY_CONFIG.set(self.retry_config)
        try:
            response = super()._generate_next_message(message, state)
        except Exception as exc:
            call.update(status="error", error_type=type(exc).__name__)
            self._record(state, error=exc)
            raise
        finally:
            _RETRY_CONFIG.reset(token)
        if not (response.has_content() or response.is_tool_call()):
            call.update(status="error", error_type="EmptyUserResponseError")
            path = self._record(state, response=response)
            # Official `UserMessage.validate()` rejects the same message one step
            # later; failing here keeps the raw provider response with the error.
            raise EmptyUserResponseError(
                f"Empty user response for {self.domain}/{self.task_id}; "
                f"evidence: {path}"
            )
        call["status"] = "completed"
        return response

    def _record(
        self,
        state: UserState,
        *,
        response: UserMessage | None = None,
        error: Exception | None = None,
    ) -> Path:
        path = self.debug_dir / f"{uuid4().hex}.json"
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
                    **(
                        {"doc": error.doc, "pos": error.pos}
                        if isinstance(error, json.JSONDecodeError)
                        else {}
                    ),
                }
                if error
                else None,
            }
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2, default=str)
        except Exception:
            logger.warning(
                "Could not write user-simulator incident %s", path, exc_info=True
            )
        return path
