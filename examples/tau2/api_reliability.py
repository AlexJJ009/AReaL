"""Bounded provider retries and a run-wide quota circuit breaker for τ²."""

import ast
import fcntl
import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from areal.utils import logging
from areal.utils.rollout_error import QUOTA_EXIT_CODE, FatalRolloutError

if TYPE_CHECKING:
    from examples.tau2.utils import Tau2EnvConfig

logger = logging.getLogger("Tau2Agent")
_T = TypeVar("_T")
_LOCAL_LOCK = RLock()
_LOCAL_STATE: dict[str, Any] = {}
_QUOTA_CODES = {"AccountQuotaExceeded", "insufficient_quota"}
_QUOTA_MESSAGE = re.compile(
    r"(?:exceeded (?:the |your )?(?:(?:daily|weekly|monthly|total) )?usage quota"
    r"|(?:(?:daily|weekly|monthly|total) )?usage quota (?:has been |is )?exceeded"
    r"|(?:exhausted|used up) (?:the |your )?(?:(?:daily|weekly|monthly|total) )?usage quota)",
    re.IGNORECASE,
)


def classify_api_error(exc: BaseException) -> Literal["quota", "transient", "other"]:
    """Prefer structured provider codes, with a narrow usage-quota fallback.

    LiteLLM sometimes keeps the body only as a Python dict in the error string.
    Walk causes/contexts so τ² infrastructure wrappers keep the classification.
    A plain reference to quota, or a per-minute token/request quota, is insufficient.
    """
    seen: set[int] = set()
    transient = False
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        body = getattr(current, "body", None)
        response = getattr(current, "response", None)
        if body is None and response is not None:
            try:
                body = response.json()
            except (ValueError, AttributeError, RuntimeError):
                pass
        if not isinstance(body, dict):
            # Parse the actual body, never search arbitrary text for a code name.
            start = message.find("{")
            if start >= 0:
                try:
                    body = ast.literal_eval(message[start:])
                except (ValueError, SyntaxError):
                    try:
                        body, _ = json.JSONDecoder().raw_decode(message[start:])
                    except ValueError:
                        pass
        error = body.get("error", body) if isinstance(body, dict) else {}
        error = error if isinstance(error, dict) else {}
        code = error.get("code", getattr(current, "code", None))
        if code in _QUOTA_CODES:
            return "quota"
        provider_message = str(error.get("message") or message)
        if _QUOTA_MESSAGE.search(provider_message):
            return "quota"
        status = getattr(current, "status_code", None) or getattr(
            response, "status_code", None
        )
        transient |= (
            code == "AccountRateLimitExceeded"
            or status in (408, 429)
            or (isinstance(status, int) and 500 <= status < 600)
            or isinstance(current, (TimeoutError, ConnectionError))
            or any(
                cls.__name__
                in {
                    "APIConnectionError",
                    "APITimeoutError",
                    "Timeout",
                    "TimeoutException",
                    "ConnectionError",
                    "NetworkError",
                }
                for cls in type(current).__mro__
            )
        )
        current = current.__cause__ or current.__context__
    return "transient" if transient else "other"


class QuotaCircuitBreaker:
    """Share quota timestamps and counters across all run workers using flock.

    Training supplies a unique state path in the shared run directory, including
    on recovery. Standalone in-process evaluation uses the process-local state.
    State and its lock must live on the same shared filesystem on all workers.
    """

    def __init__(self, config: "Tau2EnvConfig"):
        self.config = config
        self.path = (
            Path(config._quota_breaker_state_path)
            if config._quota_breaker_state_path
            else None
        )

    @contextmanager
    def _state(self, *, write: bool = True) -> Iterator[dict[str, Any]]:
        if self.path is None:
            with _LOCAL_LOCK:
                yield _LOCAL_STATE
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {}
            try:
                yield state
            finally:
                if write:
                    temporary = self.path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state))
                    temporary.replace(self.path)
                fcntl.flock(lock, fcntl.LOCK_UN)

    def snapshot(self) -> dict[str, Any]:
        with self._state(write=False) as state:
            return {
                "quota_errors": 0,
                "transient_api_errors": 0,
                "rejected_episodes": 0,
                "fatal": None,
                **state,
            }

    def check(self) -> None:
        fatal = self.snapshot()["fatal"]
        if fatal:
            raise FatalRolloutError(fatal)

    def record_rejection(self) -> None:
        """Count a discarded episode once, after all fresh-session retries."""
        with self._state() as state:
            state["rejected_episodes"] = state.get("rejected_episodes", 0) + 1

    def record(
        self, kind: str, exc: BaseException, *, now: float | None = None
    ) -> None:
        now = time.time() if now is None else now
        with self._state() as state:
            if state.get("fatal"):
                raise FatalRolloutError(state["fatal"])
            key = "quota_errors" if kind == "quota" else "transient_api_errors"
            state[key] = state.get(key, 0) + 1
            if kind == "quota":
                times = [
                    t
                    for t in state.get("quota_times", [])
                    if now - t <= self.config.quota_breaker_window_seconds
                ]
                times.append(now)
                state["quota_times"] = times
                if len(times) >= self.config.quota_breaker_threshold:
                    # Keep the provider reset time, never the transport credentials.
                    reset = re.search(
                        r"(?:will )?reset(?:s)? at ([^.\n]+)", str(exc), re.IGNORECASE
                    )
                    reset_text = reset.group(1) if reset else "not supplied by provider"
                    state["fatal"] = (
                        f"τ² quota circuit breaker: {len(times)} quota errors within "
                        f"{self.config.quota_breaker_window_seconds:g}s; provider reset: {reset_text}; "
                        f"stopping rollouts, exit code {QUOTA_EXIT_CODE}. Resume from the latest recover checkpoint."
                    )
            fatal = state.get("fatal")
        if fatal:
            logger.critical(fatal)
            raise FatalRolloutError(fatal)


def bounded_call(
    call: Callable[..., _T],
    *args: Any,
    config: "Tau2EnvConfig",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> _T:
    """Retry only transient API errors, with one wall-time budget per request.

    The existing num_retries override remains an upper bound. LiteLLM and its
    OpenAI SDK adapter receive num_retries=0; exponential_backoff_retry is owned
    here so quota errors cannot enter either library's retry loop.
    """
    retries = int(kwargs.pop("num_retries", 8))
    kwargs.pop("retry_strategy", None)
    timeout = float(kwargs.pop("timeout", 120))
    breaker = QuotaCircuitBreaker(config)
    deadline = clock() + config.api_retry_budget_seconds
    for attempt in range(retries + 1):
        breaker.check()
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("τ² API request retry budget exhausted")
        try:
            return call(*args, **kwargs, num_retries=0, timeout=min(timeout, remaining))
        except Exception as exc:
            kind = classify_api_error(exc)
            if kind in ("quota", "transient"):
                breaker.record(kind, exc)
            if kind != "transient" or attempt == retries or clock() >= deadline:
                raise
            sleep(min(2 ** min(attempt, 5), 30, max(0.0, deadline - clock())))
    raise AssertionError("Unreachable retry state")


@contextmanager
def quota_exit_guard(config: "Tau2EnvConfig | None") -> Iterator[None]:
    """Preserve the dedicated exit even if evaluator cleanup raises an RPC error."""
    try:
        yield
    except Exception as exc:
        # RPC layers may discard the Python cause type. The shared fatal state
        # remains authoritative in the controller/trainer process.
        if config is not None:
            try:
                QuotaCircuitBreaker(config).check()
            except FatalRolloutError as fatal:
                logger.critical("Fatal rollout failure: %s", fatal)
                raise SystemExit(QUOTA_EXIT_CODE) from exc
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, FatalRolloutError):
                logger.critical("Fatal rollout failure: %s", current)
                raise SystemExit(QUOTA_EXIT_CODE) from exc
            current = current.__cause__ or current.__context__
        raise


def train_with_quota_exit(trainer: Any, **kwargs: Any) -> None:
    """Convert a fatal rollout failure into the dedicated trainer process exit."""
    config = getattr(getattr(trainer, "config", None), "econfig", None)
    stopped = Event()
    monitor = None
    if config is not None and config._quota_breaker_state_path:
        breaker = QuotaCircuitBreaker(config)

        def watch_quota():
            while not stopped.wait(0.1):
                try:
                    breaker.check()
                except Exception as exc:
                    # This is the trainer/controller process, even when rollout
                    # workers are subprocesses. No RPC serialization can hide the
                    # fatal state, and pending partial batches wake immediately.
                    for name in ("rollout", "eval_rollout", "_async_eval_rollout"):
                        rollout = getattr(trainer, name, None)
                        if rollout is not None:
                            dispatcher = getattr(rollout, "dispatcher", None)
                            if dispatcher is None:
                                dispatcher = rollout.workflow_executor.dispatcher
                            dispatcher.abort(exc)
                    return

        monitor = Thread(target=watch_quota, name="tau2-quota-monitor", daemon=True)
        monitor.start()
    try:
        with quota_exit_guard(config):
            if config is not None:
                QuotaCircuitBreaker(config).check()
            trainer.train(**kwargs)
            if config is not None:
                QuotaCircuitBreaker(config).check()
    finally:
        stopped.set()
        if monitor is not None:
            monitor.join(timeout=2)


class Tau2ApiStatsMixin:
    """Report run-wide error and rejected-episode deltas per training step."""

    def _additional_stats(self) -> dict[str, float]:
        stats = super()._additional_stats()
        current = QuotaCircuitBreaker(self.config.econfig).snapshot()
        previous = getattr(self, "_tau2_api_previous", {})
        for key in ("quota_errors", "transient_api_errors", "rejected_episodes"):
            stats[f"rollout/{key}"] = current[key] - previous.get(key, 0)
        self._tau2_api_previous = current
        return stats
