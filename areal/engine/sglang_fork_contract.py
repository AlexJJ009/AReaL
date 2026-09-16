# SPDX-License-Identifier: Apache-2.0

"""SGLang fork contract checks for the AWEX colocate integration.

The AWEX colocate plugin reaches into SGLang internals that carry no API
stability guarantee (scheduler attributes, weight updater, pause/flush
semantics). Fork upgrades have repeatedly broken these assumptions
SILENTLY; incidents this guards against:

- incident 12: pause mode default flipped semantics; scheduler never paused.
- incident 13/14: release_memory_occupation / flush_cache idle gates tightened,
  asserts killed every server under retract-pause.
- incident 15: Scheduler.tp_rank moved to tp_worker; getattr default 0 routed
  train shard 0 to all 64 inference ranks (silent weight corruption).

These checks turn that class of failure into a startup error. Run
``check_static_contract()`` in the scheduler process BEFORE the Scheduler
is constructed and ``check_scheduler_contract(scheduler)`` when the plugin
binds. All violations are collected and raised together.

Escape hatch: ``AREAL_SGLANG_CONTRACT=warn`` downgrades violations to log
warnings (for bring-up of a new fork); ``off`` skips entirely.
"""

from __future__ import annotations

import os
from typing import Any

from areal.utils.logging import getLogger

logger = getLogger("SGLangForkContract")


def _mode() -> str:
    return os.environ.get("AREAL_SGLANG_CONTRACT", "strict").strip().lower()


def _report(violations: list[str], stage: str) -> None:
    if not violations:
        logger.info("[fork-contract] %s: all checks passed", stage)
        return
    lines = "\n".join(f"  - {v}" for v in violations)
    message = (
        f"[fork-contract] {stage}: {len(violations)} SGLang fork contract "
        f"violation(s) detected:\n{lines}"
    )
    if _mode() == "warn":
        logger.warning("%s\nAREAL_SGLANG_CONTRACT=warn: continuing anyway.", message)
        return
    raise RuntimeError(message)


def check_static_contract() -> None:
    """Class/module-level assumptions; call before Scheduler construction."""
    if _mode() == "off":
        return
    violations: list[str] = []

    # incident 12: /pause_generation must support mode="retract" and the
    # scheduler must implement the pause handler pair.
    try:
        from sglang.srt.managers.io_struct import PauseGenerationReqInput

        try:
            PauseGenerationReqInput(mode="retract")
        except Exception as exc:
            violations.append(
                f"PauseGenerationReqInput(mode='retract') rejected: {exc!r} "
                "(incident 12: AReaL pauses with retract before offload)"
            )
    except ImportError as exc:
        violations.append(f"PauseGenerationReqInput import failed: {exc!r}")

    try:
        from sglang.srt.managers.scheduler import Scheduler

        for method in ("pause_generation", "continue_generation", "flush_cache"):
            if not callable(getattr(Scheduler, method, None)):
                violations.append(
                    f"Scheduler.{method} missing (incident 12/14 depend on it)"
                )
        idle_gate_names = ("is_fully_idle", "_is_no_request")
        if not any(
            callable(getattr(Scheduler, name, None)) for name in idle_gate_names
        ):
            violations.append(
                "Scheduler has neither is_fully_idle nor _is_no_request "
                "(retract-pause idle-gate semantics moved again)"
            )
    except ImportError as exc:
        violations.append(f"Scheduler import failed: {exc!r}")

    # incident 13: the release_memory_occupation patch target must exist where
    # we patch it, or the patch silently no-ops and offload kills servers.
    try:
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        if not callable(
            getattr(SchedulerWeightUpdaterManager, "release_memory_occupation", None)
        ):
            violations.append(
                "SchedulerWeightUpdaterManager.release_memory_occupation "
                "missing (incident 13 patch target)"
            )
    except ImportError:
        # AReaL pins SGLang 0.5.10, where SchedulerUpdateWeightsMixin exposes
        # release_memory_occupation directly on Scheduler. The AWEX plugin
        # patches that path when the newer manager module is absent.
        try:
            from sglang.srt.managers.scheduler import Scheduler

            if not callable(getattr(Scheduler, "release_memory_occupation", None)):
                violations.append(
                    "Neither SchedulerWeightUpdaterManager nor the legacy "
                    "Scheduler.release_memory_occupation API is available"
                )
        except ImportError as exc:
            violations.append(f"Scheduler memory-release API import failed: {exc!r}")

    _report(violations, "static")


def resolve_scheduler_parallel_attr(scheduler: Any, name: str) -> int | None:
    """Read native parallel state while retaining older fork attribute layouts."""
    worker = getattr(scheduler, "tp_worker", None)
    runner = getattr(worker, "model_runner", None)
    for owner in (
        scheduler,
        getattr(scheduler, "ps", None),
        worker,
        getattr(worker, "ps", None),
        getattr(runner, "ps", None),
    ):
        value = getattr(owner, name, None)
        if value is not None:
            return int(value)
    return None


def check_scheduler_contract(scheduler: Any) -> None:
    """Instance-level assumptions; call when the plugin binds a Scheduler."""
    if _mode() == "off":
        return
    violations: list[str] = []

    # incident 12: the scheduler-loop hook gates on _engine_paused.
    if not hasattr(scheduler, "_engine_paused"):
        violations.append(
            "scheduler._engine_paused missing (incident 12: paused AWEX hook "
            "never runs; decode races weight transfer)"
        )

    # incident 15: tp_rank must be resolvable without a silent default. This is
    # the exact lookup chain awex_colocate_reader._build_model_context uses.
    tp_rank = resolve_scheduler_parallel_attr(scheduler, "tp_rank")
    if tp_rank is None:
        violations.append(
            "tp_rank not found on scheduler/worker parallel state (incident 15: "
            "a silent 0 here routes train shard 0 to every inference rank)"
        )

    # incident 13/14 patch preconditions.
    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is None or not callable(getattr(running_batch, "is_empty", None)):
        violations.append(
            "scheduler.running_batch.is_empty missing (retract-pause patches "
            "use it to distinguish retract from in_place)"
        )
    if not hasattr(scheduler, "waiting_queue"):
        violations.append("scheduler.waiting_queue missing (patch logging/guards)")
    has_direct_memory_api = all(
        callable(getattr(scheduler, name, None))
        for name in ("release_memory_occupation", "resume_memory_occupation")
    )
    if getattr(scheduler, "weight_updater", None) is None and not has_direct_memory_api:
        violations.append(
            "scheduler has neither weight_updater nor direct release/resume "
            "memory methods (incident 13 patch target unavailable)"
        )

    # Reader model access (AWEX weight write target).
    model_runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
    if model_runner is None or getattr(model_runner, "model", None) is None:
        violations.append(
            "scheduler.tp_worker.model_runner.model unreachable (AWEX reader "
            "cannot bind weight tensors)"
        )

    _report(violations, "scheduler-bind")
