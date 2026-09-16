# SPDX-License-Identifier: Apache-2.0
"""Experiment-local raw tensor audit for frozen Qwen Arena canaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from examples.swe.arena_agent import ArenaStreamAgentWorkflow

_FIELDS = (
    "input_ids",
    "logprobs",
    "loss_mask",
    "attention_mask",
    "versions",
    "turn_ids",
    "rewards",
    "original_rewards",
    "token_rewards",
)
_JOIN_FIELDS = (
    "session_id",
    "arena_task_id",
    "arena_status",
    "stream_id",
    "data_id",
    "arena_stream_name",
    "harness_outcome_code",
)


def save_raw_audit(
    root: Path,
    traj: dict[str, Any] | None,
    task_id: int,
    is_eval: bool,
    metadata: list[dict] | None,
) -> Path:
    """Save an independent CPU snapshot; never modify the returned training data."""
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise ValueError("Audit root must not be a symlink")
    tensors = (
        {}
        if traj is None
        else {
            key: traj[key].detach().cpu().clone()
            for key in _FIELDS
            if key in traj and isinstance(traj[key], torch.Tensor)
        }
    )
    joins = [
        {key: row[key] for key in _JOIN_FIELDS if key in row}
        for row in (metadata or [])
    ]
    payload = {
        "task_id": task_id,
        "is_eval": is_eval,
        "metadata": joins,
        "tensors": tensors,
        "trajectory_present": traj is not None,
    }
    stem = f"{socket.gethostname()}-{os.getpid()}-{task_id}-{uuid.uuid4().hex}"
    path = root / (stem + ".pt")
    with path.open("xb") as file:
        os.fchmod(file.fileno(), 0o600)
        torch.save(payload, file)
    manifest = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "utc_seconds": time.time(),
        "task_id": task_id,
        "is_eval": is_eval,
        "trajectory_present": traj is not None,
        "fields": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in tensors.items()
        },
        "metadata": joins,
    }
    with (root / (stem + ".json")).open("x") as file:
        os.fchmod(file.fileno(), 0o600)
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return path


def install_raw_audit(root: Path) -> None:
    """Install inside the worker where the Arena workflow is constructed."""
    from areal.infra.rpc.rtensor import RTensor
    from areal.infra.workflow_executor import WorkflowExecutor

    installed = getattr(WorkflowExecutor, "_qwen_raw_audit_root", None)
    if installed is not None:
        if installed != str(root):
            raise ValueError("Cannot change the raw audit root in a live worker")
        return
    original = WorkflowExecutor._dump_trajectory

    async def dump(self, traj, task_id, is_eval, sample_metadata=None):
        localized = RTensor.localize(traj) if traj is not None else None
        await asyncio.to_thread(
            save_raw_audit, root, localized, task_id, is_eval, sample_metadata
        )
        return await original(
            self, traj, task_id, is_eval, sample_metadata=sample_metadata
        )

    WorkflowExecutor._dump_trajectory = dump
    WorkflowExecutor._qwen_raw_audit_root = str(root)


class AuditedArenaWorkflow(ArenaStreamAgentWorkflow):
    def __init__(self, *args, **kwargs):
        root = Path(os.environ["QWEN_ARENA_RAW_AUDIT_DIR"])
        install_raw_audit(root)
        super().__init__(*args, **kwargs)
