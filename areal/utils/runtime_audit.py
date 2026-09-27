# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

_ENV_NAME = "AREAL_RUNTIME_AUDIT_DIR"
_REDACTED = "<redacted>"
_ENV_WHITELIST = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
)
_SECRET_EXACT_KEYS = (
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "passwd",
    "password",
    "secret",
)
_SECRET_SUFFIXES = (
    "_api_key",
    "_auth",
    "_authorization",
    "_bearer",
    "_credential",
    "_credentials",
    "_passwd",
    "_password",
    "_secret",
    "_token",
)


def runtime_audit_dir() -> Path | None:
    path = os.environ.get(_ENV_NAME)
    if not path:
        return None
    return Path(path)


def runtime_audit_enabled() -> bool:
    return runtime_audit_dir() is not None


def sanitize_runtime_value(value: Any) -> Any:
    return _sanitize(value)


def write_runtime_audit(stage: str, payload: dict[str, Any]) -> Path | None:
    root = runtime_audit_dir()
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    safe_stage = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stage)
    rank = os.environ.get("RANK")
    suffix = f".rank{rank}" if rank is not None else ""
    recorded_ns = time.time_ns()
    path = root / f"{safe_stage}{suffix}.pid{os.getpid()}.{recorded_ns}.json"
    record = {
        "stage": stage,
        "recorded_ns": recorded_ns,
        "pid": os.getpid(),
        "rank": _int_or_none(rank),
        "local_rank": _int_or_none(os.environ.get("LOCAL_RANK")),
        "env": {
            name: os.environ[name] for name in _ENV_WHITELIST if name in os.environ
        },
        "payload": _sanitize(payload),
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def tensor_evidence(tensor: torch.Tensor, *, digest: bool = True) -> dict[str, Any]:
    local = _local_tensor(tensor)
    evidence: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "numel": int(tensor.numel()),
        "is_meta": bool(getattr(tensor, "is_meta", False)),
    }
    if local is not tensor:
        evidence["local_shape"] = list(local.shape)
        evidence["local_numel"] = int(local.numel())
        evidence["local_device"] = str(local.device)
    if digest and not evidence["is_meta"]:
        evidence["sha256"] = _tensor_digest(local)
    return evidence


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key is not None and _is_secret_key(key):
        return _REDACTED
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _sanitize(getattr(value, field.name), field.name)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, set):
        return sorted(_sanitize(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return tensor_evidence(value, digest=False)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return f"<{type(value).__name__}>"


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized.endswith("_api_key_env"):
        return False
    return normalized in _SECRET_EXACT_KEYS or any(
        normalized.endswith(suffix) for suffix in _SECRET_SUFFIXES
    )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        return to_local()
    return tensor


def _tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu", copy=True).contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
