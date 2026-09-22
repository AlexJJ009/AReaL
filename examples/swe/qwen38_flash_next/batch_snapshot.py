# SPDX-License-Identifier: Apache-2.0
"""Opt-in, lossless CPU snapshots for diagnosing a collected training batch."""

import copy
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from areal.infra.rpc.rtensor import RTensor


def save_batch_snapshot(path: Path, batch: Any, metadata: dict[str, Any]) -> None:
    """Fetch copied remote wrappers without changing the live batch or its leases."""
    localized = RTensor.localize(copy.deepcopy(batch), preserve_tensor_aliases=True)
    memo: dict[int, torch.Tensor] = {}

    def cpu(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in memo:
                memo[id(value)] = value.detach().to(device="cpu", copy=True)
            return memo[id(value)]
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu(v) for v in value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported batch snapshot value: {type(value).__name__}")

    payload = {"schema_version": 1, "metadata": cpu(metadata), "batch": cpu(localized)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=".batch-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication refuses to overwrite an existing snapshot.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


@contextmanager
def capture_training_batches(actor, directory: Path, metadata: dict[str, Any]):
    """Capture preparation and advantage calls; indices are calls, not global steps.

    This preserves inputs for replay experiments, not optimizer/RNG state. It does
    not enable training replay or make repeated off-policy updates safe.
    """
    originals = {
        name: getattr(actor, name) for name in ("prepare_batch", "compute_advantages")
    }
    own = {name: name in vars(actor) for name in originals}
    counts = {name: 0 for name in originals}

    def wrap(name):
        def call(*args, **kwargs):
            index = counts[name]
            prefix = directory / f"{name}-{index:04d}"
            info = {**metadata, "method": name, "call_index": index}
            if name == "compute_advantages":
                batch = args[0] if args else kwargs["data"]
                save_batch_snapshot(prefix.with_suffix(".input.pt"), batch, info)
            result = originals[name](*args, **kwargs)
            save_batch_snapshot(prefix.with_suffix(".output.pt"), result, info)
            counts[name] += 1
            return result

        return call

    try:
        for name in originals:
            setattr(actor, name, wrap(name))
        yield
    finally:
        for name, original in originals.items():
            if own[name]:
                setattr(actor, name, original)
            else:
                delattr(actor, name)
