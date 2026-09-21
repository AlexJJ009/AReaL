# SPDX-License-Identifier: Apache-2.0
"""Allocate portable CUDA IPC buffers without changing training allocations."""

import re
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def cuda_ipc_allocation() -> Iterator[None]:
    """Use ordinary CUDA storage for new IPC buffers, then restore the allocator.

    Expandable IPC handles are not portable across all PyTorch versions. Scope
    this context to staging-buffer creation during serialized weight publishing,
    while training is paused: the allocator switch is process-wide. Existing
    training storage is untouched, and restoration precedes IPC serialization.
    """
    settings = torch.cuda.memory._snapshot()["allocator_settings"]
    if not settings["expandable_segments"]:
        yield
        return

    # Preserve runtime settings, not environment defaults. The setter resets
    # other options (including split size and GC threshold) on every call.
    original = settings["PYTORCH_CUDA_ALLOC_CONF"]
    # A preceding setter can omit this option while retaining its True value.
    # Replaying that string alone would leave our temporary False in effect.
    restore = original
    if not re.search(r"expandable_segments\s*:", original):
        restore = (
            f"{original},expandable_segments:True"
            if original
            else "expandable_segments:True"
        )
    staging = re.sub(r"expandable_segments\s*:\s*(True|False)", "", original)
    staging = ",".join(part for part in staging.split(",") if part.strip())
    staging = (
        f"{staging},expandable_segments:False"
        if staging
        else "expandable_segments:False"
    )
    try:
        torch.cuda.memory._set_allocator_settings(staging)
        yield
    finally:
        torch.cuda.memory._set_allocator_settings(restore)
