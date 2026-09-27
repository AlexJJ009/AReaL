# SPDX-License-Identifier: Apache-2.0
"""Compatibility CLI for shared GPU diagnostics helpers."""

from __future__ import annotations

from areal.utils.gpu_diagnostics import (
    DEFAULT_DEVICES,
    PROC_ROOT,
    _admission_from_args,
    _parse_args,
    _parse_proc_stat_starttime,
    _start_monitors,
    _start_sampler,
    _stop_monitors,
    _write_json,
    main,
    query_compute_memory_snapshot,
    read_process_owner,
    run_diagnostics,
    write_snapshot,
)

__all__ = [
    "DEFAULT_DEVICES",
    "PROC_ROOT",
    "_admission_from_args",
    "_parse_args",
    "_parse_proc_stat_starttime",
    "_start_monitors",
    "_start_sampler",
    "_stop_monitors",
    "_write_json",
    "main",
    "query_compute_memory_snapshot",
    "read_process_owner",
    "run_diagnostics",
    "write_snapshot",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
