# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for shared GPU admission helpers."""

from __future__ import annotations

from areal.utils.gpu_admission import (
    GPUAdmissionCancelled,
    GPUAdmissionConfig,
    GPUAdmissionError,
    GPUAdmissionQueryError,
    GPUAdmissionTimeout,
    GPUProcess,
    _parse_csv_rows,
    _query_gpu_index_by_uuid,
    _run_nvidia_smi,
    admission_config_from_env,
    query_compute_processes,
    scoped_busy_processes,
    wait_for_scoped_gpus_free,
)

__all__ = [
    "GPUAdmissionCancelled",
    "GPUAdmissionConfig",
    "GPUAdmissionError",
    "GPUAdmissionQueryError",
    "GPUAdmissionTimeout",
    "GPUProcess",
    "_parse_csv_rows",
    "_query_gpu_index_by_uuid",
    "_run_nvidia_smi",
    "admission_config_from_env",
    "query_compute_processes",
    "scoped_busy_processes",
    "wait_for_scoped_gpus_free",
]
