# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for FSDP checkpoint planning and format compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areal.engine.fsdp_engine import FSDPEngine


@pytest.mark.parametrize("operation", ["save", "load"])
def test_dcp_planning_uses_cpu_group_without_initializing_cuda_group(
    tmp_path, monkeypatch, operation
):
    engine = _engine_with_scheduler()
    engine.model = torch.nn.Linear(1, 1)
    cpu_group = object()
    engine._cpu_group = cpu_group
    calls = []

    def checkpoint_call(*args, **kwargs):
        assert kwargs["process_group"] is cpu_group
        calls.append(kwargs["checkpoint_id"])

    monkeypatch.setattr(f"areal.engine.fsdp_engine.dcp.{operation}", checkpoint_call)
    if operation == "save":
        engine._save_to_dcp(str(tmp_path), with_optim=True)
    else:

        class _Reader:
            def __init__(self, path):
                assert path == str(tmp_path)

            def read_metadata(self):
                return SimpleNamespace(state_dict_metadata={})

        monkeypatch.setattr("areal.engine.fsdp_engine.dcp.FileSystemReader", _Reader)
        engine._load_from_dcp(str(tmp_path), with_optim=True)
    assert calls == [str(tmp_path)]


@pytest.mark.parametrize(
    ("metadata_keys", "expected_manifest"),
    [
        ({"dcp.model.weight"}, False),
        ({"dcp.optim_uninitialized_state"}, True),
    ],
)
def test_dcp_load_uninitialized_manifest_template_follows_metadata(
    tmp_path, monkeypatch, metadata_keys, expected_manifest
):
    engine = _engine_with_scheduler()
    engine.model = torch.nn.Linear(1, 1)
    engine._cpu_group = object()
    observed = []

    class _Reader:
        def __init__(self, path):
            assert path == str(tmp_path)

        def read_metadata(self):
            return SimpleNamespace(state_dict_metadata=metadata_keys)

    def checkpoint_load(*, state_dict, checkpoint_id, process_group):
        assert checkpoint_id == str(tmp_path)
        assert process_group is engine._cpu_group
        observed.append(state_dict["dcp"]._include_uninitialized_state_manifest)

    monkeypatch.setattr("areal.engine.fsdp_engine.dcp.FileSystemReader", _Reader)
    monkeypatch.setattr("areal.engine.fsdp_engine.dcp.load", checkpoint_load)

    engine._load_from_dcp(str(tmp_path), with_optim=True)

    assert observed == [expected_manifest]


def _engine_with_scheduler() -> FSDPEngine:
    engine = FSDPEngine.__new__(FSDPEngine)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    engine.optimizer = optimizer
    engine.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )
    engine.optimizer_steps_since_init = 3
    engine._initialized = True
    engine._cpu_group = None
    engine.logger = type("_Logger", (), {"warning": lambda *_args, **_kwargs: None})()
    return engine
