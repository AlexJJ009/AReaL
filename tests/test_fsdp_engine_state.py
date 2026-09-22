# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for FSDP engine scheduler/counter checkpoint sidecars."""

from __future__ import annotations

import torch

from areal.engine.fsdp_engine import FSDPEngine


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


def test_fsdp_engine_training_state_round_trip_restores_scheduler_and_counter(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("areal.engine.fsdp_engine.dist.get_rank", lambda: 0)
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.dist.barrier", lambda group=None: None
    )

    source = _engine_with_scheduler()
    source.lr_scheduler.step()
    source.optimizer_steps_since_init = 7
    source._save_training_state(str(tmp_path))

    restored = _engine_with_scheduler()
    restored.lr_scheduler.step()
    restored.lr_scheduler.step()
    restored.optimizer_steps_since_init = 99
    restored._load_training_state(str(tmp_path))

    assert restored.optimizer_steps_since_init == 7
    assert restored.lr_scheduler.state_dict() == source.lr_scheduler.state_dict()
    assert restored.lr_scheduler.get_last_lr() == source.lr_scheduler.get_last_lr()


def test_fsdp_engine_training_state_load_keeps_legacy_checkpoint_compatible(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.dist.barrier", lambda group=None: None
    )

    engine = _engine_with_scheduler()
    engine.optimizer_steps_since_init = 4
    engine._load_training_state(str(tmp_path))

    assert engine.optimizer_steps_since_init == 4
    assert engine.lr_scheduler.last_epoch == 0
