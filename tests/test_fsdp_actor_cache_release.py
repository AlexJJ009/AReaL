# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from areal.engine.fsdp_engine import FSDPEngine
from areal.trainer.rl_trainer import PPOTrainer


class _Engine:
    def __init__(self, backend: str = "fsdp:d4t1"):
        self.config = type("Config", (), {"backend": backend})()
        self.calls = []

    def _custom_function_call(self, name, *, rpc_meta):
        self.calls.append((name, rpc_meta))


def test_fsdp_engine_release_unused_cuda_cache_uses_platform(monkeypatch):
    calls = []
    engine = FSDPEngine.__new__(FSDPEngine)
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.gc.collect", lambda: calls.append("gc")
    )
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.current_platform.empty_cache",
        lambda: calls.append("empty_cache"),
    )

    engine._release_unused_cuda_cache()

    assert calls == ["gc", "empty_cache", "gc"]


def test_trainer_releases_fsdp_actor_cache_through_existing_controller_rpc():
    trainer = PPOTrainer.__new__(PPOTrainer)
    engine = _Engine()

    trainer._release_unused_cuda_cache(engine, role="actor")

    assert engine.calls == [("_release_unused_cuda_cache", {"broadcast": False})]


def test_trainer_skips_non_fsdp_cache_release():
    trainer = PPOTrainer.__new__(PPOTrainer)
    engine = _Engine(backend="megatron:d4t1")

    trainer._release_unused_cuda_cache(engine, role="actor")

    assert engine.calls == []


def test_trainer_propagates_cache_release_rpc_failures():
    class FailingEngine(_Engine):
        def _custom_function_call(self, name, *, rpc_meta):
            raise RuntimeError(f"failed {name}")

    trainer = PPOTrainer.__new__(PPOTrainer)

    with pytest.raises(RuntimeError, match="_release_unused_cuda_cache"):
        trainer._release_unused_cuda_cache(FailingEngine(), role="actor")
