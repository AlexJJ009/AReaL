"""Offloaded engines broadcast over CPU only when they expose a CPU mirror group."""

from types import SimpleNamespace

import pytest

import areal.engine.fsdp_engine as fsdp_engine
from areal.engine.fsdp_engine import FSDPEngine
from areal.infra.rpc.guard.engine_blueprint import resolve_broadcast_target


def _engine(is_offload: bool, with_cpu_group: bool):
    fields = {
        "is_offload": is_offload,
        "context_and_model_parallel_group": "device-group",
    }
    if with_cpu_group:
        fields["cpu_model_parallel_group"] = "cpu-group"
    return SimpleNamespace(**fields)


def test_offloaded_engine_uses_cpu_mirror_group():
    group, device = resolve_broadcast_target(
        _engine(is_offload=True, with_cpu_group=True), device="cuda:0"
    )
    assert group == "cpu-group"
    assert device == "cpu"


def test_engine_without_cpu_group_keeps_device_broadcast():
    """FSDP tracks is_offload but has no CPU mirror group."""
    group, device = resolve_broadcast_target(
        _engine(is_offload=True, with_cpu_group=False), device="cuda:0"
    )
    assert group == "device-group"
    assert device == "cuda:0"


def test_resident_engine_keeps_device_broadcast():
    group, device = resolve_broadcast_target(
        _engine(is_offload=False, with_cpu_group=True), device="cuda:0"
    )
    assert group == "device-group"
    assert device == "cuda:0"


class _FakeStats:
    def __init__(self):
        self.logs = []

    def log(self, message: str) -> None:
        self.logs.append(message)


class _FakePlatform:
    def __init__(self):
        self.clear_memory_calls = 0
        self.synchronize_calls = 0

    def clear_memory(self) -> None:
        self.clear_memory_calls += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeTorchMemorySaver:
    def __init__(self):
        self.pause_calls = 0
        self.resume_calls = 0

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1


def _patch_fsdp_phase_deps(monkeypatch, *, tms_enabled: bool = True):
    platform = _FakePlatform()
    tms = _FakeTorchMemorySaver()
    barriers = []
    monkeypatch.setattr(fsdp_engine, "is_tms_enabled", lambda: tms_enabled)
    monkeypatch.setattr(fsdp_engine, "current_platform", platform)
    monkeypatch.setattr(fsdp_engine, "torch_memory_saver", tms)
    monkeypatch.setattr(
        fsdp_engine.dist,
        "barrier",
        lambda group: barriers.append(group),
    )
    return platform, tms, barriers


def test_fsdp_offload_onload_phase_transitions_are_idempotent(monkeypatch):
    platform, tms, barriers = _patch_fsdp_phase_deps(monkeypatch)
    stats = _FakeStats()
    engine = SimpleNamespace(
        is_offload=False,
        cpu_group="cpu-group",
        get_device_stats=lambda: stats,
    )

    FSDPEngine.offload(engine)
    FSDPEngine.offload(engine)

    assert engine.is_offload is True
    assert tms.pause_calls == 1
    assert platform.clear_memory_calls == 1
    assert platform.synchronize_calls == 1
    assert barriers == ["cpu-group"]
    assert stats.logs == ["before offload model", "after offload model"]

    FSDPEngine.onload(engine)
    FSDPEngine.onload(engine)

    assert engine.is_offload is False
    assert tms.resume_calls == 1
    assert platform.synchronize_calls == 2
    assert barriers == ["cpu-group", "cpu-group"]
    assert stats.logs == [
        "before offload model",
        "after offload model",
        "after onload model",
    ]


def test_fsdp_repeated_offload_still_requires_tms_enabled(monkeypatch):
    _patch_fsdp_phase_deps(monkeypatch, tms_enabled=False)
    engine = SimpleNamespace(
        is_offload=True,
        cpu_group="cpu-group",
        get_device_stats=lambda: _FakeStats(),
    )

    with pytest.raises(RuntimeError, match="torch_memory_saver requires"):
        FSDPEngine.offload(engine)


@pytest.mark.parametrize("offloaded", [False, True])
def test_destroy_resumes_tms_before_freeing_model(monkeypatch, offloaded):
    events = []
    engine = SimpleNamespace(
        _initialized=True,
        is_offload=offloaded,
        optimizer=object(),
        model=object(),
        _per_layer_optim_wrapper=None,
        own_global_group=False,
    )

    def onload():
        assert (
            engine._initialized
            and hasattr(engine, "model")
            and hasattr(engine, "optimizer")
        )
        events.append("resume")
        engine.is_offload = False

    engine.onload = onload
    monkeypatch.setattr(fsdp_engine.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(fsdp_engine.gc, "collect", lambda: None)
    monkeypatch.setattr(
        fsdp_engine.current_platform, "empty_cache", lambda: events.append("free")
    )
    FSDPEngine.destroy(engine)
    assert events == (["resume", "free"] if offloaded else ["free"])
    assert not hasattr(engine, "model") and not hasattr(engine, "optimizer")
    FSDPEngine.destroy(engine)
    assert events.count("resume") == int(offloaded)


def test_destroy_does_not_free_paused_allocations_if_resume_fails():
    def onload():
        raise RuntimeError("resume failed")

    engine = SimpleNamespace(
        _initialized=True,
        is_offload=True,
        model=object(),
        optimizer=object(),
        onload=onload,
    )
    with pytest.raises(RuntimeError, match="resume failed"):
        FSDPEngine.destroy(engine)
    assert (
        engine._initialized
        and hasattr(engine, "model")
        and hasattr(engine, "optimizer")
    )
