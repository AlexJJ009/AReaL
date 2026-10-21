# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for opt-in disjoint AReaL port pools."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def network():
    """Import the standard-library utility without importing GPU engines."""
    path = Path(__file__).resolve().parents[1] / "areal/utils/network.py"
    spec = importlib.util.spec_from_file_location("cpu_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_port_pool_env_limits_default_allocations(network, monkeypatch):
    """Worker subprocesses inherit the same pool, excluding policy ports."""
    monkeypatch.setenv("AREAL_PORT_RANGE", "21000:21099")
    monkeypatch.setattr(network, "is_port_free", lambda port: True)
    ports = network.find_free_ports(8)
    assert all(21000 <= p <= 21099 for p in ports)
    assert len(set(ports)) == 8


@pytest.mark.parametrize("value", ["bad", "21000:20000", "0:5", "65000:70000"])
def test_invalid_port_pool_rejected(network, monkeypatch, value):
    """Fail closed on invalid pools rather than silently choosing random ports."""
    monkeypatch.setenv("AREAL_PORT_RANGE", value)
    with pytest.raises(ValueError, match="AREAL_PORT_RANGE"):
        network.find_free_ports(1)


def test_explicit_port_range_preserves_existing_behavior(network, monkeypatch):
    """Explicit callers retain their range; unrelated calls use the opt-in pool."""
    monkeypatch.setenv("AREAL_PORT_RANGE", "21000:21099")
    monkeypatch.setattr(network, "is_port_free", lambda port: True)
    assert network.find_free_ports(1, port_range=(20000, 20000)) == [20000]
