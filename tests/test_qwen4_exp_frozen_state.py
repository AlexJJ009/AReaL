# SPDX-License-Identifier: Apache-2.0
"""Exercise independent snapshots and strict restoration before native GPU cycles."""

import pytest
import torch
from torch import nn

from areal.models.mcore.qwen4_exp_frozen_state import (
    restore_visual_parameters,
    snapshot_visual_parameters,
)


class Qwen4ExpForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(3, 2)
        self.model = nn.Linear(3, 2)


def test_visual_snapshot_restores_parameters_without_touching_language_weights():
    model = Qwen4ExpForConditionalGeneration()
    saved = snapshot_visual_parameters(model)
    original_ids = {name: id(p) for name, p in model.named_parameters()}
    for name, parameter in model.named_parameters():
        if name in saved:
            assert saved[name].data_ptr() != parameter.data_ptr()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(42)
    restore_visual_parameters(model, saved)
    for name, parameter in model.named_parameters():
        assert id(parameter) == original_ids[name]
        expected = saved[name] if name in saved else torch.full_like(parameter, 42)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype"])
def test_invalid_visual_snapshot_rejected_before_any_parameter_changes(fault):
    model = Qwen4ExpForConditionalGeneration()
    saved = snapshot_visual_parameters(model)
    if fault == "missing":
        del saved["visual.bias"]
    elif fault == "extra":
        saved["model.bias"] = model.model.bias.detach().clone()
    elif fault == "shape":
        saved["visual.bias"] = torch.ones(1)
    else:
        saved["visual.bias"] = saved["visual.bias"].double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(42)
    with pytest.raises(ValueError):
        restore_visual_parameters(model, saved)
    for parameter in model.parameters():
        torch.testing.assert_close(
            parameter, torch.full_like(parameter, 42), rtol=0, atol=0
        )


def test_visual_snapshot_rejects_other_model_architectures():
    with pytest.raises(TypeError):
        snapshot_visual_parameters(nn.Linear(3, 2))


def test_static_hooks_preserve_visual_and_native_buffer_lifecycle():
    from types import SimpleNamespace

    from areal.models.mcore.qwen4_exp_frozen_state import install_static_state_hooks

    events = []
    updater = SimpleNamespace(
        _export_static_state=lambda model: {"native": "buffer-state"},
        _import_static_state=lambda model, state: events.append(state["native"]),
    )
    install_static_state_hooks(updater)
    exporter = updater._export_static_state
    install_static_state_hooks(updater)
    assert updater._export_static_state is exporter
    model = Qwen4ExpForConditionalGeneration()
    initial = snapshot_visual_parameters(model)
    for _ in range(2):
        state = updater._export_static_state(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(42)
        updater._import_static_state(model, state)
        for name, parameter in model.named_parameters():
            if name in initial:
                torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
    assert events == ["buffer-state", "buffer-state"]
    with pytest.raises(ValueError, match="not saved"):
        updater._import_static_state(model, {"native": "buffer-state"})


def test_static_hooks_leave_other_architectures_with_native_state_only():
    from types import SimpleNamespace

    from areal.models.mcore.qwen4_exp_frozen_state import install_static_state_hooks

    state = {"native": "buffer-state"}
    received = []
    updater = SimpleNamespace(
        _export_static_state=lambda model: state,
        _import_static_state=lambda model, value: received.append(value),
    )
    install_static_state_hooks(updater)
    model = nn.Linear(2, 2)
    assert updater._export_static_state(model) is state
    updater._import_static_state(model, state)
    assert received == [state]
