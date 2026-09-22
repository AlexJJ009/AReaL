# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for DCP optimizer state handling."""

from __future__ import annotations

import copy

import torch
from torch import nn

from areal.engine.fsdp_utils.checkpoint import DCPState


class _PartlyUsedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = nn.Linear(2, 1, bias=False)
        self.unused = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor, *, include_unused: bool) -> torch.Tensor:
        out = self.used(x)
        if include_unused:
            out = out + self.unused(x)
        return out.sum()


def _train_step(
    model: _PartlyUsedModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    *,
    include_unused: bool,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    model(x, include_unused=include_unused).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_models_close(left: nn.Module, right: nn.Module) -> None:
    for left_param, right_param in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(left_param, right_param, rtol=0.0, atol=0.0)


def _assert_adam_state_close(
    left: torch.optim.Optimizer, right: torch.optim.Optimizer
) -> None:
    left_state = left.state_dict()["state"]
    right_state = right.state_dict()["state"]
    assert left_state.keys() == right_state.keys()
    for param_id in left_state:
        assert left_state[param_id].keys() == right_state[param_id].keys()
        for slot_name, left_value in left_state[param_id].items():
            torch.testing.assert_close(
                left_value, right_state[param_id][slot_name], rtol=0.0, atol=0.0
            )


def test_dcp_state_save_adds_cpu_zero_adam_slots_without_mutating_optimizer():
    """Missing lazy AdamW state is completed only in the exported state dict."""
    model = _PartlyUsedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=True)
    _train_step(model, optimizer, torch.tensor([[1.0, 2.0]]), include_unused=False)

    params_before = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    live_state_before = optimizer.state_dict()
    used_step_before = live_state_before["state"][0]["step"].detach().clone()
    assert len(live_state_before["state"]) == 1

    state = DCPState(model, optimizer).state_dict()

    assert set(state["optim"]["state"]) == {"used.weight", "unused.weight"}
    unused_state = state["optim"]["state"]["unused.weight"]
    assert unused_state["step"].item() == 0
    assert unused_state["exp_avg"].device.type == "cpu"
    assert unused_state["exp_avg_sq"].device.type == "cpu"
    torch.testing.assert_close(
        unused_state["exp_avg"], torch.zeros_like(unused_state["exp_avg"])
    )
    torch.testing.assert_close(
        unused_state["exp_avg_sq"], torch.zeros_like(unused_state["exp_avg_sq"])
    )

    live_state_after = optimizer.state_dict()
    assert len(live_state_after["state"]) == 1
    torch.testing.assert_close(live_state_after["state"][0]["step"], used_step_before)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, params_before[name], rtol=0.0, atol=0.0)


def test_dcp_state_partial_adam_roundtrip_matches_continuous_next_update():
    """A checkpoint with an unused trainable param resumes like continuous AdamW."""
    continuous_model = _PartlyUsedModel()
    restored_model = copy.deepcopy(continuous_model)
    continuous_optimizer = torch.optim.AdamW(
        continuous_model.parameters(), lr=0.01, fused=True
    )
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=0.01, fused=True
    )

    first_batch = torch.tensor([[1.0, 2.0]])
    next_batch = torch.tensor([[3.0, 5.0]])
    _train_step(
        continuous_model, continuous_optimizer, first_batch, include_unused=False
    )

    state = copy.deepcopy(DCPState(continuous_model, continuous_optimizer).state_dict())
    assert state["optim_uninitialized_state"] == ["unused.weight"]

    DCPState(restored_model, restored_optimizer).load_state_dict(state)
    _assert_models_close(continuous_model, restored_model)
    assert len(restored_optimizer.state_dict()["state"]) == 1

    _train_step(continuous_model, continuous_optimizer, next_batch, include_unused=True)
    _train_step(restored_model, restored_optimizer, next_batch, include_unused=True)

    _assert_models_close(continuous_model, restored_model)
    _assert_adam_state_close(continuous_optimizer, restored_optimizer)


def test_dcp_filesystem_roundtrip_drops_manifested_missing_adam_state(tmp_path):
    """Real DCP save/load keeps absent-state semantics for unused AdamW params."""
    continuous_model = _PartlyUsedModel()
    restored_model = copy.deepcopy(continuous_model)
    continuous_optimizer = torch.optim.AdamW(
        continuous_model.parameters(), lr=0.01, fused=True
    )
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=0.01, fused=True
    )

    first_batch = torch.tensor([[1.0, 2.0]])
    next_batch = torch.tensor([[3.0, 5.0]])
    _train_step(
        continuous_model, continuous_optimizer, first_batch, include_unused=False
    )

    import torch.distributed.checkpoint as dcp

    checkpoint_dir = tmp_path / "checkpoint"
    dcp.save(
        {"dcp": DCPState(continuous_model, continuous_optimizer)},
        checkpoint_id=str(checkpoint_dir),
    )
    metadata = dcp.FileSystemReader(str(checkpoint_dir)).read_metadata()
    assert "dcp.optim_uninitialized_state" in metadata.state_dict_metadata

    dcp.load(
        {"dcp": DCPState(restored_model, restored_optimizer)},
        checkpoint_id=str(checkpoint_dir),
    )

    _assert_models_close(continuous_model, restored_model)
    assert len(restored_optimizer.state_dict()["state"]) == 1

    _train_step(continuous_model, continuous_optimizer, next_batch, include_unused=True)
    _train_step(restored_model, restored_optimizer, next_batch, include_unused=True)

    _assert_models_close(continuous_model, restored_model)
    _assert_adam_state_close(continuous_optimizer, restored_optimizer)


def test_dcp_filesystem_loads_complete_old_format_without_manifest(tmp_path):
    """Old checkpoints without the optional manifest still load when complete."""
    source_model = _PartlyUsedModel()
    restored_model = copy.deepcopy(source_model)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01, fused=True)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=0.01, fused=True
    )

    first_batch = torch.tensor([[1.0, 2.0]])
    next_batch = torch.tensor([[3.0, 5.0]])
    _train_step(source_model, source_optimizer, first_batch, include_unused=True)

    import torch.distributed.checkpoint as dcp

    checkpoint_dir = tmp_path / "old-format-complete"
    dcp.save(
        {
            "dcp": DCPState(
                source_model,
                source_optimizer,
                include_uninitialized_state_manifest=False,
            )
        },
        checkpoint_id=str(checkpoint_dir),
    )
    metadata = dcp.FileSystemReader(str(checkpoint_dir)).read_metadata()
    assert "dcp.optim_uninitialized_state" not in metadata.state_dict_metadata

    dcp.load(
        {
            "dcp": DCPState(
                restored_model,
                restored_optimizer,
                include_uninitialized_state_manifest=False,
            )
        },
        checkpoint_id=str(checkpoint_dir),
    )

    _assert_models_close(source_model, restored_model)
    _assert_adam_state_close(source_optimizer, restored_optimizer)

    _train_step(source_model, source_optimizer, next_batch, include_unused=True)
    _train_step(restored_model, restored_optimizer, next_batch, include_unused=True)

    _assert_models_close(source_model, restored_model)
    _assert_adam_state_close(source_optimizer, restored_optimizer)
