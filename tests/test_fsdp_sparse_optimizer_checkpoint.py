# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for sparse AdamW state in FSDP DCP checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.api import CheckpointException

from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.fsdp_utils.checkpoint import DCPState


def _module() -> nn.ModuleDict:
    torch.manual_seed(7)
    return nn.ModuleDict(
        {
            "text": nn.Linear(3, 2, bias=False),
            "visual": nn.Linear(3, 2, bias=False),
        }
    )


def _optimizer(model: nn.ModuleDict, *, alternate_groups: bool = False):
    if alternate_groups:
        return torch.optim.AdamW(
            [
                {
                    "params": list(model["text"].parameters()),
                    "lr": 0.5,
                    "betas": (0.5, 0.6),
                    "eps": 1e-3,
                    "weight_decay": 0.0,
                },
                {
                    "params": list(model["visual"].parameters()),
                    "lr": 0.6,
                    "betas": (0.4, 0.7),
                    "eps": 1e-4,
                    "weight_decay": 0.0,
                },
            ]
        )
    return torch.optim.AdamW(
        [
            {
                "params": list(model["text"].parameters()),
                "lr": 0.01,
                "betas": (0.8, 0.9),
                "eps": 1e-6,
                "weight_decay": 0.2,
            },
            {
                "params": list(model["visual"].parameters()),
                "lr": 0.02,
                "betas": (0.7, 0.95),
                "eps": 1e-7,
                "weight_decay": 0.3,
            },
        ]
    )


def _engine(model: nn.ModuleDict, optimizer: torch.optim.Optimizer) -> FSDPEngine:
    engine = FSDPEngine.__new__(FSDPEngine)
    engine._initialized = True
    engine._cpu_group = None
    engine.model = model
    engine.optimizer = optimizer
    return engine


def _train_step(
    model: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    *,
    include_visual: bool,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    text_input = torch.tensor(
        [[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]], dtype=torch.float32
    )
    loss = model["text"](text_input).square().sum()
    if include_visual:
        visual_input = torch.tensor(
            [[-1.0, 0.5, 2.0], [1.5, -0.25, 0.75]], dtype=torch.float32
        )
        loss = loss + 0.7 * model["visual"](visual_input).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _model_state(model: nn.ModuleDict) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _optimizer_state_by_name(
    model: nn.ModuleDict, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, Any]]:
    names_by_param = dict(model.named_parameters())
    params_by_id = {id(parameter): name for name, parameter in names_by_param.items()}
    state_by_name: dict[str, dict[str, Any]] = {}
    for parameter, state in optimizer.state.items():
        state_by_name[params_by_id[id(parameter)]] = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
    return state_by_name


def _group_settings(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    keys = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")
    return [{key: group[key] for key in keys} for group in optimizer.param_groups]


def _assert_model_state_matches(
    model: nn.ModuleDict, expected: dict[str, torch.Tensor]
) -> None:
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def _assert_optimizer_state_matches(
    model: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    expected: dict[str, dict[str, Any]],
) -> None:
    actual = _optimizer_state_by_name(model, optimizer)
    assert actual.keys() == expected.keys()
    for param_name, expected_state in expected.items():
        assert actual[param_name].keys() == expected_state.keys()
        for slot_name, expected_value in expected_state.items():
            actual_value = actual[param_name][slot_name]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value


def _assert_training_state_matches(
    left_model: nn.ModuleDict,
    left_optimizer: torch.optim.Optimizer,
    right_model: nn.ModuleDict,
    right_optimizer: torch.optim.Optimizer,
) -> None:
    _assert_model_state_matches(left_model, _model_state(right_model))
    _assert_optimizer_state_matches(
        left_model,
        left_optimizer,
        _optimizer_state_by_name(right_model, right_optimizer),
    )


def test_sparse_adamw_dcp_load_restores_same_optimizer_and_prunes_unused_state(
    tmp_path,
):
    source = _module()
    optimizer = _optimizer(source)
    _train_step(source, optimizer, include_visual=False)
    assert _optimizer_state_by_name(source, optimizer).keys() == {"text.weight"}

    expected_model = _model_state(source)
    expected_optimizer = _optimizer_state_by_name(source, optimizer)
    expected_groups = _group_settings(optimizer)
    engine = _engine(source, optimizer)
    engine._save_to_dcp(str(tmp_path), with_optim=True)

    _train_step(source, optimizer, include_visual=True)
    assert _optimizer_state_by_name(source, optimizer).keys() == {
        "text.weight",
        "visual.weight",
    }

    engine._load_from_dcp(str(tmp_path), with_optim=True)

    _assert_model_state_matches(source, expected_model)
    _assert_optimizer_state_matches(source, optimizer, expected_optimizer)
    assert _group_settings(optimizer) == expected_groups


@pytest.mark.parametrize(
    "prepare_target",
    [
        pytest.param(lambda _model, _optimizer: None, id="fresh-optimizer"),
        pytest.param(
            lambda model, optimizer: _train_step(model, optimizer, include_visual=True),
            id="extra-initialized-visual-state",
        ),
    ],
)
def test_sparse_adamw_dcp_load_restores_fresh_optimizer_and_next_update_matches(
    tmp_path,
    prepare_target: Callable[[nn.ModuleDict, torch.optim.Optimizer], None],
):
    source = _module()
    source_optimizer = _optimizer(source)
    _train_step(source, source_optimizer, include_visual=False)
    expected_model = _model_state(source)
    expected_optimizer = _optimizer_state_by_name(source, source_optimizer)
    expected_groups = _group_settings(source_optimizer)
    _engine(source, source_optimizer)._save_to_dcp(str(tmp_path), with_optim=True)

    target = _module()
    target_optimizer = _optimizer(target, alternate_groups=True)
    prepare_target(target, target_optimizer)
    _engine(target, target_optimizer)._load_from_dcp(str(tmp_path), with_optim=True)

    _assert_model_state_matches(target, expected_model)
    _assert_optimizer_state_matches(target, target_optimizer, expected_optimizer)
    assert _group_settings(target_optimizer) == expected_groups

    _train_step(source, source_optimizer, include_visual=True)
    _train_step(target, target_optimizer, include_visual=True)

    _assert_training_state_matches(target, target_optimizer, source, source_optimizer)
    assert _optimizer_state_by_name(target, target_optimizer)[
        "visual.weight"
    ].keys() == {"step", "exp_avg", "exp_avg_sq"}
    torch.testing.assert_close(
        _optimizer_state_by_name(target, target_optimizer)["visual.weight"]["step"],
        torch.tensor(1.0),
        rtol=0,
        atol=0,
    )


def test_sparse_adamw_dcp_load_rejects_missing_saved_adam_moment(tmp_path):
    source = _module()
    source_optimizer = _optimizer(source)
    _train_step(source, source_optimizer, include_visual=False)
    state = DCPState(source, source_optimizer).state_dict()
    del state["optim"]["state"]["text.weight"]["exp_avg"]
    dcp.save({"dcp": state}, checkpoint_id=str(tmp_path))

    target = _module()
    target_optimizer = _optimizer(target)
    _train_step(target, target_optimizer, include_visual=False)

    with pytest.raises(CheckpointException) as exc_info:
        _engine(target, target_optimizer)._load_from_dcp(str(tmp_path), with_optim=True)
    assert "exp_avg" in repr(exc_info.value)
