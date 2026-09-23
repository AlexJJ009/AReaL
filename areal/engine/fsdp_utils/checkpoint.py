# SPDX-License-Identifier: Apache-2.0

"""FSDP checkpointing utilities for DCP (Distributed Checkpoint) integration."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


class DCPState(Stateful):
    """Wrapper for checkpointing the State using DCP.

    This class implements the Stateful protocol, so DCP will automatically call
    state_dict/load_state_dict as needed in the dcp.save/load APIs.

    It handles calling distributed state dict methods on the model and optimizer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        saved_optimizer_parameters: set[str] | None = None,
        include_uninitialized_state_manifest: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.saved_optimizer_parameters = saved_optimizer_parameters
        self._include_uninitialized_state_manifest = (
            include_uninitialized_state_manifest
        )

    def state_dict(self) -> dict[str, Any]:
        """
        Get state dict for model and optimizer using DCP utilities.
        This automatically manages FSDP FQN's and
        sets default state dict type to FSDP.SHARDED_STATE_DICT
        """
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
            (
                optimizer_state_dict,
                uninitialized_optimizer_state,
            ) = _complete_missing_adam_state_for_save(
                self.model, self.optimizer, model_state_dict, optimizer_state_dict
            )
            if self.saved_optimizer_parameters is not None:
                missing = self.saved_optimizer_parameters.difference(
                    optimizer_state_dict["state"]
                )
                if missing:
                    raise ValueError(
                        "Optimizer load template lacks saved parameter state; "
                        "restore into a fresh optimizer: " + ", ".join(sorted(missing))
                    )
                # A fresh optimizer gets dense placeholder state from PyTorch,
                # but the checkpoint may contain only parameters used so far.
                # Prune whole absent entries, never individual moment tensors:
                # incomplete state for an initialized parameter must still fail.
                optimizer_state_dict["state"] = {
                    name: value
                    for name, value in optimizer_state_dict["state"].items()
                    if name in self.saved_optimizer_parameters
                }
            state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
            if self._include_uninitialized_state_manifest or (
                uninitialized_optimizer_state
                and self.saved_optimizer_parameters is None
            ):
                state_dict["optim_uninitialized_state"] = uninitialized_optimizer_state
        else:
            state_dict = {"model": get_model_state_dict(self.model)}
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state dicts onto model and optimizer.
        """
        if self.optimizer is not None:
            optim_state = state_dict["optim"]
            # PyTorch's FQN splitter assumes every trainable parameter has
            # state, which is false for lazy AdamW state (e.g. unused vision).
            # Empty entries retain lazy initialization without inventing moments.
            optim_state = {
                **optim_state,
                "state": dict(optim_state["state"]),
            }
            for group in optim_state["param_groups"]:
                for name in group["params"]:
                    optim_state["state"].setdefault(name, {})
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=optim_state,
            )
            for parameter in list(self.optimizer.state):
                if not self.optimizer.state[parameter]:
                    del self.optimizer.state[parameter]
            _drop_adam_state_for_fqns(
                self.model,
                self.optimizer,
                list(state_dict.get("optim_uninitialized_state", [])),
            )
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
            )


def _complete_missing_adam_state_for_save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state_dict: Mapping[str, Any],
    optim_state_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Add CPU zero Adam slots for trainable params missing lazy optimizer state.

    Adam/AdamW initialize per-parameter state lazily on the first gradient. If a
    trainable parameter is unused before checkpointing, PyTorch DCP can emit it
    in ``param_groups`` without a matching ``state`` entry. Fresh DCP loads build
    a complete optimizer template and then reject the missing state. Fill only
    the returned checkpoint state dict so the live optimizer is not mutated, and
    place missing moment tensors on CPU to avoid consuming scarce GPU headroom
    during checkpoint save.
    """
    if not _is_adam_optimizer(optimizer):
        return optim_state_dict, []

    state = optim_state_dict.get("state")
    param_groups = optim_state_dict.get("param_groups")
    if not isinstance(state, dict) or not isinstance(param_groups, list):
        return optim_state_dict, []

    uninitialized_state: list[str] = []

    named_parameters = dict(model.named_parameters())
    for group in param_groups:
        params = group.get("params") if isinstance(group, dict) else None
        if not isinstance(params, list):
            continue
        for fqn in params:
            if not isinstance(fqn, str) or fqn in state:
                continue
            parameter = named_parameters.get(fqn)
            if parameter is None or not parameter.requires_grad:
                continue
            state[fqn] = _new_adam_zero_state(fqn, parameter, group, model_state_dict)
            uninitialized_state.append(fqn)
    return optim_state_dict, uninitialized_state


def _drop_adam_state_for_fqns(
    model: nn.Module, optimizer: torch.optim.Optimizer, fqns: list[str]
) -> None:
    if not fqns or not _is_adam_optimizer(optimizer):
        return
    parameters = dict(model.named_parameters())
    for fqn in dict.fromkeys(fqns):
        parameter = parameters.get(fqn)
        if parameter is not None:
            optimizer.state.pop(parameter, None)


def _is_adam_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    return isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))


def _new_adam_zero_state(
    fqn: str,
    parameter: torch.nn.Parameter,
    group: Mapping[str, Any],
    model_state_dict: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    tensor = model_state_dict.get(fqn, parameter)
    state = {
        "step": _new_adam_zero_step(parameter, group),
        "exp_avg": _zeros_like_for_checkpoint(tensor),
        "exp_avg_sq": _zeros_like_for_checkpoint(tensor),
    }
    if group.get("amsgrad", False):
        state["max_exp_avg_sq"] = _zeros_like_for_checkpoint(tensor)
    return state


def _new_adam_zero_step(
    parameter: torch.nn.Parameter, group: Mapping[str, Any]
) -> torch.Tensor:
    dtype = torch.float32
    if not group.get("fused", False) and torch.get_default_dtype() == torch.float64:
        dtype = torch.float64
    device = (
        parameter.device if group.get("capturable") or group.get("fused") else "cpu"
    )
    return torch.zeros((), dtype=dtype, device=device)


def _zeros_like_for_checkpoint(tensor: Any) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor optimizer slot source, got {type(tensor)!r}")
    try:
        return torch.zeros_like(
            tensor, device="cpu", memory_format=torch.preserve_format
        )
    except TypeError:
        return torch.zeros_like(tensor, device="cpu")
