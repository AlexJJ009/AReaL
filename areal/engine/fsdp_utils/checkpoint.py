# SPDX-License-Identifier: Apache-2.0

"""FSDP checkpointing utilities for DCP (Distributed Checkpoint) integration."""

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
    ):
        self.model = model
        self.optimizer = optimizer
        self.saved_optimizer_parameters = saved_optimizer_parameters

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
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
            )
