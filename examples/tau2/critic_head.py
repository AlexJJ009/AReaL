# SPDX-License-Identifier: Apache-2.0
"""Value-head initialisation and learning rate for the τ² critic.

The base critic head is a linear layer with weight std 0.02, so its first outputs
are about ±3 for 0/1 targets, and every Adam step moves each parameter by about the
learning rate: a head bias at the backbone lr of 5e-6 never moves. This variant
gives the head a bias set to the expected success rate, a zero weight (every token
starts at that rate), and an optional learning rate of its own (docs/research/tau2/critic/value-head.md).

The bias is switched on through ``VALUE_HEAD_BIAS_FLAG`` on the model config, which
the engine writes into the checkpoint's config.json; the plain critic loaders read
it, so SAO loads the biased head without this module.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from areal.api.cli_args import PPOCriticConfig
from areal.engine import FSDPPPOCritic
from areal.models.transformers.token_critic import (
    VALUE_HEAD_BIAS_FLAG,
    value_head_has_bias,
)
from areal.utils import logging

logger = logging.getLogger("Tau2Train")


@dataclass
class Tau2HeadCriticConfig(PPOCriticConfig):
    """Critic config carrying the head settings to the remote engine workers."""

    head_bias_init: float = 0.5
    head_lr: float | None = None

    @classmethod
    def from_critic(
        cls, critic: PPOCriticConfig, *, head_bias_init: float, head_lr: float | None
    ) -> Tau2HeadCriticConfig:
        values = {f.name: getattr(critic, f.name) for f in fields(critic) if f.init}
        return cls(**values, head_bias_init=head_bias_init, head_lr=head_lr)


def init_value_head(model: torch.nn.Module, bias_init: float) -> None:
    """Zero the head weight and set its bias so every token predicts ``bias_init``."""
    score = model.score
    if score.bias is None:
        raise ValueError(f"value head has no bias; set {VALUE_HEAD_BIAS_FLAG}")
    with torch.no_grad():
        score.weight.zero_()
        score.bias.fill_(bias_init)


def is_head_parameter(name: str) -> bool:
    return name.split(".")[-2:-1] == ["score"]


class Tau2HeadFSDPPPOCritic(FSDPPPOCritic):
    def _create_llm_actor_or_critic(self):
        # A sealed artifact or a saved biased critic already carries a trained head.
        if self._scalar_value_artifact or value_head_has_bias(self.model_config):
            return super()._create_llm_actor_or_critic()
        setattr(self.model_config, VALUE_HEAD_BIAS_FLAG, True)
        model = super()._create_llm_actor_or_critic()
        init_value_head(model, self.config.head_bias_init)
        return model

    def _optimizer_param_groups(self) -> list:
        if self.config.head_lr is None:
            return super()._optimizer_param_groups()
        if self.config.fsdp.per_layer_optim_step:
            raise ValueError("head_lr needs one optimizer over all parameter groups")
        head, body = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                (head if is_head_parameter(name) else body).append(param)
        if not head:
            raise ValueError("no trainable value-head parameters found")
        logger.info(
            "Critic head lr %s for %d tensors, backbone lr %s for %d tensors",
            self.config.head_lr,
            len(head),
            self.optimizer_config.lr,
            len(body),
        )
        return [{"params": body}, {"params": head, "lr": self.config.head_lr}]
