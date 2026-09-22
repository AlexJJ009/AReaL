# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen3.5 text-only scalar critic; strict checkpoints include score."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class TokenCriticOutput(ModelOutput):
    logits: torch.Tensor


class _LanguageModelContainer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.language_model = model


class Qwen35ScalarValueModel(PreTrainedModel):
    """Keep composite checkpoint keys while loading only its causal text model.

    The independent pretraining pipeline creates the scalar head. Online SAO
    loads the complete state strictly; missing score.weight is never tolerated.
    """

    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: PretrainedConfig, language_model: nn.Module):
        super().__init__(config)
        self.model = _LanguageModelContainer(language_model)
        parameter = next(language_model.parameters())
        self.score = nn.Linear(
            config.text_config.hidden_size,
            1,
            bias=False,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        self._no_split_modules = language_model._no_split_modules

    @classmethod
    def from_config(cls, config: PretrainedConfig, **kwargs: Any):
        config = deepcopy(config)
        text = config.text_config
        dtype = kwargs.get("dtype", text.dtype)
        if dtype is not None:
            text.dtype = dtype
        language_model = AutoModel.from_config(text, **kwargs)
        return cls(config, language_model)

    def forward(self, *args, **kwargs) -> TokenCriticOutput:
        kwargs["return_dict"] = True
        return TokenCriticOutput(
            logits=self.score(
                self.model.language_model(*args, **kwargs).last_hidden_state
            )
        )

    def gradient_checkpointing_enable(self, *args, **kwargs) -> None:
        self.model.language_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.model.language_model.gradient_checkpointing_disable()
