# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from areal.utils.save_load import get_state_dict_from_repo_id_or_path


@dataclass
class TokenCriticOutput(ModelOutput):
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class _LanguageModelContainer(nn.Module):
    """Expose composite Qwen-style ``model.language_model`` state keys."""

    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


class Qwen35TokenCriticForCausalBackbone(PreTrainedModel):
    """Token-value critic for composite Qwen3.5 checkpoints."""

    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: PretrainedConfig, language_model: nn.Module):
        super().__init__(config)
        self.model = _LanguageModelContainer(language_model)
        backbone_parameter = next(language_model.parameters())
        self.score = nn.Linear(
            _hidden_size(config),
            1,
            bias=False,
            dtype=backbone_parameter.dtype,
            device=backbone_parameter.device,
        )
        self._no_split_modules = getattr(language_model, "_no_split_modules", [])
        self._init_score_head()

    @classmethod
    def from_config(cls, config: PretrainedConfig, **kwargs: Any):
        config = deepcopy(config)
        text_config = _text_config(config)
        # Transformers' from_config changes the default torch dtype, but FLA's
        # gated norm explicitly reads config.dtype. Keep both aligned before
        # constructing modules, including meta ranks, for FSDP master weights.
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype", text_config.dtype))
        if dtype is not None:
            text_config.dtype = dtype
        language_model = AutoModel.from_config(text_config, **kwargs)
        return cls(config, language_model)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

        model = cls.from_config(config, **kwargs)
        state_dict = get_state_dict_from_repo_id_or_path(pretrained_model_name_or_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if not _is_ignored_pretrained_key(key)]
        if unexpected:
            raise RuntimeError(
                "Unexpected pretrained keys for Qwen3.5 token critic: "
                f"{unexpected[:20]}"
            )
        missing = [key for key in missing if key != "score.weight"]
        if missing:
            raise RuntimeError(
                "Missing pretrained text-backbone keys for Qwen3.5 token critic: "
                f"{missing[:20]}"
            )
        return model

    def _init_score_head(self) -> None:
        std = getattr(_text_config(self.config), "initializer_range", 0.02)
        nn.init.normal_(self.score.weight, mean=0.0, std=std)

    def forward(self, *args, **kwargs) -> TokenCriticOutput:
        kwargs["return_dict"] = True
        outputs = self.model.language_model(*args, **kwargs)
        values = self.score(outputs.last_hidden_state)
        return TokenCriticOutput(
            logits=values,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        incompatible = super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not _is_ignored_pretrained_key(key)
        ]
        missing = [key for key in incompatible.missing_keys if key != "score.weight"]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.__class__.__name__}: missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}"
            )
        if missing:
            raise RuntimeError(
                "Missing pretrained text-backbone keys for Qwen3.5 token critic: "
                f"{missing[:20]}"
            )
        if unexpected:
            raise RuntimeError(
                "Unexpected pretrained keys for Qwen3.5 token critic: "
                f"{unexpected[:20]}"
            )
        return _IncompatibleKeys(
            [key for key in incompatible.missing_keys if key == "score.weight"],
            [
                key
                for key in incompatible.unexpected_keys
                if _is_ignored_pretrained_key(key)
            ],
        )

    def gradient_checkpointing_enable(self, *args, **kwargs) -> None:
        if hasattr(self.model.language_model, "gradient_checkpointing_enable"):
            self.model.language_model.gradient_checkpointing_enable(*args, **kwargs)
        else:
            super().gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.model.language_model, "gradient_checkpointing_disable"):
            self.model.language_model.gradient_checkpointing_disable()
        else:
            super().gradient_checkpointing_disable()


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", config)


def _hidden_size(config: PretrainedConfig) -> int:
    return int(getattr(_text_config(config), "hidden_size"))


def _is_ignored_pretrained_key(key: str) -> bool:
    return (
        key.startswith("model.visual.")
        or key.startswith("visual.")
        or key.startswith("lm_head.")
        or key.startswith("mtp.")
    )
