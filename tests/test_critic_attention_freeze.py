# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from areal.api.cli_args import PPOCriticConfig, TrainEngineConfig
from areal.engine.fsdp_utils.critic_freeze import (
    apply_critic_attention_freeze,
    validate_critic_freeze_manifest,
)


class _ToyHybridBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = nn.Linear(width, width)
        self.linear_attn = nn.Linear(width, width)
        self.mlp = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(x) + self.linear_attn(x)
        return x + self.mlp(torch.relu(x))


class _ToyHybridCritic(nn.Module):
    def __init__(self, vocab_size: int = 8, width: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.layers = nn.ModuleList([_ToyHybridBlock(width)])
        self.score = nn.Linear(width, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids).mean(dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.score(x).squeeze(-1)


class _OnlySelfAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(3, 3)
        self.output = nn.Linear(3, 1)


def _clone_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def test_apply_critic_attention_freeze_freezes_only_attention_parameters():
    """Frozen attention weights stay fixed while upstream and head weights train."""
    torch.manual_seed(1)
    model = _ToyHybridCritic()
    manifest = apply_critic_attention_freeze(model, enabled=True, is_critic=True)
    before = _clone_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    prediction = model(torch.tensor([[1, 2, 3], [3, 2, 1]]))
    loss = (prediction - torch.tensor([0.25, -0.5])).pow(2).mean()
    loss.backward()

    for name, param in model.named_parameters():
        assert torch.isfinite(param).all()
        if name in manifest["frozen_parameters"]:
            assert param.grad is None
        else:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    optimizer.step()

    assert manifest == {
        "version": 1,
        "enabled": True,
        "frozen_parameters": [
            "layers.0.linear_attn.bias",
            "layers.0.linear_attn.weight",
            "layers.0.self_attn.bias",
            "layers.0.self_attn.weight",
        ],
        "frozen_numel": 40,
        "trainable_numel": 57,
    }
    for name in manifest["frozen_parameters"]:
        torch.testing.assert_close(model.get_parameter(name), before[name])
    for name in ("embed.weight", "layers.0.mlp.weight", "score.weight"):
        assert not torch.equal(model.get_parameter(name), before[name])


def test_apply_critic_attention_freeze_disabled_does_not_mutate_model():
    model = _ToyHybridCritic()
    before_requires_grad = {
        name: param.requires_grad for name, param in model.named_parameters()
    }

    manifest = apply_critic_attention_freeze(model, enabled=False, is_critic=False)

    assert manifest == {
        "version": 1,
        "enabled": False,
        "frozen_parameters": [],
        "frozen_numel": 0,
        "trainable_numel": sum(param.numel() for param in model.parameters()),
    }
    assert {
        name: param.requires_grad for name, param in model.named_parameters()
    } == before_requires_grad


def test_apply_critic_attention_freeze_rejects_actor_without_mutation():
    model = _ToyHybridCritic()
    before_requires_grad = {
        name: param.requires_grad for name, param in model.named_parameters()
    }

    with pytest.raises(ValueError, match="only be enabled for critics"):
        apply_critic_attention_freeze(model, enabled=True, is_critic=False)

    assert {
        name: param.requires_grad for name, param in model.named_parameters()
    } == before_requires_grad


def test_apply_critic_attention_freeze_rejects_missing_attention_type_before_mutation():
    model = _OnlySelfAttentionModel()
    before_requires_grad = {
        name: param.requires_grad for name, param in model.named_parameters()
    }

    with pytest.raises(ValueError, match="missing linear_attn"):
        apply_critic_attention_freeze(model, enabled=True, is_critic=True)

    assert {
        name: param.requires_grad for name, param in model.named_parameters()
    } == before_requires_grad


def test_validate_critic_freeze_manifest_allows_legacy_only_when_disabled():
    expected_disabled = {
        "version": 1,
        "enabled": False,
        "frozen_parameters": [],
        "frozen_numel": 0,
        "trainable_numel": 3,
    }
    validate_critic_freeze_manifest(expected_disabled, None)

    expected_enabled = {
        "version": 1,
        "enabled": True,
        "frozen_parameters": ["layers.0.self_attn.weight"],
        "frozen_numel": 9,
        "trainable_numel": 3,
    }
    with pytest.raises(ValueError, match="missing critic attention freeze manifest"):
        validate_critic_freeze_manifest(expected_enabled, None)


def test_validate_critic_freeze_manifest_rejects_mismatch():
    expected = {
        "version": 1,
        "enabled": True,
        "frozen_parameters": ["layers.0.self_attn.weight"],
        "frozen_numel": 9,
        "trainable_numel": 3,
    }
    saved = {
        **expected,
        "frozen_parameters": ["layers.0.linear_attn.weight"],
    }

    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_critic_freeze_manifest(expected, saved)


def test_freeze_critic_attention_config_defaults_disabled():
    assert (
        TrainEngineConfig(
            backend="fsdp:d1",
            experiment_name="test",
            trial_name="actor",
        ).freeze_critic_attention
        is False
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "fsdp:d1", "is_critic": False},
        {"backend": "archon:d1", "is_critic": True},
        {"backend": "fsdp:d1", "is_critic": True, "use_lora": True},
    ],
)
def test_freeze_critic_attention_config_rejects_invalid_roles(kwargs):
    with pytest.raises(ValueError, match="requires an FSDP critic without LoRA"):
        TrainEngineConfig(
            experiment_name="test",
            trial_name="invalid",
            freeze_critic_attention=True,
            **kwargs,
        )


def test_ppo_critic_config_accepts_fsdp_critic_attention_freeze():
    config = PPOCriticConfig(
        backend="fsdp:d4",
        experiment_name="test",
        trial_name="critic",
        is_critic=True,
        freeze_critic_attention=True,
    )

    assert config.freeze_critic_attention is True
