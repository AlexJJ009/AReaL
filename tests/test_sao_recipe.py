# SPDX-License-Identifier: Apache-2.0

"""Resolved example recipe must not silently select PPO or a fake value artifact."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from examples.math.gsm8k_sao import validate_sao_recipe

from areal.api.cli_args import PPOConfig, to_structured_cfg


@pytest.fixture
def recipe(monkeypatch, tmp_path):
    monkeypatch.setenv("SAO_MODEL_PATH", "Qwen/Qwen3.5-4B-Base")
    monkeypatch.setenv("SAO_RUN_ROOT", str(tmp_path))
    monkeypatch.delenv("SAO_VALUE_PATH", raising=False)
    path = Path(__file__).parents[1] / "examples/math/gsm8k_sao.yaml"
    return OmegaConf.to_object(to_structured_cfg(OmegaConf.load(path), PPOConfig))


def test_resolved_recipe_uses_paper_mechanisms_and_explicit_base_critic(recipe):
    validate_sao_recipe(recipe, allow_base_critic=True)
    assert recipe.train_dataset.batch_size == 128
    assert recipe.critic.path == recipe.actor.path
    assert recipe.actor.should_compute_prox_logp() is False
    with pytest.raises(ValueError, match="pretrained value_contract"):
        validate_sao_recipe(recipe)


@pytest.mark.parametrize("role, lr", [("actor", 6e-6), ("critic", 1e-6)])
def test_recipe_rejects_wrong_online_learning_rate(recipe, role, lr):
    getattr(recipe, role).optimizer.lr = lr
    with pytest.raises(ValueError, match="paper LR"):
        validate_sao_recipe(recipe, allow_base_critic=True)


def test_recipe_rejects_synthetic_artifact_as_pretrained(recipe):
    recipe.critic.value_contract = {"require_pretrained": False}
    with pytest.raises(ValueError, match="pretrained qualification"):
        validate_sao_recipe(recipe)


def test_recipe_rejects_clipped_value_loss(recipe):
    recipe.critic.eps_clip = 0.5
    with pytest.raises(ValueError, match="MSE"):
        validate_sao_recipe(recipe, allow_base_critic=True)
