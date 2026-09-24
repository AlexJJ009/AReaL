# SPDX-License-Identifier: Apache-2.0

"""Resolved example recipe must not silently select PPO or a fake value artifact."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from examples.math.gsm8k_sao import validate_sao_recipe
from scripts.sao.async_eval import SaoPPOConfig

from areal.api.cli_args import to_structured_cfg


@pytest.fixture
def recipe(monkeypatch, tmp_path):
    monkeypatch.setenv("SAO_MODEL_PATH", "Qwen/Qwen3.5-4B-Base")
    monkeypatch.setenv("SAO_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("SAO_CRITIC_PATH", "Qwen/Qwen3.5-4B-Base")
    path = Path(__file__).parents[1] / "examples/math/gsm8k_sao.yaml"
    return OmegaConf.to_object(to_structured_cfg(OmegaConf.load(path), SaoPPOConfig))


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


def test_recipe_allows_opt_in_critic_attention_freeze(recipe):
    assert not recipe.critic.freeze_critic_attention
    assert not recipe.actor.freeze_critic_attention
    recipe.critic.freeze_critic_attention = True
    recipe.critic.__post_init__()
    validate_sao_recipe(recipe, allow_base_critic=True)


def test_recipe_rejects_clipped_value_loss(recipe):
    recipe.critic.eps_clip = 0.5
    with pytest.raises(ValueError, match="MSE"):
        validate_sao_recipe(recipe, allow_base_critic=True)


def test_dedicated_validation_recipe(recipe):
    from scripts.sao.async_eval import AsyncEvalPPOTrainer

    assert recipe.actor.backend == "fsdp:d4p1t1"
    assert recipe.critic.backend == recipe.actor.backend
    assert recipe.critic.scheduling_strategy.target == "actor"
    assert recipe.rollout.backend == "sglang:d3p1t1"
    assert recipe.evaluation_rollout.backend == "sglang:d1p1t1"
    assert recipe.eval_gconfig.n_samples == 2
    assert recipe.valid_dataset.split == "test"
    assert recipe.train_dataset.scheduling_spec is None
    assert recipe.valid_dataset.scheduling_spec is None
    assert recipe.saver.freq_steps == recipe.evaluator.freq_steps == 20
    AsyncEvalPPOTrainer._validate_save_eval_sync(recipe)
    recipe.evaluator.freq_steps = 21
    with pytest.raises(ValueError, match="frequencies"):
        AsyncEvalPPOTrainer._validate_save_eval_sync(recipe)


def test_recipe_uses_8k_generation_with_consistent_budgets(recipe):
    assert recipe.gconfig.max_new_tokens == recipe.eval_gconfig.max_new_tokens == 8192
    assert recipe.gconfig.max_tokens == recipe.sglang.context_length == 9216
    assert recipe.actor.mb_spec.max_tokens_per_mb == 9216
    assert recipe.critic.mb_spec.max_tokens_per_mb == 9216


def test_recipe_requires_no_warmup_or_critic_only_stage(recipe):
    assert recipe.critic.optimizer.warmup_steps == 0
    assert recipe.actor.optimizer.warmup_steps == 0
    assert recipe.num_critic_only_steps == 0
    recipe.critic.optimizer.warmup_steps = 10
    with pytest.raises(ValueError, match="warmup"):
        validate_sao_recipe(recipe, allow_base_critic=True)
