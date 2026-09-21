# SPDX-License-Identifier: Apache-2.0
"""SAO PPO alignment tests for the official GSM8K recipe."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from examples.math.sao_ppo import validate_contract

from areal.api.cli_args import PPOConfig, parse_cli_args, to_structured_cfg
from areal.trainer.ppo.validation import verify_gamma_one_episodic_returns
from areal.utils.lr_scheduler import get_num_warmup_steps

REPO_ROOT = Path(__file__).resolve().parents[1]
SAO_CONFIG = REPO_ROOT / "examples/math/sao_ppo.yaml"
GSM8K_PPO_CONFIG = REPO_ROOT / "examples/math/gsm8k_ppo.yaml"


def _set_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAO_TRIAL_NAME", "unit-test")
    monkeypatch.setenv("SAO_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("SAO_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("SAO_DATA_PATH", "openai/gsm8k")
    monkeypatch.setenv("SAO_ARTIFACT_ROOT", str(tmp_path / "artifacts"))


def _compose_config(path: Path) -> PPOConfig:
    cfg, _ = parse_cli_args(["--config", str(path)])
    cfg = to_structured_cfg(cfg, PPOConfig)
    config = OmegaConf.to_object(cfg)
    assert isinstance(config, PPOConfig)
    return config


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def test_sao_yaml_inherits_official_gsm8k_ppo_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SAO config keeps one source of truth except fixed warmup steps."""
    _set_env(monkeypatch, tmp_path)

    sao = _compose_config(SAO_CONFIG)
    official = _compose_config(GSM8K_PPO_CONFIG)

    assert sao.critic is not None
    assert sao.actor.optimizer is not None
    assert official.actor.optimizer is not None
    assert sao.critic.optimizer == sao.actor.optimizer
    for field in (
        "type",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "lr_scheduler_type",
        "gradient_clipping",
        "warmup_steps_proportion",
    ):
        assert getattr(sao.actor.optimizer, field) == getattr(
            official.actor.optimizer, field
        )
    assert official.actor.optimizer.warmup_steps is None
    assert sao.actor.optimizer.warmup_steps == 5
    assert get_num_warmup_steps(sao.actor.optimizer, 135) == 5
    assert sao.actor.eps_clip == official.actor.eps_clip == 0.4
    assert sao.actor.reward_scaling == official.actor.reward_scaling == 10.0
    assert sao.actor.reward_bias == official.actor.reward_bias == -0.5
    assert sao.actor.reward_clip == official.actor.reward_clip == 20.0
    assert sao.actor.recompute_logprob is official.actor.recompute_logprob is True
    assert sao.actor.use_decoupled_loss is official.actor.use_decoupled_loss is True

    rejection = sao.actor.rejection_sampling
    official_rejection = official.actor.rejection_sampling
    assert rejection == official_rejection
    assert rejection is not None
    assert (
        rejection.level,
        rejection.action,
        rejection.metric,
        rejection.upper,
        rejection.lower,
    ) == ("token", "mask", "ratio", 5.0, None)


def test_validate_contract_accepts_composed_sao_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live entrypoint contract accepts the resolved SAO config."""
    _set_env(monkeypatch, tmp_path)

    validate_contract(_compose_config(SAO_CONFIG))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("actor.optimizer.lr", 2.0e-5, "actor optimizer"),
        ("actor.optimizer.warmup_steps", 4, "actor optimizer"),
        ("critic.optimizer.weight_decay", 0.0, "critic optimizer"),
        ("critic.optimizer.warmup_steps", 4, "critic optimizer"),
        ("actor.eps_clip", 0.2, "actor.eps_clip"),
        ("actor.reward_scaling", 1.0, "actor.reward_scaling"),
        ("actor.recompute_logprob", False, "actor.recompute_logprob"),
    ],
)
def test_validate_contract_rejects_official_recipe_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    """Contract validation fails closed when official PPO knobs drift."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    if field.startswith("critic."):
        assert config.critic is not None
        config.critic.optimizer = copy.deepcopy(config.actor.optimizer)
    _set_nested(config, field, value)

    with pytest.raises(ValueError, match=error):
        validate_contract(config)


def test_return_probe_uses_official_reward_transform_for_terminal_outcomes() -> None:
    """Raw 0/1 outcomes become -5/+5 with the official GSM8K reward transform."""
    group = {
        "values": torch.zeros(2, 3),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "terminated": torch.tensor([True, True]),
        "truncated": torch.tensor([False, False]),
        "rewards": torch.tensor([0.0, 1.0]),
        "returns": torch.tensor([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]]),
        "loss_mask": torch.ones(2, 3, dtype=torch.bool),
    }

    report = verify_gamma_one_episodic_returns(
        [group],
        reward_scaling=10.0,
        reward_bias=-0.5,
        reward_clip=20.0,
    )

    assert report["passed"] is True
    assert report["terminated"] == 2
    assert report["truncated"] == 0
    assert report["max_abs_error"] == 0.0


def test_return_probe_clips_transformed_reward_before_truncated_bootstrap() -> None:
    """Reward clipping is applied before adding the real final-token bootstrap."""
    group = {
        "values": torch.tensor([[0.0, 0.0, 7.0], [0.0, 0.0, 2.0]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "terminated": torch.tensor([False, False]),
        "truncated": torch.tensor([True, True]),
        "rewards": torch.tensor([3.0, -3.0]),
        "returns": torch.tensor([[27.0, 27.0, 27.0], [-18.0, -18.0, -18.0]]),
        "loss_mask": torch.ones(2, 3, dtype=torch.bool),
    }

    report = verify_gamma_one_episodic_returns(
        [group],
        reward_scaling=10.0,
        reward_bias=-0.5,
        reward_clip=20.0,
    )

    assert report["passed"] is True
    assert report["terminated"] == 0
    assert report["truncated"] == 2
    assert report["max_abs_error"] == 0.0
