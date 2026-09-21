# SPDX-License-Identifier: Apache-2.0
"""SAO GRPO review-candidate contract tests."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset, DatasetDict
from omegaconf import OmegaConf

from examples.math.sao_grpo import validate_contract, write_step_count_evidence

from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg
from areal.utils.data import Normalization
from areal.utils.lr_scheduler import get_num_warmup_steps

REPO_ROOT = Path(__file__).resolve().parents[1]
SAO_CONFIG = REPO_ROOT / "examples/math/sao_grpo.yaml"
GSM8K_GRPO_CONFIG = REPO_ROOT / "examples/math/gsm8k_grpo.yaml"


def _set_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAO_TRIAL_NAME", "unit-test")
    monkeypatch.setenv("SAO_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("SAO_MODEL_PATH", "Qwen/Qwen3.5-4B-Base")
    monkeypatch.setenv("SAO_DATA_PATH", str(tmp_path / "dataset"))


def _compose_config(path: Path) -> GRPOConfig:
    cfg, _ = parse_cli_args(["--config", str(path)])
    cfg = to_structured_cfg(cfg, GRPOConfig)
    config = OmegaConf.to_object(cfg)
    assert isinstance(config, GRPOConfig)
    return config


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def test_sao_yaml_inherits_official_gsm8k_grpo_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SAO config preserves official GRPO optimizer and loss knobs."""
    _set_env(monkeypatch, tmp_path)

    sao = _compose_config(SAO_CONFIG)
    official = _compose_config(GSM8K_GRPO_CONFIG)

    assert sao.critic is None
    assert sao.ref is None
    assert sao.teacher is None
    assert sao.actor.optimizer is not None
    assert official.actor.optimizer is not None
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
    assert sao.actor.optimizer.lr == 6.0e-6
    assert sao.actor.optimizer.weight_decay == 0.017
    assert sao.actor.optimizer.warmup_steps == 5
    assert sao.actor.optimizer.warmup_steps_proportion == 0.001
    assert get_num_warmup_steps(sao.actor.optimizer, 134) == 5
    assert sao.actor.eps_clip == official.actor.eps_clip == 0.4
    assert sao.actor.reward_scaling == official.actor.reward_scaling == 10.0
    assert sao.actor.reward_bias == official.actor.reward_bias == -0.5
    assert sao.actor.recompute_logprob is official.actor.recompute_logprob is True
    assert sao.actor.use_decoupled_loss is official.actor.use_decoupled_loss is True
    assert sao.actor.kl_ctl == official.actor.kl_ctl == 0.0

    assert sao.actor.rejection_sampling == official.actor.rejection_sampling
    assert sao.actor.reward_norm == official.actor.reward_norm
    assert sao.actor.adv_norm == official.actor.adv_norm
    assert sao.actor.reward_norm.group_size == sao.gconfig.n_samples == 4
    assert sao.train_dataset.drop_last is True
    assert (sao.actor.adv_norm.mean_level, sao.actor.adv_norm.std_level) == (
        "batch",
        "batch",
    )


def test_validate_contract_accepts_composed_sao_grpo_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live entrypoint contract accepts the resolved SAO GRPO config."""
    _set_env(monkeypatch, tmp_path)

    validate_contract(_compose_config(SAO_CONFIG))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("actor.optimizer.lr", 1.0e-5, "actor.optimizer.lr"),
        ("actor.optimizer.weight_decay", 0.0, "actor.optimizer.weight_decay"),
        ("actor.optimizer.warmup_steps", None, "actor.optimizer.warmup_steps"),
        ("actor.optimizer.warmup_steps_proportion", 0.0, "warmup_steps_proportion"),
        ("actor.eps_clip", 0.2, "actor.eps_clip"),
        ("actor.reward_scaling", 1.0, "actor.reward_scaling"),
        ("actor.reward_bias", 0.0, "actor.reward_bias"),
        ("actor.recompute_logprob", False, "actor.recompute_logprob"),
        ("actor.use_decoupled_loss", False, "actor.use_decoupled_loss"),
        ("actor.kl_ctl", 0.1, "actor.kl_ctl"),
    ],
)
def test_validate_contract_rejects_official_recipe_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    """Contract validation fails closed when official GRPO knobs drift."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    _set_nested(config, field, value)

    with pytest.raises(ValueError, match=error):
        validate_contract(config)


def test_validate_contract_rejects_critic_ref_and_group_size_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SAO review candidate is actor-only and keeps reward groups at N."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    config_with_critic = copy.deepcopy(config)
    config_with_critic.critic = copy.deepcopy(config.actor)
    with pytest.raises(ValueError, match="critic"):
        validate_contract(config_with_critic)

    config_with_ref = copy.deepcopy(config)
    config_with_ref.ref = copy.deepcopy(config.actor)
    with pytest.raises(ValueError, match="ref"):
        validate_contract(config_with_ref)

    config.actor.reward_norm.group_size = 8
    with pytest.raises(ValueError, match="group_size"):
        validate_contract(config)


def test_step_count_evidence_records_134_step_epoch_and_five_step_warmup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dataset-size evidence catches prompt, sample, tail, and warmup drift."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    train_rows = 17157
    train = Dataset.from_dict(
        {
            "source_id": [f"dapo:{i}" for i in range(train_rows)],
            "prompt": ["p"] * train_rows,
            "answer": ["1"] * train_rows,
        }
    )
    valid = Dataset.from_dict(
        {
            "source_id": [f"test:{i}" for i in range(700)],
            "prompt": ["p"] * 700,
            "answer": ["1"] * 700,
        }
    )
    DatasetDict({"train": train, "test": valid}).save_to_disk(tmp_path / "dataset")

    payload = write_step_count_evidence(
        config, train, valid, tmp_path / "run" / "evidence"
    )

    assert payload["train_dataset_rows"] == 17157
    assert payload["valid_dataset_rows"] == 700
    assert payload["train_prompts_per_step"] == 128
    assert payload["samples_per_prompt"] == 4
    assert payload["expected_optimizer_steps"] == 134
    assert payload["expected_consumed_prompts"] == 17152
    assert payload["dropped_tail_prompts"] == 5
    assert payload["expected_train_trajectories"] == 68608
    assert payload["last_step_prompt_count"] == 128
    assert payload["last_step_trajectory_count"] == 512
    assert payload["resolved_warmup_steps"] == 5


def test_group_reward_norm_requires_whole_n4_prompt_groups() -> None:
    """Splitting a GRPO group into singletons destroys relative reward signal."""
    norm = Normalization(_compose_config(GSM8K_GRPO_CONFIG).actor.reward_norm)
    rewards = norm(
        torch.tensor([0.0, 1.0, 1.0, 0.0]),
        group_sizes=[4],
    )
    singleton_rewards = norm(
        torch.tensor([0.0, 1.0, 1.0, 0.0]),
        group_sizes=[1, 1, 1, 1],
    )

    assert rewards[0] < 0
    assert rewards[1] > 0
    assert torch.count_nonzero(singleton_rewards) == 0


def test_small_preflight_does_not_relax_formal_eval_count(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    config.train_dataset.batch_size = 4
    train, valid = list(range(9)), list(range(5))
    report = write_step_count_evidence(
        config, train, valid, tmp_path / "probe", preflight=True
    )
    assert report["expected_optimizer_steps"] == 2
    assert report["dropped_tail_prompts"] == 1
    with pytest.raises(ValueError, match="700"):
        write_step_count_evidence(config, train, valid, tmp_path / "formal")
