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

from examples.math.sao_grpo import (
    _check_actor_step_metrics,
    validate_contract,
    write_step_count_evidence,
)
from scripts.sao.async_eval import SaoGRPOConfig

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
    cfg = to_structured_cfg(cfg, SaoGRPOConfig if path == SAO_CONFIG else GRPOConfig)
    config = OmegaConf.to_object(cfg)
    assert isinstance(config, GRPOConfig)
    return config


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _actor_metrics(
    *, grad_norm: float, adv_min: float, adv_max: float, valid_tokens: float = 8.0
) -> dict[str, float]:
    return {
        "ppo_actor/update/grad_norm": grad_norm,
        "ppo_actor/update/update_successful": 1.0,
        "ppo_actor/update/optimizer_steps_since_init": 2.0,
        "ppo_actor/advantages/min": adv_min,
        "ppo_actor/advantages/max": adv_max,
        "ppo_actor/update/n_valid_tokens": valid_tokens,
        "ppo_actor/update/actor_loss/avg": 0.0,
    }


def test_actor_metric_guard_accepts_zero_gradient_only_for_zero_advantage() -> None:
    payload = _check_actor_step_metrics(
        _actor_metrics(grad_norm=0.0, adv_min=0.0, adv_max=0.0),
        completed_step=2,
    )

    assert payload["zero_gradient_allowed"] == {
        "reason": "zero_advantage",
        "advantages_min": 0.0,
        "advantages_max": 0.0,
        "n_valid_tokens": 8.0,
    }


def test_actor_metric_guard_rejects_unqualified_zero_gradient() -> None:
    with pytest.raises(RuntimeError, match="zero gradient"):
        _check_actor_step_metrics(
            _actor_metrics(grad_norm=0.0, adv_min=-1.0, adv_max=1.0),
            completed_step=2,
        )
    with pytest.raises(RuntimeError, match="zero gradient"):
        _check_actor_step_metrics(
            _actor_metrics(grad_norm=0.0, adv_min=0.0, adv_max=0.0, valid_tokens=0.0),
            completed_step=2,
        )


def test_actor_metric_guard_still_rejects_nan_and_skipped_updates() -> None:
    with pytest.raises(RuntimeError, match="zero gradient"):
        _check_actor_step_metrics(
            _actor_metrics(grad_norm=float("nan"), adv_min=0.0, adv_max=0.0),
            completed_step=2,
        )
    data = _actor_metrics(grad_norm=0.0, adv_min=0.0, adv_max=0.0)
    data["ppo_actor/update/update_successful"] = 0.0
    with pytest.raises(RuntimeError, match="skipped"):
        _check_actor_step_metrics(data, completed_step=2)


def test_sao_yaml_applies_miles_grpo_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SAO config pins the reviewed Miles GRPO optimizer and loss knobs."""
    _set_env(monkeypatch, tmp_path)

    sao = _compose_config(SAO_CONFIG)
    official = _compose_config(GSM8K_GRPO_CONFIG)

    assert sao.critic is None
    assert sao.ref is None
    assert sao.teacher is None
    assert sao.actor.optimizer is not None
    assert sao.actor.optimizer.type == "adam"
    assert sao.actor.optimizer.lr == 1.0e-6
    assert sao.actor.optimizer.weight_decay == 0.1
    assert sao.actor.optimizer.beta1 == 0.9
    assert sao.actor.optimizer.beta2 == 0.98
    assert sao.actor.optimizer.eps == 1.0e-8
    assert sao.actor.optimizer.lr_scheduler_type == "constant"
    assert sao.actor.optimizer.gradient_clipping == 1.0
    assert sao.actor.optimizer.warmup_steps == 0
    assert sao.actor.optimizer.warmup_steps_proportion == 0.0
    assert get_num_warmup_steps(sao.actor.optimizer, 536) == 0
    assert sao.actor.eps_clip == 0.2
    assert sao.actor.eps_clip_higher == 0.28
    assert sao.actor.reward_scaling == 1.0
    assert sao.actor.reward_bias == 0.0
    assert sao.actor.recompute_logprob is official.actor.recompute_logprob is True
    assert sao.actor.use_decoupled_loss is official.actor.use_decoupled_loss is True
    assert sao.actor.kl_ctl == official.actor.kl_ctl == 0.0
    assert sao.actor.discount == 1.0
    assert sao.actor.gae_lambda == 1.0
    assert sao.actor.dtype == "bfloat16"
    assert sao.actor.optimizer_dtype == "float32"

    assert sao.actor.rejection_sampling == official.actor.rejection_sampling
    assert sao.actor.reward_norm is not None
    assert sao.actor.reward_norm.mean_level == "group"
    assert sao.actor.reward_norm.std_level == "group"
    assert sao.actor.reward_norm.group_size == sao.gconfig.n_samples == 8
    assert sao.actor.adv_norm is None
    assert sao.train_dataset.drop_last is True
    assert sao.train_dataset.batch_size == 32
    assert sao.gconfig.max_new_tokens == 8192
    assert sao.gconfig.max_tokens == 9216
    assert sao.eval_gconfig.n_samples == 2
    assert sao.eval_gconfig.max_new_tokens == 8192
    assert sao.rollout.max_head_offpolicyness == 2


def test_validate_contract_accepts_composed_sao_grpo_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live entrypoint contract accepts the resolved SAO GRPO config."""
    _set_env(monkeypatch, tmp_path)

    validate_contract(_compose_config(SAO_CONFIG))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("actor.backend", "fsdp:d3p1t1", "actor.backend"),
        ("rollout.backend", "sglang:d4p1t1", "rollout.backend"),
        ("evaluation_rollout.backend", "sglang:d2p1t1", "evaluation_rollout.backend"),
        ("cluster.n_gpus_per_node", 7, "cluster.n_gpus_per_node"),
        ("actor.optimizer.lr", 1.0e-5, "actor.optimizer.lr"),
        ("actor.optimizer.lr", 6.0e-6, "actor.optimizer.lr"),
        ("actor.optimizer.weight_decay", 0.0, "actor.optimizer.weight_decay"),
        ("actor.optimizer.weight_decay", 0.017, "actor.optimizer.weight_decay"),
        ("actor.optimizer.beta2", 0.999, "actor.optimizer.beta2"),
        ("actor.optimizer.warmup_steps", 5, "actor.optimizer.warmup_steps"),
        ("actor.optimizer.warmup_steps_proportion", 0.001, "warmup_steps_proportion"),
        ("actor.eps_clip", 0.4, "actor.eps_clip"),
        ("actor.eps_clip_higher", None, "actor.eps_clip_higher"),
        ("actor.reward_scaling", 10.0, "actor.reward_scaling"),
        ("actor.reward_bias", -0.5, "actor.reward_bias"),
        ("actor.recompute_logprob", False, "actor.recompute_logprob"),
        ("actor.use_decoupled_loss", False, "actor.use_decoupled_loss"),
        ("actor.kl_ctl", 0.1, "actor.kl_ctl"),
        ("actor.discount", 0.99, "actor.discount"),
        ("actor.gae_lambda", 0.95, "actor.gae_lambda"),
        ("actor.dtype", "float32", "actor.dtype"),
        ("actor.optimizer_dtype", "bfloat16", "actor.optimizer_dtype"),
        ("train_dataset.batch_size", 128, "train_dataset.batch_size"),
        ("gconfig.n_samples", 4, "gconfig.n_samples"),
        ("eval_gconfig.n_samples", 4, "eval_gconfig.n_samples"),
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

    config.actor.reward_norm.group_size = 4
    with pytest.raises(ValueError, match="group_size"):
        validate_contract(config)

    config = _compose_config(SAO_CONFIG)
    config.actor.adv_norm = copy.deepcopy(
        _compose_config(GSM8K_GRPO_CONFIG).actor.adv_norm
    )
    with pytest.raises(ValueError, match="adv_norm"):
        validate_contract(config)


def test_step_count_evidence_records_536_step_epoch_and_zero_warmup(
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
    assert payload["train_prompts_per_step"] == 32
    assert payload["samples_per_prompt"] == 8
    assert payload["expected_optimizer_steps"] == 536
    assert payload["expected_consumed_prompts"] == 17152
    assert payload["dropped_tail_prompts"] == 5
    assert payload["expected_train_trajectories"] == 137216
    assert payload["last_step_prompt_count"] == 32
    assert payload["last_step_trajectory_count"] == 256
    assert payload["resolved_warmup_steps"] == 0


def test_group_reward_norm_requires_whole_n8_prompt_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Splitting a GRPO group into singletons destroys relative reward signal."""
    _set_env(monkeypatch, tmp_path)

    norm = Normalization(_compose_config(SAO_CONFIG).actor.reward_norm)
    rewards = norm(
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.25, 0.75, 0.5, 0.5]),
        group_sizes=[8],
    )
    singleton_rewards = norm(
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.25, 0.75, 0.5, 0.5]),
        group_sizes=[1, 1, 1, 1, 1, 1, 1, 1],
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


def test_native_allocator_separates_training_rollout_and_eval(monkeypatch, tmp_path):
    """Exercise native allocation on CPU; no workers or CUDA devices are started."""
    from areal.api.alloc_mode import ModelAllocation
    from areal.infra.scheduler.local import LocalScheduler

    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    scheduler = object.__new__(LocalScheduler)
    scheduler.gpu_devices = list(range(8))
    scheduler._gpu_counter = 0
    allocations = {}
    for role, backend in (
        ("actor", config.actor.backend),
        ("rollout", config.rollout.backend),
        ("evaluation", config.evaluation_rollout.backend),
    ):
        strategy = ModelAllocation.from_str(backend).parallel
        allocations[role] = scheduler._allocate_gpus(strategy.world_size)
    assert allocations == {
        "actor": [0, 1, 2, 3],
        "rollout": [4, 5, 6],
        "evaluation": [7],
    }
    assert len(set(sum(allocations.values(), []))) == 8
    assert config.gconfig.n_samples == config.actor.reward_norm.group_size == 8
    assert config.evaluation_rollout.scheduling_spec[0].gpu == 1


def test_dedicated_eval_cannot_colocate_or_drop_gpu_reservation(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    config.evaluation_rollout.scheduling_strategy.type = "colocation"
    with pytest.raises(ValueError, match="separate GPU"):
        validate_contract(config)
    config.evaluation_rollout.scheduling_strategy.type = "separation"
    config.evaluation_rollout.scheduling_spec[0].gpu = 0
    with pytest.raises(ValueError, match="one GPU"):
        validate_contract(config)


@pytest.mark.parametrize("role", ["actor", "rollout"])
def test_training_and_rollout_cannot_share_devices(monkeypatch, tmp_path, role):
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    getattr(config, role).scheduling_strategy.type = "colocation"
    with pytest.raises(ValueError, match="separate GPU"):
        validate_contract(config)
