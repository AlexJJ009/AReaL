# SPDX-License-Identifier: Apache-2.0
"""SAO PPO entrypoint contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

import examples.math.sao_ppo as sao_ppo

from areal.api.cli_args import PPOConfig, parse_cli_args, to_structured_cfg
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.validation import verify_gamma_one_episodic_returns
from areal.utils.lr_scheduler import get_num_warmup_steps

REPO_ROOT = Path(__file__).resolve().parents[1]
SAO_CONFIG = REPO_ROOT / "examples/math/sao_ppo.yaml"


def _write_hf_checkpoint(path: Path, keys: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    shard = "model-00001-of-00001.safetensors"
    tensors = {
        key: torch.zeros((1, 4), dtype=torch.bfloat16)
        if key == "score.weight"
        else torch.zeros((4, 4), dtype=torch.float32)
        for key in keys
    }
    save_file(tensors, path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: shard for key in keys}}) + "\n",
        encoding="utf-8",
    )


def _set_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    actor_path = tmp_path / "actor-base"
    critic_path = tmp_path / "critic-pretrained"
    _write_hf_checkpoint(actor_path, ["model.embed_tokens.weight"])
    _write_hf_checkpoint(critic_path, ["model.embed_tokens.weight", "score.weight"])
    monkeypatch.setenv("SAO_TRIAL_NAME", "unit-test")
    monkeypatch.setenv("SAO_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("SAO_MODEL_PATH", str(actor_path))
    monkeypatch.setenv("SAO_CRITIC_PATH", str(critic_path))
    monkeypatch.setenv("SAO_DATA_PATH", "openai/gsm8k")
    monkeypatch.setenv("SAO_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return actor_path, critic_path


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


def test_sao_yaml_resolves_current_async_ppo_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The checked-in SAO config resolves to the current PPO launch contract."""
    actor_path, critic_path = _set_env(monkeypatch, tmp_path)

    config = _compose_config(SAO_CONFIG)

    assert config.critic is not None
    assert config.actor.path == str(actor_path)
    assert config.critic.path == str(critic_path)
    assert config.actor.path != config.critic.path
    assert config.actor.init_from_scratch is False
    assert config.critic.init_from_scratch is False
    assert config.gconfig.n_samples == 8
    assert config.eval_gconfig.n_samples == 2
    assert config.evaluator.eval_before_train is True
    assert config.evaluator.freq_steps == 50
    assert config.saver.freq_steps == 50
    assert config.recover.freq_steps == 50
    assert config.train_dataset.batch_size == 16
    assert config.num_critic_only_steps == 0
    assert config.stats_logger.wandb.mode == "online"
    assert config.actor.gae_lambda == 0.95
    assert config.actor.critic_gae_lambda == 1.0
    assert config.actor.reward_scaling == 1.0
    assert config.actor.reward_bias == 0.0
    assert config.actor.reward_norm is None
    assert config.actor.adv_norm is None
    assert config.actor.eps_clip == 0.2
    assert config.actor.eps_clip_higher == 0.28
    assert config.critic.eps_clip == 0.2
    assert config.actor.recompute_logprob is True
    assert config.actor.use_decoupled_loss is True
    assert config.actor.prox_logp_method == "recompute"
    assert config.actor.loss_reduction == "sequence_mean"
    assert config.critic.loss_reduction == "sequence_mean"
    assert config.actor.ppo_n_minibatches == 1
    assert config.critic.ppo_n_minibatches == 1

    assert config.actor.optimizer is not None
    assert config.critic.optimizer is not None
    for optimizer, lr in (
        (config.actor.optimizer, 1.0e-6),
        (config.critic.optimizer, 5.0e-6),
    ):
        assert optimizer.type == "adam"
        assert optimizer.lr == lr
        assert optimizer.weight_decay == 0.01
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.98
        assert optimizer.eps == 1.0e-8
        assert optimizer.lr_scheduler_type == "constant"
        assert optimizer.gradient_clipping == 1.0
        assert optimizer.warmup_steps == 0
        assert optimizer.warmup_steps_proportion == 0.0
        assert get_num_warmup_steps(optimizer, 1073) == 0

    rejection = config.actor.rejection_sampling
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

    sao_ppo.validate_contract(_compose_config(SAO_CONFIG))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("actor.gae_lambda", 1.0, "actor.gae_lambda"),
        ("actor.loss_reduction", "token_mean", "actor.loss_reduction"),
        ("critic.loss_reduction", "token_mean", "critic.loss_reduction"),
        ("actor.critic_gae_lambda", 0.95, "actor.critic_gae_lambda"),
        ("actor.reward_bias", -0.5, "actor.reward_bias"),
        ("actor.adv_norm", object(), "actor.adv_norm"),
        ("actor.eps_clip", 0.4, "actor.eps_clip"),
        ("actor.eps_clip_higher", None, "actor.eps_clip_higher"),
        ("actor.recompute_logprob", False, "actor.recompute_logprob"),
        ("actor.prox_logp_method", "reuse_train_logp", "actor.prox_logp_method"),
        ("actor.optimizer.lr", 2.0e-6, "actor optimizer"),
        ("actor.optimizer.warmup_steps_proportion", 0.001, "actor optimizer"),
        ("actor.optimizer.warmup_steps", 5, "actor optimizer"),
        ("critic.optimizer.warmup_steps", 5, "critic optimizer"),
        ("critic.optimizer.lr", 1.0e-6, "critic optimizer"),
        ("critic.eps_clip", 0.5, "critic.eps_clip"),
        ("gconfig.n_samples", 4, "gconfig.n_samples"),
        ("train_dataset.batch_size", 128, "train_dataset.batch_size"),
        ("num_critic_only_steps", 1, "num_critic_only_steps"),
        ("eval_gconfig.n_samples", 4, "eval_gconfig.n_samples"),
        ("evaluator.freq_steps", 20, "evaluator.freq_steps"),
        ("evaluator.eval_before_train", False, "evaluator.eval_before_train"),
        ("saver.freq_steps", 20, "saver.freq_steps"),
        ("recover.freq_steps", 20, "recover.freq_steps"),
    ],
)
def test_validate_contract_rejects_sao_recipe_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    """Contract validation fails closed when launch-critical PPO knobs drift."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    _set_nested(config, field, value)

    with pytest.raises(ValueError, match=error):
        sao_ppo.validate_contract(config)


def test_validate_contract_allows_smaller_preflight_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preflight keeps the old small-batch escape hatch for smoke checks."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    config.gconfig.n_samples = 1
    config.train_dataset.batch_size = 2

    sao_ppo.validate_contract(config, preflight=True)


def test_validate_contract_rejects_same_actor_and_critic_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Base actor snapshot must not be used as the pretrained critic."""
    actor_path, _critic_path = _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    assert config.critic is not None
    config.critic.path = str(actor_path)

    with pytest.raises(ValueError, match="must differ"):
        sao_ppo.validate_contract(config)


def test_validate_hf_checkpoint_paths_requires_critic_score_weight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The critic path must be a real safetensors HF value head checkpoint."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    sao_ppo.validate_hf_checkpoint_paths(config)

    bad_critic = tmp_path / "base-without-score-head"
    _write_hf_checkpoint(bad_critic, ["model.embed_tokens.weight"])
    assert config.critic is not None
    config.critic.path = str(bad_critic)

    with pytest.raises(ValueError, match="score.weight"):
        sao_ppo.validate_hf_checkpoint_paths(config)


def test_check_config_does_not_create_run_root_or_trainer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--check-config parses and validates locally without launching or saving."""
    _set_env(monkeypatch, tmp_path)
    run_root = tmp_path / "run"

    def fail_trainer(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("--check-config must not instantiate PPOTrainer")

    monkeypatch.setattr(sao_ppo, "PPOTrainer", fail_trainer)

    sao_ppo.main(["--config", str(SAO_CONFIG), "--check-config"])

    assert not run_root.exists()
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["critic"]["path"] == str(tmp_path / "critic-pretrained")


def test_actor_dual_lambda_path_writes_raw_critic_returns_without_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Actor lambda .95 can coexist with critic lambda 1 raw 0/1 targets."""
    _set_env(monkeypatch, tmp_path)
    config = _compose_config(SAO_CONFIG)
    actor = PPOActor(config.actor, engine=object())
    batch = {
        "input_ids": torch.arange(10, dtype=torch.long).view(2, 5),
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "loss_mask": torch.tensor(
            [[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.float32
        ),
        "logprobs": torch.zeros(2, 5, dtype=torch.float32),
        "values": torch.tensor(
            [[0.0, 0.2, 0.4, 0.6, 999.0], [0.0, -0.2, -0.4, -0.6, 777.0]],
            dtype=torch.float32,
        ),
        "rewards": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "terminated": torch.tensor([False, False]),
        "truncated": torch.tensor([True, True]),
        "bootstrap_mask": torch.tensor([False, False]),
    }

    result = actor._compute_advantages(batch)

    expected_returns = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 999.0], [1.0, 1.0, 1.0, 1.0, 777.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result["returns"], expected_returns, rtol=0.0, atol=1e-6)
    report = verify_gamma_one_episodic_returns(
        [result],
        reward_scaling=1.0,
        reward_bias=0.0,
        reward_clip=20.0,
    )
    assert report["passed"] is True
    assert report["terminated"] == 0
    assert report["truncated"] == 2
    assert report["bootstrapped"] == 0
    assert report["max_abs_error"] <= 1e-6


def test_return_probe_uses_raw_reward_for_terminal_outcomes() -> None:
    """Raw 0/1 outcomes remain 0/1 under the SAO reward contract."""
    group = {
        "values": torch.zeros(2, 3),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "terminated": torch.tensor([True, True]),
        "truncated": torch.tensor([False, False]),
        "bootstrap_mask": torch.tensor([False, False]),
        "rewards": torch.tensor([0.0, 1.0]),
        "returns": torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "loss_mask": torch.ones(2, 3, dtype=torch.bool),
    }

    report = verify_gamma_one_episodic_returns(
        [group],
        reward_scaling=1.0,
        reward_bias=0.0,
        reward_clip=20.0,
    )

    assert report["passed"] is True
    assert report["terminated"] == 2
    assert report["truncated"] == 0
    assert report["bootstrapped"] == 0
    assert report["max_abs_error"] == 0.0


def test_return_probe_bootstrap_is_explicit_and_measured() -> None:
    """The shared oracle still reports bootstrap use for truncated trajectories."""
    group = {
        "values": torch.tensor([[0.0, 0.0, 7.0], [0.0, 0.0, 2.0]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "terminated": torch.tensor([False, False]),
        "truncated": torch.tensor([True, True]),
        "bootstrap_mask": torch.tensor([True, False]),
        "rewards": torch.tensor([1.0, 0.0]),
        "returns": torch.tensor([[8.0, 8.0, 8.0], [0.0, 0.0, 0.0]]),
        "loss_mask": torch.ones(2, 3, dtype=torch.bool),
    }

    report = verify_gamma_one_episodic_returns(
        [group],
        reward_scaling=1.0,
        reward_bias=0.0,
        reward_clip=20.0,
    )

    assert report["passed"] is True
    assert report["terminated"] == 0
    assert report["truncated"] == 2
    assert report["bootstrapped"] == 1
    assert report["max_abs_error"] == 0.0
