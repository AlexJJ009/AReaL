# SPDX-License-Identifier: Apache-2.0
"""CPU checks for PPO whitening, ADR-0012 Qwen3 heads and the K2 recipe."""

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from examples.tau2 import critic_head, evaluation, ppo_data, train
from examples.tau2.utils import Tau2PPOConfig
from scripts.tau2 import train_critic
from tests.test_sao_dual_lambda import TinyTrainEngine, rollout
from tests.test_tau2_critic_head import (
    _head_engine,
    _plain_critic_engine,
    _save_like_engine,
    _values,
)
from tests.test_tau2_ppo import _config, pools  # noqa: F401

from areal.api.cli_args import NormConfig, load_expr_config
from areal.models.transformers.token_critic import VALUE_HEAD_BIAS_FLAG
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.critic import PPOCritic


def test_ppo_whitening_contract_refuses_missing_or_nonbatch(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    for norm in (
        None,
        NormConfig(mean_level="group", std_level="group"),
        NormConfig(mean_level="batch", std_level="group"),
        NormConfig(mean_leave1out=True),
        NormConfig(std_unbiased=False),
        NormConfig(eps=1e-5),
    ):
        config.actor.adv_norm = norm
        with pytest.raises(ValueError, match="batch action-token advantage whitening"):
            train.validate_ppo_contract(config)


def test_ppo_contract_allows_only_unit_or_sao_actor_lambda(monkeypatch, tmp_path):
    """Actor lambda may be 1 or SAO alpha=1.5; the critic target stays lambda=1."""
    config = _config(monkeypatch, tmp_path)
    train.validate_ppo_contract(config)
    config.actor.gae_lambda = train.SAO_ACTOR_LAMBDA
    config.actor.gae_lambda_kwargs = {"alpha": 1.5}
    train.validate_ppo_contract(config)
    for gae_lambda, kwargs, critic_lambda in (
        (train.SAO_ACTOR_LAMBDA, {"alpha": 2.0}, 1.0),
        (train.SAO_ACTOR_LAMBDA, {"alpha": 1.5}, None),
        (0.95, {}, 1.0),
    ):
        config.actor.gae_lambda = gae_lambda
        config.actor.gae_lambda_kwargs = kwargs
        config.actor.critic_gae_lambda = critic_lambda
        with pytest.raises(ValueError, match="SAO length-adaptive"):
            train.validate_ppo_contract(config)


def test_ppo_actor_loss_consumes_global_whitening_with_raw_returns(
    monkeypatch, tmp_path
):
    config = _config(monkeypatch, tmp_path)
    engine = TinyTrainEngine()
    actor = PPOActor(config.actor, engine)
    batch = rollout()
    batch["attention_mask"][1] = torch.tensor([1, 1, 1, 0, 0])
    batch["loss_mask"][1] = torch.tensor([0, 1, 1, 0, 0])
    batch["prox_logp"] = torch.full((2, 5), -2.77)
    data = actor._compute_advantages(batch)
    raw = torch.tensor([0.8, 0.6, 0.4, -0.3, -0.5])
    expected = (raw - raw.mean()) / (raw.std(unbiased=True) + 1e-8)
    mask = data["loss_mask"].bool()
    expected_returns = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    torch.testing.assert_close(
        data["returns"][mask], expected_returns, rtol=1e-6, atol=1e-7
    )
    seen = []
    original = actor_module.ppo_actor_loss_fn

    def loss(**kwargs):
        seen.append(kwargs["advantages"][kwargs["loss_mask"]].clone())
        return original(**kwargs)

    monkeypatch.setattr(actor_module, "ppo_actor_loss_fn", loss)
    actor.ppo_update([deepcopy(data)])
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected, rtol=1e-6, atol=1e-7)
    critic_engine = TinyTrainEngine(critic=True)
    data["values"] = data["values"].detach()
    PPOCritic(config.critic, critic_engine).ppo_update([data])
    torch.testing.assert_close(
        critic_engine.consumed[0][mask], expected_returns, rtol=1e-6, atol=1e-7
    )


def _launcher_config(tmp_path, monkeypatch, launcher, *args, **env):
    _config(monkeypatch, tmp_path)
    fake = tmp_path / "python"
    fake.write_text(
        '#!/usr/bin/env python3\nimport json,sys,os\nprint(json.dumps([sys.argv[1:], os.environ["TAU2_ACTOR_PATH"]]))\n'
    )
    fake.chmod(0o755)
    result = subprocess.run(
        ["bash", f"scripts/tau2/{launcher}", "unit", *args],
        env={
            **os.environ,
            "AREAL_PYTHON": str(fake),
            "TAU2_PPO_ARTIFACT_ROOT": str(tmp_path),
            "TAU2_AREAL_PACK": str(tmp_path / "pack"),
            **env,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    command, actor_path = json.loads(result.stdout)
    monkeypatch.setenv("TAU2_ACTOR_PATH", actor_path)
    entry, *argv = command
    argv.remove("--check-config")
    config, _ = load_expr_config(argv, Tau2PPOConfig)
    return entry, config


def test_ppo_launchers_default_to_zero_warmup_and_smoke_final_only(
    tmp_path, monkeypatch
):
    for mode in ("smoke", "formal"):
        _, config = _launcher_config(
            tmp_path, monkeypatch, "launch_ppo.sh", "check-config", mode
        )
        assert config.num_critic_only_steps == 0
        assert config.evaluator.eval_before_train == (mode == "formal")
        train.validate_ppo_contract(config)
    _, config = _launcher_config(
        tmp_path,
        monkeypatch,
        "launch_ppo.sh",
        "smoke",
        "--check-config",
        TAU2_CRITIC_WARMUP_STEPS="1",
    )
    assert config.num_critic_only_steps == 1


def test_ppo_launcher_cadence_parameter_controls_eval_and_saver(tmp_path, monkeypatch):
    _, config = _launcher_config(
        tmp_path,
        monkeypatch,
        "launch_ppo.sh",
        "formal",
        "--check-config",
        TAU2_EVAL_EVERY="12",
    )
    train.validate_ppo_contract(config)
    assert (
        config.evaluator.freq_steps
        == config.saver.freq_steps
        == config.recover.freq_steps
        == 12
    )
    config.total_train_steps = 48
    config.effective_episodes = 5976
    summary = ppo_data.ppo_config_summary(config, [], [])
    assert summary["eval_steps"] == [0, 12, 24, 36, 48]
    assert summary["actor_update_steps"] == 48


def test_ppo_head_contract_requires_training_prior_only_for_backbone(
    monkeypatch, tmp_path
):
    config = _config(monkeypatch, tmp_path)
    config.critic_head_bias_init = None
    with pytest.raises(ValueError, match="expected success rate"):
        train.validate_ppo_contract(config)
    config.critic.path = str(tmp_path / "saved-critic")
    train.validate_ppo_contract(config)


def test_ppo_critic_factory_initialises_backbone_but_preserves_checkpoint(
    monkeypatch, tmp_path
):
    config = _config(monkeypatch, tmp_path)
    trainer = evaluation.Tau2PPOTrainer.__new__(evaluation.Tau2PPOTrainer)
    trainer.config = config
    fresh, saved = object(), object()
    init = MagicMock(return_value=fresh)
    plain = MagicMock(return_value=saved)
    monkeypatch.setattr(critic_head, "create_head_critic", init)
    monkeypatch.setattr(evaluation.Tau2AsyncEvalTrainer, "_create_critic", plain)
    assert trainer._create_critic(config.critic, "alloc") is fresh
    config.critic.path = str(tmp_path / "saved-critic")
    assert trainer._create_critic(config.critic, "alloc") is saved
    assert init.call_count == plain.call_count == 1


def test_head_factory_carries_settings_to_workers(monkeypatch):
    config = SimpleNamespace(critic_head_bias_init=0.62, critic_head_lr=None)
    trainer = SimpleNamespace(config=config)
    from areal.api.cli_args import PPOCriticConfig

    monkeypatch.setattr("areal.utils.environ.is_single_controller", lambda: False)
    with pytest.raises(ValueError, match="fsdp backend"):
        critic_head.create_head_critic(
            trainer, PPOCriticConfig(), SimpleNamespace(backend="megatron")
        )
    engine = MagicMock()
    constructor = MagicMock(return_value=engine)
    monkeypatch.setattr(critic_head, "Tau2HeadFSDPPPOCritic", constructor)
    critic_head.create_head_critic(
        trainer, PPOCriticConfig(), SimpleNamespace(backend="fsdp", parallel="parallel")
    )
    observed = constructor.call_args.kwargs["config"]
    assert observed.head_bias_init == 0.62 and observed.head_lr is None
    engine.create_process_group.assert_called_once_with(parallel_strategy="parallel")


def _qwen3_backbone(tmp_path):
    path = tmp_path / "backbone"
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path


def test_qwen3_head_init_flag_and_saved_critic_roundtrip(tmp_path, monkeypatch):
    path = _qwen3_backbone(tmp_path)
    engine = _head_engine(path, monkeypatch, head_bias_init=0.6204819277108434)
    ids = torch.tensor([[1, 2, 3]])
    assert engine.model.score.bias is not None
    assert engine.model.config.num_labels == engine.model_config.num_labels == 1
    assert getattr(engine.model.config, VALUE_HEAD_BIAS_FLAG) is True
    torch.testing.assert_close(
        _values(engine, ids),
        torch.full((1, 3, 1), 0.6204819277108434),
        rtol=0,
        atol=1e-7,
    )
    with torch.no_grad():
        engine.model.score.weight.fill_(0.1)
        engine.model.score.bias.fill_(0.4)
    checkpoint = tmp_path / "saved"
    _save_like_engine(engine, checkpoint)
    plain = _plain_critic_engine(checkpoint, monkeypatch)
    resumed = _head_engine(checkpoint, monkeypatch, head_bias_init=0.99)
    for loaded in (plain, resumed):
        torch.testing.assert_close(
            loaded.model.score.weight, engine.model.score.weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            _values(loaded, ids), _values(engine, ids), rtol=0, atol=0
        )


def test_qwen3_memory_efficient_construction_has_scalar_mean_head(
    tmp_path, monkeypatch
):
    path = _qwen3_backbone(tmp_path)
    engine = _head_engine(path, monkeypatch, head_bias_init=0.62)
    engine.config.fsdp.memory_efficient_load = True
    engine = type(engine)(config=engine.config)
    model = engine._create_llm_actor_or_critic()
    assert model.score.weight.shape == (1, 16)
    assert model.config.num_labels == 1
    torch.testing.assert_close(model.score.weight, torch.zeros(1, 16), rtol=0, atol=0)
    torch.testing.assert_close(
        model.score.bias, torch.tensor([0.62]), rtol=0, atol=1e-7
    )


def test_qwen3_unflagged_saved_checkpoint_plain_loader_keeps_head(
    tmp_path, monkeypatch
):
    path = _qwen3_backbone(tmp_path)
    engine = _plain_critic_engine(path, monkeypatch)
    with torch.no_grad():
        engine.model.score.weight.fill_(0.123)
        engine.model.score.bias.fill_(0.456)
    checkpoint = tmp_path / "legacy-critic"
    _save_like_engine(engine, checkpoint)
    loaded = _plain_critic_engine(checkpoint, monkeypatch)
    torch.testing.assert_close(
        loaded.model.score.weight, engine.model.score.weight, rtol=0, atol=0
    )
    torch.testing.assert_close(
        loaded.model.score.bias, engine.model.score.bias, rtol=0, atol=0
    )


def test_k2_launcher_preserves_protocol_and_parameterises_budget(tmp_path, monkeypatch):
    entry, config = _launcher_config(
        tmp_path, monkeypatch, "launch_critic_k2.sh", "--check-config"
    )
    assert Path(entry).name == "train_critic_head.py"
    assert "Qwen3-4B-Instruct-2507@cdbee" in config.actor.path
    assert config.critic.freeze_critic_attention is False
    assert config.critic_head_bias_init == pytest.approx(927 / 1494)
    assert config.critic_head_lr is None
    assert config.actor.optimizer.lr == 0
    assert config.gconfig.temperature == 1
    assert (
        config.experiment_mode == "formal"
        and config.econfig.benchmark_variant == "fixed"
    )
    assert config.rollout.agent.tool_call_parser == "qwen25"
    assert config.critic.optimizer.lr == 5e-6
    assert config.train_batch_episodes == 16 and config.total_train_epochs == 2
    train_critic.fill_derived_train_steps(config, train_rows=747)
    train_critic.validate_tau2_critic_config(config, train_rows=747, validation_rows=60)
    assert config.total_train_steps == config.num_critic_only_steps == 94
    _, changed = _launcher_config(
        tmp_path,
        monkeypatch,
        "launch_critic_k2.sh",
        "--check-config",
        TAU2_CRITIC_LR="2e-6",
        TAU2_CRITIC_BATCH="32",
        TAU2_CRITIC_EPOCHS="3",
        TAU2_EVAL_EVERY="12",
    )
    train_critic.fill_derived_train_steps(changed, train_rows=747)
    train_critic.validate_tau2_critic_config(
        changed, train_rows=747, validation_rows=60
    )
    assert changed.critic.optimizer.lr == 2e-6
    assert changed.total_train_steps == 72
    assert changed.evaluator.freq_steps == changed.saver.freq_steps == 12


def test_ppo_smoke_refuses_intermediate_evaluation(pools, monkeypatch):  # noqa: F811
    _, _, packs = pools
    monkeypatch.setattr(train.benchmark_variant, "validate_task_files", lambda *a: None)
    with pytest.raises(ValueError, match="only at final step"):
        train.main(
            [
                "--config",
                "examples/tau2/config_ppo.yaml",
                "--check-config",
                f"train_task_packs=[{packs[0]}]",
                "experiment_mode=qualification",
                "total_train_epochs=1",
                "total_train_steps=2",
                "train_batch_episodes=2",
                "train_dataset.batch_size=2",
                "evaluator.eval_before_train=false",
                "evaluator.freq_steps=1",
                "saver.freq_steps=1",
            ]
        )
