# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the τ² mean-initialised critic head (README 16.14.3)."""

# Optional model packages must be checked before dependent imports.
# ruff: noqa: E402

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from examples.tau2.critic_head import (
    Qwen35BiasedTokenCritic,
    Tau2HeadCriticConfig,
    Tau2HeadFSDPPPOCritic,
    init_value_head,
    is_head_parameter,
)
from scripts.tau2.train_critic_head import validate_head_config
from tests.test_sao_qwen35_critic import (
    _force_qwen35_torch_reference_cpu,  # noqa: F401
    _make_tiny_saved_critic,
)

from areal.api.cli_args import OptimizerConfig, PPOCriticConfig
from areal.infra.rpc.serialization import deserialize_value, serialize_value


def _head_engine(save_dir, monkeypatch, **head):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.load_hf_tokenizer", lambda _path: MagicMock()
    )
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.current_platform",
        SimpleNamespace(
            device_type="cpu",
            set_device=lambda *_args, **_kwargs: None,
            set_numa_affinity=lambda *_args, **_kwargs: None,
        ),
    )
    config = Tau2HeadCriticConfig(
        experiment_name="test",
        trial_name="critic",
        path=str(save_dir),
        is_critic=True,
        dtype="float32",
        optimizer_dtype="float32",
        attn_impl="eager",
        optimizer=OptimizerConfig(lr=5e-6, weight_decay=0.0),
        **head,
    )
    engine = Tau2HeadFSDPPPOCritic(config)
    engine.get_device_stats = lambda: SimpleNamespace(
        log=lambda *_args, **_kwargs: None
    )
    engine.logger = MagicMock()
    engine._create_device_model()
    return engine


def test_head_engine_starts_every_token_at_the_bias(tmp_path, monkeypatch):
    config, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    engine = _head_engine(save_dir, monkeypatch, head_bias_init=0.8)

    assert isinstance(engine.model, Qwen35BiasedTokenCritic)
    input_ids = torch.randint(0, config.text_config.vocab_size, (2, 6))
    values = engine._extract_model_forward_output(
        engine.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    )
    torch.testing.assert_close(values, torch.full((2, 6, 1), 0.8))


def test_zero_weight_head_still_learns(tmp_path):
    config, _, _, _ = _make_tiny_saved_critic(tmp_path)
    model = Qwen35BiasedTokenCritic.from_config(config)
    init_value_head(model, 0.5)
    input_ids = torch.randint(0, config.text_config.vocab_size, (2, 6))
    values = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    (values.logits - 1.0).pow(2).mean().backward()

    assert model.score.weight.grad.abs().sum() > 0
    assert model.score.bias.grad.item() == pytest.approx(-1.0)


def test_biased_head_loads_backbone_checkpoint_without_head_keys(tmp_path):
    _, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    model = Qwen35BiasedTokenCritic.from_pretrained(save_dir)
    assert {"score.weight", "score.bias"} <= set(model.state_dict())


def test_head_lr_splits_param_groups(tmp_path, monkeypatch):
    _, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    engine = _head_engine(save_dir, monkeypatch, head_bias_init=0.5, head_lr=1e-4)
    body, head = engine._optimizer_param_groups()

    assert head["lr"] == 1e-4 and "lr" not in body
    assert {id(p) for p in head["params"]} == {
        id(engine.model.score.weight),
        id(engine.model.score.bias),
    }
    trainable = [p for p in engine.model.parameters() if p.requires_grad]
    assert len(body["params"]) + len(head["params"]) == len(trainable)

    optimizer = torch.optim.AdamW(engine._optimizer_param_groups(), lr=5e-6)
    assert [g["lr"] for g in optimizer.param_groups] == [5e-6, 1e-4]


def test_no_head_lr_keeps_one_param_group(tmp_path, monkeypatch):
    _, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    engine = _head_engine(save_dir, monkeypatch, head_bias_init=0.5)
    params = engine._optimizer_param_groups()
    assert all(isinstance(p, torch.nn.Parameter) for p in params)


def test_is_head_parameter_matches_fsdp_and_hf_names():
    assert is_head_parameter("score.weight")
    assert is_head_parameter("score.bias")
    assert not is_head_parameter("model.language_model.layers.0.mlp.score_proj.weight")
    assert not is_head_parameter("model.language_model.embed_tokens.weight")


def test_head_config_survives_rpc_serialization():
    base = PPOCriticConfig(experiment_name="e", trial_name="t", path="/m")
    config = Tau2HeadCriticConfig.from_critic(base, head_bias_init=0.7, head_lr=1e-4)
    restored = deserialize_value(serialize_value(config))

    assert type(restored) is Tau2HeadCriticConfig
    assert (restored.head_bias_init, restored.head_lr) == (0.7, 1e-4)
    assert restored.path == "/m" and restored.is_critic == base.is_critic


@pytest.mark.parametrize("bias", [None, -0.1, 1.5])
def test_validate_rejects_missing_or_out_of_range_bias(bias):
    with pytest.raises(ValueError, match="critic_head_bias_init"):
        validate_head_config(
            SimpleNamespace(critic_head_bias_init=bias, critic_head_lr=None)
        )


def test_validate_rejects_nonpositive_head_lr():
    with pytest.raises(ValueError, match="critic_head_lr"):
        validate_head_config(
            SimpleNamespace(critic_head_bias_init=0.5, critic_head_lr=0)
        )
