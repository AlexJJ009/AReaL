# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the τ² mean-initialised critic head (docs/research/tau2/critic/value-head.md)."""

# Optional model packages must be checked before dependent imports.
# ruff: noqa: E402

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors.torch import save_file

from examples.tau2.critic_head import (
    Tau2HeadCriticConfig,
    Tau2HeadFSDPPPOCritic,
    init_value_head,
    is_head_parameter,
)
from scripts.sao import check_critic_head as preflight
from scripts.tau2.train_critic_head import validate_head_config
from tests.test_sao_qwen35_critic import (
    _force_qwen35_torch_reference_cpu,  # noqa: F401
    _make_tiny_saved_critic,
)

from areal.api.cli_args import OptimizerConfig, PPOCriticConfig, TrainEngineConfig
from areal.engine.fsdp_engine import FSDPEngine
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.models.transformers.scalar_value import Qwen35ScalarValueModel
from areal.models.transformers.token_critic import (
    VALUE_HEAD_BIAS_FLAG,
    Qwen35TokenCriticForCausalBackbone,
)


def _patch_cpu_engine(monkeypatch):
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


def _build(engine):
    engine.get_device_stats = lambda: SimpleNamespace(
        log=lambda *_args, **_kwargs: None
    )
    engine.logger = MagicMock()
    engine._create_device_model()
    return engine


def _head_engine(save_dir, monkeypatch, **head):
    _patch_cpu_engine(monkeypatch)
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
    return _build(Tau2HeadFSDPPPOCritic(config))


def _plain_critic_engine(path, monkeypatch):
    """The engine SAO uses to load an unsealed critic checkpoint."""
    _patch_cpu_engine(monkeypatch)
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="sao-critic",
        path=str(path),
        is_critic=True,
        dtype="float32",
        optimizer_dtype="float32",
        attn_impl="eager",
    )
    return _build(FSDPEngine(config))


def _values(engine, input_ids):
    return engine._extract_model_forward_output(
        engine.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    )


def _save_like_engine(engine, path):
    """Mirror FSDPEngine._save_model_to_hf: weights from the model, config.json
    from the engine's model_config, tokenizer files from the actor."""
    path.mkdir()
    state = {k: v.contiguous() for k, v in engine.model.state_dict().items()}
    save_file(state, path / "model.safetensors")
    engine.model_config.save_pretrained(path)
    (path / "tokenizer.json").write_text('{"model":{"vocab":{}}}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_head_engine_starts_every_token_at_the_bias(tmp_path, monkeypatch):
    config, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    engine = _head_engine(save_dir, monkeypatch, head_bias_init=0.8)

    assert isinstance(engine.model, Qwen35TokenCriticForCausalBackbone)
    assert getattr(engine.model_config, VALUE_HEAD_BIAS_FLAG) is True
    input_ids = torch.randint(0, config.text_config.vocab_size, (2, 6))
    torch.testing.assert_close(_values(engine, input_ids), torch.full((2, 6, 1), 0.8))


def test_zero_weight_head_still_learns(tmp_path):
    config, _, _, _ = _make_tiny_saved_critic(tmp_path)
    setattr(config, VALUE_HEAD_BIAS_FLAG, True)
    model = Qwen35TokenCriticForCausalBackbone.from_config(config)
    init_value_head(model, 0.5)
    input_ids = torch.randint(0, config.text_config.vocab_size, (2, 6))
    values = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    (values.logits - 1.0).pow(2).mean().backward()

    assert model.score.weight.grad.abs().sum() > 0
    assert model.score.bias.grad.item() == pytest.approx(-1.0)


def test_unflagged_config_keeps_the_bias_free_head(tmp_path):
    config, _, _, _ = _make_tiny_saved_critic(tmp_path)
    assert Qwen35TokenCriticForCausalBackbone.from_config(config).score.bias is None
    assert Qwen35ScalarValueModel.from_config(config).score.bias is None
    with pytest.raises(ValueError, match="no bias"):
        init_value_head(Qwen35TokenCriticForCausalBackbone.from_config(config), 0.5)


def test_saved_biased_critic_reloads_in_the_plain_sao_engine(tmp_path, monkeypatch):
    config, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    trained = _head_engine(save_dir, monkeypatch, head_bias_init=0.8)
    with torch.no_grad():  # stand in for training: a nonzero weight, moved bias
        trained.model.score.weight.normal_(std=0.5)
        trained.model.score.bias.fill_(0.6)
    checkpoint = tmp_path / "critic-step"
    _save_like_engine(trained, checkpoint)

    loaded = _plain_critic_engine(checkpoint, monkeypatch)

    input_ids = torch.randint(0, config.text_config.vocab_size, (2, 6))
    torch.testing.assert_close(loaded.model.score.bias, trained.model.score.bias)
    torch.testing.assert_close(_values(loaded, input_ids), _values(trained, input_ids))
    report = preflight.check_critic_head(checkpoint)
    assert report["head"]["bias"]["value"] == pytest.approx(0.6)


def test_restarting_head_engine_from_a_biased_critic_keeps_its_head(
    tmp_path, monkeypatch
):
    _, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    trained = _head_engine(save_dir, monkeypatch, head_bias_init=0.8)
    with torch.no_grad():
        trained.model.score.weight.normal_(std=0.5)
        trained.model.score.bias.fill_(0.6)
    checkpoint = tmp_path / "critic-step"
    _save_like_engine(trained, checkpoint)

    restarted = _head_engine(checkpoint, monkeypatch, head_bias_init=0.2)

    torch.testing.assert_close(restarted.model.score.weight, trained.model.score.weight)
    torch.testing.assert_close(restarted.model.score.bias, trained.model.score.bias)


def test_bias_free_loader_rejects_a_biased_checkpoint(tmp_path):
    config, _, _, _ = _make_tiny_saved_critic(tmp_path)
    setattr(config, VALUE_HEAD_BIAS_FLAG, True)
    biased_state = Qwen35TokenCriticForCausalBackbone.from_config(config).state_dict()
    setattr(config, VALUE_HEAD_BIAS_FLAG, False)
    plain = Qwen35TokenCriticForCausalBackbone.from_config(config)

    with pytest.raises(RuntimeError, match="score.bias"):
        plain.load_state_dict(biased_state)


def test_scalar_value_model_loads_a_biased_state_strictly(tmp_path):
    config, _, _, _ = _make_tiny_saved_critic(tmp_path)
    setattr(config, VALUE_HEAD_BIAS_FLAG, True)
    source = Qwen35TokenCriticForCausalBackbone.from_config(config)
    with torch.no_grad():
        source.score.bias.fill_(0.3)
    model = Qwen35ScalarValueModel.from_config(config)
    model.load_state_dict(source.state_dict(), strict=True)
    torch.testing.assert_close(model.score.bias, torch.tensor([0.3]))


def test_preflight_flag_name_matches_the_model_flag():
    assert preflight.VALUE_HEAD_BIAS_FLAG == VALUE_HEAD_BIAS_FLAG


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
