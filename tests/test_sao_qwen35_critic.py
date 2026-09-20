# SPDX-License-Identifier: Apache-2.0
# Optional model packages must be checked before dependent imports.
# ruff: noqa: E402

import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors.torch import save_file
from transformers import AutoConfig

from areal.api.cli_args import TrainEngineConfig
from areal.engine.fsdp_engine import FSDPEngine, _use_qwen35_token_critic_adapter
from areal.models.transformers.token_critic import (
    Qwen35TokenCriticForCausalBackbone,
)


def _allow_real_qwen35_fla_norm(fn):
    fn.allow_real_qwen35_fla_norm = True
    return fn


def _require_qwen35():
    if importlib.util.find_spec("transformers.models.qwen3_5") is None:
        pytest.skip("Transformers build does not provide Qwen3.5 model classes")


@pytest.fixture(autouse=True)
def _force_qwen35_torch_reference_cpu(monkeypatch, request):
    if getattr(request.node.function, "allow_real_qwen35_fla_norm", False):
        return
    _require_qwen35()
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35

    monkeypatch.setattr(qwen35, "FusedRMSNormGated", None)
    monkeypatch.setattr(qwen35, "causal_conv1d_fn", None)
    monkeypatch.setattr(
        qwen35,
        "causal_conv1d_update",
        qwen35.torch_causal_conv1d_update,
    )
    monkeypatch.setattr(
        qwen35,
        "chunk_gated_delta_rule",
        qwen35.torch_chunk_gated_delta_rule,
    )
    monkeypatch.setattr(
        qwen35,
        "fused_recurrent_gated_delta_rule",
        qwen35.torch_recurrent_gated_delta_rule,
    )
    monkeypatch.setattr(qwen35, "is_fast_path_available", False)


def _write_tiny_qwen35_config(tmp_path, *, text_dtype: str = "float32"):
    _require_qwen35()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 100,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_output_gate": True,
            "dtype": text_dtype,
            "eos_token_id": 2,
            "full_attention_interval": 4,
            "head_dim": 4,
            "hidden_act": "silu",
            "hidden_size": 16,
            "initializer_range": 0.02,
            "intermediate_size": 32,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 4,
            "max_position_embeddings": 256,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
            "use_cache": False,
            "vocab_size": 128,
            "mamba_ssm_dtype": "float32",
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
            },
        },
        "tie_word_embeddings": True,
        "video_token_id": 101,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 8,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 16,
            "model_type": "qwen3_5",
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": 16,
            "patch_size": 4,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
        },
        "vision_end_token_id": 103,
        "vision_start_token_id": 102,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)


def _make_tiny_saved_critic(tmp_path):
    _require_qwen35()
    config = _write_tiny_qwen35_config(tmp_path)
    model = Qwen35TokenCriticForCausalBackbone.from_config(config)
    save_dir = tmp_path / "base"
    save_dir.mkdir()
    config.save_pretrained(save_dir)
    state = model.state_dict()
    pretrained_state = {
        key: value
        for key, value in state.items()
        if key.startswith("model.language_model.")
    }
    pretrained_state["lm_head.weight"] = pretrained_state[
        "model.language_model.embed_tokens.weight"
    ].clone()
    pretrained_state["mtp.layers.0.norm.weight"] = torch.ones(
        config.text_config.hidden_size
    )
    save_file(pretrained_state, save_dir / "model.safetensors")
    return config, save_dir, model, pretrained_state


def test_engine_selects_qwen35_critic_without_changing_actor(tmp_path):
    config = _write_tiny_qwen35_config(tmp_path)
    critic_config = TrainEngineConfig(
        experiment_name="test",
        trial_name="critic",
        path=str(tmp_path),
        is_critic=True,
    )
    actor_config = TrainEngineConfig(
        experiment_name="test",
        trial_name="actor",
        path=str(tmp_path),
        is_critic=False,
    )

    assert _use_qwen35_token_critic_adapter(critic_config, config)
    assert not _use_qwen35_token_critic_adapter(actor_config, config)


def test_qwen35_critic_forward_backward_dtype_mask_and_state_keys(tmp_path):
    config, _, _, _ = _make_tiny_saved_critic(tmp_path)
    assert config.text_config.layer_types.count("linear_attention") == 3
    assert config.text_config.layer_types.count("full_attention") == 1

    model = Qwen35TokenCriticForCausalBackbone.from_config(config)
    input_ids = torch.randint(0, config.text_config.vocab_size, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -2:] = 0

    values = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert values.shape == (2, 7, 1)
    assert values.dtype == next(model.parameters()).dtype

    loss_mask = attention_mask.bool()
    loss = values.squeeze(-1).float()[loss_mask].pow(2).mean()
    loss.backward()

    assert model.score.weight.grad is not None
    backbone_grad = model.model.language_model.embed_tokens.weight.grad
    assert backbone_grad is not None
    assert torch.isfinite(backbone_grad).all()

    keys = set(model.state_dict())
    assert "score.weight" in keys
    assert "model.language_model.embed_tokens.weight" in keys
    assert "lm_head.weight" not in keys


def test_qwen35_critic_pretrained_load_adds_only_score_head(tmp_path):
    _, save_dir, _, pretrained_state = _make_tiny_saved_critic(tmp_path)
    loaded = Qwen35TokenCriticForCausalBackbone.from_pretrained(save_dir)
    loaded_state = loaded.state_dict()

    torch.testing.assert_close(
        loaded_state["model.language_model.embed_tokens.weight"],
        pretrained_state["model.language_model.embed_tokens.weight"],
    )
    assert "score.weight" in loaded_state


def test_qwen35_critic_memory_efficient_preload_keeps_score_for_broadcast(tmp_path):
    config, _, model, pretrained_state = _make_tiny_saved_critic(tmp_path)
    incompatible = model.load_state_dict(pretrained_state, strict=False)

    assert incompatible.missing_keys == ["score.weight"]
    assert incompatible.unexpected_keys == [
        "lm_head.weight",
        "mtp.layers.0.norm.weight",
    ]

    full_state = model.state_dict()
    assert "score.weight" in full_state
    assert full_state["score.weight"].shape == (1, config.text_config.hidden_size)


@_allow_real_qwen35_fla_norm
def test_qwen35_critic_requested_dtype_overrides_bf16_config_for_fla_norm_meta(
    tmp_path,
    monkeypatch,
):
    config = _write_tiny_qwen35_config(tmp_path, text_dtype="bfloat16")
    assert config.text_config.dtype == torch.bfloat16

    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("meta"))

    with torch.device("meta"):
        model = Qwen35TokenCriticForCausalBackbone.from_config(
            config,
            dtype=torch.float32,
        )

    dtypes = {param.dtype for param in model.parameters()}
    assert dtypes == {torch.float32}
    norm_weights = {
        name: param
        for name, param in model.named_parameters()
        if name.endswith("linear_attn.norm.weight")
    }
    assert norm_weights
    assert all(param.is_meta for param in norm_weights.values())
    assert all(param.dtype == torch.float32 for param in norm_weights.values())
    assert model.score.weight.is_meta
    assert model.score.weight.dtype == torch.float32


@torch.no_grad()
def test_qwen35_critic_save_reload_preserves_values_with_explicit_adapter(tmp_path):
    config, _, _, _ = _make_tiny_saved_critic(tmp_path)
    model = Qwen35TokenCriticForCausalBackbone.from_config(config)
    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 6))
    attention_mask = torch.ones_like(input_ids)
    expected = model(input_ids=input_ids, attention_mask=attention_mask).logits

    save_path = tmp_path / "critic"
    model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    loaded = Qwen35TokenCriticForCausalBackbone.from_pretrained(save_path)
    actual = loaded(input_ids=input_ids, attention_mask=attention_mask).logits

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fsdp_engine_qwen35_critic_uses_text_token_value_path(tmp_path, monkeypatch):
    config, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.load_hf_tokenizer",
        lambda _path: MagicMock(),
    )
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.current_platform",
        SimpleNamespace(
            device_type="cpu",
            set_device=lambda *_args, **_kwargs: None,
            set_numa_affinity=lambda *_args, **_kwargs: None,
        ),
    )

    engine = FSDPEngine(
        TrainEngineConfig(
            experiment_name="test",
            trial_name="critic",
            path=str(save_dir),
            is_critic=True,
            dtype="float32",
            optimizer_dtype="float32",
            attn_impl="eager",
        )
    )
    engine.get_device_stats = lambda: SimpleNamespace(
        log=lambda *_args, **_kwargs: None
    )
    engine.logger = MagicMock()
    engine._create_device_model()

    assert isinstance(engine.model, Qwen35TokenCriticForCausalBackbone)
    assert engine.processor is None
    assert engine.is_vision_model

    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 5))
    outputs = engine.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
    )
    values = engine._extract_model_forward_output(outputs)
    assert values.shape == (1, 5, 1)
