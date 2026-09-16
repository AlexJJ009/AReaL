# SPDX-License-Identifier: Apache-2.0
"""Focused tests of new branches, using the real installed AWEX base classes."""

from types import SimpleNamespace

import pytest
import torch

from areal.models.mcore.qwen4_exp_awex import (
    _MIXER_WEIGHTS,
    _REPLICATED_LAYER_WEIGHTS,
    build_mcore_converter,
    build_sglang_converter,
    build_sharding_strategy,
)


@pytest.fixture
def writer():
    cls = build_mcore_converter()
    instance = cls.__new__(cls)
    instance.rank_info = SimpleNamespace(
        pp_rank=1, pp_size=2, attn_tp_size=4, attn_tp_rank=2
    )
    instance.hf_config = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    instance.tf_config = SimpleNamespace()
    instance.infer_atten_tp_size = 4
    instance._pp_stage_layer_id_map = {(1, 0): {0: 24}, (1, 1): {0: 36}}
    return instance


@pytest.mark.parametrize("suffix", sorted(_REPLICATED_LAYER_WEIGHTS))
@pytest.mark.parametrize("vp,global_layer", [(0, 24), (1, 36)])
def test_replicated_mapping_pp_vpp_matches_inference(writer, suffix, vp, global_layer):
    parameter = torch.randn(3, 5)
    mcore_suffix = suffix.replace("self_attn.indexer.", "self_attention.indexer.")
    actual = writer.convert_param(
        f"module.language_model.decoder.layers.0.{mcore_suffix}", parameter, vp_stage=vp
    )
    canonical = f"model.layers.{global_layer}.{suffix}"
    assert actual[0][0] == canonical
    assert actual[0][1] is parameter
    cls = build_sglang_converter()
    reader = cls.__new__(cls)
    sglang_suffix = suffix.replace("self_attn.indexer.", "indexer.")
    received = reader.convert_param(
        f"model.language_model.layers.{global_layer}.{sglang_suffix}", parameter
    )
    assert received[0][0] == canonical
    assert received[0][1] is parameter
    strategy_cls = build_sharding_strategy()
    strategy = strategy_cls.__new__(strategy_cls)
    from awex.sharding.param_sharding import ShardingType

    assert strategy.get_sharding_strategy(canonical) == (ShardingType.NO_SHARDING, 0, 1)


@pytest.mark.parametrize("suffix", sorted(_MIXER_WEIGHTS))
def test_final_mixer_keeps_weights_without_norm_offset(writer, suffix):
    parameter = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.bfloat16)
    actual = writer.convert_param(f"language_model.decoder.{suffix}", parameter)
    assert actual[0][0] == f"model.{suffix}"
    torch.testing.assert_close(actual[0][1], parameter, rtol=0, atol=0)


def test_gdn_norm_does_not_inherit_qwen35_offset(writer):
    parameter = torch.tensor([0.5, 1.0, 1.5], dtype=torch.bfloat16)
    actual = writer._convert_attention_param(
        "self_attention.out_norm.weight", parameter, "0"
    )
    assert actual[0][0] == "linear_attn.norm.weight"
    torch.testing.assert_close(actual[0][1], parameter, rtol=0, atol=0)


@pytest.mark.parametrize(
    "suffix",
    [
        "ple.ple_embedding.ngram_embedding.weight",
        "self_attention.indexer.unknown.weight",
        "attn_hyper_connection.unknown.weight",
    ],
)
def test_unimplemented_state_rejected_before_generic_fallback(writer, suffix):
    with pytest.raises(NotImplementedError):
        writer.convert_param(f"decoder.layers.0.{suffix}", torch.ones(2, 2))


def test_missing_pp_map_fails_instead_of_using_local_layer_id(writer):
    with pytest.raises(ValueError, match="Missing pp stage"):
        writer.convert_param(
            "decoder.layers.0.attn_hyper_connection.hc_norm.weight",
            torch.ones(4),
            vp_stage=2,
        )


def test_gdn_writer_slices_repacked_tensor_using_training_tp(writer):
    from areal.models.mcore.qwen4_exp_awex_layout import Qwen4ExpGDNLayout

    full = torch.arange(16480 * 3, dtype=torch.float32).reshape(16480, 3)
    # Stand in only for the TP gather; conversion and training-rank selection
    # execute the actual production methods inherited from AWEX.
    writer._full_tp_tensor = lambda parameter: full
    actual = writer._convert_attention_param(
        "self_attention.in_proj.weight", full.chunk(4)[2], "0"
    )
    qkvz, ba = Qwen4ExpGDNLayout(16, 48, 128, 128).pack_input(full, 4, 4)
    assert [name for name, _ in actual] == [
        "linear_attn.in_proj_qkvz.weight",
        "linear_attn.in_proj_ba.weight",
    ]
    for (_, parameter), reference in zip(actual, (qkvz, ba)):
        torch.testing.assert_close(parameter, reference.chunk(4)[2], rtol=0, atol=0)


@pytest.mark.parametrize("component,rows", [("qkvz", 16384), ("ba", 96)])
def test_actual_decoupled_gdn_entry_points(writer, component, rows):
    from areal.models.mcore.qwen4_exp_awex_layout import Qwen4ExpGDNLayout

    full = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    writer._full_tp_tensor = lambda parameter: full
    actual = writer._convert_attention_param(
        f"self_attention.in_proj_{component}.weight", full.chunk(4)[2], "0"
    )
    expected = Qwen4ExpGDNLayout(16, 48, 128, 128).pack_decoupled(full, 4, 4, component)
    assert actual[0][0] == f"linear_attn.in_proj_{component}.weight"
    torch.testing.assert_close(actual[0][1], expected.chunk(4)[2], rtol=0, atol=0)


def test_gdn_a_log_expands_bf16_values_to_native_sglang_float32(writer):
    parameter = torch.tensor([-2.25, 0.5, 1.75], dtype=torch.bfloat16)
    name, actual = writer._convert_attention_param(
        "self_attention.A_log", parameter, "0"
    )[0]
    assert name == "linear_attn.A_log"
    assert actual.dtype == torch.float32
    assert parameter.dtype == torch.bfloat16
    torch.testing.assert_close(actual, torch.tensor([-2.25, 0.5, 1.75]), rtol=0, atol=0)


def test_explicit_registration_resolves_native_factories_without_fallback(monkeypatch):
    from awex.models.registry import (
        ModelRegistry,
        _resolve_converter,
        get_sharding_strategy,
    )

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex

    monkeypatch.setattr(ModelRegistry, "models", {})
    register_qwen4_exp_awex()
    register_qwen4_exp_awex()
    config = ModelRegistry.get_model_config("Qwen4ExpForConditionalGeneration")
    assert (
        _resolve_converter(config["mcore_converter"], None) is build_mcore_converter()
    )
    assert (
        _resolve_converter(config["sglang_converter"], None) is build_sglang_converter()
    )
    assert (
        get_sharding_strategy("Qwen4ExpForConditionalGeneration")
        is build_sharding_strategy()
    )


def test_registration_rejects_unrelated_upstream_adapter(monkeypatch):
    from awex.models.registry import ModelRegistry

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex

    existing = {"mcore_converter": object()}
    monkeypatch.setattr(
        ModelRegistry, "models", {"Qwen4ExpForConditionalGeneration": existing}
    )
    with pytest.raises(ValueError, match="already registered"):
        register_qwen4_exp_awex()
    assert ModelRegistry.models["Qwen4ExpForConditionalGeneration"] is existing


def test_bound_frozen_contract_excludes_exact_table_and_preserved_visual(writer):
    from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract

    table_name = "model.layers.24.ple.ple_embedding.ngram_embedding.weight"
    visual_name = "model.visual.patch_embed.proj.weight"
    contract = Qwen4ExpFrozenContract(
        "a" * 64, frozenset({table_name}), frozenset({visual_name}), True, True
    )
    parameter = torch.nn.Parameter(
        torch.ones(4, 3, dtype=torch.bfloat16), requires_grad=False
    )
    original = {table_name: parameter}
    writer.bind_frozen_contract(contract, original, frozenset({table_name}))
    mcore_name = "module.language_model.decoder.layers.0.ple.ple_embedding.ngram_embedding.weight"
    assert writer.convert_param(mcore_name, parameter.detach(), vp_stage=0) == []
    reader_cls = build_sglang_converter()
    reader = reader_cls.__new__(reader_cls)
    visual = torch.nn.Parameter(torch.ones(3, 4))
    reader.bind_frozen_contract(
        contract, {**original, visual_name: visual}, frozenset({visual_name})
    )
    assert reader.convert_param(table_name, parameter.detach()) == []
    assert reader.convert_param("visual.patch_embed.proj.weight", visual) == []
    qsa_name = "model.layers.24.self_attn.indexer.index_qk_proj.weight"
    result = reader.convert_param(
        "model.layers.24.indexer.index_qk_proj.weight", visual
    )
    assert result[0][0] == qsa_name
    assert result[0][1] is visual
    # Writer payloads are detached, so checking their requires_grad would miss this.
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        writer.convert_param(mcore_name, parameter.detach(), vp_stage=0)


def test_native_registry_binds_every_reader_and_refreshes_original_parameters(
    monkeypatch,
):
    from awex.models.registry import ModelRegistry, get_infer_weights_converter

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex
    from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract

    table_name = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
    visual_name = "model.visual.patch_embed.proj.weight"
    contract = Qwen4ExpFrozenContract(
        "a" * 64, frozenset({table_name}), frozenset({visual_name}), True, True
    )
    parameters = {
        table_name: torch.nn.Parameter(
            torch.ones(2, 3, dtype=torch.bfloat16), requires_grad=False
        ),
        visual_name: torch.nn.Parameter(torch.ones(3, 4)),
    }
    bound = []

    def bind(converter):
        converter.bind_frozen_contract(
            contract, dict(parameters), frozenset({visual_name})
        )
        bound.append(converter)

    monkeypatch.setattr(ModelRegistry, "models", {})
    register_qwen4_exp_awex(sglang_binder=bind)
    register_qwen4_exp_awex(sglang_binder=bind)
    args = (
        "sglang",
        "Qwen4ExpForConditionalGeneration",
        SimpleNamespace(num_attention_heads=24, num_key_value_heads=2),
        SimpleNamespace(tp_rank=0, ep_rank=0),
        SimpleNamespace(tp_size=4, ep_size=1, device_backend="cpu"),
    )
    metadata = get_infer_weights_converter(*args)
    payload = get_infer_weights_converter(*args)
    assert bound == [metadata, payload]
    for converter in (metadata, payload):
        assert (
            converter.convert_param(table_name, parameters[table_name].detach()) == []
        )
    replacement = torch.nn.Parameter(
        torch.zeros(2, 3, dtype=torch.bfloat16), requires_grad=False
    )
    parameters[table_name] = replacement
    payload.refresh_frozen_contract()
    assert payload._qwen4_original_parameters[table_name] is replacement
    # Recovery to an invalid model must invalidate the previously valid binding.
    parameters[table_name] = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="trainable"):
        payload.refresh_frozen_contract()
    with pytest.raises(NotImplementedError):
        payload.convert_param(table_name, replacement.detach())


def test_native_registry_rejects_binder_that_does_not_bind(monkeypatch):
    from awex.models.registry import ModelRegistry, get_infer_weights_converter

    from areal.models.mcore.qwen4_exp_awex import register_qwen4_exp_awex

    monkeypatch.setattr(ModelRegistry, "models", {})
    register_qwen4_exp_awex(sglang_binder=lambda converter: None)
    with pytest.raises(ValueError, match="did not bind"):
        get_infer_weights_converter(
            "sglang",
            "Qwen4ExpForConditionalGeneration",
            SimpleNamespace(num_attention_heads=24, num_key_value_heads=2),
            SimpleNamespace(tp_rank=0, ep_rank=0),
            SimpleNamespace(tp_size=4, ep_size=1, device_backend="cpu"),
        )
