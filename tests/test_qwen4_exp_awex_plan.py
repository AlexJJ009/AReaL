# SPDX-License-Identifier: Apache-2.0
"""Exercise native AWEX metadata/planner on CPU, without inventing offsets."""

from types import SimpleNamespace

import pytest
import torch

from areal.models.mcore.qwen4_exp_awex import build_sharding_strategy
from areal.models.mcore.qwen4_exp_awex_layout import Qwen4ExpGDNLayout


def _metadata(raw):
    from awex.meta.meta_resolver import ParamMetaResolver

    class Resolver(ParamMetaResolver):
        def get_model_arch_name(self):
            return "Qwen4ExpForConditionalGeneration"

        def get_parameters_meta(self):
            return self._build_params_meta()

        def _get_params_raw_meta(self):
            return raw

        def _get_sharding_info(self, name, rank_info, param_meta):
            strategy = build_sharding_strategy()(
                engine_name="sglang" if rank_info.is_infer else "mcore",
                enable_dp_attention=False,
                enable_dp_lm_head=False,
                moe_dense_tp_size=rank_info.tp_size,
                tp_size=rank_info.tp_size,
                ep_size=1,
                ep_tp_size=1,
                rank_info=rank_info,
            )
            return strategy.get_sharding_strategy(name)

    return Resolver(SimpleNamespace(num_hidden_layers=48)).get_parameters_meta()


def _rank(tp, rank, pp, pp_rank, dp, dp_rank, inference):
    from awex.sharding.rank_info import RankInfo

    global_rank = dp_rank * pp * tp + pp_rank * tp + rank
    return RankInfo(
        tp_rank=rank,
        tp_size=tp,
        pp_rank=pp_rank,
        pp_size=pp,
        dp_rank=dp_rank,
        dp_size=dp,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=rank,
        attn_tp_size=tp,
        attn_dp_rank=dp_rank,
        world_size=tp * pp * dp,
        global_rank=global_rank,
        local_rank=global_rank % 8,
        engine_rank=0,
        is_infer=inference,
    )


def _raw(ranks, full_weights, tp):
    entries, tensors = [], {}
    for rank in ranks:
        params = []
        for name, full in full_weights.items():
            replicated = "hyper_connection" in name
            value = full if replicated else full.chunk(tp, dim=0)[rank.tp_rank]
            params.append(
                dict(
                    name=name,
                    shape=tuple(value.shape),
                    numel=value.numel(),
                    dtype=value.dtype,
                )
            )
            tensors[name, rank.global_rank] = value.clone()
        entries.append(
            dict(
                rank_info=rank,
                params_meta=params,
                model_arch_name="Qwen4ExpForConditionalGeneration",
            )
        )
    return entries, tensors


@pytest.mark.parametrize(
    "train_tp,infer_tp,dp,owner_pp",
    [(4, 4, 1, 1), (8, 4, 2, 3), (2, 4, 2, 1), (4, 8, 1, 1)],
)
def test_native_metadata_plan_reconstructs_every_destination_once(
    train_tp, infer_tp, dp, owner_pp
):
    from awex.transfer.transfer_plan import TransferPlanBuilder

    layout = Qwen4ExpGDNLayout(16, 48, 128, 128)
    source = torch.arange(16480 * 3, dtype=torch.float32).reshape(16480, 3)
    qkvz, ba = layout.pack_input(source, train_tp, infer_tp)
    conv = layout.pack_conv(
        torch.arange(10240 * 4, dtype=torch.float32).reshape(10240, 1, 4),
        train_tp,
        infer_tp,
    )
    weights = {
        "model.layers.24.linear_attn.in_proj_qkvz.weight": qkvz,
        "model.layers.24.linear_attn.in_proj_ba.weight": ba,
        "model.layers.24.linear_attn.conv1d.weight": conv,
        "model.layers.24.attn_hyper_connection.input_mix_weight_down.weight": torch.arange(
            15, dtype=torch.float32
        ).reshape(3, 5),
    }
    pp = owner_pp + 1
    train_ranks = [
        _rank(train_tp, rank, pp, owner_pp, dp, replica, False)
        for replica in range(dp)
        for rank in range(train_tp)
    ]
    infer_ranks = [_rank(infer_tp, rank, 1, 0, 1, 0, True) for rank in range(infer_tp)]
    train_raw, train_tensors = _raw(train_ranks, weights, train_tp)
    infer_raw, expected = _raw(infer_ranks, weights, infer_tp)
    train_meta, infer_meta = _metadata(train_raw), _metadata(infer_raw)
    for meta in train_meta + infer_meta:
        assert tuple(meta.global_shape) == tuple(weights[meta.name].shape)
    builder = TransferPlanBuilder(
        infer_world_size=infer_tp,
        train_world_size=train_tp * pp * dp,
        num_infer_engines=1,
        strict_param_key_match=True,
    )
    ops = builder.build_weights_mapping_operations(infer_meta, train_meta)
    destinations = {key: torch.empty_like(value) for key, value in expected.items()}
    written = {
        key: torch.zeros_like(value, dtype=torch.bool)
        for key, value in expected.items()
    }
    for op in ops:
        src_key = (op.send_shard_meta.name, op.send_rank - infer_tp)
        dst_key = (op.recv_shard_meta.name, op.recv_rank)
        assert not written[dst_key][op.inf_slices].any()
        destinations[dst_key][op.inf_slices].copy_(
            train_tensors[src_key][op.train_slices]
        )
        written[dst_key][op.inf_slices] = True
    for key in expected:
        assert written[key].all(), key
        torch.testing.assert_close(destinations[key], expected[key], rtol=0, atol=0)
