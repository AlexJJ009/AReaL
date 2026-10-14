# SPDX-License-Identifier: Apache-2.0

"""CPU wire-metadata and precision checks; these do not emulate CUDA/NCCL."""

from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from areal.api.alloc_mode import ModelAllocation
from areal.api.io_struct import WeightUpdateMeta
from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.sglang_remote import SGLangBackend


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_text", "qwen3"])
@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("group_count", [1, 2])
def test_xccl_metadata_and_values_preserve_qwen35_sglang_a_log(
    monkeypatch, model_type, backend, storage_dtype, group_count
):
    """Exercise the sender, including bucket boundaries and both group paths."""
    prefix = "model.language_model.layers.0.linear_attn."
    # Values deliberately not representable in bf16, unlike integers/zeros.
    a_log = torch.linspace(0.1001, 3.1001, 32, dtype=torch.float32).to(storage_dtype)
    parameters = [
        (prefix + "dt_bias", a_log.clone()),
        (prefix + "A_log", a_log.clone()),
        (prefix + "conv1d.weight", torch.ones(8192, 1, 4, dtype=storage_dtype)),
        # A >1 MiB parameter forces a middle bucket and the final drain path.
        (
            "model.language_model.embed_tokens.weight",
            torch.ones(1024, 1024, dtype=storage_dtype),
        ),
        (prefix.replace("layers.0", "layers.1") + "A_log", a_log.clone()),
    ]
    meta = WeightUpdateMeta.from_fsdp_xccl(
        gen_allocation=ModelAllocation.from_str(f"{backend}:d1t1")
    )
    meta.weight_chunked_mem_mb = 1
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.config = SimpleNamespace(use_lora=False, dtype="bfloat16")
    engine.model_config = SimpleNamespace(model_type=model_type)
    engine.model = SimpleNamespace(named_parameters=lambda: iter(parameters))
    engine.is_vision_model = False
    engine._initialized = True
    engine._cpu_group = object()
    engine.weight_update_master_addr = "localhost"
    engine.weight_update_master_port = 12345
    engine.weight_update_groups = [object() for _ in range(group_count)]
    engine.weight_update_group_names = [
        f"update_weight_group_{i}" for i in range(group_count)
    ]
    # Leave real bucket construction/serialization intact; replace only I/O.
    engine._get_full_tensor = lambda param: param.detach()
    received = {group: [] for group in engine.weight_update_groups}
    pending = {group: [] for group in engine.weight_update_groups}
    events = []

    def submit(update_meta, specs):
        group = engine.weight_update_groups[
            engine.weight_update_group_names.index(update_meta.nccl_group_name)
        ]
        assert not pending[group]
        payload = (
            SGLangBackend()
            .build_distributed_weight_update_requests(update_meta, specs)
            .requests[0]
            .payload
        )
        assert "load_format" not in payload
        for name, dtype, shape in zip(
            payload["names"], payload["dtypes"], payload["shapes"]
        ):
            pending[group].append(
                (name, torch.empty(shape, dtype=getattr(torch, dtype)))
            )
        future = Future()
        future.set_result(None)
        return future

    def broadcast(tensor, *, src, group, async_op):
        assert src == 0
        name, wire_tensor = pending[group].pop(0)
        assert wire_tensor.shape == tensor.shape
        assert wire_tensor.dtype == tensor.dtype
        wire_tensor.copy_(tensor)
        received[group].append((name, wire_tensor))
        return SimpleNamespace(wait=lambda: None) if async_op else None

    engine.rollout_engine = SimpleNamespace(
        pause_generation=lambda: events.append("pause"),
        continue_generation=lambda: events.append("continue"),
        update_weights_from_distributed=submit,
    )
    monkeypatch.setattr("areal.engine.fsdp_engine.dist.get_rank", lambda: 0)
    monkeypatch.setattr("areal.engine.fsdp_engine.dist.barrier", lambda **kwargs: None)
    monkeypatch.setattr("areal.engine.fsdp_engine.dist.broadcast", broadcast)
    monkeypatch.setattr(
        "areal.engine.fsdp_engine.current_platform",
        SimpleNamespace(device_type="cpu", synchronize=lambda: None),
    )

    engine._update_weights_from_distributed(meta)

    assert events == ["pause", "continue"]
    for group in engine.weight_update_groups:
        assert not pending[group]
        assert [name for name, _ in received[group]] == [name for name, _ in parameters]
        for (name, actual), (_, source) in zip(received[group], parameters):
            preserve = (
                model_type.startswith("qwen3_5")
                and backend == "sglang"
                and name.endswith(".linear_attn.A_log")
            )
            expected = source if preserve else source.to(torch.bfloat16)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # SGLang's A_log destination is fp32; round-tripping must not
            # silently lose fp32 actor values on the protected path.
            if preserve:
                torch.testing.assert_close(
                    actual.float(), source.float(), rtol=0, atol=0
                )
