# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optimizer_roundtrip_preserves_cpu_ownership_and_adam_updates(monkeypatch):
    monkeypatch.delenv("AWEX_OPT_OFFLOAD_VIA_HDO", raising=False)
    params = [torch.nn.Parameter(torch.ones(4, device=d)) for d in ("cpu", "cuda")]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    opts = [torch.optim.AdamW([p], lr=0.01) for p in params]
    ref_opts = [torch.optim.AdamW([p], lr=0.01) for p in refs]
    wrapped = [
        SimpleNamespace(optimizer=o, shard_fp32_from_float16_groups=[[p]])
        for o, p in zip(opts, params)
    ]
    adapter = MegatronWeightResidency(
        SimpleNamespace(
            optimizer=SimpleNamespace(chained_optimizers=wrapped),
            device=torch.device("cuda"),
        )
    )
    for _ in range(3):
        for p, r, o, ro in zip(params, refs, opts, ref_opts):
            p.grad = torch.full_like(p, 0.5)
            r.grad = torch.full_like(r, 0.5)
            o.step()
            ro.step()
        cpu_moment = opts[0].state[params[0]]["exp_avg"]
        adapter._offload_optimizer_states()
        assert all(p.device.type == "cpu" for p in params)
        adapter._reload_optimizer_states()
        adapter._reload_optimizer_states()
        assert opts[0].state[params[0]]["exp_avg"] is cpu_moment
        for p, r, o, ro in zip(params, refs, opts, ref_opts):
            assert p.device == r.device
            torch.testing.assert_close(p, r)
            for key in ("exp_avg", "exp_avg_sq"):
                actual, expected = o.state[p][key], ro.state[r][key]
                assert actual.device == expected.device
                torch.testing.assert_close(actual, expected)
