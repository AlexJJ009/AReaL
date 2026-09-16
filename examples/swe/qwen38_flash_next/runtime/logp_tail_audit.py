# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-microbatch logp tails; host copies occur after optimizer.step."""

import functools
import os
import runpy
from pathlib import Path

import torch


def collect_tail(new, data, k=64):
    old = data["logprobs"].detach().reshape(-1)
    new = new.detach().reshape(-1)
    mask = data["loss_mask"].detach().bool().reshape(-1)
    assert old.shape == new.shape == mask.shape
    delta = (old - new).abs()
    finite = torch.isfinite(delta)
    score = torch.where(mask, torch.where(finite, delta, torch.inf), -torch.inf)
    values, indices = torch.topk(score, min(k, score.numel()))
    selected = mask[indices]
    thresholds = delta.new_tensor([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    counts = torch.stack([((delta > t) & mask & finite).sum() for t in thresholds])
    result = {
        "shape": tuple(data["logprobs"].shape),
        "index": indices,
        "selected": selected,
        "old_logp": old[indices],
        "new_logp": new[indices],
        "abs_diff": values,
        "n_masked": mask.sum(),
        "n_nonfinite": (mask & ~finite).sum(),
        "finite_abs_sum": torch.where(mask & finite, delta, 0.0).sum(),
        "thresholds": thresholds,
        "counts_above": counts,
    }
    for key in ("input_ids", "versions", "position_ids"):
        source = data.get(key)
        if isinstance(source, torch.Tensor) and source.numel() == old.numel():
            source = source.detach().reshape(-1)
            offsets = torch.arange(-4, 6, device=indices.device)
            context = indices[:, None] + offsets[None, :]
            result[key + "_context"] = source[context.clamp(0, source.numel() - 1)]
            result["context_offsets"] = offsets
            result["context_in_bounds"] = (context >= 0) & (context < source.numel())
    if isinstance(data.get("input_ids"), torch.Tensor):
        result["full_input_ids"] = data["input_ids"].detach().clone()
    return result


def main():
    import guarded_actor as guarded
    from megatron.core import parallel_state as mpu

    import areal.trainer.ppo.actor as actor

    pending = []
    original_loss = actor.grpo_loss_fn

    @functools.wraps(original_loss)
    def audited_loss(logprobs, entropy, input_data, *args, **kwargs):
        result = original_loss(logprobs, entropy, input_data, *args, **kwargs)
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            record = collect_tail(logprobs, input_data)
            record["current_version"] = kwargs.get("current_version")
            pending.append(record)
        return result

    actor.grpo_loss_fn = audited_loss
    factory = guarded.runtime.diagnostic.engine.get_megatron_optimizer

    def audited_factory(*args, **kwargs):
        opt = factory(*args, **kwargs)
        step = opt.step
        count = 0

        def audited_step(*a, **kw):
            nonlocal count
            result = step(*a, **kw)
            count += 1
            if pending:
                root = Path(os.environ["QWEN_LOGP_TAIL_DIR"])
                root.mkdir(parents=True, exist_ok=True)
                records = [
                    {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in record.items()
                    }
                    for record in pending
                ]
                path = (
                    root
                    / f"step-{count:04d}-rank-{os.environ.get('RANK', 'unknown')}.pt"
                )
                temp = path.with_suffix(".tmp")
                torch.save(
                    {
                        "records": records,
                        "optimizer_successful": bool(result[0]),
                        "microbatch_count": len(records),
                        "index_convention": "loss-position; target token AND behavior version use context offset +1",
                        "scope": "all microbatches on this last-PP TP0 rank",
                    },
                    temp,
                )
                temp.replace(path)
                pending.clear()
            return result

        opt.step = audited_step
        return opt

    guarded.runtime.diagnostic.engine.get_megatron_optimizer = audited_factory
    runpy.run_module("areal.infra.rpc.rpc_server", run_name="__main__")


if __name__ == "__main__":
    main()
