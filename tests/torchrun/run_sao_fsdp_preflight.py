# SPDX-License-Identifier: Apache-2.0
"""GPU preflight for SAO FSDP2 actor/critic full-update evidence.

Run only on an allocated GPU host, for example:

    torchrun --standalone --nproc-per-node=4 \
      tests/torchrun/run_sao_fsdp_preflight.py --role actor

The script intentionally uses the native FSDPEngine and PPO losses. It does not
mock FSDP or replace the engine.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    PPOCriticConfig,
)
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform
from areal.trainer.ppo.actor import grpo_loss_fn as actor_ppo_loss_fn
from areal.trainer.ppo.critic import ppo_loss_fn as critic_ppo_loss_fn
from areal.utils.constants import PROX_LOGP_METHOD_REUSE_TRAIN_LOGP

DEFAULT_PROMPT_LEN = 1024
DEFAULT_RESPONSE_LEN = 8192


def _setup_distributed() -> None:
    if dist.is_initialized():
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def _json_rank0(payload: dict[str, Any]) -> None:
    if dist.get_rank() == 0:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


def _torch_device() -> torch.device:
    return torch.device(
        f"{current_platform.device_type}:{current_platform.current_device()}"
    )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _tensor_fingerprint(tensor: torch.Tensor, sample: int = 4096) -> float:
    local = _local_tensor(tensor).detach().float().flatten()
    if local.numel() == 0:
        local_sum = torch.zeros((), device=_torch_device())
    else:
        stride = max(local.numel() // sample, 1)
        local_sum = local[::stride][:sample].sum()
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    return float(local_sum.cpu())


def _sample_weight_keys(engine: FSDPEngine, limit: int) -> list[str]:
    keys: list[str] = ["score.weight"] if engine.config.is_critic else []
    for name, param in engine.model.named_parameters():
        if param.requires_grad and "visual" not in name and name not in keys:
            keys.append(name)
        if len(keys) >= limit:
            break
    return keys


def _sample_params(engine: FSDPEngine, keys: list[str]) -> dict[str, torch.Tensor]:
    named = dict(engine.model.named_parameters())
    return {key: named[key] for key in keys}


def _grad_evidence(
    params: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | bool]]:
    out: dict[str, dict[str, float | bool]] = {}
    for key, param in params.items():
        grad = param.grad
        if grad is None:
            out[key] = {"present": False, "finite": False, "norm": 0.0}
            continue
        local_grad = _local_tensor(grad.detach()).float()
        finite = torch.isfinite(local_grad).all().to(torch.int32)
        sq_norm = torch.sum(local_grad * local_grad)
        dist.all_reduce(sq_norm, op=dist.ReduceOp.SUM)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        out[key] = {
            "present": True,
            "finite": bool(finite.cpu()),
            "norm": float(torch.sqrt(sq_norm).cpu()),
        }
    return out


def _make_batch(
    *,
    role: str,
    device: torch.device,
    seq_len: int,
    prompt_len: int,
    vocab_size: int,
    rank: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260920 + rank)
    high = max(vocab_size - 1, 2048)
    input_ids = torch.randint(
        32,
        high,
        (1, seq_len),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.bool)
    loss_mask = torch.zeros((1, seq_len), device=device, dtype=torch.bool)
    loss_mask[:, prompt_len - 1 : seq_len - 1] = True
    token_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
    response_scale = torch.linspace(0.5, 1.25, seq_len, device=device).unsqueeze(0)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }
    if role == "actor":
        batch.update(
            {
                "logprobs": torch.zeros(
                    (1, seq_len), device=device, dtype=torch.float32
                ),
                "advantages": torch.where(loss_mask, response_scale, 0.0),
                "kl_rewards": torch.zeros(
                    (1, seq_len), device=device, dtype=torch.float32
                ),
                "tot_rewards": torch.ones((1,), device=device, dtype=torch.float32),
                "rewards": torch.ones((1,), device=device, dtype=torch.float32),
            }
        )
    else:
        old_values = torch.sin(token_idx / 173.0).unsqueeze(0)
        returns = old_values + torch.where(loss_mask, response_scale * 0.25, 0.0)
        batch.update(
            {
                "values": old_values,
                "returns": returns,
            }
        )
    return batch


def _make_engine(args: argparse.Namespace) -> FSDPEngine:
    mb_spec = MicroBatchSpec(max_tokens_per_mb=args.seq_len, granularity=1)
    optimizer = OptimizerConfig(
        type="adam",
        lr=args.lr,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        eps=1.0e-8,
        lr_scheduler_type="constant",
        gradient_clipping=1.0,
    )
    common = dict(
        experiment_name="sao_fsdp_preflight",
        trial_name=f"sao_fsdp_preflight_{args.role}",
        path=args.model_path,
        backend=f"fsdp:d{dist.get_world_size()}p1t1",
        dtype="bfloat16",
        optimizer_dtype="float32",
        gradient_checkpointing=True,
        attn_impl="flash_attention_2",
        logprobs_chunk_size=args.logprobs_chunk_size,
        mb_spec=mb_spec,
        optimizer=optimizer,
    )
    if args.role == "actor":
        config = PPOActorConfig(
            **common,
            ppo_n_minibatches=1,
            eps_clip=0.2,
            eps_clip_higher=None,
            kl_ctl=0.0,
            adv_norm=None,
            reward_norm=None,
            use_decoupled_loss=False,
            recompute_logprob=False,
            rejection_sampling=None,
            importance_sampling_level="token",
            overlong_reward_penalty=False,
            use_sapo_loss=False,
            use_cispo_loss=False,
        )
    else:
        config = PPOCriticConfig(
            **common,
            is_critic=True,
            ppo_n_minibatches=1,
            eps_clip=0.5,
        )

    alloc = ModelAllocation.from_str(config.backend)
    engine = FSDPEngine(config)
    engine.create_process_group(parallel_strategy=alloc.parallel)
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(
            total_train_epochs=1,
            dataset_size=dist.get_world_size(),
            train_batch_size=dist.get_world_size(),
        ),
    )
    return engine


def _train_once(
    engine: FSDPEngine,
    args: argparse.Namespace,
    params: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, dict[str, float | bool]]]:
    if args.role == "actor":
        loss_fn = functools.partial(
            actor_ppo_loss_fn,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            use_decoupled_loss=False,
            rejection_sampling=None,
            importance_sampling_level="token",
            prox_logp_method=PROX_LOGP_METHOD_REUSE_TRAIN_LOGP,
            use_sapo_loss=False,
            use_cispo_loss=False,
        )
    else:
        loss_fn = functools.partial(critic_ppo_loss_fn, eps_clip=0.5)
    batch = _make_batch(
        role=args.role,
        device=engine.device,
        seq_len=args.seq_len,
        prompt_len=args.prompt_len,
        vocab_size=int(getattr(engine.model_config, "vocab_size", 151936)),
        rank=dist.get_rank(),
    )

    def loss_weight_fn(x):
        return x["loss_mask"].count_nonzero()

    stats = engine.train_batch(batch, loss_fn=loss_fn, loss_weight_fn=loss_weight_fn)
    return stats, _grad_evidence(params)


def _runtime_readback(engine: FSDPEngine, args: argparse.Namespace) -> dict[str, Any]:
    sample_param = next(
        param for param in engine.model.parameters() if param.requires_grad
    )
    local = _local_tensor(sample_param)
    return {
        "role": args.role,
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "device": str(engine.device),
        "cuda_device_name": torch.cuda.get_device_name(engine.device),
        "model_path": args.model_path,
        "model_type": getattr(engine.model_config, "model_type", None),
        "architectures": getattr(engine.model_config, "architectures", None),
        "backend": engine.config.backend,
        "dtype": engine.config.dtype,
        "optimizer_dtype": engine.config.optimizer_dtype,
        "attn_impl": engine.config.attn_impl,
        "gradient_checkpointing": engine.config.gradient_checkpointing,
        "param_storage_dtype": str(local.dtype),
        "is_critic": engine.config.is_critic,
        "seq_len": args.seq_len,
        "prompt_len": args.prompt_len,
        "response_len": args.seq_len - args.prompt_len,
        "tokens_per_rank": args.seq_len,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def _write_trace_summary(
    *,
    trace_dir: str,
    role: str,
    step_elapsed_s: float,
    peak_mem_bytes: int,
    stats: dict[str, float],
    grad: dict[str, dict[str, float | bool]],
    fingerprints: dict[str, dict[str, float | bool]],
    sampled_keys: list[str],
) -> None:
    if dist.get_rank() != 0:
        return
    trace_root = Path(trace_dir)
    kernel_names: set[str] = set()
    evidence_terms = {
        "gdn": False,
        "fla": False,
        "causalconv": False,
        "fullattnbackward": False,
    }
    for trace_file in trace_root.rglob("*.json"):
        text = trace_file.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        evidence_terms["gdn"] |= "gateddeltanet" in lowered or "gdn" in lowered
        evidence_terms["fla"] |= "fla" in lowered
        evidence_terms["causalconv"] |= (
            "causal_conv" in lowered or "causalconv" in lowered
        )
        evidence_terms["fullattnbackward"] |= (
            "flash" in lowered and "backward" in lowered
        ) or "scaled_dot_product" in lowered
        for needle in (
            "flash",
            "attention",
            "gateddeltanet",
            "causal_conv",
            "causalconv",
            "fla",
            "adam",
            "nccl",
        ):
            if needle in lowered:
                kernel_names.add(needle)
    summary = {
        "role": role,
        "status": "profiled_measured_update",
        "trace_dir": trace_dir,
        "elapsed_step_s": step_elapsed_s,
        "peak_mem_bytes": peak_mem_bytes,
        "stats": stats,
        "gradients": grad,
        "parameter_fingerprints": fingerprints,
        "sampled_source_weight_keys": sampled_keys,
        "kernel_name_terms_seen": sorted(kernel_names),
        "qwen35_evidence_terms": evidence_terms,
    }
    (trace_root / "rank0_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _optional_checkpoint_probe(
    engine: FSDPEngine, params: dict[str, torch.Tensor]
) -> dict[str, Any]:
    rank = dist.get_rank()
    tmpdir = tempfile.mkdtemp(prefix="sao_fsdp_preflight_dcp_") if rank == 0 else None
    holder = [tmpdir]
    dist.broadcast_object_list(holder, src=0)
    tmpdir = holder[0]
    try:
        before = {key: _tensor_fingerprint(param) for key, param in params.items()}
        engine._save_to_dcp(tmpdir, with_optim=True)
        engine._load_from_dcp(tmpdir, with_optim=True)
        after = {key: _tensor_fingerprint(param) for key, param in params.items()}
        return {
            "enabled": True,
            "path": tmpdir,
            "fingerprints_match": all(before[key] == after[key] for key in before),
        }
    finally:
        dist.barrier()
        if rank == 0 and tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["actor", "critic"], required=True)
    parser.add_argument("--model-path", default=os.environ.get("SAO_MODEL_PATH"))
    parser.add_argument(
        "--seq-len", type=int, default=DEFAULT_PROMPT_LEN + DEFAULT_RESPONSE_LEN
    )
    parser.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    parser.add_argument("--lr", type=float, default=5.0e-6)
    parser.add_argument("--logprobs-chunk-size", type=int, default=1024)
    parser.add_argument("--sample-weight-keys", type=int, default=6)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--checkpoint-probe", action="store_true")
    args = parser.parse_args()

    if args.seq_len <= args.prompt_len:
        raise ValueError("--seq-len must be larger than --prompt-len")
    if not args.model_path or not os.path.exists(args.model_path):
        raise FileNotFoundError(args.model_path)

    _setup_distributed()
    rank = dist.get_rank()
    allreduce_probe = torch.tensor([rank + 1.0], device=_torch_device())
    dist.all_reduce(allreduce_probe, op=dist.ReduceOp.SUM)
    _json_rank0(
        {
            "status": "nccl_allreduce_ok",
            "world_size": dist.get_world_size(),
            "sum_rank_plus_one": float(allreduce_probe.cpu()),
        }
    )

    engine = _make_engine(args)
    engine.train()
    sampled_keys = _sample_weight_keys(engine, args.sample_weight_keys)
    sampled_params = _sample_params(engine, sampled_keys)
    readback = _runtime_readback(engine, args)
    gathered_readback = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_readback, readback)
    _json_rank0({"status": "runtime_readback", "ranks": gathered_readback})

    warm_stats, warm_grad = _train_once(engine, args, sampled_params)
    current_platform.synchronize()
    _json_rank0(
        {
            "status": "warmup_update_complete",
            "stats": warm_stats,
            "gradients": warm_grad,
        }
    )

    before = {key: _tensor_fingerprint(param) for key, param in sampled_params.items()}
    trace_dir = args.trace_dir if rank == 0 else None
    if trace_dir is None and rank == 0:
        trace_dir = tempfile.mkdtemp(prefix=f"sao_fsdp_preflight_{args.role}_")
    trace_holder = [trace_dir]
    dist.broadcast_object_list(trace_holder, src=0)
    trace_dir = trace_holder[0]
    torch.cuda.reset_peak_memory_stats(engine.device)
    current_platform.synchronize()
    start = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ):
        stats, grad = _train_once(engine, args, sampled_params)
    current_platform.synchronize()
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(engine.device)
    after = {key: _tensor_fingerprint(param) for key, param in sampled_params.items()}
    fingerprints = {
        key: {
            "before": before[key],
            "after": after[key],
            "changed": before[key] != after[key],
        }
        for key in sampled_keys
    }
    any_changed = any(item["changed"] for item in fingerprints.values())
    if not any_changed:
        raise RuntimeError(
            "No sampled parameter fingerprint changed after optimizer step"
        )
    if not all(item["finite"] and item["norm"] > 0 for item in grad.values()):
        raise RuntimeError(f"Non-finite or zero sampled gradients: {grad}")

    checkpoint_probe = (
        _optional_checkpoint_probe(engine, sampled_params)
        if args.checkpoint_probe
        else {"enabled": False}
    )
    _write_trace_summary(
        trace_dir=trace_dir,
        role=args.role,
        step_elapsed_s=elapsed,
        peak_mem_bytes=peak_mem,
        stats=stats,
        grad=grad,
        fingerprints=fingerprints,
        sampled_keys=sampled_keys,
    )
    _json_rank0(
        {
            "status": "sao_fsdp_preflight_ok",
            "role": args.role,
            "trace_dir": trace_dir,
            "elapsed_step_s": elapsed,
            "peak_mem_bytes": peak_mem,
            "stats": stats,
            "gradients": grad,
            "parameter_fingerprints": fingerprints,
            "sampled_source_weight_keys": sampled_keys,
            "checkpoint_probe": checkpoint_probe,
        }
    )
    engine.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
