"""Actor-update startup probe: does one acceleration option start and train?

Run with torchrun from the repository root, e.g.
  torchrun --nproc_per_node=4 actor_probe.py --model PATH --variant baseline
Variants: baseline (FSDP, flash_attention_2, packing, grad checkpointing),
ulysses2 (context parallel 2), tree (tree training), archon, archon_compile.
Each rank trains on synthetic episodes shaped like tau2 rollouts: groups of
sequences sharing a system-prompt-sized prefix. Prints one JSON line on rank 0.
"""

import argparse
import json
import os
import time
import traceback

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec, ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.infra.platforms import current_platform

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--variant", default="baseline")
p.add_argument("--seqs", type=int, default=8, help="sequences per DP rank")
p.add_argument("--group", type=int, default=4, help="sequences sharing a prefix")
p.add_argument("--prefix", type=int, default=3000)
p.add_argument("--min_len", type=int, default=6000)
p.add_argument("--max_len", type=int, default=16000)
p.add_argument("--steps", type=int, default=3)
a = p.parse_args()

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl")
current_platform.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

sp = 2 if a.variant == "ulysses2" else 1
dp = world // sp
result = {
    "variant": a.variant,
    "model": os.path.basename(a.model.rstrip("/")),
    "world": world,
}


def batch(seed: int):
    g = torch.Generator().manual_seed(seed + (rank // sp))
    rows = []
    for i in range(a.seqs):
        if i % a.group == 0:
            prefix = torch.randint(10000, 50000, (a.prefix,), generator=g)
        n = int(torch.randint(a.min_len, a.max_len, (1,), generator=g))
        rows.append(
            torch.cat(
                [prefix, torch.randint(10000, 50000, (n - a.prefix,), generator=g)]
            )
        )
    L = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    mask = torch.zeros(len(rows), L, dtype=torch.bool)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
        mask[i, : len(r)] = True
    loss_mask = mask.clone()
    loss_mask[:, : a.prefix] = False
    return {
        k: v.to(dev)
        for k, v in dict(
            input_ids=ids, attention_mask=mask, loss_mask=loss_mask
        ).items()
    }


def loss_fn(logprobs, entropy, input_data, **kwargs):
    return -logprobs.float().mean()


try:
    archon = a.variant.startswith("archon")
    config = TrainEngineConfig(
        backend=f"{'archon' if archon else 'fsdp'}:d{dp}c{sp}"
        if sp > 1
        else f"{'archon' if archon else 'fsdp'}:d{dp}",
        experiment_name="actor-probe",
        trial_name=a.variant,
        path=a.model,
        dtype="bfloat16",
        attn_impl="flash_attention_2",
        gradient_checkpointing=True,
        mb_spec=MicroBatchSpec(n_mbs=1, max_tokens_per_mb=32768),
        optimizer=OptimizerConfig(lr=1e-7),
        enable_tree_training=a.variant == "tree",
        # Tree training builds block masks over padded microbatches.
        pad_to_maximum=a.variant == "tree",
    )
    if a.variant == "archon_compile":
        config.archon.enable_compile = True
    if a.variant in ("archon_full", "archon_full_compile"):
        # Whole-block recompute, like the FSDP path's HF gradient checkpointing.
        config.archon.ac_mode = "full"
        config.archon.enable_compile = a.variant == "archon_full_compile"
    if archon:
        from areal.experimental.engine.archon_engine import ArchonEngine as Engine
    else:
        from areal.engine import FSDPEngine as Engine
    engine = Engine(config)
    engine.create_process_group(
        parallel_strategy=ParallelStrategy(
            data_parallel_size=dp, context_parallel_size=sp
        )
    )
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(total_train_epochs=1, dataset_size=64, train_batch_size=8),
    )
    engine.train()
    times = []
    for step in range(a.steps):
        data = batch(step)
        torch.cuda.synchronize()
        t0 = time.time()
        stats = engine.train_batch(
            input_=data,
            loss_fn=loss_fn,
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    tokens = int(data["attention_mask"].sum()) * dp
    result.update(
        ok=True,
        step_seconds=[round(t, 2) for t in times],
        tokens_per_step=tokens,
        tokens_per_second_after_warmup=round(
            tokens / (sum(times[1:]) / max(1, len(times) - 1))
        ),
        max_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
        grad_norm=(stats or {}).get("update_successful", None)
        if isinstance(stats, dict)
        else None,
    )
    engine.destroy()
except Exception as e:
    result.update(
        ok=False,
        error=f"{type(e).__name__}: {e}"[:800],
        trace=traceback.format_exc()[-1500:],
    )
if rank == 0:
    print("PROBE_RESULT " + json.dumps(result), flush=True)
dist.destroy_process_group()
