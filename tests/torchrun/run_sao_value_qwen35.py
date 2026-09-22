"""M5 Qwen3.5 recurrent isolation, actual microbatches and strict value reload."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from tests.test_sao_value_checkpoint import make_artifact

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.cli_args import (
    FSDPEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    PPOCriticConfig,
)
from areal.engine.fsdp_engine import FSDPPPOCritic
from areal.trainer.ppo.value_checkpoint import validate_value_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    engine = None
    try:
        if rank == 0:
            make_artifact(args.output, qwen35=True, linear=True)
        dist.broadcast_object_list([True], src=0)
        path = args.output / "value"
        manifest = validate_value_artifact(path)
        cfg = PPOCriticConfig(
            experiment_name="sao-value-qwen35",
            trial_name="microbatch-and-reload",
            path=str(path),
            backend=f"fsdp:d{world}",
            is_critic=True,
            attn_impl="flash_attention_2",
            dtype="bfloat16",
            optimizer_dtype="float32",
            disable_dropout=True,
            fsdp=FSDPEngineConfig(memory_efficient_load=True),
            optimizer=OptimizerConfig(lr=5e-6, weight_decay=0, warmup_steps=0),
            mb_spec=MicroBatchSpec(n_mbs=2, max_tokens_per_mb=128),
            ppo_n_minibatches=1,
            eps_clip=1000,
            value_contract={
                "identity": manifest["identity"],
                "protocol": manifest["protocol"],
                "require_pretrained": False,
            },
        )
        engine = FSDPPPOCritic(cfg)
        engine.create_process_group()
        engine.initialize(
            None,
            FinetuneSpec(
                total_train_epochs=1, dataset_size=24, train_batch_size=3 * world
            ),
        )
        rows = []
        for size in [5, 9, 17]:
            ids = (
                torch.arange(size, device=engine.device).remainder(29).unsqueeze(0) + 2
            )
            rows.append(
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "loss_mask": torch.ones_like(ids),
                }
            )
        batch = engine._normalize_batch_input(rows)[0]
        prepared = engine._prepare_mb_list(copy.deepcopy(batch))
        actual = len(prepared.mbs)
        assert actual == 3, (
            f"requested2, actual={actual}; Qwen3.5 must isolate three recurrent sequences"
        )
        assert all(mb["cu_seqlens"].numel() == 2 for mb in prepared.mbs)
        values = engine.compute_values(rows)
        for row, value in zip(rows, values):
            alone = engine.compute_values([row])[0]
            torch.testing.assert_close(value, alone, rtol=0, atol=0)
        changed = copy.deepcopy(rows)
        changed[0]["input_ids"].fill_(20)
        changed_values = engine.compute_values(changed)
        for a, b in zip(values[1:], changed_values[1:]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        raw_stats = []
        train_batch = engine.train_batch

        def record(*args, **kwargs):
            stats = train_batch(*args, **kwargs)
            raw_stats.append(stats)
            return stats

        engine.train_batch = record
        steps = []
        engine.optimizer.register_step_post_hook(lambda *unused: steps.append(1))
        for row, value in zip(rows, values):
            row["values"] = value
            row["returns"] = torch.ones_like(value)
        summary = engine.ppo_update(rows)
        assert steps == [1] and raw_stats[0]["num_micro_batches"] == 3
        assert summary["successful"] == 1 and summary["effective"] == 1
        before = engine.compute_values(rows)
        export = args.output / "reload"
        engine.save(
            SaveLoadMeta(
                path=str(export),
                weight_format="hf",
                with_optim=False,
                tokenizer=engine.tokenizer,
            )
        )
        with torch.no_grad():
            for param in engine.model.parameters():
                param.zero_()
        engine.load(
            SaveLoadMeta(path=str(export), weight_format="hf", with_optim=False)
        )
        for a, b in zip(before, engine.compute_values(rows)):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.002)
        result = {
            "rank": rank,
            "requested_microbatches": 2,
            "actual_microbatches": actual,
            "optimizer_steps": len(steps),
            "stats": raw_stats,
            "isolation": "batch equals separate forwards; changing neighbor leaves others unchanged",
            "reload": True,
            "scope": "tiny Qwen3.5 real FLA/FA2 FSDP, not 4B checkpoint qualification",
        }
        (args.output / f"rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    finally:
        if engine is not None:
            engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
