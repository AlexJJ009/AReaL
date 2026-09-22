"""M6 compare actual FSDP DIS gradients across DP and microbatch splits."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from tests.torchrun.run_sao_update import build_model

from areal.api import FinetuneSpec
from areal.api.cli_args import (
    FSDPEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
)
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.trainer.ppo.dis import DirectDISLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatches", type=int, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    engine = None
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            build_model(args.output / "model")
        dist.broadcast_object_list([True], src=0)
        cfg = PPOActorConfig(
            path=str(args.output / "model"),
            experiment_name="sao-dis-reduction",
            trial_name="fixed-input",
            backend=f"fsdp:d{world}",
            attn_impl="flash_attention_2",
            dtype="bfloat16",
            optimizer_dtype="float32",
            fsdp=FSDPEngineConfig(memory_efficient_load=False),
            disable_dropout=True,
            mb_spec=MicroBatchSpec(n_mbs=args.microbatches, max_tokens_per_mb=128),
            optimizer=OptimizerConfig(
                type="sgd",
                lr=0.001,
                weight_decay=0,
                gradient_clipping=1e9,
                warmup_steps=0,
            ),
            use_direct_dis_loss=True,
            recompute_logprob=False,
            use_decoupled_loss=False,
            kl_ctl=0,
            ppo_n_minibatches=1,
        )
        engine = FSDPPPOActor(cfg)
        engine.create_process_group()
        engine.initialize(
            None,
            FinetuneSpec(total_train_epochs=1, dataset_size=16, train_batch_size=4),
        )
        rows = []
        for index, size in enumerate([4, 9, 6, 15]):
            if index % world != rank:
                continue
            ids = (
                torch.arange(size, device=engine.device).remainder(29).unsqueeze(0) + 2
            )
            mask = torch.ones_like(ids)
            mask[:, -1] = 0
            behavior = torch.full_like(ids, -3.5, dtype=torch.float32)
            behavior[:, 1] = -10  # One rejected token remains in denominator.
            advantage = torch.full_like(behavior, 1.0 if index % 2 else -0.5)
            rows.append(
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "loss_mask": mask,
                    "logprobs": behavior,
                    "advantages": advantage,
                }
            )
        gradients = {}

        def capture(optimizer, args, kwargs):
            for name, param in engine.model.named_parameters():
                gradients[name] = param.grad.full_tensor().detach().cpu().clone()

        engine.optimizer.register_step_pre_hook(capture)
        stats = engine.train_batch(
            rows,
            loss_fn=DirectDISLoss(),
            loss_weight_fn=lambda row: row["loss_mask"].count_nonzero(),
        )
        assert stats["update_successful"] == 1
        if rank == 0:
            torch.save(gradients, args.output / "gradients.pt")
            (args.output / "result.json").write_text(
                json.dumps(
                    {
                        "world": world,
                        "microbatches": args.microbatches,
                        "stats": stats,
                        "test_optimizer": "SGD lr.001 for comparison only; production paper LR unchanged",
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        if engine is not None:
            engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
