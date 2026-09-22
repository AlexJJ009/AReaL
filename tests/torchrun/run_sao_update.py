"""Bounded real FSDP2 M4 qualification with a CPU-created tiny dense model."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.cli_args import (
    FSDPEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    PPOActorConfig,
    PPOCriticConfig,
)
from areal.engine.fsdp_engine import FSDPPPOActor, FSDPPPOCritic
from areal.trainer.ppo.update import update_critic_before_actor


def build_model(path):
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=1,
            attention_dropout=0.0,
        )
    )
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0")
        ),
        unk_token="t0",
        pad_token="t0",
        eos_token="t1",
    )
    tokenizer.save_pretrained(path)


def snapshot(engine):
    return {
        name: value.full_tensor().detach().cpu().clone()
        for name, value in engine.model.named_parameters()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--independent-value", action="store_true")
    parser.add_argument("--direct-dis", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    actor = critic = None
    try:
        model_path = args.output / "initial_model"
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            build_model(model_path)
            if args.independent_value:
                from tests.test_sao_value_checkpoint import make_artifact

                make_artifact(args.output)
        ready = [str(model_path)]
        dist.broadcast_object_list(ready, src=0)
        common = dict(
            experiment_name="sao-m4-qualification",
            trial_name=f"dp{world}-mb{args.microbatches}",
            path=ready[0],
            backend=f"fsdp:d{world}",
            attn_impl=args.attention,
            dtype=args.dtype,
            optimizer_dtype="float32",
            disable_dropout=True,
            mb_spec=MicroBatchSpec(n_mbs=args.microbatches, max_tokens_per_mb=64),
            fsdp=FSDPEngineConfig(memory_efficient_load=False),
            ppo_n_minibatches=1,
        )

        def optimizer(lr):
            return OptimizerConfig(
                lr=lr,
                weight_decay=0.0,
                gradient_clipping=1.0,
                lr_scheduler_type="constant",
                warmup_steps=0,
            )

        actor = FSDPPPOActor(
            PPOActorConfig(
                **common,
                optimizer=optimizer(1e-6),
                kl_ctl=0,
                discount=1,
                gae_lambda="areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae",
                gae_lambda_kwargs={"alpha": 1.5},
                critic_gae_lambda=1.0,
                recompute_logprob=False,
                use_decoupled_loss=False,
                use_direct_dis_loss=args.direct_dis,
            )
        )
        critic = FSDPPPOCritic(
            PPOCriticConfig(
                **{
                    **common,
                    "path": str(args.output / "value")
                    if args.independent_value
                    else ready[0],
                },
                optimizer=optimizer(5e-6),
                is_critic=True,
                eps_clip=None if args.direct_dis else 1000,
            )
        )
        ft = FinetuneSpec(
            total_train_epochs=1, dataset_size=16, train_batch_size=2 * world
        )
        for engine in (actor, critic):
            engine.create_process_group()
            torch.manual_seed(19)
            engine.initialize(None, ft)
        events = []
        actor.optimizer.register_step_post_hook(
            lambda *unused: events.append("actor.optimizer")
        )
        critic.optimizer.register_step_post_hook(
            lambda *unused: events.append("critic.optimizer")
        )
        raw = []
        for offset, size in enumerate([4, 6]):
            ids = torch.tensor([[2 + rank + offset] * (size - 1) + [1]])
            mask = torch.ones_like(ids)
            mask[:, 0] = 0
            raw.append(
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "loss_mask": mask,
                    "logprobs": torch.full_like(ids, -3.5, dtype=torch.float32),
                    "rewards": torch.tensor([float(offset == 0)]),
                    "terminated": torch.tensor([True]),
                    "truncated": torch.tensor([False]),
                }
            )
        # Match DistRolloutCoordinator.prepare_batch's SPMD device transfer.
        raw = [
            {key: value.to(actor.device) for key, value in row.items()} for row in raw
        ]
        for row, values in zip(raw, critic.compute_values(raw)):
            row["values"] = values
        if args.independent_value:
            from areal.trainer.ppo.value_checkpoint import create_scalar_value_model

            with torch.device(actor.device):
                oracle = create_scalar_value_model(
                    critic.config.path,
                    dtype=torch.bfloat16,
                    attn_implementation=args.attention,
                ).eval()
            for row in raw:
                expected = (
                    oracle(
                        input_ids=row["input_ids"], attention_mask=row["attention_mask"]
                    )
                    .logits.squeeze(-1)
                    .float()
                )
                torch.testing.assert_close(
                    row["values"].float(), expected, rtol=0.01, atol=0.002
                )
            del oracle
        fixed = actor.compute_advantages(copy.deepcopy(raw))
        before = {"actor": snapshot(actor), "critic": snapshot(critic)}
        targets = [row["returns"].clone() for row in fixed]
        report = update_critic_before_actor(actor, critic, raw, fixed, 2)
        assert events == ["critic.optimizer", "critic.optimizer", "actor.optimizer"]
        assert actor.optimizer.param_groups[0]["lr"] == 1e-6
        assert critic.optimizer.param_groups[0]["lr"] == 5e-6
        after = {"actor": snapshot(actor), "critic": snapshot(critic)}
        changed = {}
        for role in ("actor", "critic"):
            changed[role] = sum(
                not torch.equal(before[role][key], value)
                for key, value in after[role].items()
            )
            assert changed[role] > 0
        for row, target in zip(fixed, targets):
            torch.testing.assert_close(row["returns"], target, rtol=0, atol=0)
        reload_checked = False
        if args.independent_value:
            saved_values = critic.compute_values(raw)
            export = args.output / "value_reload"
            critic.save(
                SaveLoadMeta(
                    path=str(export),
                    weight_format="hf",
                    with_optim=False,
                    tokenizer=critic.tokenizer,
                )
            )
            with torch.no_grad():
                for parameter in critic.model.parameters():
                    parameter.zero_()
            critic.load(
                SaveLoadMeta(path=str(export), weight_format="hf", with_optim=False)
            )
            for actual, expected in zip(critic.compute_values(raw), saved_values):
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
            reload_checked = True
        all_masked_checked = False
        if args.direct_dis:
            # Adam already has moments: an all-DIS-masked update must not apply
            # momentum or weight decay, even though backward produces zeros.
            before_masked = snapshot(actor)
            masked = copy.deepcopy(raw)
            for row in masked:
                row["logprobs"].fill_(-10.0)
            masked_batch = actor.compute_advantages(masked)
            old_event_count = len(events)
            masked_report = actor.ppo_update(masked_batch)
            assert masked_report["all_masked"] == 1 and masked_report["effective"] == 0
            assert len(events) == old_event_count
            for name, value in snapshot(actor).items():
                torch.testing.assert_close(value, before_masked[name], rtol=0, atol=0)
            all_masked_checked = True
        if rank == 0:
            torch.save({"before": before, "after": after}, args.output / "weights.pt")
            (args.output / "result.json").write_text(
                json.dumps(
                    {
                        "events": events,
                        "report": report,
                        "changed_parameters": changed,
                        "world_size": world,
                        "microbatches": args.microbatches,
                        "actor_lr": 1e-6,
                        "critic_lr": 5e-6,
                        "dtype": args.dtype,
                        "attention": args.attention,
                        "independent_value": args.independent_value,
                        "value_reload_checked": reload_checked,
                        "direct_dis": args.direct_dis,
                        "all_masked_optimizer_skip_checked": all_masked_checked,
                        "scope": "M4 synthetic real FSDP; not SAO end-to-end or value qualification",
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        for engine in (critic, actor):
            if engine is not None:
                engine.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
