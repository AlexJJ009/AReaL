# SPDX-License-Identifier: Apache-2.0

"""Dense math SAO recipe, using the production PPO trainer and RLVR workflow."""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from scripts.sao.async_eval import AsyncEvalPPOTrainer, SaoPPOConfig

from areal.api.cli_args import PPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.trainer.ppo.value_checkpoint import (
    validate_ppo_value_export,
    validate_value_artifact,
)
from areal.utils.hf_utils import load_hf_tokenizer


def validate_sao_recipe(config: PPOConfig, *, allow_base_critic: bool = False) -> None:
    """Reject silent fallbacks before starting workers or allocating GPUs.

    The base-critic option is an explicit cold start with an untrained scalar
    head. It does not grant pretrained value-model qualification.
    """
    actor, critic = config.actor, config.critic
    if critic is None or not critic.is_critic:
        raise ValueError("SAO requires a scalar critic")
    if (
        not actor.use_direct_dis_loss
        or config.critic_updates_before_actor != 2
        or actor.critic_gae_lambda != 1.0
        or actor.gae_lambda != "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"
        or actor.gae_lambda_kwargs != {"alpha": 1.5}
        or actor.dis_epsilon_low != 0.3
        or actor.dis_epsilon_high != 5.0
        or config.gconfig.n_samples != 1
    ):
        raise ValueError("SAO requires Direct DIS, paper lambdas, bounds, n=1 and K=2")
    if (
        actor.discount != 1.0
        or actor.adv_norm is not None
        or actor.reward_norm is not None
        or actor.reward_scaling != 1.0
        or actor.reward_bias != 0.0
        or critic.eps_clip is not None
        or config.teacher is not None
        or getattr(config, "num_critic_only_steps", 0)
    ):
        raise ValueError(
            "SAO recipe requires raw rewards, gamma=1, MSE and no critic-only stage"
        )
    for role, engine, lr in (("actor", actor, 1e-6), ("critic", critic, 5e-6)):
        opt = engine.optimizer
        if (
            opt is None
            or opt.lr != lr
            or opt.lr_scheduler_type != "constant"
            or opt.warmup_steps not in (None, 0)
            or opt.warmup_steps_proportion not in (None, 0)
            or engine.ppo_n_minibatches != 1
        ):
            raise ValueError(
                f"SAO {role} requires paper LR, constant schedule, "
                "no LR warmup and one train batch"
            )
    contract = critic.value_contract
    if contract is None:
        if Path(critic.path, "export-manifest.json").is_file():
            if critic.init_from_scratch or critic.use_lora:
                raise ValueError("PPO critic export requires full pretrained loading")
            validate_ppo_value_export(critic.path, actor.path)
            return
        if not allow_base_critic or critic.path != actor.path:
            raise ValueError(
                "Supply a pretrained value_contract or PPO export, or explicitly allow the actor's Base checkpoint as critic"
            )
        return
    if contract.get("require_pretrained") is not True:
        raise ValueError(
            "Online independent value checkpoints must require pretrained qualification"
        )
    protocol = contract.get("protocol", {})
    if any(
        protocol.get(key) != expected
        for key, expected in {
            "discount": actor.discount,
            "target_horizon": config.gconfig.max_new_tokens,
            "thinking": False,
            "termination": "finite-budget-terminal",
        }.items()
    ):
        raise ValueError(
            "Value protocol differs from this finite-budget, non-thinking recipe"
        )
    validate_value_artifact(
        critic.path,
        expected_identity=contract["identity"],
        expected_protocol=protocol,
        require_pretrained=True,
    )


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--allow-base-critic", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--startup-checkpoint-step", type=int, default=0)
    options, remaining = parser.parse_known_args(args)
    config, _ = load_expr_config(remaining, SaoPPOConfig)
    validate_sao_recipe(config, allow_base_critic=options.allow_base_critic)
    AsyncEvalPPOTrainer._validate_save_eval_sync(config)
    if options.check_config:
        sys.stdout.write(OmegaConf.to_yaml(OmegaConf.structured(config), resolve=True))
        return
    evidence = Path(config.cluster.fileroot) / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "resolved-config.json").write_text(
        json.dumps(dataclasses.asdict(config), indent=2) + "\n"
    )
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train = get_custom_dataset(
        split=config.train_dataset.split,
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid = None
    if config.valid_dataset is not None:
        valid = get_custom_dataset(
            split=config.valid_dataset.split,
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )
    kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
        length_stop_is_terminal=True,
    )
    workflow = "areal.workflow.rlvr.RLVRWorkflow"
    if "source_id" in train.column_names:
        workflow = "areal.workflow.sao_math.AuditedMathWorkflow"
        kwargs.update(
            reward_fn="areal.reward.math_prd.math_prd_reward_fn",
            audit_dir=str(evidence / "samples"),
        )
    with AsyncEvalPPOTrainer(
        config, train_dataset=train, valid_dataset=valid
    ) as trainer:
        trainer._snapshot_evidence = (
            workflow == "areal.workflow.sao_math.AuditedMathWorkflow"
        )
        save_training_state = trainer._save_training_state

        def save_with_startup_probe(*, epoch, epoch_step, global_step, force=False):
            probe = global_step + 1 == options.startup_checkpoint_step
            save_training_state(
                epoch=epoch,
                epoch_step=epoch_step,
                global_step=global_step,
                force=force or probe,
            )
            if probe:
                (evidence / "startup-checkpoint.json").write_text(
                    json.dumps({"step": global_step + 1, "saved": True}) + "\n"
                )

        trainer._save_training_state = save_with_startup_probe
        commit = trainer.stats_logger.commit

        def commit_with_evidence(epoch, step, global_step, data):
            result = commit(epoch, step, global_step, data)
            step_dir = evidence / "steps"
            step_dir.mkdir(exist_ok=True)
            path = step_dir / f"{global_step + 1}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "step": global_step + 1,
                        "completed_ns": time.time_ns(),
                        "metrics": data,
                    }
                )
                + "\n"
            )
            temporary.replace(path)
            return result

        trainer.stats_logger.commit = commit_with_evidence
        trainer.train(
            workflow=workflow,
            workflow_kwargs=kwargs,
            eval_workflow=workflow if valid is not None else None,
            eval_workflow_kwargs={**kwargs, "gconfig": config.eval_gconfig},
        )


if __name__ == "__main__":
    main(sys.argv[1:])
