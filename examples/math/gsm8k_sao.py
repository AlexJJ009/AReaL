# SPDX-License-Identifier: Apache-2.0

"""Dense math SAO recipe, using the production PPO trainer and RLVR workflow."""

import argparse
import sys

from areal import PPOTrainer
from areal.api.cli_args import PPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.trainer.ppo.value_checkpoint import validate_value_artifact
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
            "SAO recipe requires raw rewards, gamma=1, MSE and no online warmup"
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
                f"SAO {role} requires paper LR, constant schedule, no warmup and one train batch"
            )
    contract = critic.value_contract
    if contract is None:
        if not allow_base_critic or critic.path != actor.path:
            raise ValueError(
                "Supply a pretrained value_contract or explicitly allow the actor's Base checkpoint as critic"
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
    options, remaining = parser.parse_known_args(args)
    config, _ = load_expr_config(remaining, PPOConfig)
    validate_sao_recipe(config, allow_base_critic=options.allow_base_critic)
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
    with PPOTrainer(config, train_dataset=train, valid_dataset=valid) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow"
            if valid is not None
            else None,
            eval_workflow_kwargs={**kwargs, "gconfig": config.eval_gconfig},
        )


if __name__ == "__main__":
    main(sys.argv[1:])
