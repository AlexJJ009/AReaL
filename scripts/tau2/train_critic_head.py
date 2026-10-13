# SPDX-License-Identifier: Apache-2.0
"""τ² critic fit with a biased, mean-initialised value head (README 16.14.3).

Same as ``train_critic.py`` except the critic engine: the head gets a bias set to
``critic_head_bias_init``, a zero weight, and the learning rate ``critic_head_lr``.
"""

from __future__ import annotations

import argparse
import sys

from examples.tau2.critic_head import (
    Tau2HeadCriticConfig,
    Tau2HeadFSDPPPOCritic,
)
from examples.tau2.utils import Tau2PPOConfig
from scripts.tau2 import train_critic

from areal.api.cli_args import load_expr_config
from areal.utils.environ import is_single_controller


def validate_head_config(config) -> None:
    bias = config.critic_head_bias_init
    if bias is None or not 0.0 <= bias <= 1.0:
        raise ValueError(
            "critic_head_bias_init must be the expected success rate in [0, 1]"
        )
    if config.critic_head_lr is not None and config.critic_head_lr <= 0:
        raise ValueError("critic_head_lr must be positive")


class Tau2HeadCriticTrainer(train_critic.Tau2CriticOnlineTrainer):
    def _create_critic(self, critic_config, alloc):
        if alloc.backend != "fsdp":
            raise ValueError(f"head critic needs the fsdp backend, got {alloc.backend}")
        validate_head_config(self.config)
        config = Tau2HeadCriticConfig.from_critic(
            critic_config,
            head_bias_init=self.config.critic_head_bias_init,
            head_lr=self.config.critic_head_lr,
        )
        if is_single_controller():
            critic = Tau2HeadFSDPPPOCritic.as_controller(config, self.scheduler)
        else:
            critic = Tau2HeadFSDPPPOCritic(config=config)
        critic.create_process_group(parallel_strategy=alloc.parallel)
        return critic


def main(argv: list[str]) -> None:
    # Fail on a missing head setting before any rollout, including --check-config.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args(argv)
    overrides = [arg for arg in overrides if arg != "--check-config"]
    config, _ = load_expr_config(["--config", args.config, *overrides], Tau2PPOConfig)
    validate_head_config(config)
    train_critic.main(argv, trainer_cls=Tau2HeadCriticTrainer)


if __name__ == "__main__":
    main(sys.argv[1:])
