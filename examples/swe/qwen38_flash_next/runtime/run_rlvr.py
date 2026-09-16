# SPDX-License-Identifier: Apache-2.0
"""GSM8K entrypoint with the existing Flash-Next SGLang startup contract."""

import os
import secrets
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    os.environ.setdefault("QWEN_GSM8K_TRIAL", "acceptance256-v1")
    os.environ.setdefault(
        "QWEN_GSM8K_OUTPUT", str(root / "runs" / os.environ["QWEN_GSM8K_TRIAL"])
    )
    os.environ.setdefault("QWEN_GSM8K_PROXY_KEY", secrets.token_urlsafe(32))
    sys.path[:0] = os.environ["QWEN_ACTOR_PYTHONPATH"].split(os.pathsep)

    from omegaconf import OmegaConf

    from areal.api.cli_args import GRPOConfig, to_structured_cfg

    config_path = root.parent / "rlvr_gsm8k_256k.yaml"
    if sys.argv[1:] == ["--check"]:
        config = OmegaConf.to_object(
            to_structured_cfg(OmegaConf.load(config_path), GRPOConfig)
        )
        assert config.gconfig.max_tokens == config.sglang.context_length == 262144
        assert config.actor.mb_spec.max_tokens_per_mb == 262144
        assert config.gconfig.n_samples == 8
        assert config.train_dataset.batch_size == 16
        assert config.rollout.queue_size > config.train_dataset.batch_size
        assert config.actor.weight_update_mode == "awex"
        assert not config.enable_offload and config.sglang.enable_memory_saver
        assert config.recover.freq_steps == 5 and not config.recover.no_save_optim
        for key in (
            "QWEN_REPO",
            "QWEN_MODEL",
            "QWEN_GSM8K_DATA",
            "QWEN_ACTOR_IMAGE",
            "QWEN_ROLLOUT_IMAGE",
            "QWEN_AWEX_FROZEN_CONTRACT",
        ):
            assert Path(os.environ[key]).exists(), key
        from areal.utils.logging import getLogger

        getLogger("QwenRecipe").info("GRPO schema, 256K limits, AWEX and paths checked")
        return

    import examples.math.gsm8k_rl as entry

    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.infra.scheduler.slurm import SlurmScheduler

    original_trainer = entry.PPOTrainer

    class GSM8KTrainer(original_trainer):
        def _init_scheduler(self):
            return SlurmScheduler(
                exp_config=self.config, container_mounts=os.environ["QWEN_MOUNTS"]
            )

        def __init__(self, *args, **kwargs):
            kwargs["valid_dataset"] = None
            super().__init__(*args, **kwargs)

        def train(self, **kwargs):
            kwargs["workflow"] = "gsm8k_math_agent.MathAgent"
            kwargs["eval_workflow"] = None
            return super().train(**kwargs)

    entry.PPOTrainer = GSM8KTrainer
    train = entry.main

    original = RemoteSGLangEngine.as_controller

    def controller_factory(*args, **kwargs):
        controller = original(*args, **kwargs)
        initialize = controller.initialize

        def initialize_model(**init_kwargs):
            server_args = dict(init_kwargs["server_args"])
            if server_args.get("tp_size") != 4 or server_args.get("ep_size") != 4:
                raise ValueError("Flash-Next baseline requires SGLang TP4/EP4")
            server_args.update(
                linear_attn_prefill_backend="flashinfer",
                linear_attn_decode_backend="flashinfer",
                ple_offload_embedding=True,
            )
            return initialize(**{**init_kwargs, "server_args": server_args})

        controller.initialize = initialize_model
        return controller

    RemoteSGLangEngine.as_controller = staticmethod(controller_factory)
    train(["--config", str(config_path), *sys.argv[1:]])


if __name__ == "__main__":
    main()
