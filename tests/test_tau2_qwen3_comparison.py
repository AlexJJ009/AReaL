# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the PPO-matched Qwen3 comparison recipes."""

import os
import subprocess
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from datasets import Dataset

from examples.tau2 import ppo_data, task_pack, train
from examples.tau2.evaluation import Tau2AsyncEvalTrainer, Tau2PPOTrainer
from examples.tau2.utils import Tau2PPOConfig

from areal.api.cli_args import load_expr_config


def _config(monkeypatch, tmp_path, algorithm, budget="default"):
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("TAU2_TRIAL_NAME", "comparison-test")
    monkeypatch.delenv("TAU2_ACTOR_PATH", raising=False)
    monkeypatch.delenv("TAU2_CRITIC_PATH", raising=False)
    suffix = "" if algorithm == "ppo" else "_qwen3"
    overrides = (
        [
            "train_dataset.batch_size=16",
            "train_batch_episodes=128",
            "total_train_epochs=1",
        ]
        if algorithm == "grpo" and budget == "matched"
        else []
    )
    return load_expr_config(
        ["--config", f"examples/tau2/config_{algorithm}{suffix}.yaml", *overrides],
        Tau2PPOConfig,
    )[0]


@pytest.mark.parametrize("algorithm", ["sao", "grpo"])
def test_matched_config_preserves_ppo_optimizers_and_parser(
    monkeypatch, tmp_path, algorithm
):
    """The separate recipes inherit the PPO settings outside algorithm choices."""
    ppo = _config(monkeypatch, tmp_path, "ppo")
    config = _config(monkeypatch, tmp_path, algorithm)
    pack = tmp_path / "pack"
    pack.mkdir()
    config.train_task_packs = [str(pack)]
    assert train.validate_tau2_recipe(config) == ("airline", "retail")
    Tau2AsyncEvalTrainer.validate_config(config)
    assert config.actor.optimizer == ppo.actor.optimizer
    assert config.actor.path == ppo.actor.path
    assert config.actor.eps_clip == 0.2
    assert config.actor.eps_clip_higher == (0.28 if algorithm == "grpo" else 0.2)
    assert config.rollout.agent == ppo.rollout.agent
    assert (
        replace(
            config.evaluation_rollout,
            experiment_name=ppo.experiment_name,
            scheduling_spec=ppo.evaluation_rollout.scheduling_spec,
        )
        == ppo.evaluation_rollout
    )
    assert config.actor.per_episode_microbatching
    assert config.train_batch_episodes == (64 if algorithm == "grpo" else 128)
    assert (
        replace(config.evaluator, experiment_name=ppo.experiment_name) == ppo.evaluator
    )
    if algorithm == "sao":
        assert config.critic.freeze_critic_attention
        assert config.critic.optimizer.lr == 5e-6
        assert (
            replace(config.critic.optimizer, lr=ppo.critic.optimizer.lr)
            == ppo.critic.optimizer
        )
        assert (
            replace(
                config.critic,
                experiment_name=ppo.experiment_name,
                freeze_critic_attention=ppo.critic.freeze_critic_attention,
                optimizer=ppo.critic.optimizer,
            )
            == ppo.critic
        )
        assert config.critic_head_bias_init == ppo.critic_head_bias_init
        assert config.total_train_epochs == 8
        assert config.actor.adv_norm is None
        broken = deepcopy(config)
        broken.critic.eps_clip = 0.2
        with pytest.raises(ValueError, match="SAO requires"):
            train.validate_tau2_recipe(broken)
    else:
        assert config.critic is None
        assert not config.enable_offload
        assert config.gconfig.n_samples == 8
        assert config.train_dataset.batch_size == 8
        assert config.total_train_epochs == 2


@pytest.mark.parametrize(
    "algorithm,budget,steps,episodes",
    [
        ("ppo", "default", 48, 5976),
        ("sao", "default", 48, 5976),
        ("grpo", "default", 188, 11952),
        ("grpo", "matched", 47, 5976),
    ],
)
def test_check_config_combined_pool_finite_tail_and_eval_without_workers(
    monkeypatch, tmp_path, algorithm, budget, steps, episodes
):
    """The real entrypoint binds full coverage and Telecom eval before any workers."""
    config = _config(monkeypatch, tmp_path, algorithm, budget)
    pack = tmp_path / "pack"
    pack.mkdir()
    monkeypatch.setattr(task_pack, "domains", lambda _: ["airline", "retail"])
    monkeypatch.setattr(
        task_pack,
        "task_ids",
        lambda _, domain: [f"p{i}" for i in range(300 if domain == "airline" else 348)],
    )
    monkeypatch.setattr(train.benchmark_variant, "validate_task_files", lambda *a: None)

    def official(**kwargs):
        split = kwargs["split"]
        counts = (
            {"airline": 30, "retail": 69}
            if split == "train"
            else {"airline": 20, "retail": 40, "telecom": 40}
        )
        return Dataset.from_list(
            [
                {
                    "domain": d,
                    "task_id": f"{split}-{i}",
                    "split": split,
                    "source_id": f"tau2:{d}:{split}-{i}",
                }
                for d in kwargs["domain"]
                for i in range(counts[d])
            ]
        )

    monkeypatch.setattr(train, "get_tau2_dataset", official)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU preflight must not resolve snapshots or create workers")

    monkeypatch.setattr(train, "resolve_actor_snapshot", forbidden)
    monkeypatch.setattr(Tau2AsyncEvalTrainer, "__init__", forbidden)
    configs = []
    monkeypatch.setattr(
        train.logger,
        "info",
        lambda message, *args: configs.append(args[0])
        if message.startswith("Resolved config")
        else None,
    )
    suffix = "" if algorithm == "ppo" else "_qwen3"
    args = [
        "--config",
        f"examples/tau2/config_{algorithm}{suffix}.yaml",
        f"train_task_packs=[{pack}]",
        "--check-config",
    ]
    if algorithm == "grpo" and budget == "matched":
        args.extend(
            [
                "train_dataset.batch_size=16",
                "train_batch_episodes=128",
                "total_train_epochs=1",
            ]
        )
    train.main(args)
    from omegaconf import OmegaConf

    resolved = OmegaConf.create(configs[-1])
    assert resolved.total_train_steps == steps
    assert resolved.effective_episodes == episodes
    assert sum(resolved.domain_effective_episodes.values()) == (
        5976 if algorithm == "grpo" else 747
    )
    config.train_task_packs = [str(pack)]
    kwargs = {"domain": tuple(config.domains), "provenance": {}, "seed": 42}
    rows = ppo_data.build_ppo_train_dataset(config, kwargs)
    assert len(rows) == 747
    assert set(
        Counter((r["domain"], r["task_id"], r["task_pack"]) for r in rows).values()
    ) == {1}
    batch = config.train_dataset.batch_size
    assert 747 % batch == (
        3
        if algorithm == "grpo" and budget == "default"
        else 11
        if algorithm == "grpo"
        else 107
    )
    with pytest.raises(ValueError, match="complete epochs"):
        train.main([*args, "total_train_steps=1"])

    # Exercise runtime selection, stopping at the constructor before any worker.
    class SelectedTrainer(Exception):
        pass

    selected = []

    def intercept_init(self, *args, **kwargs):
        selected.append(type(self))
        raise SelectedTrainer

    monkeypatch.setattr(train, "resolve_actor_snapshot", lambda _: None)
    monkeypatch.setattr(Tau2PPOTrainer, "__init__", intercept_init)
    monkeypatch.setattr(Tau2AsyncEvalTrainer, "__init__", intercept_init)
    with pytest.raises(SelectedTrainer):
        train.main([arg for arg in args if arg != "--check-config"])
    assert selected == [Tau2AsyncEvalTrainer if algorithm == "grpo" else Tau2PPOTrainer]


def test_sao_backbone_resolution_selects_adr0012_head(monkeypatch, tmp_path):
    """Matched SAO reaches the same initialized head engine as PPO at runtime."""
    from examples.tau2 import critic_head

    config = _config(monkeypatch, tmp_path, "sao")
    config.train_task_packs = ["pack"]
    monkeypatch.setattr(train, "resolve_pinned_hf_snapshot", lambda _: "/local/actor")
    train.resolve_actor_snapshot(config)
    assert config.critic.path == config.actor.path == "/local/actor"
    sentinel = object()
    seen = []
    monkeypatch.setattr(
        critic_head, "create_head_critic", lambda *a: seen.append(a) or sentinel
    )
    trainer = Tau2PPOTrainer.__new__(Tau2PPOTrainer)
    trainer.config = config
    assert trainer._create_critic(config.critic, "allocation") is sentinel
    assert seen == [(trainer, config.critic, "allocation")]


@pytest.mark.parametrize(
    "algorithm,budget", [("sao", None), ("grpo", "default"), ("grpo", "matched")]
)
def test_launcher_check_config_skips_runtime_and_preserves_overrides(
    monkeypatch, tmp_path, algorithm, budget
):
    """A fake Python command makes accidental credential/runtime setup fail visibly."""
    executable = tmp_path / "python"
    executable.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    env = {
        **os.environ,
        "AREAL_PYTHON": str(executable),
        "TAU2_PREP_ARTIFACT_ROOT": str(tmp_path),
        "TAU2_AREAL_PACK": str(tmp_path / "pack"),
    }
    for name in (
        "SAO_ARTIFACT_ROOT",
        "HF_HUB_CACHE",
        "TAU2_DATA_DIR",
        "TAU2_DEEPSEEK_ENV_FILE",
    ):
        env.pop(name, None)
    result = subprocess.run(
        [
            "bash",
            "scripts/tau2/launch_qwen3.sh",
            algorithm,
            "trial",
            "check-config",
            *([budget] if budget else []),
            "train_dataset.batch_size=32" if budget else "critic.optimizer.lr=1e-5",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    argv = result.stdout.splitlines()
    assert str(Path(f"examples/tau2/config_{algorithm}_qwen3.yaml").resolve()) in argv
    assert "--check-config" in argv
    override = "train_dataset.batch_size=32" if budget else "critic.optimizer.lr=1e-5"
    assert argv[-1] == override
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "trial"))
    monkeypatch.setenv("TAU2_TRIAL_NAME", "trial")
    config = load_expr_config(
        [arg for arg in argv if arg != "--check-config"], Tau2PPOConfig
    )[0]
    if budget:
        assert config.train_dataset.batch_size == 32
    else:
        assert config.critic.optimizer.lr == 1e-5
    if budget:
        assert f"train_dataset.batch_size={8 if budget == 'default' else 16}" in argv
        assert f"total_train_epochs={2 if budget == 'default' else 1}" in argv


def test_proposed_grpo_group_size_can_change_with_prompt_batch(monkeypatch, tmp_path):
    """Paired group/batch overrides remain usable while the episode unit stays128."""
    config = _config(monkeypatch, tmp_path, "grpo", "matched")
    config.train_task_packs = [str(tmp_path)]
    config.gconfig.n_samples = 4
    config.actor.reward_norm.group_size = 4
    config.train_dataset.batch_size = 32
    train.validate_tau2_recipe(config)
    monkeypatch.setattr(task_pack, "domains", lambda _: ["airline", "retail"])
    monkeypatch.setattr(task_pack, "task_ids", lambda *_: ["pack-task"])
    monkeypatch.setattr(
        train,
        "get_tau2_dataset",
        lambda **kw: Dataset.from_list(
            [
                {
                    "domain": d,
                    "task_id": "official-task",
                    "source_id": f"tau2:{d}:official-task",
                }
                for d in kw["domain"]
            ]
        ),
    )
    kwargs = {"domain": tuple(config.domains), "provenance": {}, "seed": 42}
    rows = list(ppo_data.build_ppo_train_dataset(config, kwargs))
    assert len(rows) == 4
    assert config.domain_effective_episodes == {"airline": 8, "retail": 8}
    ppo = _config(monkeypatch, tmp_path, "ppo")
    ppo.train_task_packs = [str(tmp_path)]
    assert rows == list(ppo_data.build_ppo_train_dataset(ppo, kwargs))
    config.train_dataset.batch_size = 16
    with pytest.raises(ValueError, match="prompt-group count"):
        train.validate_tau2_recipe(config)
