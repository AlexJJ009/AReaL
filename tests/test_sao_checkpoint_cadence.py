# SPDX-License-Identifier: Apache-2.0
"""Regression for actor/critic saves sharing a single logical step clock."""

from types import SimpleNamespace

from areal.api import FinetuneSpec
from areal.api.cli_args import SaverConfig
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils.saver import Saver


def test_actor_and_critic_save_together_every_twenty_steps_and_at_tail(
    tmp_path, monkeypatch
):
    saved = {"actor": [], "critic": []}
    actor = SimpleNamespace(save=lambda meta: saved["actor"].append(meta.path))
    critic = SimpleNamespace(save=lambda meta: saved["critic"].append(meta.path))
    config = SaverConfig(
        experiment_name="cadence",
        trial_name="paired",
        fileroot=str(tmp_path),
        mode="sync",
        freq_steps=20,
        freq_epochs=1,
        freq_secs=None,
    )
    saver = Saver(
        config,
        FinetuneSpec(total_train_epochs=1, dataset_size=45, train_batch_size=1),
    )
    monkeypatch.setattr(saver, "_should_use_async", lambda engine: False)
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.saver = saver
    trainer.actor = actor
    trainer.critic = critic
    trainer.tokenizer = None
    trainer.processor = None

    for step in range(45):
        trainer._save_hf(epoch=0, epoch_step=step, global_step=step)

    for role in ("actor", "critic"):
        assert len(saved[role]) == 3
        assert [path.split("globalstep")[-1] for path in saved[role]] == [
            "19",
            "39",
            "44",
        ]


def test_controller_critic_metrics_are_exported_without_overwriting_actor(monkeypatch):
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.actor = SimpleNamespace(
        export_stats=lambda: {"grad_norm": 1.0, "optimizer_steps": 2}
    )
    trainer.critic = SimpleNamespace(
        export_stats=lambda: {"grad_norm": 0.3, "optimizer_steps": 2}
    )
    trainer.rollout = SimpleNamespace(export_stats=lambda: {"reward": 0.5})
    trainer.eval_rollout = None
    recorded = []
    trainer.stats_logger = SimpleNamespace(
        commit=lambda *args: recorded.append(args[-1])
    )
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)

    trainer._export_and_commit_stats(0, 1, 1)

    assert recorded == [
        {
            "grad_norm": 1.0,
            "optimizer_steps": 2,
            "critic/grad_norm": 0.3,
            "critic/optimizer_steps": 2,
            "reward": 0.5,
        }
    ]
