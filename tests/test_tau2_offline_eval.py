# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for checkpoint-only formal PPO training."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from examples.tau2 import train
from examples.tau2.evaluation import Tau2PPOTrainer
from examples.tau2.utils import Tau2PPOConfig
from tests.test_tau2_ppo import _config

from areal.api.cli_args import SaverConfig
from areal.api.io_struct import FinetuneSpec
from areal.trainer.async_eval import AsyncEvalPPOTrainer
from areal.utils.saver import Saver


def test_offline_eval_switch_defaults_off_and_accepts_override():
    config = OmegaConf.structured(Tau2PPOConfig)
    assert config.offline_eval_only is False
    assert OmegaConf.merge(config, {"offline_eval_only": True}).offline_eval_only


@pytest.mark.parametrize("offline", [False, True])
def test_dedicated_role_and_proxy_only_initialized_online(monkeypatch, offline):
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(offline_eval_only=offline)
    controller = Mock()
    parent_init = Mock(
        side_effect=lambda: setattr(trainer, "_async_eval_rollout", controller)
    )
    monkeypatch.setattr(
        AsyncEvalPPOTrainer, "_init_dedicated_eval_rollout", parent_init
    )
    trainer._init_dedicated_eval_rollout()
    assert parent_init.call_count == int(not offline)
    assert controller.start_proxy.call_count == int(not offline)


@pytest.mark.parametrize(
    "method,args,kwargs",
    [
        ("_evaluate_fn", ("workflow", {}), {}),
        ("_evaluate", ("workflow", {}, 0, 9, 9), {}),
        (
            "_enqueue_eval",
            (),
            dict(
                version=10,
                checkpoint_path="unused",
                eval_workflow="workflow",
                eval_workflow_kwargs={},
                snapshot=False,
            ),
        ),
        ("_drain_evaluations", (), {}),
    ],
)
@pytest.mark.parametrize("offline", [False, True])
def test_all_eval_paths_delegate_only_online(
    monkeypatch, method, args, kwargs, offline
):
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(offline_eval_only=offline)
    parent = Mock()
    monkeypatch.setattr(AsyncEvalPPOTrainer, method, parent)
    getattr(trainer, method)(*args, **kwargs)
    assert parent.call_count == int(not offline)


def test_offline_formal_contract_keeps_eval_set_but_ignores_eval_schedule(
    monkeypatch, tmp_path
):
    config = _config(monkeypatch, tmp_path)
    config.offline_eval_only = True
    config.evaluator.eval_before_train = False
    config.evaluator.freq_steps = None
    config.saver.freq_steps = 10
    train.validate_tau2_recipe(config)
    Tau2PPOTrainer.validate_config(config)
    config.offline_eval_only = False
    with pytest.raises(ValueError):
        train.validate_tau2_recipe(config)
    with pytest.raises(ValueError, match="frequencies to match"):
        Tau2PPOTrainer.validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "async"),
        ("freq_steps", None),
        ("freq_steps", 0),
        ("freq_epochs", 1),
        ("freq_secs", 60),
    ],
)
def test_offline_requires_sync_positive_step_checkpoint_cadence(
    monkeypatch, tmp_path, field, value
):
    config = _config(monkeypatch, tmp_path)
    config.offline_eval_only = True
    setattr(config.saver, field, value)
    with pytest.raises(ValueError, match="Offline evaluation"):
        Tau2PPOTrainer.validate_config(config)


@pytest.mark.parametrize("steps", [20, 23])
def test_offline_saves_actor_and_critic_at_cadence_and_final(
    monkeypatch, tmp_path, steps
):
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(offline_eval_only=True, total_train_steps=steps)
    trainer.saver = Saver(
        SaverConfig(
            freq_steps=10,
            freq_epochs=None,
            freq_secs=None,
            mode="sync",
            fileroot=str(tmp_path),
        ),
        FinetuneSpec(total_train_epochs=1, dataset_size=steps, train_batch_size=1),
    )
    trainer.actor, trainer.critic = Mock(), Mock()
    trainer.tokenizer = trainer.processor = None
    trainer._save_recover_checkpoint = Mock()
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)
    for step in range(steps):
        trainer._save_training_state(epoch=0, epoch_step=step, global_step=step)
    expected = sorted({10, 20, steps})
    for engine, name in [(trainer.actor, "default"), (trainer.critic, "critic")]:
        metas = [call.args[0] for call in engine.save.call_args_list]
        assert [
            int(meta.path.rsplit("globalstep", 1)[1]) + 1 for meta in metas
        ] == expected
        assert all(
            f"/{name}/" in meta.path
            and meta.weight_format == "hf"
            and not meta.with_optim
            for meta in metas
        )
    assert trainer._save_recover_checkpoint.call_args.kwargs["force"] is True


def test_gpu_evidence_records_disabled_online_eval(tmp_path):
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(
        offline_eval_only=True, cluster=SimpleNamespace(fileroot=str(tmp_path))
    )
    trainer.scheduler = SimpleNamespace(gpu_devices=list(range(8)), _workers={})
    trainer._write_gpu_allocation()
    evidence = json.loads((tmp_path / "evidence/gpu-allocation.json").read_text())
    assert evidence["online_eval_enabled"] is False
    assert evidence["offline_eval_only"] is True
    assert "dedicated-eval" not in evidence["roles"]


def test_offline_full_init_has_no_eval_controller_or_executor(monkeypatch, tmp_path):
    from areal.trainer.rl_trainer import PPOTrainer

    config = _config(monkeypatch, tmp_path)
    config.offline_eval_only = True
    trainer = object.__new__(Tau2PPOTrainer)
    sampler = SimpleNamespace(seed=None)
    trainer.config = config
    trainer.scheduler = SimpleNamespace(gpu_devices=list(range(7)), _workers={})

    def init(self, config, train_dataset, valid_dataset):
        self.recover_info = None
        self._online_mode = False
        self.train_dataloader = SimpleNamespace(sampler=sampler)
        self.rollout = SimpleNamespace(prepare_batch=Mock())
        self.valid_dataloader = valid_dataset

    monkeypatch.setattr(PPOTrainer, "_init_impl", init)
    factory = Mock(side_effect=AssertionError("must not create eval controller"))
    monkeypatch.setattr(
        "areal.trainer.async_eval.RemoteSGLangEngine.as_controller", factory
    )
    trainer._init_impl(
        config, train_dataset=[], valid_dataset=[{"task_id": "held-out"}]
    )
    assert trainer._async_eval_rollout is None
    assert trainer._async_eval_executor is None
    assert trainer._async_eval_futures == []
    assert trainer.valid_dataloader == [{"task_id": "held-out"}]
    assert sampler.seed == config.seed
    factory.assert_not_called()


@pytest.mark.parametrize("recovered", [False, True])
def test_offline_train_skips_initial_and_recovery_eval(monkeypatch, recovered):
    from areal.trainer.rl_trainer import PPOTrainer

    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(
        offline_eval_only=True, evaluator=SimpleNamespace(eval_before_train=True)
    )
    trainer.recover_info = object() if recovered else None
    trainer.valid_dataloader = ["held-out"]
    trainer.evaluator = Mock()
    trainer._enqueue_eval = Mock(side_effect=AssertionError("must not enqueue"))
    trainer._catch_up_recovered_eval = Mock(
        side_effect=AssertionError("must not catch up")
    )
    parent_train = Mock()
    monkeypatch.setattr(PPOTrainer, "train", parent_train)
    trainer.train(
        workflow="train-workflow",
        eval_workflow="eval-workflow",
        eval_workflow_kwargs={},
    )
    assert parent_train.call_args.kwargs["eval_workflow"] is None
    trainer.evaluator.freq_ctl.check.assert_not_called()
    trainer._enqueue_eval.assert_not_called()
    trainer._catch_up_recovered_eval.assert_not_called()


def test_offline_summary_records_empty_eval_schedule(monkeypatch, tmp_path):
    from examples.tau2.ppo_data import ppo_config_summary

    config = _config(monkeypatch, tmp_path)
    config.offline_eval_only = True
    config.total_train_steps = 23
    config.evaluator.freq_steps = None
    summary = ppo_config_summary(config, [], [])
    assert summary["eval_steps"] == []
    assert summary["eval_before_train"] is False
    assert summary["eval_every_global_steps"] is None
    assert summary["saver_every_global_steps"] == config.saver.freq_steps
