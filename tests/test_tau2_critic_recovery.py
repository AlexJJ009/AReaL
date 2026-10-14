# SPDX-License-Identifier: Apache-2.0
"""CPU recovery round trips with real torchdata state and recording engines."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from datasets import Dataset

from scripts.tau2 import train_critic
from tests.test_ppo_warmup_recovery import (
    _assert_state_equal,
    _FakeEngine,
    _FakeInferenceEngine,
    _info,
    _Stateful,
    _write_recover_tree,
)

from areal import PPOTrainer
from areal.api.cli_args import TrainDatasetConfig, load_expr_config
from areal.api.io_struct import StepInfo, WeightUpdateMeta
from areal.trainer import rl_trainer


def _config():
    return SimpleNamespace(
        seed=42,
        total_train_steps=12,
        num_critic_only_steps=12,
        critic_updates_before_actor=0,
        critic=SimpleNamespace(),
        actor=SimpleNamespace(optimizer=SimpleNamespace(lr=0.0)),
        train_dataset=TrainDatasetConfig(
            batch_size=4, shuffle=True, drop_last=False, num_workers=0
        ),
    )


def _loader(trainer):
    return trainer._create_dataloader(
        Dataset.from_list([{"id": i} for i in range(13)]),
        trainer.config.train_dataset,
        rank=0,
        world_size=1,
    )


def _stub_init(monkeypatch, *, handler=None, recover_info=None):
    """Replace worker startup only; recovery and dataloader stay real."""

    def init(self, config, *args, skip_recovery_weight_update=False, **kwargs):
        self.config = config
        self.train_dataloader = _loader(self)
        self.actor = _FakeEngine(0.0)
        self.critic = _FakeEngine(0.0)
        self.rollout = _FakeInferenceEngine()
        self.saver, self.evaluator, self.stats_logger = (
            _Stateful(),
            _Stateful(),
            _Stateful(),
        )
        self.recover_info = recover_info
        if handler is not None:
            # Omitting the new keyword on the default path also lets this
            # fixture run against the pre-fix RecoverHandler implementation.
            options = {}
            if skip_recovery_weight_update:
                options["skip_inference_weight_update"] = True
            self.recover_info = handler.load(
                {"default": self.actor, "critic": self.critic},
                self.saver,
                self.evaluator,
                self.stats_logger,
                self.train_dataloader,
                inference_engine=self.rollout,
                weight_update_meta=WeightUpdateMeta.from_disk("exp", "trial", "unused"),
                **options,
            )
        elif recover_info is not None:
            self.train_dataloader.load_state_dict(recover_info.dataloader_info)

    monkeypatch.setattr(PPOTrainer, "_init_impl", init)


def _trainer(config=None, cls=train_critic.Tau2CriticOnlineTrainer):
    trainer = object.__new__(cls)
    trainer._init_impl(config or _config())
    return trainer


def _stub_workers(monkeypatch, handler):
    """Exercise PPOTrainer._init_impl itself, replacing external services only."""
    actor, critic = _FakeEngine(0.0), _FakeEngine(0.0)
    for engine in (actor, critic):
        engine.data_parallel_rank = 0
        engine.data_parallel_world_size = 1
        engine.initialize = lambda **kwargs: None
    rollout = _FakeInferenceEngine()
    rollout.staleness_manager = None
    for method in (
        "_apply_dte_config_envvars",
        "_validate_cfg",
        "_amend_xccl_weight_update_envvar",
        "_amend_deterministic_envvar",
        "_init_scheduler",
        "_save_initial_lora_weights",
        "_offload_model",
        "_config_perf_tracer",
        "_apply_initial_offload_policy",
    ):
        monkeypatch.setattr(PPOTrainer, method, lambda *args, **kwargs: None)
    monkeypatch.setattr(PPOTrainer, "_is_actor_rollout_colocated", lambda *args: False)
    monkeypatch.setattr(PPOTrainer, "_is_v1_awex_colocate", lambda *args: False)
    monkeypatch.setattr(PPOTrainer, "_create_train_engine", lambda *args: actor)
    monkeypatch.setattr(PPOTrainer, "_create_critic", lambda *args: critic)
    monkeypatch.setattr(PPOTrainer, "_init_rollout", lambda *args, **kwargs: rollout)
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    monkeypatch.setattr(rl_trainer.logging, "setup_file_logging", lambda *args: None)
    monkeypatch.setattr(
        rl_trainer, "load_hf_processor_and_tokenizer", lambda *args: (None, None)
    )

    class StateStore(_Stateful):
        def __init__(self, *args, **kwargs):
            super().__init__()

        @staticmethod
        def get_log_path(*args):
            return "unused"

    for name in ("Saver", "Evaluator", "StatsLogger"):
        monkeypatch.setattr(rl_trainer, name, StateStore)
    monkeypatch.setattr(rl_trainer, "RecoverHandler", lambda *args, **kwargs: handler)


@pytest.mark.parametrize("critic_only", [True, False], ids=["critic", "policy"])
def test_recovery_restores_state_and_transfers_only_for_policy(
    monkeypatch, tmp_path, critic_only
):
    """Assert update_weights itself, alongside real Adam/model/loader state."""
    with monkeypatch.context() as setup:
        _stub_init(setup)
        original = _trainer()
    original._next_strict_rows(original.train_dataloader)
    original.critic.train_step()
    critic_state = deepcopy(original.critic.state_dict())
    version = 0 if critic_only else 1
    warmup = 12 if critic_only else 0
    handler = _write_recover_tree(
        tmp_path,
        global_step=0,
        trainer_state={"num_critic_only_steps": warmup, "policy_version": version},
        optimizer_roles={"critic"} if critic_only else None,
        engine_states={
            "default": original.actor.state_dict(),
            "critic": critic_state,
        },
    )
    info = _info(
        global_step=0,
        trainer_state={"num_critic_only_steps": warmup, "policy_version": version},
    )
    if critic_only:
        info.trainer_state["_recover_optimizer_roles"] = ["critic"]
    info.dataloader_info = original.train_dataloader.state_dict()
    info.dump(handler.recover_info_path("exp", "trial", str(tmp_path)))
    _stub_workers(monkeypatch, handler)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "total_train_steps=12",
            f"num_critic_only_steps={warmup}",
            "train_dataset.batch_size=4",
            "train_dataset.shuffle=true",
            "actor.weight_update_mode=disk",
        ],
        train_critic.Tau2PPOConfig,
    )
    cls = train_critic.Tau2CriticOnlineTrainer if critic_only else PPOTrainer
    resumed = object.__new__(cls)
    resumed._init_impl(
        config, train_dataset=Dataset.from_list([{"id": i} for i in range(13)])
    )

    assert resumed.actor.updated_versions == ([] if critic_only else [version])
    assert resumed.rollout.pause_calls == (0 if critic_only else 1)
    assert resumed.rollout.resume_calls == (0 if critic_only else 1)
    assert (
        resumed.actor.versions
        == resumed.critic.versions
        == resumed.rollout.versions
        == [version]
    )
    assert resumed.actor.loaded_with_optim == [not critic_only]
    assert resumed.critic.loaded_with_optim == [True]
    _assert_state_equal(resumed.critic.state_dict(), critic_state)
    _assert_state_equal(resumed.actor.param, original.actor.param)
    _assert_state_equal(resumed.train_dataloader.next_iter_state, info.dataloader_info)
    assert resumed.saver.loaded == info.saver_info
    assert resumed.evaluator.loaded == info.evaluator_info
    assert resumed.stats_logger.loaded == info.stats_logger_info
    assert handler.freq_ctl.state_dict() == info.checkpoint_info
    assert resumed.recover_info.trainer_state == info.trainer_state
    if critic_only:
        assert resumed._tau2_next_train_step == 1
        assert resumed._next_strict_rows(
            resumed.train_dataloader
        ) == original._next_strict_rows(original.train_dataloader)


@pytest.mark.parametrize("bad_state", ["version", "optimizer_roles"])
def test_recovery_skip_rejects_unsafe_saved_state(monkeypatch, tmp_path, bad_state):
    """A caller cannot suppress transfer for a changed or trainable actor."""
    handler = _write_recover_tree(
        tmp_path,
        global_step=0,
        trainer_state={"policy_version": 1 if bad_state == "version" else 0},
        optimizer_roles={"critic"} if bad_state == "version" else None,
    )
    _stub_init(monkeypatch, handler=handler)
    with pytest.raises(
        ValueError,
        match="policy_version=0"
        if bad_state == "version"
        else "critic-only optimizer roles",
    ):
        _trainer()


@pytest.mark.parametrize(
    "bad_config", ["lr", "window", "sao", "no_critic", "no_limit", "zero_steps"]
)
def test_critic_recovery_skip_rejects_unsafe_config(monkeypatch, bad_config):
    """Reject an unsafe skip before starting workers or touching checkpoints."""
    _stub_init(monkeypatch)
    config = _config()
    if bad_config == "lr":
        config.actor.optimizer.lr = 1e-5
    elif bad_config == "window":
        config.num_critic_only_steps = 1
    elif bad_config == "sao":
        config.critic_updates_before_actor = 1
    elif bad_config == "no_critic":
        config.critic = None
    elif bad_config == "no_limit":
        config.total_train_steps = None
    else:
        config.total_train_steps = 0
    with pytest.raises(ValueError, match="frozen actor|every step to be critic-only"):
        _trainer(config)


@pytest.mark.parametrize(
    "saved_steps",
    [2, 4, 6, 8],
    ids=["mid_epoch0", "boundary1", "mid_epoch1", "boundary2"],
)
def test_resumed_sampler_matches_uninterrupted_epochs(monkeypatch, saved_steps):
    """Real torchdata restoration must preserve tail batches and later epochs."""
    _stub_init(monkeypatch)
    original = _trainer()
    for _ in range(saved_steps):
        original._next_strict_rows(original.train_dataloader)
    info = _info()
    info.last_step_info = StepInfo(
        epoch=(saved_steps - 1) // 4,
        epoch_step=(saved_steps - 1) % 4,
        global_step=saved_steps - 1,
        steps_per_epoch=4,
    )
    info.dataloader_info = deepcopy(original.train_dataloader.state_dict())
    _stub_init(monkeypatch, recover_info=info)
    resumed = _trainer()
    assert resumed._tau2_next_train_step == saved_steps
    for _ in range(12 - saved_steps):
        expected = original._next_strict_rows(original.train_dataloader)
        actual = resumed._next_strict_rows(resumed.train_dataloader)
        assert actual == expected
        assert resumed._tau2_train_epoch == original._tau2_train_epoch
