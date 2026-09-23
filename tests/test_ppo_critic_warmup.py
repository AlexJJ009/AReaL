# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for PPO critic-only warmup trainer semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import PPOConfig
from areal.api.io_struct import WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.trainer.rl_trainer import PPOTrainer


class _Workflow(RolloutWorkflow):
    async def arun_episode(self, _engine, data):
        return data


class _DeviceStats:
    def log(self, _message: str) -> None:
        pass


class _StatefulListDataLoader:
    batch_size = 1

    def __init__(self, steps: int):
        self.steps = steps
        self.loaded_state = None

    def __len__(self) -> int:
        return self.steps

    def state_dict(self) -> dict:
        return {"cursor": 0}

    def load_state_dict(self, state: dict) -> None:
        self.loaded_state = state


class _ActorConfig:
    use_direct_dis_loss = False
    _version = "v1"
    weight_update_mode = "disk"

    def should_compute_prox_logp(self) -> bool:
        return False


class _StatsLogger:
    def __init__(self):
        self.commits = []

    def commit(self, epoch: int, step: int, global_step: int, data: dict) -> None:
        self.commits.append(
            {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "data": dict(data),
            }
        )

    def state_dict(self) -> dict:
        return {"commits": len(self.commits)}

    def close(self) -> None:
        pass


class _RecoverHandler:
    def __init__(self):
        self.dumps = []
        self.snapshots = []

    def dump(self, *args, **kwargs) -> None:
        self.dumps.append({"args": args, "kwargs": kwargs})
        engines, step_info = args[0], args[1]
        self.snapshots.append(
            SimpleNamespace(
                last_step_info=step_info,
                trainer_state=dict(kwargs["trainer_state"]),
                engines={name: engine.state_dict() for name, engine in engines.items()},
            )
        )


class _Saver:
    is_async = False

    def __init__(self):
        self.waits = 0

    def maybe_wait_for_staging(self) -> None:
        self.waits += 1

    def state_dict(self) -> dict:
        return {"waits": self.waits}

    def finalize(self) -> None:
        pass


class _Evaluator:
    def state_dict(self) -> dict:
        return {}


class _StalenessManager:
    def __init__(self):
        self.consumed_without_update = 0
        self.recovered_versions = []

    def on_batch_consumed_without_update(self) -> None:
        self.consumed_without_update += 1

    def on_version_recovered(self, version: int) -> None:
        self.recovered_versions.append(version)


class _Rollout:
    def on_batch_consumed_without_update(self):
        self.staleness_manager.on_batch_consumed_without_update()

    def __init__(self):
        self.pause_calls = 0
        self.resume_calls = 0
        self.versions = []
        self.staleness_manager = _StalenessManager()

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def set_version(self, version: int) -> None:
        self.versions.append(version)

    def get_input_recovery_state(self) -> dict:
        return {
            "outstanding": [{"id": "inflight"}],
            "buffer": [{"id": "prefetched"}],
        }

    def export_stats(self) -> dict:
        return {"rollout/paused": self.pause_calls}

    def save_perf_tracer(self, step: int) -> None:
        pass

    def destroy(self) -> None:
        pass


class _TrainEngine:
    def __init__(self, *, initial: float):
        self.param = torch.nn.Parameter(torch.tensor([initial], dtype=torch.float32))
        self.optimizer = torch.optim.AdamW([self.param], lr=0.1, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )
        self.ppo_update_steps = []
        self.scheduler_steps = []
        self.weight_update_versions = []
        self.versions = []
        self.clear_calls = 0

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def prepare_batch(self, *_args, **_kwargs) -> list[dict]:
        return [{"loss_scale": torch.tensor([1.0], dtype=torch.float32)}]

    def compute_advantages(self, batch: list[dict]) -> list[dict]:
        return batch

    def compute_values(self, batch: list[dict]) -> list[torch.Tensor]:
        return [torch.zeros(1, dtype=torch.float32) for _ in batch]

    def ppo_update(self, batch: list[dict]) -> None:
        step_id = len(self.ppo_update_steps)
        self.ppo_update_steps.append(step_id)
        self.optimizer.zero_grad()
        loss = self.param.mul(batch[0]["loss_scale"]).sum()
        loss.backward()
        self.optimizer.step()

    def step_lr_scheduler(self) -> None:
        self.scheduler_steps.append(len(self.scheduler_steps))
        self.scheduler.step()

    def update_weights(self, meta: WeightUpdateMeta) -> None:
        self.weight_update_versions.append(meta.version)

    def set_version(self, version: int) -> None:
        self.versions.append(version)

    def get_device_stats(self) -> _DeviceStats:
        return _DeviceStats()

    def clear_batches(self, *_args) -> None:
        self.clear_calls += 1

    def export_stats(self) -> dict:
        return {"param": float(self.param.detach()[0])}

    def save_perf_tracer(self, step: int) -> None:
        pass

    def destroy(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {
            "param": self.param.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load(self, state: dict) -> None:
        self.param.data.copy_(state["param"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


def _make_trainer(
    *,
    total_steps: int,
    num_critic_only_steps: int,
    critic_updates_before_actor: int = 0,
) -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        critic_updates_before_actor=critic_updates_before_actor,
        total_train_epochs=1,
        total_train_steps=total_steps,
        num_critic_only_steps=num_critic_only_steps,
        gconfig=SimpleNamespace(
            n_samples=1,
            reward_normalization=False,
            drop_incomplete_group=False,
        ),
        actor=_ActorConfig(),
        ref=None,
        teacher=None,
        dynamic_bs=False,
        memory_profiler=None,
    )
    trainer.recover_info = None
    trainer.train_dataloader = _StatefulListDataLoader(total_steps)
    trainer.actor = _TrainEngine(initial=1.0)
    trainer.critic = _TrainEngine(initial=2.0)
    trainer.ref = None
    trainer.teacher = None
    trainer.rollout = _Rollout()
    trainer.eval_rollout = None
    trainer.data_controller = None
    trainer.saver = _Saver()
    trainer.evaluator = _Evaluator()
    trainer.stats_logger = _StatsLogger()
    trainer.recover_handler = _RecoverHandler()
    trainer.weight_update_meta = WeightUpdateMeta(
        type="disk", path="/tmp/weight_update"
    )
    trainer.tokenizer = None
    trainer.processor = None
    trainer._should_offload_rollout = False
    trainer._should_offload_actor = False
    trainer._should_offload_critic = False
    trainer._should_offload_ref = False
    trainer._should_offload_teacher = False
    trainer._proxy_started = False
    trainer._save_hf_calls = []
    trainer._eval_calls = []

    def _save_hf(
        *, epoch: int, epoch_step: int, global_step: int, force: bool = False
    ) -> None:
        trainer._save_hf_calls.append((epoch, epoch_step, global_step, force))

    def _evaluate(**kwargs) -> None:
        trainer._eval_calls.append(kwargs)

    trainer._save_hf = _save_hf
    trainer._evaluate = _evaluate
    return trainer


def _assert_state_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_state_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_state_equal(actual_item, expected_item)
        return
    assert actual == expected


def _load_snapshot(trainer: PPOTrainer, snapshot: SimpleNamespace) -> None:
    trainer.actor.load(snapshot.engines["default"])
    trainer.critic.load(snapshot.engines["critic"])
    trainer.recover_info = snapshot


def test_ppo_config_defaults_to_no_critic_only_warmup():
    config = PPOConfig()

    assert config.num_critic_only_steps == 0


def test_validate_cfg_rejects_dynamic_batching_with_critic_only_warmup():
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        critic_updates_before_actor=0,
        num_critic_only_steps=2,
        dynamic_bs=True,
        actor=SimpleNamespace(_version="v1", weight_update_mode="disk"),
        rollout=SimpleNamespace(
            _version="v1",
            scheduling_strategy=SimpleNamespace(type="separation", target=None),
        ),
    )

    with pytest.raises(ValueError, match="dynamic_bs=False"):
        trainer._validate_cfg()


def test_validate_cfg_rejects_actor_rollout_colocation_with_critic_only_warmup():
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        critic_updates_before_actor=0,
        num_critic_only_steps=2,
        dynamic_bs=False,
        actor=SimpleNamespace(
            _version="v1",
            weight_update_mode="disk",
            scheduling_strategy=SimpleNamespace(type="colocation", target="rollout"),
        ),
        rollout=SimpleNamespace(
            _version="v1",
            scheduling_strategy=SimpleNamespace(type="separation", target=None),
        ),
    )

    with pytest.raises(ValueError, match="separate actor and rollout"):
        trainer._validate_cfg()


def test_train_with_default_zero_warmup_keeps_original_policy_update_semantics():
    trainer = _make_trainer(total_steps=3, num_critic_only_steps=0)

    trainer.train(workflow=_Workflow())

    assert len(trainer.actor.ppo_update_steps) == 3
    assert len(trainer.actor.scheduler_steps) == 3
    assert trainer.actor.weight_update_versions == [1, 2, 3]
    assert trainer.actor.versions == [1, 2, 3]
    assert trainer.critic.versions == [1, 2, 3]
    assert trainer.rollout.versions == [1, 2, 3]
    assert len(trainer._eval_calls) == 3
    assert all(
        "ppo/policy_version" not in commit["data"]
        for commit in trainer.stats_logger.commits
    )


def test_train_weight_only_initial_step_uses_cumulative_versions_with_fresh_updates():
    trainer = _make_trainer(total_steps=6, num_critic_only_steps=0)

    trainer.train(workflow=_Workflow(), initial_step=2)

    assert len(trainer.actor.ppo_update_steps) == 4
    assert trainer.actor.ppo_update_steps == [0, 1, 2, 3]
    assert trainer.actor.weight_update_versions == [3, 4, 5, 6]
    assert trainer.actor.versions == [2, 3, 4, 5, 6]
    assert trainer.critic.versions == [2, 3, 4, 5, 6]
    assert trainer.rollout.versions == [2, 3, 4, 5, 6]
    assert [commit["global_step"] for commit in trainer.stats_logger.commits] == [
        2,
        3,
        4,
        5,
    ]
    assert [call[:3] for call in trainer._save_hf_calls] == [
        (0, 2, 2),
        (0, 3, 3),
        (0, 4, 4),
        (0, 5, 5),
    ]
    assert not any(call[3] for call in trainer._save_hf_calls)
    assert trainer._optimizer_steps_base == 2


def test_initial_step_policy_version_syncs_single_controller_staleness(monkeypatch):
    trainer = _make_trainer(total_steps=6, num_critic_only_steps=0)
    rollout = SimpleNamespace(
        versions=[],
        staleness_manager=_StalenessManager(),
        set_version=lambda version: rollout.versions.append(version),
    )
    trainer.rollout = rollout
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)

    trainer._apply_initial_step_policy_version(2)

    assert trainer.rollout.versions == [2]
    assert trainer.rollout.staleness_manager.recovered_versions == [2]


def test_train_rejects_initial_step_when_native_recovery_loaded():
    trainer = _make_trainer(total_steps=6, num_critic_only_steps=0)
    trainer.recover_info = SimpleNamespace(
        last_step_info=SimpleNamespace(
            next=lambda: SimpleNamespace(global_step=2),
        )
    )

    with pytest.raises(ValueError, match="weight-only starts"):
        trainer.train(workflow=_Workflow(), initial_step=2)


def test_train_native_recovery_at_terminal_step_remains_noop():
    trainer = _make_trainer(total_steps=6, num_critic_only_steps=0)
    trainer.recover_info = SimpleNamespace(
        last_step_info=SimpleNamespace(
            next=lambda: SimpleNamespace(global_step=6),
        )
    )

    trainer.train(workflow=_Workflow())

    assert trainer.actor.ppo_update_steps == []
    assert trainer.stats_logger.commits == []


def test_train_skips_actor_and_weight_updates_for_first_ten_warmup_steps():
    trainer = _make_trainer(total_steps=12, num_critic_only_steps=10)

    trainer.train(workflow=_Workflow())

    assert len(trainer.actor.ppo_update_steps) == 2
    assert len(trainer.actor.scheduler_steps) == 2
    assert trainer.actor.weight_update_versions == [1, 2]
    assert trainer.actor.versions == [1, 2]
    assert trainer.critic.versions == [1, 2]
    assert trainer.rollout.versions == [1, 2]
    assert len(trainer.critic.ppo_update_steps) == 12
    assert len(trainer.critic.scheduler_steps) == 12
    assert len(trainer._eval_calls) == 2
    assert trainer.rollout.staleness_manager.consumed_without_update == 10
    assert not torch.equal(trainer.actor.param.detach(), torch.tensor([1.0]))
    assert trainer.actor.lr > trainer.critic.lr

    committed_versions = [
        commit["data"]["ppo/policy_version"] for commit in trainer.stats_logger.commits
    ]
    assert committed_versions == [0] * 10 + [1, 2]


@pytest.mark.parametrize(
    ("total_steps", "expected_actor_updates", "expected_versions"),
    [
        (9, 0, []),
        (10, 0, []),
        (11, 1, [1]),
    ],
)
def test_train_warmup_boundary_freezes_actor_until_step_n_plus_one(
    total_steps: int,
    expected_actor_updates: int,
    expected_versions: list[int],
):
    trainer = _make_trainer(total_steps=total_steps, num_critic_only_steps=10)

    actor_initial = trainer.actor.state_dict()
    trainer.train(workflow=_Workflow())

    assert len(trainer.actor.ppo_update_steps) == expected_actor_updates
    assert trainer.actor.weight_update_versions == expected_versions
    assert trainer.actor.versions == expected_versions
    if expected_actor_updates == 0:
        _assert_state_equal(trainer.actor.state_dict(), actor_initial)
    assert len(trainer.critic.ppo_update_steps) == total_steps


def test_train_short_critic_only_run_keeps_policy_frozen_and_checkpoints_zero_version():
    trainer = _make_trainer(total_steps=3, num_critic_only_steps=10)
    actor_initial = trainer.actor.state_dict()

    trainer.train(workflow=_Workflow())

    torch.testing.assert_close(
        trainer.actor.param.detach(), torch.tensor([1.0]), rtol=0.0, atol=0.0
    )
    assert trainer.actor.lr == 0.1
    _assert_state_equal(trainer.actor.state_dict(), actor_initial)
    assert trainer.actor.weight_update_versions == []
    assert trainer.actor.versions == []
    assert trainer.rollout.versions == []
    assert len(trainer.critic.ppo_update_steps) == 3
    assert len(trainer._eval_calls) == 0

    trainer_state = trainer.recover_handler.dumps[-1]["kwargs"]["trainer_state"]
    assert trainer_state == {
        "policy_version": 0,
        "num_critic_only_steps": 10,
        "optimizer_steps_base": 0,
    }


def test_save_recover_checkpoint_keeps_plain_ppo_rollout_inputs_out_of_checkpoint():
    trainer = _make_trainer(total_steps=1, num_critic_only_steps=0)

    trainer._save_recover_checkpoint(epoch=0, epoch_step=0, global_step=0)

    dump = trainer.recover_handler.dumps[-1]["kwargs"]
    assert dump["rollout_input_state"] is None
    assert dump["trainer_state"] == {
        "policy_version": 1,
        "num_critic_only_steps": 0,
    }


def test_save_recover_checkpoint_records_sao_raw_input_replay_policy():
    trainer = _make_trainer(
        total_steps=1,
        num_critic_only_steps=0,
        critic_updates_before_actor=2,
    )

    trainer._save_recover_checkpoint(epoch=0, epoch_step=0, global_step=0)

    dump = trainer.recover_handler.dumps[-1]["kwargs"]
    assert dump["rollout_input_state"] == {
        "outstanding": [{"id": "inflight"}],
        "buffer": [{"id": "prefetched"}],
    }
    assert dump["trainer_state"] == {
        "policy_version": 1,
        "num_critic_only_steps": 0,
        "rollout_recovery_policy": "replay_raw_inputs_discard_generated_trajectories",
    }


def test_train_resume_from_warmup_checkpoint_matches_uninterrupted_state():
    uninterrupted = _make_trainer(total_steps=12, num_critic_only_steps=10)
    uninterrupted.train(workflow=_Workflow())

    first_leg = _make_trainer(total_steps=5, num_critic_only_steps=10)
    first_leg.train(workflow=_Workflow())
    snapshot = first_leg.recover_handler.snapshots[-1]
    assert snapshot.last_step_info.global_step == 4
    assert snapshot.trainer_state == {
        "policy_version": 0,
        "num_critic_only_steps": 10,
        "optimizer_steps_base": 0,
    }

    resumed = _make_trainer(total_steps=12, num_critic_only_steps=10)
    _load_snapshot(resumed, snapshot)
    resumed.train(workflow=_Workflow())

    _assert_state_equal(resumed.actor.state_dict(), uninterrupted.actor.state_dict())
    _assert_state_equal(resumed.critic.state_dict(), uninterrupted.critic.state_dict())
    assert resumed.actor.weight_update_versions == [1, 2]
    assert resumed.actor.versions == [1, 2]
    assert resumed.critic.versions == [1, 2]
    assert resumed.rollout.versions == [1, 2]
    assert len(resumed.actor.ppo_update_steps) == 2
    assert len(resumed.critic.ppo_update_steps) == 7
