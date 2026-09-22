# SPDX-License-Identifier: Apache-2.0
"""Recovery tests for PPO critic-only warmup trainer state."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from areal.api.cli_args import RecoverConfig
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.utils.recover import RecoverHandler, RecoverInfo
from areal.utils.saver import Saver
from areal.utils.timeutil import EpochStepTimeFreqCtl


class _Stateful:
    def __init__(self, state: dict | None = None):
        self.state = state or {}
        self.loaded = None

    def state_dict(self) -> dict:
        return self.state

    def load_state_dict(self, state: dict) -> None:
        self.loaded = state


class _DataLoader(_Stateful):
    pass


class _FakeEngine:
    def __init__(self, initial: float | None = None):
        self.loaded_paths = []
        self.connected = []
        self.updated_versions = []
        self.versions = []
        self.param = None
        self.optimizer = None
        self.scheduler = None
        if initial is not None:
            self.param = torch.nn.Parameter(
                torch.tensor([initial], dtype=torch.float32)
            )
            self.optimizer = torch.optim.AdamW([self.param], lr=0.1, weight_decay=0.01)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=0.5
            )

    def load(self, meta) -> None:
        self.loaded_paths.append(meta.path)
        state_path = Path(meta.path) / "state.pt"
        if state_path.exists():
            assert self.param is not None
            assert self.optimizer is not None
            assert self.scheduler is not None
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.param.data.copy_(state["param"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])

    def save(self, meta) -> None:
        Path(meta.path).mkdir(parents=True, exist_ok=True)
        if self.param is not None:
            torch.save(self.state_dict(), Path(meta.path) / "state.pt")

    def connect_engine(self, inference_engine, meta: WeightUpdateMeta) -> None:
        self.connected.append((inference_engine, meta.version))

    def update_weights(self, meta: WeightUpdateMeta) -> None:
        self.updated_versions.append(meta.version)

    def set_version(self, version: int) -> None:
        self.versions.append(version)

    def train_step(self) -> None:
        assert self.param is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        self.optimizer.zero_grad()
        self.param.sum().backward()
        self.optimizer.step()
        self.scheduler.step()

    def state_dict(self) -> dict:
        assert self.param is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        return {
            "param": self.param.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }


class _FakeInferenceEngine:
    def __init__(self):
        self.pause_calls = 0
        self.resume_calls = 0
        self.versions = []

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def set_version(self, version: int) -> None:
        self.versions.append(version)


def _handler(tmp_path: Path, *, mode: str = "on") -> RecoverHandler:
    config = RecoverConfig(
        experiment_name="exp",
        trial_name="trial",
        fileroot=str(tmp_path),
        mode=mode,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=4,
        train_batch_size=1,
    )
    return RecoverHandler(config, ft_spec)


def _info(
    *,
    global_step: int = 4,
    trainer_state: dict | None = None,
) -> RecoverInfo:
    return RecoverInfo(
        last_step_info=StepInfo(
            epoch=0,
            epoch_step=global_step,
            global_step=global_step,
            steps_per_epoch=8,
        ),
        saver_info={"saver": "state"},
        evaluator_info={"evaluator": "state"},
        stats_logger_info={"stats": "state"},
        dataloader_info={"loader": "state"},
        checkpoint_info=EpochStepTimeFreqCtl(
            freq_epoch=None,
            freq_step=None,
            freq_sec=None,
        ).state_dict(),
        trainer_state=trainer_state or {},
    )


def _write_recover_tree(
    tmp_path: Path,
    *,
    trainer_state: dict | None = None,
    global_step: int = 4,
    roles: tuple[str, ...] = ("default", "critic"),
    engine_states: dict[str, dict] | None = None,
) -> RecoverHandler:
    handler = _handler(tmp_path)
    recover_info_path = Path(
        RecoverHandler.recover_info_path("exp", "trial", str(tmp_path))
    )
    _info(global_step=global_step, trainer_state=trainer_state).dump(
        str(recover_info_path)
    )
    for role in roles:
        checkpoint_path = Path(
            Saver.get_recover_checkpoint_path("exp", "trial", str(tmp_path), name=role)
        )
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        if engine_states is not None and role in engine_states:
            torch.save(engine_states[role], checkpoint_path / "state.pt")
    return handler


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


def test_recover_info_roundtrips_trainer_state(tmp_path):
    source = _info(
        global_step=6,
        trainer_state={"policy_version": 3, "num_critic_only_steps": 4},
    )

    source.dump(str(tmp_path))
    loaded = RecoverInfo.load(str(tmp_path))

    assert loaded.last_step_info.global_step == 6
    assert loaded.trainer_state == {
        "policy_version": 3,
        "num_critic_only_steps": 4,
    }


def test_recover_info_loads_legacy_checkpoint_without_trainer_state(tmp_path):
    source = _info(global_step=2, trainer_state={})
    source.dump(str(tmp_path))
    (tmp_path / "trainer_state.json").unlink()

    loaded = RecoverInfo.load(str(tmp_path))

    assert loaded.trainer_state == {}


def test_recover_load_rejects_warmup_config_mismatch(tmp_path):
    handler = _write_recover_tree(
        tmp_path,
        trainer_state={"policy_version": 3, "num_critic_only_steps": 2},
        global_step=4,
    )

    with pytest.raises(ValueError, match="num_critic_only_steps"):
        handler.load(
            {"default": _FakeEngine(), "critic": _FakeEngine()},
            _Stateful(),
            _Stateful(),
            _Stateful(),
            _DataLoader(),
            expected_trainer_state={"num_critic_only_steps": 3},
        )


def test_recover_load_pushes_saved_policy_version_to_actor_critic_and_rollout(tmp_path):
    handler = _write_recover_tree(
        tmp_path,
        trainer_state={"policy_version": 3, "num_critic_only_steps": 2},
        global_step=4,
    )
    actor = _FakeEngine()
    critic = _FakeEngine()
    rollout = _FakeInferenceEngine()
    weight_update_meta = WeightUpdateMeta(type="disk", path=str(tmp_path / "weights"))
    saver = _Stateful()
    evaluator = _Stateful()
    stats_logger = _Stateful()
    dataloader = _DataLoader()

    recover_info = handler.load(
        {"default": actor, "critic": critic},
        saver,
        evaluator,
        stats_logger,
        dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
        expected_trainer_state={"num_critic_only_steps": 2},
    )

    assert recover_info is not None
    assert len(actor.loaded_paths) == 1
    assert len(critic.loaded_paths) == 1
    assert actor.connected == [(rollout, 3)]
    assert actor.updated_versions == [3]
    assert actor.versions == [3]
    assert critic.versions == [3]
    assert rollout.versions == [3]
    assert rollout.pause_calls == 1
    assert rollout.resume_calls == 1
    assert saver.loaded == {"saver": "state"}
    assert evaluator.loaded == {"evaluator": "state"}
    assert stats_logger.loaded == {"stats": "state"}
    assert dataloader.loaded == {"loader": "state"}


def test_recover_load_restores_real_torch_state_for_actor_and_critic(tmp_path):
    source_actor = _FakeEngine(initial=1.0)
    source_critic = _FakeEngine(initial=2.0)
    for _ in range(2):
        source_actor.train_step()
    for _ in range(5):
        source_critic.train_step()
    handler = _write_recover_tree(
        tmp_path,
        trainer_state={"policy_version": 3, "num_critic_only_steps": 2},
        global_step=4,
        engine_states={
            "default": source_actor.state_dict(),
            "critic": source_critic.state_dict(),
        },
    )
    actor = _FakeEngine(initial=-1.0)
    critic = _FakeEngine(initial=-2.0)
    rollout = _FakeInferenceEngine()

    handler.load(
        {"default": actor, "critic": critic},
        _Stateful(),
        _Stateful(),
        _Stateful(),
        _DataLoader(),
        inference_engine=rollout,
        weight_update_meta=WeightUpdateMeta(
            type="disk", path=str(tmp_path / "weights")
        ),
        expected_trainer_state={"num_critic_only_steps": 2},
    )

    _assert_state_equal(actor.state_dict(), source_actor.state_dict())
    _assert_state_equal(critic.state_dict(), source_critic.state_dict())
    assert actor.updated_versions == [3]
    assert actor.versions == [3]
    assert critic.versions == [3]
    assert rollout.versions == [3]


@pytest.mark.parametrize(
    ("trainer_state", "expected_state"),
    [
        (
            {"policy_version": 0, "num_critic_only_steps": 10},
            {"num_critic_only_steps": 10},
        ),
        ({}, {"num_critic_only_steps": 0}),
    ],
)
def test_recover_missing_critic_checkpoint_fails_loudly(
    tmp_path, trainer_state, expected_state
):
    handler = _write_recover_tree(
        tmp_path,
        trainer_state=trainer_state,
        global_step=4,
        roles=("default",),
    )

    with pytest.raises(RuntimeError, match="partially restored"):
        handler.load(
            {"default": _FakeEngine(), "critic": _FakeEngine()},
            _Stateful(),
            _Stateful(),
            _Stateful(),
            _DataLoader(),
            expected_trainer_state=expected_state,
        )


def test_warmup_recovery_metadata_missing_refuses_fresh_start(tmp_path):
    handler = _handler(tmp_path)
    recover_info_path = Path(
        RecoverHandler.recover_info_path("exp", "trial", str(tmp_path))
    )
    recover_info_path.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Invalid critic-only recovery metadata"):
        handler.load(
            {"default": _FakeEngine()},
            _Stateful(),
            _Stateful(),
            _Stateful(),
            _DataLoader(),
            expected_trainer_state={"num_critic_only_steps": 10},
        )


def test_recover_load_treats_legacy_missing_trainer_state_as_zero_warmup(tmp_path):
    handler = _write_recover_tree(tmp_path, trainer_state={}, global_step=1)
    recover_info_path = Path(
        RecoverHandler.recover_info_path("exp", "trial", str(tmp_path))
    )
    (recover_info_path / "trainer_state.json").unlink()
    actor = _FakeEngine()
    rollout = _FakeInferenceEngine()

    handler.load(
        {"default": actor},
        _Stateful(),
        _Stateful(),
        _Stateful(),
        _DataLoader(),
        inference_engine=rollout,
        weight_update_meta=WeightUpdateMeta(
            type="disk", path=str(tmp_path / "weights")
        ),
        expected_trainer_state={"num_critic_only_steps": 0},
    )

    assert actor.updated_versions == [2]
    assert actor.versions == [2]
    assert rollout.versions == [2]
