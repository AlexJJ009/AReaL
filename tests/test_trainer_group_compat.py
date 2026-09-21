from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.api.cli_args import GenerationHyperparameters
from areal.trainer import rl_trainer


@pytest.mark.parametrize(
    "options",
    [
        {"reward_normalization_use_std": False},
        {"keep_partial_group_on_error": True, "drop_incomplete_group": True},
    ],
)
def test_group_config_conflicting_options_raise(options):
    with pytest.raises(ValueError):
        GenerationHyperparameters(**options).validate_group_compatibility()


@pytest.mark.parametrize("version,single", [("v2", True), ("v1", False)])
@pytest.mark.parametrize("entry", ["train", "eval"])
def test_group_compat_unsupported_execution_raises(monkeypatch, version, single, entry):
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: single)
    config = GenerationHyperparameters(keep_partial_group_on_error=True)
    trainer = rl_trainer.PPOTrainer.__new__(rl_trainer.PPOTrainer)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(_version=version),
        gconfig=config,
        eval_gconfig=config,
    )
    with pytest.raises(ValueError, match="single-controller v1"):
        if entry == "train":
            trainer.train(None)
        else:
            trainer._evaluate_fn(None, None)


@pytest.mark.parametrize("keep_partial", [False, True])
def test_eval_requires_full_group_independent_of_retention(monkeypatch, keep_partial):
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    trainer = rl_trainer.PPOTrainer.__new__(rl_trainer.PPOTrainer)
    trainer.config = SimpleNamespace(
        rollout=SimpleNamespace(_version="v1"),
        eval_gconfig=GenerationHyperparameters(
            n_samples=2, keep_partial_group_on_error=keep_partial
        ),
    )
    trainer.actor, trainer.eval_rollout = Mock(), Mock()
    trainer.valid_dataloader = [[{"prompt": "test"}]]
    trainer._evaluate_fn(None, None)
    kwargs = trainer.eval_rollout.submit.call_args.kwargs
    assert kwargs.get("keep_partial_group_on_error", False) is keep_partial
    assert kwargs.get("legacy_reward_normalization", False) is False
    assert kwargs["min_usable_group_size"] == 2


@pytest.mark.parametrize("method", ["prepare_batch", "rollout_batch"])
def test_train_controller_default_group_options_preserve_call_contract(method):
    from areal.infra.controller.train_controller import TrainController

    controller = TrainController.__new__(TrainController)
    rollout = Mock()
    controller.rollout = rollout
    getattr(controller, method)([], None, {})
    kwargs = getattr(rollout, method).call_args.kwargs
    assert not {
        "keep_partial_group_on_error",
        "legacy_reward_normalization",
        "reward_normalization_use_std",
    }.intersection(kwargs)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("method", ["submit", "prepare_batch", "rollout_batch"])
def test_v1_wrappers_forward_group_compatibility_options(backend, method):
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.engine.vllm_remote import RemotevLLMEngine

    cls = RemoteSGLangEngine if backend == "sglang" else RemotevLLMEngine
    engine = cls.__new__(cls)
    engine._engine = Mock()
    options = dict(
        keep_partial_group_on_error=True,
        legacy_reward_normalization=True,
        reward_normalization_use_std=False,
    )
    getattr(engine, method)({} if method == "submit" else [], None, **options)
    kwargs = getattr(engine._engine, method).call_args.kwargs
    assert {key: kwargs[key] for key in options} == options
