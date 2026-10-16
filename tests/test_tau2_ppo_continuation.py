# SPDX-License-Identifier: Apache-2.0
"""CPU regression for a completed finite-epoch PPO recovery."""

import random
from types import SimpleNamespace

import pytest
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from examples.tau2.contracts import build_sampling_schedule
from examples.tau2.evaluation import Tau2AsyncEvalTrainer
from tests.test_rollout_recovery_inputs import _FiniteRollout

from areal.api import StepInfo
from areal.trainer.async_eval import AsyncEvalPPOTrainer
from areal.trainer.rollout_batch import FiniteEpochBatcher


@pytest.mark.parametrize("prefetch_eof", [False, True])
def test_completed_epoch_recovery_matches_scratch_steps_48_through_119(
    monkeypatch, prefetch_eof
):
    """Use the actual sampler, loader state and finite dispatcher across EOF."""
    rows = [{"domain": "airline", "id": i} for i in range(375)] + [
        {"domain": "retail", "id": i} for i in range(375, 747)
    ]
    rows = build_sampling_schedule(
        rows,
        domain_effective_episodes={"airline": 375, "retail": 372},
        algorithm="ppo",
        seed=42,
    )
    random.Random(42).shuffle(rows)

    def loader():
        result = StatefulDataLoader(
            rows,
            batch_size=128,
            sampler=DistributedSampler(rows, 1, 0, shuffle=True),
            collate_fn=list,
        )
        return result

    scratch_loader = loader()
    scratch_loader.sampler.seed = 42
    scratch = FiniteEpochBatcher(_FiniteRollout([]))
    expected = [scratch(scratch_loader) for _ in range(120)]

    original_loader = loader()
    original_loader.sampler.seed = 42
    original = FiniteEpochBatcher(_FiniteRollout([]))
    assert [original(original_loader) for _ in range(48)] == expected[:48]
    # Cover both torchdata cursor forms at a completed epoch boundary.
    original_loader.sampler.set_epoch(7)
    iterator = iter(original_loader)
    for _ in range(6):
        next(iterator)
    if prefetch_eof:
        with pytest.raises(StopIteration):
            next(iterator)
    saved = original_loader.state_dict()
    assert saved["_iterator_finished"] is prefetch_eof

    restored_loader = loader()
    restored_loader.load_state_dict(saved)
    trainer = object.__new__(Tau2AsyncEvalTrainer)
    trainer.train_dataloader = restored_loader
    trainer.rollout = _FiniteRollout([])
    trainer.recover_info = SimpleNamespace(
        last_step_info=StepInfo(
            epoch=7, epoch_step=5, global_step=47, steps_per_epoch=6
        ),
        rollout_input_state={"outstanding": [], "buffer": []},
    )
    config = SimpleNamespace(seed=42)
    trainer.config = config
    monkeypatch.setattr(AsyncEvalPPOTrainer, "_init_impl", lambda *a, **kw: None)
    monkeypatch.setattr(Tau2AsyncEvalTrainer, "validate_config", lambda *a: None)
    # Keep the real init/wrapper and replace trajectory-only DP dispatch.
    monkeypatch.setattr(
        trainer, "_prepare_training_batch", lambda prepare, *a, **kw: prepare(*a, **kw)
    )
    trainer._init_impl(config)
    resumed = [trainer.rollout.prepare_batch(restored_loader) for _ in range(72)]

    assert resumed == expected[48:120]
    assert resumed[0] != expected[0]
    assert resumed[0] != expected[42]
