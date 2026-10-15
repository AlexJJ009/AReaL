# SPDX-License-Identifier: Apache-2.0
"""CPU regression for actor/critic residency across real trainer iterations."""

from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.test_ppo_critic_warmup import _make_trainer, _TrainEngine, _Workflow


class _ResidencyEngine(_TrainEngine):
    """Model the engine's persistent compute onload and temporary IO onload."""

    def __init__(self, role: str, resident: set[str], events: list, step: list[int]):
        super().__init__(initial=1.0)
        self.role = role
        self.resident = resident
        self.events = events
        self.step = step
        self.requires_resident_transfer = False
        self.config = SimpleNamespace(backend="fsdp")

    def _record(self, operation: str) -> None:
        self.events.append((self.step[0], self.role, operation, sorted(self.resident)))

    def onload(self) -> None:
        self.resident.add(self.role)
        self._record("onload")

    def offload(self) -> None:
        self.resident.discard(self.role)
        self._record("offload")

    def _release_unused_cuda_cache(self) -> None:
        self._record("release_unused_cuda_cache")

    def _compute(self, operation: str) -> None:
        # Like FSDPEngine._ensure_ready(), compute leaves the engine resident.
        if self.role not in self.resident:
            self.onload()
        self._record(operation)

    @contextmanager
    def _temporary_onload(self):
        # save/update_weights/export_stats restore residency on exit, but only
        # when they entered offloaded (_offload_aware_context semantics).
        was_offloaded = self.role not in self.resident
        if was_offloaded:
            self.onload()
        try:
            yield
        finally:
            if was_offloaded:
                self.offload()

    def prepare_batch(self, *args, **kwargs) -> list[dict]:
        self.step[0] += 1
        return [
            {
                "loss_scale": torch.ones(1),
                "terminated": True,
                "truncated": False,
            }
        ]

    def compute_values(self, batch: list[dict]) -> list[torch.Tensor]:
        self._compute("compute_values")
        return super().compute_values(batch)

    def compute_logp(self, batch: list[dict]) -> list[torch.Tensor]:
        self._compute("compute_logp")
        return [torch.zeros(1) for _ in batch]

    def compute_advantages(self, batch: list[dict]) -> list[dict]:
        # Advantage construction is tensor-only: SAO calls this while the
        # critic is resident, before its before_actor_update handoff.
        self._record("compute_advantages")
        result = deepcopy(batch)
        for row in result:
            row["returns"] = torch.ones(1)
        return result

    def ppo_update(self, batch: list[dict]) -> dict:
        self._compute("ppo_update")
        super().ppo_update(batch)
        return {
            "attempted": 1.0,
            "successful": 1.0,
            "skipped": 0.0,
            "effective": 1.0,
        }

    def update_weights(self, meta) -> None:
        if self.requires_resident_transfer:
            # AWEX bypasses _offload_aware_context for its weight transfer.
            assert self.role in self.resident
        with self._temporary_onload():
            self._record("update_weights")
            super().update_weights(meta)
        if self.requires_resident_transfer:
            # AWEX releases actor memory at the end of the transfer.
            self.offload()

    def save(self) -> None:
        with self._temporary_onload():
            self._record("save")

    def export_stats(self) -> dict:
        with self._temporary_onload():
            self._record("export_stats")
            return super().export_stats()


@pytest.mark.parametrize(
    ("mode", "actor_offload"),
    [
        (mode, offload)
        for mode in ("ppo", "sao", "warmup", "grpo")
        for offload in (False, True)
    ]
    + [("awex_ppo", False), ("awex_sao", False)],
)
def test_train_two_steps_with_offload_keeps_one_role_resident(
    monkeypatch, actor_offload: bool, mode: str
):
    # Use the production train(), initial offload policy, finalization window,
    # saves, stats export and (for SAO) update_critic_before_actor(). Only engine
    # work is fake; no GPU, workers, rollout servers or external APIs are used.
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)
    trainer = _make_trainer(
        total_steps=2,
        num_critic_only_steps=1 if mode == "warmup" else 0,
        critic_updates_before_actor=2 if mode in ("sao", "awex_sao") else 0,
    )
    events = []
    step = [0]
    resident = {"actor"} if mode == "grpo" else {"actor", "critic"}
    trainer.actor = _ResidencyEngine("actor", resident, events, step)
    trainer.critic = (
        None if mode == "grpo" else _ResidencyEngine("critic", resident, events, step)
    )
    trainer.config.actor.should_compute_prox_logp = lambda: True
    trainer._should_offload_actor = actor_offload
    trainer._should_offload_critic = trainer.critic is not None
    if mode.startswith("awex"):
        trainer.config.actor.weight_update_mode = "awex"
        trainer.rollout.pause_generation_sync = lambda: None
        trainer.rollout.offload = lambda tags: None
        trainer.actor.requires_resident_transfer = True

    def save_roles(**kwargs) -> None:
        trainer.actor.save()
        if trainer.critic is not None:
            trainer.critic.save()

    trainer._save_hf = save_roles
    trainer._apply_initial_offload_policy()

    trainer.train(workflow=_Workflow())

    # Check after both complete iterations, so RED captures the step-2 failure
    # even when step 1 already overlaps. Check transitions and IO as well as
    # compute: a transient double onload also violates the memory contract.
    assert step[0] == 2
    assert len(trainer.stats_logger.commits) == 2
    violations = [event for event in events if len(event[3]) > 1]
    assert not violations, f"Simultaneously resident roles: {violations}"
    for step_id in (1, 2):
        compute = [event[1:3] for event in events if event[0] == step_id]
        assert ("actor", "compute_logp") in compute
        logp_index = compute.index(("actor", "compute_logp"))
        advantage_index = compute.index(("actor", "compute_advantages"))
        assert logp_index < advantage_index
        if mode == "grpo":
            assert ("actor", "release_unused_cuda_cache") not in compute
        else:
            # PPO must clear logp temporaries before its actor update. SAO and
            # warmup hand the GPU to the critic next, so clear before onload.
            next_phase = (
                ("critic", "onload")
                if trainer.config.critic_updates_before_actor
                or (mode == "warmup" and step_id == 1)
                else ("actor", "ppo_update")
            )
            next_phase_index = compute.index(next_phase, advantage_index + 1)
            assert ("actor", "release_unused_cuda_cache") in compute[
                advantage_index + 1 : next_phase_index
            ], f"Actor logp cache retained into {next_phase}: {compute}"
        assert compute.count(("actor", "ppo_update")) == int(
            mode != "warmup" or step_id == 2
        )
        assert compute.count(("critic", "ppo_update")) == (
            0
            if mode == "grpo"
            else 2
            if trainer.config.critic_updates_before_actor
            else 1
        )
        assert compute.count(("critic", "compute_values")) == (
            0
            if mode == "grpo"
            else 2
            if trainer.config.critic_updates_before_actor
            else 1
        )
        if trainer.config.critic_updates_before_actor:
            assert compute.index(("critic", "ppo_update")) < compute.index(
                ("actor", "ppo_update")
            )
        elif mode in ("ppo", "awex_ppo"):
            assert compute.index(("actor", "ppo_update")) < compute.index(
                ("critic", "ppo_update")
            )
    assert trainer.actor.weight_update_versions == ([1] if mode == "warmup" else [1, 2])
