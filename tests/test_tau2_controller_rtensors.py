# SPDX-License-Identifier: Apache-2.0
"""Controller regressions using real RTensors and the production RPC helpers."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from examples.tau2.evaluation import Tau2PPOTrainer
from examples.tau2.user_reliability import UserReliability, scenario_id

from areal.infra.controller.train_controller import TrainController
from areal.infra.rpc import rtensor
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.trainer.rollout_batch import FiniteEpochBatcher, pad_groups_for_dispatch


class InProcessBackend:
    """Shard store with observable reads; RTensor itself is never mocked."""

    def __init__(self):
        self.tensors = {}
        self.fetched = []

    def store(self, tensor):
        ident = f"shard-{len(self.tensors)}"
        self.tensors[ident] = tensor.clone()
        return ident

    def fetch(self, shards):
        self.fetched.extend(s.shard_id for s in shards)
        return [self.tensors[s.shard_id].clone() for s in shards]

    async def delete(self, node_addr, shard_ids):
        for ident in shard_ids:
            self.tensors.pop(ident, None)


@pytest.fixture
def backend(monkeypatch):
    storage = InProcessBackend()
    # Isolate the cache without touching any RTensor methods.
    monkeypatch.setattr(rtensor, "_backend", storage)
    monkeypatch.setattr(rtensor, "_fetch_buffer", {})
    return storage


def scenario(task="a"):
    return {"domain": "airline", "task_id": task}


def group(width=5):
    shape = (1, width)
    return {
        "input_ids": torch.arange(width).reshape(shape),
        "attention_mask": torch.ones(shape, dtype=torch.bool),
        "loss_mask": torch.ones(shape, dtype=torch.bool),
        "critic_loss_mask": torch.ones(shape, dtype=torch.bool),
        "action_origin_mask": torch.ones(shape, dtype=torch.int64),
        "rewards": torch.tensor([1.0]),
        "token_rewards": torch.ones(shape),
        "bootstrap_mask": torch.tensor([False]),
        "terminated": torch.tensor([True]),
        "truncated": torch.tensor([False]),
        "user_boundary_types": torch.tensor([[0] * (width - 1) + [3]]),
        "user_boundary_flags": torch.ones(shape, dtype=torch.bool),
        "user_boundary_detector_ids": torch.ones(shape, dtype=torch.int32),
        "user_boundary_gate_suppressed": torch.ones(shape, dtype=torch.bool),
        "user_reliability_id": torch.tensor([scenario_id(scenario())]),
        "user_reliability_weight": torch.ones(shape),
        "critic_prefix_input_ids": torch.tensor([[99, 98]]),
        "critic_prefix_lengths": torch.tensor([2]),
        "logprobs": torch.zeros(shape),
    }


def dispatch(groups, dp_size=4):
    """Controller partition -> RPC wire roundtrip -> engine ingress localize."""
    controller = object.__new__(TrainController)
    controller.train_alloc = SimpleNamespace(parallel=SimpleNamespace(dp_size=dp_size))
    args, kwargs, indices = controller._prepare_dispatch(groups)
    assert not kwargs
    shards = [RTensor.localize(deserialize_value(serialize_value(s))) for s in args[0]]
    return shards, indices


@pytest.mark.parametrize("kind", ["remote", "mixed", "local"])
@pytest.mark.parametrize("count", [1, 3, 4])
def test_tail_padding_dispatch_preserves_real_groups_and_zeroes_copies(
    backend, kind, count
):
    """No controller reads; mixed payloads retain context and every zero mask."""
    originals = [group(width) for width in range(5, 5 + count)]
    groups = deepcopy(originals)
    if kind != "local":
        groups = RTensor.remotize(groups, "in-process")
    if kind == "mixed":
        groups[0]["user_reliability_weight"] = originals[0][
            "user_reliability_weight"
        ].clone()
    physical, padding = pad_groups_for_dispatch(groups, 4)
    assert padding == (-count) % 4
    assert not backend.fetched
    assert "is_padding" not in groups[0]
    if kind != "local":
        assert all(g["input_ids"].data.is_meta for g in groups)
    shards, indices = dispatch(physical)
    assert [len(s) for s in shards] == [1] * 4
    restored = [None] * 4
    for shard, idxs in zip(shards, indices, strict=True):
        for row, idx in zip(shard, idxs, strict=True):
            restored[idx] = row
    for original, row in zip(originals, restored[:count], strict=True):
        for key, value in original.items():
            torch.testing.assert_close(row[key], value, rtol=0, atol=0)
    zero_keys = (
        "loss_mask",
        "critic_loss_mask",
        "action_origin_mask",
        "rewards",
        "token_rewards",
        "bootstrap_mask",
        "user_boundary_types",
        "user_boundary_flags",
        "user_boundary_detector_ids",
        "user_boundary_gate_suppressed",
        "user_reliability_weight",
    )
    for row in restored[count:]:
        assert row["is_padding"].all()
        for key in zero_keys:
            assert not row[key].count_nonzero(), key
            assert row[key].dtype == originals[0][key].dtype
        for key in (
            "input_ids",
            "attention_mask",
            "critic_prefix_input_ids",
            "critic_prefix_lengths",
        ):
            torch.testing.assert_close(row[key], originals[0][key], rtol=0, atol=0)


def test_m2_apply_fetches_only_ids_and_dispatches_nonunity_weights(backend):
    """Past variance produces real actor weights, then padding zeros the copy."""
    state = UserReliability([scenario(), scenario("b")], warmup_epochs=0)
    for task, xs in [("a", [0.0, 1.0]), ("b", [0.0, 0.0])]:
        for x in xs:
            state.record(1, scenario_id(scenario(task)), x)
    state.begin_epoch(2)
    groups = RTensor.remotize([group()], "in-process")
    stats = state.apply(groups)
    assert backend.fetched == [groups[0]["user_reliability_id"].shard.shard_id]
    assert groups[0]["input_ids"].data.is_meta
    assert stats["user_reliability/airline/weight_mean"] != 1
    physical, _ = pad_groups_for_dispatch(groups, 4)
    shards, _ = dispatch(physical)
    weights = [r["user_reliability_weight"] for shard in shards for r in shard]
    assert sum(bool(w.count_nonzero()) for w in weights) == 1
    torch.testing.assert_close(
        weights[0],
        torch.full((1, 5), stats["user_reliability/airline/weight_mean"]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("status", ["terminal", "truncated", "no_boundary", "padding"])
def test_m2_observe_reads_remote_fresh_critic_values_and_small_keys(backend, status):
    """Model the engine's returned list[RTensor], including excluded episodes."""
    state = UserReliability([scenario()])
    state.begin_epoch(1)
    data = group()
    data["is_padding"] = torch.full((1, 5), status == "padding", dtype=torch.bool)
    if status == "truncated":
        data["terminated"].fill_(False)
        data["truncated"].fill_(True)
    if status == "no_boundary":
        data["user_boundary_types"].zero_()
    groups = RTensor.remotize([data], "in-process")
    # engine_blueprint remotizes compute_values' bare tensor list, then the
    # trainer attaches the returned handles to the original rollout groups.
    values = deserialize_value(
        serialize_value(RTensor.remotize([torch.full((1, 5), 0.25)], "critic"))
    )
    groups[0]["values"] = values[0]
    assert isinstance(values[0], RTensor) and values[0].data.is_meta
    state.observe(groups)
    assert state.records == (
        [[1, scenario_id(scenario()), 0.75]] if status == "terminal" else []
    )
    allowed = {
        "user_reliability_id",
        "user_boundary_types",
        "rewards",
        "values",
        "terminated",
        "truncated",
        "is_padding",
    }
    assert set(backend.fetched) <= {groups[0][key].shard.shard_id for key in allowed}
    for key in set(groups[0]) - allowed:
        assert groups[0][key].data.is_meta, key


@pytest.mark.parametrize("enabled", [False, True])
def test_tau2_controller_hooks_finite_epoch_stats_and_m2(backend, tmp_path, enabled):
    """Exercise recipe entrypoints with RTensors and DP=4, with M2 on/off."""
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(
        user_reliability_weight=enabled,
        resolve_min_usable_group_size=lambda: 1,
        max_skipped_tasks_per_epoch=0,
        experiment_name="test",
        trial_name="rtensor",
        cluster=SimpleNamespace(fileroot=str(tmp_path)),
        econfig=SimpleNamespace(_quota_breaker_state_path=str(tmp_path / "quota.json")),
    )
    trainer.actor = SimpleNamespace(parallel_strategy=SimpleNamespace(dp_size=4))
    groups = RTensor.remotize([group()], "in-process")
    rollout = SimpleNamespace(
        prepare_batch=lambda *a, **kw: groups, data_generator=None
    )
    trainer.rollout = rollout
    trainer._user_reliability = UserReliability([scenario()])
    physical = trainer._prepare_training_batch(FiniteEpochBatcher(rollout))
    assert trainer._batch_counts["tau2_batch/padding_groups"] == 3
    assert trainer._batch_counts["tau2_batch/real_episodes"] == 1
    assert trainer._batch_counts["tau2_batch/real_prompts"] == 1
    for row in physical:
        row["values"] = RTensor.remotize(
            torch.full(row["input_ids"].shape, 0.25), "critic"
        )
    trainer._process_rollout_batch(physical)
    assert trainer._user_reliability.records == (
        [[1, scenario_id(scenario()), 0.75]] if enabled else []
    )
    if not enabled:
        assert not backend.fetched
    stats = trainer._additional_stats()
    assert stats["tau2_batch/real_episodes"] == 1
    assert stats["tau2_batch/padding_groups"] == 3
    assert stats["rollout/quota_errors"] == 0
    assert ("user_reliability/airline/weight_mean" in stats) is enabled
