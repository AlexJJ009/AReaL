"""CPU tail dispatch oracles through production GAE, PPO losses and SGD."""

import copy
from math import gcd
from types import SimpleNamespace

import pytest
import torch

from examples.tau2.evaluation import Tau2AsyncEvalTrainer
from tests.test_sao_dual_lambda import TinyTrainEngine, make_actor

from areal.api.cli_args import MicroBatchSpec, NormConfig, PPOCriticConfig
from areal.infra.controller.train_controller import _dispatch_tensors
from areal.trainer.ppo.critic import PPOCritic
from areal.trainer.ppo.critic_input import prepare_critic_input
from areal.utils import stats_tracker
from areal.utils.data import (
    Normalization,
    concat_padded_tensors,
    drop_non_model_forward_metadata,
    pack_tensor_dict,
    split_padded_tensor_dict_into_mb_list,
)


@pytest.fixture(autouse=True)
def isolate_stats():
    stats_tracker.export_all()
    yield
    stats_tracker.export_all()


def groups(count, *, boundaries=True, prefix=True):
    result = []
    for i in range(count):
        length = 7 + i % 4
        size = 1 + i % 2
        ids = (torch.arange(length).expand(size, -1) + i) % 15
        mask = torch.ones_like(ids)
        mask[:, [0, 3]] = 0
        data = {
            "input_ids": ids.clone(),
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "loss_mask": mask,
            "logprobs": torch.full_like(ids, -2.77, dtype=torch.float32),
            "rewards": torch.arange(size).float() * 0.2 + (i + 1) / count,
            "terminated": torch.ones(size, dtype=torch.bool),
            "truncated": torch.zeros(size, dtype=torch.bool),
            "bootstrap_mask": torch.zeros(size, dtype=torch.bool),
            "episode_ids": torch.zeros_like(ids),
            "token_rewards": torch.zeros_like(ids, dtype=torch.float32),
        }
        if boundaries:
            types = torch.zeros_like(ids, dtype=torch.int32)
            types[:, 2] = 1
            types[:, -1] = 3
            data.update(
                user_boundary_types=types,
                user_boundary_flags=types != 0,
                user_boundary_detector_ids=(types != 0).to(torch.int32),
                user_boundary_gate_suppressed=torch.zeros_like(ids, dtype=torch.bool),
                user_td_beta_flagged=torch.full((size,), 0.5),
            )
        if prefix:
            data.update(
                critic_prefix_input_ids=torch.full((size, 1 + i % 3), i + 1),
                critic_prefix_lengths=torch.full((size,), 1 + i % 3),
            )
        result.append(data)
    return result


def dispatch(data):
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            resolve_min_usable_group_size=lambda: 1, max_skipped_tasks_per_epoch=0
        ),
        actor=SimpleNamespace(parallel_strategy=SimpleNamespace(dp_size=4)),
        rollout=SimpleNamespace(),
    )
    physical = Tau2AsyncEvalTrainer._prepare_training_batch(trainer, lambda **kw: data)
    return physical, trainer._batch_counts


@pytest.mark.parametrize("count", [1, 2, 5, 7, 106, 107, 128])
def test_tail_physical_groups_are_minimal(count):
    physical, counts = dispatch(groups(count))
    assert count <= len(physical) <= count + 3
    assert len(physical) == count + (-count) % 4
    assert counts["tau2_batch/padding_groups"] == (-count) % 4
    assert counts["tau2_batch/real_prompts"] == count
    assert counts["tau2_batch/real_episodes"] == sum(1 + i % 2 for i in range(count))
    assert counts["tau2_batch/physical_prompt_groups"] == len(physical)
    assert "tau2_batch/dispatch_replication_factor" not in counts
    shards, _ = _dispatch_tensors(physical, dp_size=4)
    assert len({len(shard) for shard in shards}) == 1


def test_tail_copies_have_independent_tensor_storage_and_zero_masks():
    real = groups(5)
    snapshot = copy.deepcopy(real)
    physical, _ = dispatch(real)
    copies = physical[len(real) :]
    assert copies
    for dummy in copies:
        for key, tensor in dummy.items():
            if not torch.is_tensor(tensor):
                continue
            for original in real:
                if key in original:
                    assert tensor.data_ptr() != original[key].data_ptr(), key
            for other in copies:
                if other is not dummy:
                    assert tensor.data_ptr() != other[key].data_ptr(), key
        assert not dummy["loss_mask"].any()
        assert not dummy["user_boundary_types"].any()
        assert not dummy["user_boundary_flags"].any()
        assert not dummy["token_rewards"].any()
        assert dummy["attention_mask"].any()
        dummy["input_ids"].fill_(0)
    for before, after in zip(snapshot, real):
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def run_update(data, *, unbiased, reward_norm=False, packed=False):
    actor = make_actor(1.0, normalize=False)
    actor.adv_norm = Normalization(NormConfig(std_unbiased=unbiased, eps=1e-8))
    if reward_norm:
        actor.reward_norm = Normalization(NormConfig(std_unbiased=unbiased, eps=1e-8))
    critic_engine = TinyTrainEngine(critic=True)
    critic = PPOCritic(
        PPOCriticConfig(ppo_n_minibatches=1, eps_clip=0.5), critic_engine
    )
    for row, values in zip(data, critic.compute_values(data)):
        row["values"] = values
    advantages = actor.compute_advantages(data)
    # Also exercise the production packing boundary before the loss callbacks.
    if packed:
        original_train_batch = actor.engine.train_batch
        original_critic_train_batch = critic.engine.train_batch
        actor.engine.train_batch = lambda mb, **kw: original_train_batch(
            pack_tensor_dict(mb), **kw
        )
        critic.engine.train_batch = lambda mb, **kw: original_critic_train_batch(
            pack_tensor_dict(mb), **kw
        )
    stats_tracker.export_all()
    actor.ppo_update(copy.deepcopy(advantages))
    critic.ppo_update(copy.deepcopy(advantages))
    stats = stats_tracker.export_all()
    return {
        "advantages": advantages,
        "actor_loss": actor.engine.losses[0],
        "critic_loss": critic_engine.losses[0],
        "actor_grad": actor.engine.model.weight.grad,
        "critic_grad": critic_engine.model.weight.grad,
        "actor_weights": actor.engine.model.weight.detach(),
        "critic_weights": critic_engine.model.weight.detach(),
        "stats": stats,
    }


@pytest.mark.parametrize("count", [5, 7])
@pytest.mark.parametrize("unbiased", [False, True])
@pytest.mark.parametrize("boundaries,prefix", [(False, False), (True, True)])
@pytest.mark.parametrize("packed", [False, True])
def test_tail_losses_whitening_gradients_and_updates_equal_real_batch(
    count, unbiased, boundaries, prefix, packed
):
    real = groups(count, boundaries=boundaries, prefix=prefix)
    physical, _ = dispatch(copy.deepcopy(real))
    padded = run_update(physical, unbiased=unbiased, packed=packed)
    baseline = run_update(copy.deepcopy(real), unbiased=unbiased, packed=packed)
    for key in (
        "actor_loss",
        "critic_loss",
        "actor_grad",
        "critic_grad",
        "actor_weights",
        "critic_weights",
    ):
        torch.testing.assert_close(padded[key], baseline[key], rtol=2e-6, atol=2e-7)
    for expected, actual in zip(baseline["advantages"], padded["advantages"][:count]):
        for key in ("advantages", "returns", "loss_mask"):
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-6, atol=2e-7)
    for dummy in padded["advantages"][count:]:
        assert not dummy["loss_mask"].any()
        assert not dummy.get("critic_loss_mask", dummy["loss_mask"]).any()
    replicas = 4 // gcd(count, 4)
    replicated = run_update(
        copy.deepcopy(real * replicas), unbiased=unbiased, packed=packed
    )
    if not unbiased:
        for key in ("actor_loss", "critic_loss", "actor_grad", "critic_grad"):
            torch.testing.assert_close(
                padded[key], replicated[key], rtol=2e-6, atol=2e-7
            )
    else:
        # Sample std is not replication invariant: s_r/s = sqrt(r*(N-1)/(r*N-1)).
        n = sum(
            row["loss_mask"].count_nonzero().item() for row in baseline["advantages"]
        )
        scale = ((replicas * n - 1) / (replicas * (n - 1))) ** 0.5
        for original, repeated in zip(baseline["advantages"], replicated["advantages"]):
            torch.testing.assert_close(
                repeated["advantages"],
                original["advantages"] * scale,
                rtol=2e-6,
                atol=2e-7,
            )
        assert not torch.allclose(
            baseline["actor_grad"], replicated["actor_grad"], rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            padded["critic_grad"], replicated["critic_grad"], rtol=2e-6, atol=2e-7
        )


def test_tail_reward_normalization_and_logged_statistics_exclude_padding():
    real = groups(5)
    physical, _ = dispatch(copy.deepcopy(real))
    padded = run_update(physical, unbiased=True, reward_norm=True)
    baseline = run_update(real, unbiased=True, reward_norm=True)
    torch.testing.assert_close(
        padded["actor_grad"], baseline["actor_grad"], rtol=2e-6, atol=2e-7
    )
    torch.testing.assert_close(
        padded["critic_grad"], baseline["critic_grad"], rtol=2e-6, atol=2e-7
    )
    expected, actual = baseline["stats"], padded["stats"]
    assert actual.keys() == expected.keys()
    for key in expected:
        if expected[key] is None:
            assert actual[key] is None, key
        else:
            assert actual[key] == pytest.approx(expected[key], rel=2e-6, abs=2e-7), key


def test_tail_marker_survives_critic_prefix_microbatch_and_packing():
    physical, _ = dispatch(groups(5))
    combined = concat_padded_tensors(physical)
    critic = prepare_critic_input(combined)
    # Prefix columns must inherit dummy eligibility, including when prefix widths differ.
    padding_rows = critic["is_padding"].all(-1)
    assert padding_rows.sum() == sum(row["input_ids"].shape[0] for row in physical[5:])
    assert not critic["loss_mask"][padding_rows].any()
    mbs = split_padded_tensor_dict_into_mb_list(critic, MicroBatchSpec(n_mbs=3))
    for mb in mbs.mbs:
        packed = pack_tensor_dict(mb)
        assert not packed["loss_mask"][packed["is_padding"]].any()
        forward = dict(packed)
        drop_non_model_forward_metadata(forward)
        assert "is_padding" not in forward


def _normalization_worker(rank, rendezvous):
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=12),
    )
    try:
        # A small tail can leave an entire DP rank with only padding.
        values = torch.tensor([[1.0, 3.0]])
        mask = torch.full_like(values, rank == 0, dtype=torch.bool)
        result = Normalization(NormConfig(std_unbiased=True, eps=0))(values, mask)
        expected = (
            torch.tensor([[-(2**-0.5), 2**-0.5]])
            if rank == 0
            else torch.zeros_like(values)
        )
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_tail_all_padding_rank_still_joins_whitening_collectives(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_normalization_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_tail_adaptive_lambda_accepts_only_marked_empty_rows():
    actor = make_actor(1.0)
    from areal.trainer.ppo.lambda_fn import resolve_gae_lambda_fn

    actor.gae_lambda = "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"
    actor.gae_lambda_fn, actor._gae_lambda_is_custom = resolve_gae_lambda_fn(
        actor.gae_lambda
    )
    actor.gae_lambda_kwargs = {"alpha": 1.5}
    real = groups(5, boundaries=False, prefix=False)
    physical, _ = dispatch(copy.deepcopy(real))
    reference = actor.compute_advantages(copy.deepcopy(real))
    padded = actor.compute_advantages(physical)
    for expected, actual in zip(reference, padded[:5]):
        torch.testing.assert_close(
            actual["advantages"], expected["advantages"], rtol=1e-6, atol=1e-7
        )
    for dummy in padded[5:]:
        assert not dummy["loss_mask"].any()
    real[0]["loss_mask"].zero_()
    with pytest.raises(RuntimeError, match="Empty action episode"):
        actor.compute_advantages(real)


def test_tail_scalar_counts_defer_host_sync_until_export(monkeypatch):
    """Padding diagnostics must not add a device-to-host sync per microbatch."""
    tracker = stats_tracker.get("tail_scalar_test")

    def premature_float(value):
        raise AssertionError("scalar logging synchronized before export")

    value = torch.tensor(7.0, requires_grad=True)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__float__", premature_float)
        tracker.scalar(real_tokens=value)
    with torch.no_grad():
        value.add_(10)
    assert not tracker.stats["tail_scalar_test/real_tokens"][0].requires_grad
    assert tracker.export()["tail_scalar_test/real_tokens"] == 7
