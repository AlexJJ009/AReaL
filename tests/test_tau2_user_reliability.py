# SPDX-License-Identifier: Apache-2.0
"""CPU oracles for past-only scenario reliability and actor-only weighting."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from examples.tau2.evaluation import Tau2PPOTrainer
from examples.tau2.user_reliability import UserReliability, scenario_id
from examples.tau2.utils import Tau2PPOConfig
from tests.test_tau2_user_boundaries import _interaction
from tests.test_user_td_split import actor, batch

from areal.experimental.openai.types import AgentWorkflowResult
from areal.trainer.ppo.critic import ppo_loss_fn
from areal.trainer.rollout_batch import pad_groups_for_dispatch
from areal.utils import stats_tracker


@pytest.fixture(autouse=True)
def isolated_statistics():
    stats_tracker.export_all()
    yield
    stats_tracker.export_all()


def schedule():
    return [
        {"domain": domain, "task_id": task, "task_pack": pack}
        for domain, task, pack in [
            ("airline", "a", None),
            ("airline", "b", None),
            ("airline", "c", "pack"),
            ("telecom", "a", "pack"),
            ("telecom", "b", "pack"),
        ]
    ]


def seeded():
    state = UserReliability(schedule())
    for epoch, xs in [(1, [0.0, 0.0, 0.25, 0.0, 0.0]), (2, [1.0, 0.5, None, 0.0, 0.0])]:
        for row, x in zip(schedule(), xs, strict=True):
            if x is not None:
                state.record(epoch, scenario_id(row), x)
    return state


def test_formulas_match_offline_numpy_reference_and_normalize_scheduled_scenarios():
    """Same ddof=1, n-weighted k=2 shrinkage and domain tau as analyze_noise.py."""
    state = seeded()
    table = state.begin_epoch(3)
    prior = np.mean([np.var([0.0, 1.0], ddof=1), np.var([0.0, 0.5], ddof=1)])
    sigma = np.array([(2 * 0.5 + 2 * prior) / 4, (2 * 0.125 + 2 * prior) / 4, prior])
    tau = np.mean(sigma)
    raw = tau / (tau + sigma)
    for i, row in enumerate(table[:3]):
        assert row["sigma2"] == pytest.approx(sigma[i])
        assert row["tau"] == pytest.approx(tau)
        assert row["raw_w"] == pytest.approx(raw[i])
        assert row["weight"] == pytest.approx(raw[i] / raw.mean())
    assert [r["n"] for r in table] == [2, 2, 1, 2, 2]
    assert table[2]["s2"] is None
    assert table[3]["weight"] == table[4]["weight"] == 1
    for domain in ("airline", "telecom"):
        assert np.mean(
            [r["weight"] for r in table if r["scenario"][0] == domain]
        ) == pytest.approx(1)


def test_past_only_weights_freeze_during_epoch_and_recovery_roundtrip():
    state = seeded()
    before = deepcopy(state.begin_epoch(3))
    state.record(3, scenario_id(schedule()[0]), 100.0)
    assert state.begin_epoch(3) == before
    recovered = UserReliability(schedule())
    recovered.load_state_dict(json.loads(json.dumps(state.state_dict())))
    assert recovered.state_dict() == state.state_dict()
    assert recovered.begin_epoch(3) == before
    assert recovered.begin_epoch(4) == state.begin_epoch(4)
    assert recovered.table != before


@pytest.mark.parametrize("epoch", [1, 2])
def test_warmup_keeps_unity_even_with_history(epoch):
    assert all(r["weight"] == 1 for r in seeded().begin_epoch(epoch))


def test_no_history_and_single_observation_domain_fall_back_to_unity():
    state = UserReliability(schedule())
    state.record(1, scenario_id(schedule()[0]), 1)
    assert all(r["weight"] == 1 and r["tau"] is None for r in state.begin_epoch(3))


def test_task_pack_and_telecom_task_ids_remain_distinct():
    rows = schedule() + [{**schedule()[0], "task_pack": "pack"}]
    assert len({scenario_id(row) for row in rows}) == len(rows)
    state = UserReliability(rows + rows)
    assert len(state.begin_epoch(3)) == len(rows)


def test_terminal_record_uses_raw_boundary_value_excludes_truncation_and_padding():
    state = UserReliability(schedule())
    state.begin_epoch(1)
    data = batch(enabled=True)
    data["user_reliability_id"] = torch.tensor([scenario_id(schedule()[0])])
    groups, _ = pad_groups_for_dispatch([data], 2)
    groups[1]["user_boundary_types"][0, 14] = 3  # Even stale dummy metadata is ignored.
    state.observe(groups)
    assert state.state_dict()["records"] == [[1, scenario_id(schedule()[0]), 0.625]]
    for mutation in ("truncated", "no_boundary"):
        other = deepcopy(data)
        if mutation == "truncated":
            other["truncated"].fill_(True)
            other["terminated"].fill_(False)
        else:
            other["user_boundary_types"].zero_()
        state.observe([other])
    assert len(state.state_dict()["records"]) == 1


def test_actor_scaling_after_normalization_keeps_critic_targets_loss_and_padding():
    unweighted = batch(enabled=True)
    weighted = deepcopy(unweighted)
    weighted["user_reliability_weight"] = torch.full_like(weighted["values"], 0.5)
    policy = actor(1.0)
    policy.adv_norm = lambda adv, mask, **kw: adv + mask * 0.25
    plain = policy._compute_advantages(unweighted)
    result = policy._compute_advantages(weighted)
    torch.testing.assert_close(
        result["advantages"], plain["advantages"] * 0.5, rtol=0, atol=0
    )
    for name in ("returns", "values", "loss_mask", "critic_loss_mask"):
        torch.testing.assert_close(result[name], plain[name], rtol=0, atol=0)
    torch.testing.assert_close(
        ppo_loss_fn(result["values"], result, eps_clip=None),
        ppo_loss_fn(plain["values"], plain, eps_clip=None),
        rtol=0,
        atol=0,
    )
    dummy = deepcopy(weighted)
    dummy["is_padding"] = torch.ones_like(dummy["loss_mask"], dtype=torch.bool)
    dummy["user_boundary_types"].zero_()
    dummy["loss_mask"].zero_()
    dummy["user_reliability_weight"].zero_()
    assert not policy._compute_advantages(dummy)["advantages"].count_nonzero()


def test_identity_metadata_survives_episode_tensor_export():
    interaction = _interaction([0, 0], [1, 1])
    ident = scenario_id(schedule()[0])
    interaction.apply_episode_result(
        AgentWorkflowResult(
            reward=1,
            terminated=True,
            truncated=False,
            metadata={"user_reliability_id": ident},
        ),
        episode_id=1,
    )
    assert interaction.to_tensor_dict()["user_reliability_id"].tolist() == [ident]


def test_switch_off_hooks_leave_data_and_recover_state_unchanged():
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(user_reliability_weight=False)
    groups = [batch(enabled=True)]
    original = deepcopy(groups)
    trainer._process_rollout_batch(groups)
    assert trainer._extra_recovery_trainer_state() == {}
    for name, value in original[0].items():
        torch.testing.assert_close(groups[0][name], value, rtol=0, atol=0)
    assert set(groups[0]) == set(original[0])


def test_trainer_recover_checkpoint_contains_records_and_frozen_weights(monkeypatch):
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(
        user_reliability_weight=True,
        num_critic_only_steps=0,
        critic_updates_before_actor=0,
        rollout=SimpleNamespace(use_data_generator=True),
    )
    trainer._user_reliability = seeded()
    trainer._user_reliability.begin_epoch(3)
    trainer.actor = Mock()
    trainer.critic = Mock()
    trainer.rollout = Mock()
    trainer.recover_handler = Mock()
    trainer.train_dataloader = [1, 2]
    trainer.saver = trainer.evaluator = trainer.stats_logger = Mock()
    trainer.tokenizer = trainer.processor = None
    monkeypatch.setattr("areal.trainer.rl_trainer.is_single_controller", lambda: True)
    trainer._save_recover_checkpoint(2, 0, 4)
    saved = trainer.recover_handler.dump.call_args.kwargs["trainer_state"]
    assert saved["user_reliability"] == trainer._user_reliability.state_dict()


def test_config_defaults_and_privileged_offline_combination(monkeypatch, tmp_path):
    from examples.tau2.train import validate_tau2_recipe
    from tests.test_tau2_ppo import _config

    default = OmegaConf.structured(Tau2PPOConfig)
    assert default.user_reliability_weight is False
    assert default.user_reliability_shrink_k == 2
    assert default.user_reliability_warmup_epochs == 2
    config = _config(monkeypatch, tmp_path)
    config.user_reliability_weight = config.train_user_boundary_values = True
    config.critic_user_scenario = config.offline_eval_only = True
    validate_tau2_recipe(config)
    Tau2PPOTrainer.validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("user_reliability_shrink_k", -1),
        ("user_reliability_shrink_k", float("nan")),
        ("user_reliability_warmup_epochs", -1),
        ("train_user_boundary_values", False),
        ("algorithm", "grpo"),
    ],
)
def test_config_rejects_invalid_reliability_settings(
    monkeypatch, tmp_path, field, value
):
    from tests.test_tau2_ppo import _config

    config = _config(monkeypatch, tmp_path)
    config.user_reliability_weight = config.train_user_boundary_values = True
    setattr(config, field, value)
    with pytest.raises(ValueError, match="reliability"):
        config.validate_user_td_split()


def test_future_epoch_records_are_excluded_before_first_freeze():
    state = seeded()
    expected = deepcopy(state.begin_epoch(3))
    fresh = seeded()
    fresh.record(3, scenario_id(schedule()[0]), 100)
    fresh.record(4, scenario_id(schedule()[1]), -100)
    assert fresh.begin_epoch(3) == expected


def test_epoch_tables_recreated_after_recovery_and_metadata_removed_from_forward(
    tmp_path,
):
    from areal.utils.data import drop_non_model_forward_metadata

    state = seeded()
    table = state.begin_epoch(3, tmp_path / "first")
    restored = UserReliability(schedule())
    restored.load_state_dict(state.state_dict())
    restored.begin_epoch(3, tmp_path / "resumed")
    path = "user-reliability/epoch-0003.jsonl"
    assert (tmp_path / "first" / path).read_bytes() == (
        tmp_path / "resumed" / path
    ).read_bytes()
    assert [
        json.loads(line)
        for line in (tmp_path / "first" / path).read_text().splitlines()
    ] == table
    inputs = {
        "input_ids": torch.zeros(1, 4),
        "user_reliability_id": torch.tensor([1]),
        "user_reliability_weight": torch.ones(1, 4),
    }
    drop_non_model_forward_metadata(inputs)
    assert set(inputs) == {"input_ids"}


def test_finite_batch_preparation_logs_domain_stats_and_keeps_padding_zero(tmp_path):
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.config = SimpleNamespace(
        user_reliability_weight=True,
        resolve_min_usable_group_size=lambda: 1,
        max_skipped_tasks_per_epoch=2,
        experiment_name="reliability",
        trial_name="test",
        cluster=SimpleNamespace(fileroot=str(tmp_path)),
    )
    trainer._user_reliability = seeded()
    trainer.rollout = SimpleNamespace()
    trainer.actor = SimpleNamespace(parallel_strategy=SimpleNamespace(dp_size=4))
    groups = []
    for row in schedule()[:3]:
        data = batch(enabled=True)
        data["user_reliability_id"] = torch.tensor([scenario_id(row)])
        groups.append(data)
    prepare = Mock(return_value=groups)
    prepare.epoch = 2
    physical = trainer._prepare_training_batch(prepare)
    assert len(physical) == 4
    assert not physical[-1]["user_reliability_weight"].count_nonzero()
    ws = [float(g["user_reliability_weight"][0, 0]) for g in groups]
    stats = trainer._reliability_stats
    assert stats["user_reliability/airline/weight_mean"] == pytest.approx(np.mean(ws))
    assert stats["user_reliability/airline/weight_std"] == pytest.approx(np.std(ws))
    assert stats["user_reliability/airline/scenarios_with_history"] == 3
    assert stats["user_reliability/telecom/episodes"] == 0
    trainer._process_rollout_batch(physical)
    assert len(trainer._user_reliability.records) == 12


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.asyncio
async def test_workflow_emits_reliability_identity_only_when_enabled(
    monkeypatch, tmp_path, enabled
):
    from examples.tau2.utils import Tau2RunInfo
    from tests.test_tau2_official_runtime import _workflow_with_fake_runner

    async def run(runner, task):
        return Tau2RunInfo.model_construct(
            reward=1,
            messages=[],
            reward_info=None,
            error_type=None,
            stop_reason="user_stop",
        )

    workflow = _workflow_with_fake_runner(monkeypatch, tmp_path, run)
    workflow.user_reliability_weight = enabled
    result = await workflow.run(
        {"task_id": "a"}, base_url="http://policy.invalid/v1", api_key="test"
    )
    assert ("user_reliability_id" in result.metadata) == enabled
    if enabled:
        assert result.metadata["user_reliability_id"] == scenario_id(
            {"domain": workflow.econfig.domain, "task_id": "a"}
        )


@pytest.mark.parametrize("recover", [False, True])
def test_trainer_initialization_restores_reliability_state(monkeypatch, recover):
    from examples.tau2.evaluation import Tau2AsyncEvalTrainer

    state = seeded()
    state.begin_epoch(3)
    saved = json.loads(json.dumps({"user_reliability": state.state_dict()}))
    trainer = object.__new__(Tau2PPOTrainer)
    trainer.train_dataloader = SimpleNamespace(dataset=schedule())
    trainer.recover_info = SimpleNamespace(trainer_state=saved) if recover else None
    monkeypatch.setattr(Tau2AsyncEvalTrainer, "_init_impl", lambda *a, **kw: None)
    config = SimpleNamespace(
        user_reliability_weight=True,
        user_reliability_shrink_k=2,
        user_reliability_warmup_epochs=2,
    )
    trainer._init_impl(config)
    if recover:
        assert trainer._user_reliability.state_dict() == state.state_dict()
    else:
        assert trainer._user_reliability.records == []


def test_recover_info_disk_roundtrip_preserves_current_and_next_epoch_weights(tmp_path):
    from tests.test_ppo_warmup_recovery import _info

    from areal.utils.recover import RecoverInfo

    state = seeded()
    table = deepcopy(state.begin_epoch(3))
    state.record(3, scenario_id(schedule()[0]), -1.0)
    info = _info(trainer_state={"user_reliability": state.state_dict()})
    info.dump(str(tmp_path))
    restored_info = RecoverInfo.load(str(tmp_path))
    restored = UserReliability(schedule())
    restored.load_state_dict(restored_info.trainer_state["user_reliability"])
    assert restored.begin_epoch(3) == table
    assert restored.begin_epoch(4) == state.begin_epoch(4)


def test_unseen_scenario_gets_domain_prior_and_uneven_schedule_is_deduplicated():
    rows = schedule() + [{"domain": "airline", "task_id": "unseen", "task_pack": None}]
    state = UserReliability(rows + [rows[0]] * 4)
    for epoch, ident, x in seeded().records:
        state.record(epoch, ident, x)
    table = state.begin_epoch(3)
    unseen = next(row for row in table if row["n"] == 0)
    assert unseen["sigma2"] == 0.3125
    assert unseen["s2"] is None
    assert len(table) == len(rows)
    assert np.mean(
        [row["weight"] for row in table if row["scenario"][0] == "airline"]
    ) == pytest.approx(1)


def test_actor_loss_gradient_scales_per_episode_with_clipping_and_packing():
    from areal.trainer.ppo.actor import grpo_loss_fn
    from areal.utils.data import pack_tensor_dict

    weighted = batch(enabled=True)
    weighted["user_reliability_weight"] = torch.full_like(weighted["values"], 0.4)
    plain = actor(1.0)._compute_advantages(batch(enabled=True))
    result = actor(1.0)._compute_advantages(weighted)
    for packed in (False, True):
        losses, grads = [], []
        for data in (plain, result):
            stats_tracker.export_all()
            data = deepcopy(data)
            data["prox_logp"] = torch.zeros_like(data["logprobs"])
            if packed:
                data = pack_tensor_dict(data)
            logp = torch.full_like(data["logprobs"], 0.1, requires_grad=True)
            loss = grpo_loss_fn(
                logp,
                torch.zeros_like(logp),
                data,
                eps_clip=0.2,
                eps_clip_higher=0.2,
                c_clip=None,
                use_decoupled_loss=True,
            )
            loss.backward()
            losses.append(loss.detach())
            grads.append(logp.grad)
        torch.testing.assert_close(losses[1], losses[0] * 0.4, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(grads[1], grads[0] * 0.4, rtol=1e-6, atol=1e-7)
