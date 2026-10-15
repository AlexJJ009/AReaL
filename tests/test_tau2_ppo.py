# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the τ² PPO recipe, finite coverage and frozen-policy warmup."""

from collections import Counter
from copy import deepcopy

import pytest
from datasets import Dataset

from examples.tau2 import ppo_data, task_pack, train
from examples.tau2.evaluation import Tau2AsyncEvalTrainer
from examples.tau2.utils import Tau2PPOConfig
from tests.test_ppo_critic_warmup import (
    _assert_state_equal,
    _make_trainer,
    _Workflow,
)

from areal.api.cli_args import load_expr_config
from areal.trainer.rl_trainer import PPOTrainer


def _config(monkeypatch, tmp_path):
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    monkeypatch.setenv("TAU2_TRIAL_NAME", "ppo-test")
    monkeypatch.delenv("TAU2_ACTOR_PATH", raising=False)
    monkeypatch.delenv("TAU2_CRITIC_PATH", raising=False)
    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_ppo.yaml"], Tau2PPOConfig
    )
    return config


def test_ppo_default_contract_accepts_full_critic_and_async_loss(monkeypatch, tmp_path):
    """The shipped PPO config must select the intended engine/loss and defaults."""
    config = _config(monkeypatch, tmp_path)
    assert train.validate_tau2_recipe(config) == ("airline", "retail")
    Tau2AsyncEvalTrainer.validate_config(config)
    assert config.algorithm == "ppo"
    assert config.critic.path == config.actor.path
    assert "Qwen3-4B-Instruct-2507@" in config.actor.path
    assert config.critic.optimizer.lr == 1e-5
    assert config.actor.optimizer.lr == 1e-6
    assert config.critic.optimizer.warmup_steps is None
    assert config.critic.optimizer.warmup_steps_proportion == 0.015


def test_ppo_contract_wrong_combinations_refuse(monkeypatch, tmp_path):
    """Each incompatible switch is independently refused before any workers."""
    config = _config(monkeypatch, tmp_path)
    invalid = [
        ("actor.path", "Qwen/Qwen3.5-4B@wrong-policy"),
        ("critic", None),
        ("critic.is_critic", False),
        ("actor.use_direct_dis_loss", True),
        ("actor.use_sapo_loss", True),
        ("actor.use_cispo_loss", True),
        ("actor.eps_clip", 0.1),
        ("actor.eps_clip_higher", 0.28),
        ("actor.c_clip", 3.0),
        ("actor.gae_lambda", "areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae"),
        ("actor.critic_gae_lambda", 0.95),
        ("actor.discount", 0.99),
        ("actor.reward_bias", 1.0),
        ("actor.overlong_reward_penalty", True),
        ("critic.freeze_critic_attention", True),
        ("critic.use_lora", True),
        ("critic.eps_clip", None),
        ("critic.mask_no_eos_with_zero", True),
        ("actor.kl_ctl", 0.001),
        ("critic_updates_before_actor", 2),
        ("actor.use_decoupled_loss", False),
        ("actor.recompute_logprob", False),
        ("actor.prox_logp_method", "reuse_train_logp"),
        ("actor.importance_sampling_level", "sequence"),
        ("actor.rejection_sampling.upper", 4.0),
        ("rollout.max_head_offpolicyness", 3),
        ("rollout.agent.tool_call_parser", "qwen3_coder"),
        ("evaluation_rollout.agent.tool_call_parser", "qwen3_coder"),
        ("eval_gconfig.temperature", 1.0),
        ("eval_gconfig.n_samples", 1),
        ("evaluator.freq_steps", 0),
        ("evaluator.eval_before_train", False),
    ]
    for path, value in invalid:
        broken = deepcopy(config)
        obj = broken
        *parents, name = path.split(".")
        for part in parents:
            obj = getattr(obj, part)
        setattr(obj, name, value)
        with pytest.raises(ValueError):
            train.validate_tau2_recipe(broken)


def test_ppo_preflight_runs_runtime_checks_and_saver_sync(monkeypatch, tmp_path):
    """Preflight uses actual trainer guards, including run 527's failure."""
    config = _config(monkeypatch, tmp_path)
    for field, value in [
        ("freq_steps", 5),
        ("freq_epochs", 1),
        ("freq_secs", 60),
        ("mode", "async"),
    ]:
        broken = deepcopy(config)
        setattr(broken.saver, field, value)
        with pytest.raises(ValueError, match="Dedicated async eval"):
            Tau2AsyncEvalTrainer.validate_config(broken)
    config.num_critic_only_steps = 1
    config.dynamic_bs = True
    with pytest.raises(ValueError, match="dynamic_bs=False"):
        Tau2AsyncEvalTrainer.validate_config(config)
    config.dynamic_bs = False
    config.critic.ppo_n_minibatches = 2
    with pytest.raises(ValueError, match="Finite-tail critic"):
        Tau2AsyncEvalTrainer.validate_config(config)


class _WarmupTrainer(PPOTrainer):
    # Exercise the real τ² opt-in hook with the native training loop.
    _evaluate_during_critic_warmup = Tau2AsyncEvalTrainer._evaluate_during_critic_warmup


def test_ppo_warmup_freezes_actor_optimizer_push_and_rollout_policy():
    """Real CPU Adam state stays unchanged until N; critic and eval keep advancing."""
    trainer = _make_trainer(total_steps=12, num_critic_only_steps=10)
    trainer.__class__ = _WarmupTrainer
    trainer.config.algorithm = "ppo"
    seen = []
    original = trainer.actor.prepare_batch
    initial = deepcopy(trainer.actor.state_dict())

    def prepare(*args, **kwargs):
        seen.append(
            (deepcopy(trainer.actor.state_dict()), trainer.rollout.versions.copy())
        )
        return original(*args, **kwargs)

    trainer.actor.prepare_batch = prepare
    trainer.train(workflow=_Workflow())
    for actor_state, versions in seen[:11]:
        _assert_state_equal(actor_state, initial)
        assert versions == []
    assert len(trainer.actor.ppo_update_steps) == 2
    assert len(trainer.actor.scheduler_steps) == 2
    assert trainer.actor.weight_update_versions == [1, 2]
    assert trainer.rollout.versions == [1, 2]
    assert len(trainer.critic.ppo_update_steps) == 12
    assert len(trainer._eval_calls) == 12
    assert [c["data"]["ppo/critic_only"] for c in trainer.stats_logger.commits] == [
        1
    ] * 10 + [0, 0]
    assert [c["data"]["ppo/policy_version"] for c in trainer.stats_logger.commits] == [
        0
    ] * 10 + [1, 2]


def test_ppo_zero_warmup_preserves_native_updates():
    """N=0 has exactly the native optimizer, scheduler and version sequence."""
    native = _make_trainer(total_steps=3, num_critic_only_steps=0)
    ppo = _make_trainer(total_steps=3, num_critic_only_steps=0)
    ppo.__class__ = _WarmupTrainer
    ppo.config.algorithm = "ppo"
    native.train(workflow=_Workflow())
    ppo.train(workflow=_Workflow())
    _assert_state_equal(ppo.actor.state_dict(), native.actor.state_dict())
    _assert_state_equal(ppo.critic.state_dict(), native.critic.state_dict())
    assert (
        ppo.actor.weight_update_versions
        == native.actor.weight_update_versions
        == [1, 2, 3]
    )
    assert len(ppo.actor.scheduler_steps) == 3
    assert all("ppo/critic_only" not in c["data"] for c in ppo.stats_logger.commits)


@pytest.fixture
def pools(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    packs = [tmp_path / "pack-a", tmp_path / "pack-b"]
    for pack in packs:
        pack.mkdir()
    config.train_task_packs = list(map(str, packs))
    config.domains = ["airline", "retail", "telecom"]
    kwargs = {
        "domain": tuple(config.domains),
        "provenance": {},
        "variant": "fixed",
        "seed": 42,
    }
    monkeypatch.setattr(
        task_pack,
        "domains",
        lambda p: ["airline", "retail"] if p == str(packs[0]) else ["telecom"],
    )
    monkeypatch.setattr(
        task_pack,
        "task_ids",
        lambda p, d: ["same-id", "p1"] if d == "airline" else ["same-id"],
    )

    def official(**kw):
        return Dataset.from_list(
            [
                {
                    "domain": d,
                    "task_id": "same-id",
                    "split": "train",
                    "source_id": f"tau2:{d}:same-id",
                }
                for d in kw["domain"]
            ]
        )

    monkeypatch.setattr(train, "get_tau2_dataset", official)
    return config, kwargs, packs


def test_ppo_multi_pack_schedule_preserves_every_source_equally(pools):
    """Optional Telecom and official tasks share one uniform repeated schedule."""
    config, kwargs, packs = pools
    config.samples_per_task_per_epoch = 3
    rows = list(ppo_data.build_ppo_train_dataset(config, kwargs))
    counts = Counter((r["domain"], r["task_id"], r["task_pack"]) for r in rows)
    assert len(counts) == 6 and set(counts.values()) == {3}
    assert config.domain_effective_episodes == {"airline": 9, "retail": 6, "telecom": 3}
    assert len({r["source_id"] for r in rows}) == 6
    assert {r["task_pack"] for r in rows} == {None, *map(str, packs)}
    assert not any(r["domain"] == "telecom" and r["task_pack"] is None for r in rows)
    assert rows == list(ppo_data.build_ppo_train_dataset(config, kwargs))


def test_ppo_nonuniform_budget_and_truncation_refuse(pools):
    """Budgets, drop-last, duplicate packs and task limits cannot bias coverage."""
    config, kwargs, packs = pools
    for budgets in [
        {"airline": 3},
        {"airline": 6, "retail": 2, "telecom": 1},
        {"airline": 2, "retail": 2, "telecom": 1},
    ]:
        config.domain_effective_episodes = budgets
        with pytest.raises(ValueError, match="repeat every task equally"):
            ppo_data.build_ppo_train_dataset(config, kwargs)
    config.domain_effective_episodes = {}
    config.task_limit = 5
    with pytest.raises(ValueError, match="must not truncate"):
        ppo_data.build_ppo_train_dataset(config, kwargs)
    config.task_limit = None
    config.train_dataset.drop_last = True
    with pytest.raises(ValueError, match="drop_last=false"):
        ppo_data.build_ppo_train_dataset(config, kwargs)
    config.train_dataset.drop_last = False
    config.train_task_packs.append(str(packs[0]))
    with pytest.raises(ValueError, match="Duplicate task pack"):
        ppo_data.build_ppo_train_dataset(config, kwargs)


def test_ppo_check_config_uses_runtime_guards_without_workers(pools, monkeypatch):
    """The command path itself catches cadence and truncated formal coverage."""
    _, _, packs = pools
    monkeypatch.setattr(train.benchmark_variant, "validate_task_files", lambda *a: None)
    monkeypatch.setattr(
        ppo_data,
        "build_ppo_eval_dataset",
        lambda *a: Dataset.from_list(
            [
                {
                    "domain": "airline",
                    "task_id": "test-only",
                    "source_id": "tau2:airline:test-only",
                }
            ]
        ),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("check-config tried to resolve weights or initialize workers")

    monkeypatch.setattr(train, "resolve_actor_snapshot", forbidden)
    monkeypatch.setattr(Tau2AsyncEvalTrainer, "__init__", forbidden)
    args = [
        "--config",
        "examples/tau2/config_ppo.yaml",
        "--check-config",
        f"train_task_packs=[{packs[0]}]",
        "num_critic_only_steps=0",
    ]
    with pytest.raises(ValueError, match="frequencies to match"):
        train.main([*args, "saver.freq_steps=5"])
    with pytest.raises(ValueError, match="complete epochs"):
        train.main([*args, "total_train_steps=1"])
    train.main(args)


def test_ppo_eval_composes_fixed_airline_retail_and_official_telecom(
    monkeypatch, tmp_path
):
    """Evaluation is test-only and independent of training domains/packs."""
    config = _config(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        ppo_data.benchmark_variant, "validate_task_files", lambda *a: calls.append(a)
    )

    def official(**kwargs):
        assert kwargs["split"] == "test" and kwargs["experiment_mode"] == "formal"
        return Dataset.from_list(
            [
                {
                    "domain": d,
                    "task_id": str(i),
                    "split": "test",
                    "source_id": f"tau2:{d}:{i}",
                }
                for d in kwargs["domain"]
                for i in range({"airline": 20, "retail": 40, "telecom": 40}[d])
            ]
        )

    monkeypatch.setattr(train, "get_tau2_dataset", official)
    rows = ppo_data.build_ppo_eval_dataset(
        config, {"domain": ("airline", "retail"), "variant": "fixed"}
    )
    assert Counter(r["domain"] for r in rows) == {
        "airline": 20,
        "retail": 40,
        "telecom": 40,
    }
    assert calls == [("fixed", ("airline", "retail", "telecom"))]
    assert len(rows) * config.eval_gconfig.n_samples == 400
    assert benchmark_telecom_is_official()


def benchmark_telecom_is_official():
    return (
        ppo_data.benchmark_variant.TASK_FILE_SHA256["fixed"]["telecom"]
        == ppo_data.benchmark_variant.TASK_FILE_SHA256["official"]["telecom"]
    )


def test_ppo_eval_wrong_size_refuses(monkeypatch, tmp_path):
    """Missing Telecom rows cannot silently change the evaluation denominator."""
    config = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ppo_data.benchmark_variant, "validate_task_files", lambda *a: None
    )
    monkeypatch.setattr(
        train,
        "get_tau2_dataset",
        lambda **kw: Dataset.from_list([{"domain": "airline", "task_id": "x"}]),
    )
    with pytest.raises(ValueError, match="20/40/40"):
        ppo_data.build_ppo_eval_dataset(config, {})


def test_ppo_critic_checkpoint_override_survives_actor_resolution(
    monkeypatch, tmp_path
):
    """A pretrained scalar checkpoint must not be replaced by the actor backbone."""
    config = _config(monkeypatch, tmp_path)
    checkpoint = tmp_path / "critic-checkpoint"
    checkpoint.mkdir()
    config.critic.path = str(checkpoint)
    monkeypatch.setattr(
        train, "resolve_pinned_hf_snapshot", lambda _: "/resolved/actor"
    )
    train.resolve_actor_snapshot(config)
    assert config.actor.path == "/resolved/actor"
    assert config.critic.path == str(checkpoint)


def test_critic_only_full_parameter_recipe_accepts_original_settings(
    monkeypatch, tmp_path
):
    """Run 480's configuration admits the single attention-freezing change."""
    from scripts.tau2.train_critic import (
        fill_derived_train_steps,
        validate_tau2_critic_config,
    )

    _config(monkeypatch, tmp_path)
    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "critic.freeze_critic_attention=false",
            "actor.path=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554",
            "rollout.agent.tool_call_parser=qwen25",
            "econfig.benchmark_variant=fixed",
        ],
        Tau2PPOConfig,
    )
    fill_derived_train_steps(config, train_rows=747)
    validate_tau2_critic_config(config, train_rows=747, validation_rows=60)
    assert config.critic.freeze_critic_attention is False
    assert config.total_train_steps == config.num_critic_only_steps == 94
    assert config.critic.optimizer.lr == 5e-6
    assert config.train_batch_episodes == 16
