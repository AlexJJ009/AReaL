# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for tau2 online critic training helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from scripts.tau2 import train_critic

from areal.api.cli_args import load_expr_config
from areal.trainer.ppo.critic import ppo_loss_fn


def _domain_for_index(index: int) -> str:
    if index < 30:
        return "airline"
    if index < 104:
        return "retail"
    return "telecom"


def _task_row(index: int, critic_split: str, split: str = "train") -> dict:
    domain = _domain_for_index(index)
    return {
        "domain": domain,
        "task_id": f"{split}-{index}",
        "split": split,
        "critic_split": critic_split,
        "source_id": f"tau2:{domain}:{split}-{index}",
        "policy_id": "Qwen/Qwen3.5-4B",
        "policy_revision": "policy0-revision",
        "simulator_id": "deepseek-flash",
    }


def _patched_tau2_datasets(monkeypatch) -> None:
    def fake_get_tau2_dataset(**kwargs):
        split = kwargs["split"]
        experiment_mode = kwargs["experiment_mode"]
        rows = [
            _task_row(index, "dev" if index % 5 == 0 and index < 180 else "train")
            for index in range(178)
        ]
        if split == "train" and experiment_mode == "formal":
            selected = rows
        elif split == "train":
            selected = [row for row in rows if row["critic_split"] == "train"][:142]
        elif split == "dev":
            selected = [row for row in rows if row["critic_split"] == "dev"][:36]
        elif split == "test":
            selected = [_task_row(index, "test", split="test") for index in range(100)]
        else:
            raise AssertionError(f"unexpected split {split}")
        return Dataset.from_list(selected)

    monkeypatch.setattr(train_critic, "get_tau2_dataset", fake_get_tau2_dataset)


def _critic_row(index: int, critic_split: str = "dev") -> dict:
    return {
        "domain": _domain_for_index(index),
        "task_id": f"train-{index}",
        "split": "train",
        "critic_split": critic_split,
        "episode_id": 1000 + index,
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "loss_mask": [0, 1, 0, 1],
        "reward": float(index % 2),
        "terminated": True,
        "truncated": False,
    }


def test_build_tau2_critic_datasets_uses_official_formal_train_and_test(
    monkeypatch, tmp_path
):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"],
        train_critic.Tau2PPOConfig,
    )

    train_dataset, validation_dataset, summary = (
        train_critic.build_tau2_critic_datasets(config)
    )

    assert len(train_dataset) == 178
    assert len(validation_dataset) == 100
    assert summary["source"] == "official_tau2_train_tasks"
    assert summary["experiment_mode"] == "formal"
    assert summary["validation_split"] == "test"
    assert summary["test_rows"] == 100
    assert summary["policy_provenance"] == {
        "policy_id": "Qwen/Qwen3.5-4B",
        "policy_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "simulator_id": "deepseek-flash",
    }


def test_production_config_resolves_formal_online_training_contract(
    monkeypatch, tmp_path
):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))

    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"],
        train_critic.Tau2PPOConfig,
    )
    train_dataset, validation_dataset, _ = train_critic.build_tau2_critic_datasets(
        config
    )
    train_critic.fill_derived_train_steps(config, train_rows=len(train_dataset))
    train_critic.validate_tau2_critic_config(
        config,
        train_rows=len(train_dataset),
        validation_rows=len(validation_dataset),
    )

    assert len(train_dataset) == 178
    assert len(validation_dataset) == 100
    assert config.total_train_steps == 24
    assert config.total_train_epochs == 2
    assert config.num_critic_only_steps == 24
    assert config.algorithm == "critic"
    assert config.experiment_mode == "formal"
    assert config.train_dataset.batch_size == 16
    assert config.valid_dataset is None
    assert config.actor.backend == "fsdp:d4p1t1"
    assert config.rollout.backend == "sglang:d4p1t1"
    assert config.rollout.max_head_offpolicyness == 2
    assert config.rollout.max_concurrent_rollouts == 16
    assert config.gconfig.seed is None
    assert config.sglang.max_running_requests == 4
    assert config.sglang.mem_fraction_static == pytest.approx(0.65)
    assert config.critic is not None
    assert config.actor.offload is True
    assert config.rollout.deterministic_sampling is True
    assert config.critic.offload is True
    assert config.critic.freeze_critic_attention is True
    assert config.critic.optimizer.lr == pytest.approx(5.0e-6)
    assert config.actor.disable_dropout is True
    assert config.critic.disable_dropout is True
    assert config.actor.ppo_n_minibatches == 1
    assert config.critic.ppo_n_minibatches == 1
    assert config.critic.loss_reduction == "token_mean"
    assert config.critic.eps_clip is None
    assert config.econfig.enable_thinking is False
    assert config.gconfig.max_tokens == 32768
    assert config.gconfig.max_new_tokens == 4096


def test_check_config_does_not_resolve_model_snapshot(monkeypatch, tmp_path, capsys):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))

    def fail_snapshot_resolution(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("snapshot resolution should not run in --check-config")

    monkeypatch.setattr(
        train_critic,
        "resolve_training_snapshots",
        fail_snapshot_resolution,
    )

    train_critic.main(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "--check-config",
        ]
    )

    captured = capsys.readouterr()
    assert "model_snapshot_resolution: 0" in captured.out
    assert "train_rows: 178" in captured.out
    assert "validation_rows: 100" in captured.out
    assert "validation_split: test" in captured.out


def test_tune_override_keeps_train_dev_split_and_best_selection(monkeypatch, tmp_path):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))

    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "experiment_mode=tune",
        ],
        train_critic.Tau2PPOConfig,
    )
    train_dataset, validation_dataset, summary = (
        train_critic.build_tau2_critic_datasets(config)
    )
    train_critic.fill_derived_train_steps(config, train_rows=len(train_dataset))
    train_critic.validate_tau2_critic_config(
        config,
        train_rows=len(train_dataset),
        validation_rows=len(validation_dataset),
    )

    assert len(train_dataset) == 142
    assert len(validation_dataset) == 36
    assert config.total_train_steps == 18
    assert config.num_critic_only_steps == 18
    assert summary["experiment_mode"] == "tune"
    assert summary["validation_split"] == "dev"
    assert summary["dev_rows"] == 36


def test_tune_override_allows_bounded_two_step_probe(monkeypatch, tmp_path):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))

    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "experiment_mode=tune",
            "total_train_steps=2",
            "num_critic_only_steps=2",
        ],
        train_critic.Tau2PPOConfig,
    )
    train_dataset, validation_dataset, _ = train_critic.build_tau2_critic_datasets(
        config
    )
    train_critic.fill_derived_train_steps(config, train_rows=len(train_dataset))
    train_critic.validate_tau2_critic_config(
        config,
        train_rows=len(train_dataset),
        validation_rows=len(validation_dataset),
    )

    assert len(train_dataset) == 142
    assert len(validation_dataset) == 36
    assert config.total_train_steps == 2
    assert config.num_critic_only_steps == 2


def test_formal_override_rejects_partial_training_steps(monkeypatch, tmp_path):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))

    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "total_train_steps=2",
            "num_critic_only_steps=2",
        ],
        train_critic.Tau2PPOConfig,
    )
    train_dataset, validation_dataset, _ = train_critic.build_tau2_critic_datasets(
        config
    )

    with pytest.raises(ValueError, match="Formal tau2 critic fit requires"):
        train_critic.validate_tau2_critic_config(
            config,
            train_rows=len(train_dataset),
            validation_rows=len(validation_dataset),
        )


def test_formal_validation_hook_does_not_select_best_from_test(tmp_path, monkeypatch):
    calls = {"saves": 0, "commits": 0}

    def fake_evaluate(*args, **kwargs):  # noqa: ARG001
        return {
            "completed_step": 5,
            "summary": {
                "mse": 0.5,
                "macro_mse": 0.5,
                "explained_variance_defined": False,
                "explained_variance": None,
                "n_tokens": 2,
            },
            "by_domain": {},
            "grad_norm": {"overall": 1.0},
        }

    monkeypatch.setattr(train_critic, "evaluate_tau2_critic", fake_evaluate)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            total_train_steps=5,
            saver=SimpleNamespace(freq_steps=5),
        ),
        stats_logger=SimpleNamespace(
            commit=lambda *args: calls.__setitem__("commits", calls["commits"] + 1)
        ),
        saver=SimpleNamespace(
            save=lambda *args, **kwargs: calls.__setitem__("saves", calls["saves"] + 1)
        ),
        critic=object(),
        tokenizer=None,
        processor=None,
    )

    train_critic.install_tau2_validation_hook(
        trainer,
        [],
        evidence_dir=tmp_path,
        validation_steps=(5,),
        best=None,
        select_best=False,
    )
    trainer.stats_logger.commit(0, 4, 4, {})

    assert calls == {"saves": 0, "commits": 1}
    assert not (tmp_path / "best-validation.json").exists()


def test_tune_validation_hook_can_select_best_from_dev(tmp_path, monkeypatch):
    calls = {"saves": 0, "commits": 0}

    def fake_evaluate(*args, **kwargs):  # noqa: ARG001
        return {
            "completed_step": 5,
            "summary": {
                "mse": 0.5,
                "macro_mse": 0.5,
                "explained_variance_defined": False,
                "explained_variance": None,
                "n_tokens": 2,
            },
            "by_domain": {},
            "grad_norm": {"overall": 1.0},
        }

    monkeypatch.setattr(train_critic, "evaluate_tau2_critic", fake_evaluate)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            total_train_steps=5,
            saver=SimpleNamespace(freq_steps=5),
        ),
        stats_logger=SimpleNamespace(
            commit=lambda *args: calls.__setitem__("commits", calls["commits"] + 1)
        ),
        saver=SimpleNamespace(
            save=lambda *args, **kwargs: calls.__setitem__("saves", calls["saves"] + 1)
        ),
        critic=object(),
        tokenizer=None,
        processor=None,
    )

    train_critic.install_tau2_validation_hook(
        trainer,
        [],
        evidence_dir=tmp_path,
        validation_steps=(5,),
        best=None,
        select_best=True,
    )
    trainer.stats_logger.commit(0, 4, 4, {})

    assert calls == {"saves": 1, "commits": 1}
    assert (tmp_path / "best-validation.json").exists()


def test_critic_config_rejects_ignored_task_limit(monkeypatch, tmp_path):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "experiment_mode=tune",
            "+task_limit=1",
        ],
        train_critic.Tau2PPOConfig,
    )
    train_dataset, validation_dataset, _ = train_critic.build_tau2_critic_datasets(
        config
    )
    train_critic.fill_derived_train_steps(config, train_rows=len(train_dataset))

    with pytest.raises(ValueError, match="task_limit"):
        train_critic.validate_tau2_critic_config(
            config,
            train_rows=len(train_dataset),
            validation_rows=len(validation_dataset),
        )


def test_fixed_validation_cache_rejects_hash_mismatch_before_torch_load(
    monkeypatch, tmp_path
):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"],
        train_critic.Tau2PPOConfig,
    )
    validation_rows = [
        {"domain": "airline", "task_id": "test-0", "source_id": "tau2:airline:test-0"}
    ]
    manifest = train_critic._fixed_validation_manifest(
        config,
        validation_rows,
        workflow="wf",
        workflow_kwargs={},
    )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "fixed-validation-rollouts.pt").write_bytes(b"not a torch file")
    train_critic._write_json_atomic(
        evidence_dir / "fixed-validation-manifest.json",
        dict(manifest, rollouts_sha256="0" * 64),
    )
    trainer = SimpleNamespace(config=config, actor=SimpleNamespace())

    with pytest.raises(RuntimeError, match="hash mismatch"):
        train_critic.load_or_collect_fixed_validation_rollouts(
            trainer,
            validation_rows,
            evidence_dir=evidence_dir,
            workflow="wf",
            workflow_kwargs={},
        )


def test_fixed_validation_cache_rejects_malformed_tensor_shapes(monkeypatch, tmp_path):
    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"],
        train_critic.Tau2PPOConfig,
    )
    validation_rows = [
        {"domain": "airline", "task_id": "test-0", "source_id": "tau2:airline:test-0"}
    ]
    bad_rollouts = [
        {
            "domain": "airline",
            "task_id": "test-0",
            "source_id": "tau2:airline:test-0",
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
            "reward": 1.0,
        }
    ]
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    rollouts_path = evidence_dir / "fixed-validation-rollouts.pt"
    torch.save(bad_rollouts, rollouts_path)
    manifest = train_critic._fixed_validation_manifest(
        config,
        validation_rows,
        workflow="wf",
        workflow_kwargs={},
    )
    train_critic._write_json_atomic(
        evidence_dir / "fixed-validation-manifest.json",
        dict(manifest, rollouts_sha256=train_critic._file_sha256(rollouts_path)),
    )
    trainer = SimpleNamespace(config=config, actor=SimpleNamespace())

    with pytest.raises(RuntimeError, match="tensor shape mismatch"):
        train_critic.load_or_collect_fixed_validation_rollouts(
            trainer,
            validation_rows,
            evidence_dir=evidence_dir,
            workflow="wf",
            workflow_kwargs={},
        )


def test_final_summary_rejects_nonzero_recovered_policy_version(monkeypatch, tmp_path):
    from areal.api.io_struct import StepInfo
    from areal.utils.recover import RecoverInfo

    _patched_tau2_datasets(monkeypatch)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    config, _ = load_expr_config(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "experiment_mode=tune",
            "total_train_steps=2",
            "num_critic_only_steps=2",
        ],
        train_critic.Tau2PPOConfig,
    )
    train_dataset, _, data_summary = train_critic.build_tau2_critic_datasets(config)
    train_critic.fill_derived_train_steps(config, train_rows=len(train_dataset))
    evidence_dir = tmp_path / "evidence"
    (evidence_dir / "train-metrics").mkdir(parents=True)
    train_critic.write_json(
        evidence_dir / "validation-step-000002.json",
        {"completed_step": 2},
    )
    train_critic.write_json(
        evidence_dir / "train-metrics" / "step-000002.json",
        {"completed_step": 2},
    )
    paths = train_critic._final_step_paths(config, train_rows=len(train_dataset))
    for key in (
        "final_critic_checkpoint",
        "recover_actor_checkpoint",
        "recover_critic_checkpoint",
    ):
        Path(paths[key]).mkdir(parents=True)
    recover_info = RecoverInfo(
        last_step_info=StepInfo(
            epoch=0, epoch_step=1, global_step=1, steps_per_epoch=9
        ),
        saver_info={},
        evaluator_info={},
        stats_logger_info={},
        dataloader_info={},
        checkpoint_info={},
        trainer_state={"num_critic_only_steps": 2, "policy_version": 1},
    )
    recover_info.dump(paths["recover_info"])

    with pytest.raises(RuntimeError, match="policy_version=0"):
        train_critic._final_summary_payload(
            config,
            evidence_dir=evidence_dir,
            data_summary=data_summary,
            select_best=True,
        )


def test_critic_validation_cadence_includes_step0_and_final_step():
    assert train_critic._validation_steps(
        SimpleNamespace(total_train_steps=18, evaluator=SimpleNamespace(freq_steps=5)),
        "default",
    ) == (0, 5, 10, 15, 18)
    assert train_critic._cadence_steps(18, 5) == (5, 10, 15, 18)
    assert train_critic._cadence_steps(20, 5) == (5, 10, 15, 20)


def test_fixed_dev_collection_uses_eval_rollout_without_train_capacity_credit():
    class FakeRollout:
        def __init__(self):
            self.started = 0
            self.batches = []
            self.pending = []
            self.submissions = []

        def start_proxy(self):
            self.started += 1

        def submit(self, **kwargs):
            self.submissions.append(kwargs)
            self.pending.append(kwargs["data"])

        def wait(self, count):
            rows, self.pending = self.pending, []
            assert len(rows) == count
            self.batches.append([row["task_id"] for row in rows])
            return [
                {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
                    "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.bool),
                    "reward": 1.0,
                    "truncated": False,
                }
                for _ in rows
            ]

    class TrainRollout:
        def __init__(self):
            self.consumed = 0

        def rollout_batch(self, *args, **kwargs):  # noqa: ARG002
            raise AssertionError("fixed dev collection must not use train rollout")

        def on_batch_consumed_without_update(self):
            self.consumed += 1

    class FakeActor:
        def __init__(self):
            self.cleared = 0

        def clear_batches(self, *rows):
            self.cleared += len(rows)

    eval_rollout = FakeRollout()
    train_rollout = TrainRollout()
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            train_dataset=SimpleNamespace(batch_size=2), total_train_steps=2
        ),
        rollout=train_rollout,
        eval_rollout=eval_rollout,
        actor=FakeActor(),
    )
    dev_rows = [
        {"domain": "airline", "task_id": "a", "source_id": "tau2:airline:a"},
        {"domain": "retail", "task_id": "r", "source_id": "tau2:retail:r"},
        {"domain": "telecom", "task_id": "t", "source_id": "tau2:telecom:t"},
    ]

    rows = train_critic.collect_tau2_validation_rollouts(
        trainer,
        dev_rows,
        workflow="examples.tau2.agent.Tau2AgentWorkflow",
        workflow_kwargs={},
    )

    assert len(rows) == 3
    assert eval_rollout.started == 1
    assert eval_rollout.batches == [["a", "r"], ["t"]]
    assert [s["task_id"] for s in eval_rollout.submissions] == [4, 5, 6]
    assert all(s["is_eval"] for s in eval_rollout.submissions)
    assert train_rollout.consumed == 0
    assert trainer.actor.cleared == 3
    assert [row["task_id"] for row in rows] == ["a", "r", "t"]
    assert all(row["input_ids"].device.type == "cpu" for row in rows)


def test_validation_offloads_actor_and_wraps_critic_lifecycle(tmp_path):
    events = []

    class FakeCritic:
        parallel_strategy = SimpleNamespace(dp_size=1)

        def compute_values(self, batch):
            return [torch.tensor([[0.0, 0.25, 0.5]]) for _ in batch]

        def grad_norm(self, batch):  # noqa: ARG002
            return {"grad_norm": 1.0}

        def clear_batches(self, *args):  # noqa: ARG002
            return None

    actor = object()
    critic = FakeCritic()
    trainer = SimpleNamespace(
        actor=actor,
        critic=critic,
        _should_offload_actor=True,
        _should_offload_critic=True,
    )

    def transition(engine, *, role):
        events.append((role, "actor" if engine is actor else "critic"))

    trainer._offload_model = transition
    trainer._onload_model = transition
    row = {
        "domain": "airline",
        "task_id": "a",
        "source_id": "tau2:airline:a",
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "loss_mask": [0, 1, 1],
        "reward": 0.0,
        "truncated": False,
    }

    report = train_critic.evaluate_tau2_critic(
        trainer,
        [row],
        evidence_dir=tmp_path,
        completed_step=0,
    )

    assert report["completed_step"] == 0
    assert events == [
        ("actor", "actor"),
        ("critic", "critic"),
        ("critic", "critic"),
    ]


def test_strict_train_batch_preserves_order_and_restarts_epoch():
    class FakeSampler:
        def __init__(self):
            self.epochs = []
            self.seed = None

        def set_epoch(self, epoch):
            self.epochs.append(epoch)

    class FakeDataLoader:
        batch_size = 2

        def __init__(self):
            self.sampler = FakeSampler()
            self.batches = [
                [
                    {"domain": "airline", "task_id": "a", "source_id": "s:a"},
                    {"domain": "retail", "task_id": "r", "source_id": "s:r"},
                ],
                [
                    {"domain": "telecom", "task_id": "t", "source_id": "s:t"},
                ],
            ]

        def __iter__(self):
            return iter(self.batches)

    class FakeRollout:
        def __init__(self):
            self.calls = []
            self.pending = []

        def submit(self, *, data, task_id, **kwargs):
            self.calls.append((data["task_id"], task_id, kwargs))
            self.pending.append({"task_id": data["task_id"]})

        def wait(self, count):
            results = self.pending[:count]
            del self.pending[:count]
            return results

    trainer = object.__new__(train_critic.Tau2CriticOnlineTrainer)
    trainer.rollout = FakeRollout()
    trainer.actor = SimpleNamespace(parallel_strategy=SimpleNamespace(dp_size=4))
    trainer.config = SimpleNamespace(
        gconfig=SimpleNamespace(n_samples=1),
        train_dataset=SimpleNamespace(batch_size=2),
    )
    trainer._tau2_train_epoch = 0
    trainer._tau2_train_iterator = None
    trainer._tau2_next_train_step = 0
    dataloader = FakeDataLoader()

    first = trainer._strict_prepare_training_batch(
        dataloader,
        workflow="wf",
        workflow_kwargs={"x": 1},
        group_size=1,
        reward_normalization=False,
        drop_incomplete_group=False,
    )
    second = trainer._strict_prepare_training_batch(dataloader, workflow="wf")
    third = trainer._strict_prepare_training_batch(dataloader, workflow="wf")

    assert dataloader.sampler.epochs == [0, 1]
    assert trainer.rollout.calls[:5] == [
        (
            "a",
            0,
            {
                "workflow": "wf",
                "workflow_kwargs": {"x": 1},
                "should_accept_fn": None,
                "group_size": 1,
                "reward_normalization": False,
                "drop_incomplete_group": False,
            },
        ),
        (
            "r",
            1,
            {
                "workflow": "wf",
                "workflow_kwargs": {"x": 1},
                "should_accept_fn": None,
                "group_size": 1,
                "reward_normalization": False,
                "drop_incomplete_group": False,
            },
        ),
        (
            "t",
            2,
            {
                "workflow": "wf",
                "workflow_kwargs": {},
                "should_accept_fn": None,
                "group_size": 1,
                "reward_normalization": False,
                "drop_incomplete_group": False,
            },
        ),
        (
            "a",
            4,
            {
                "workflow": "wf",
                "workflow_kwargs": {},
                "should_accept_fn": None,
                "group_size": 1,
                "reward_normalization": False,
                "drop_incomplete_group": False,
            },
        ),
        (
            "r",
            5,
            {
                "workflow": "wf",
                "workflow_kwargs": {},
                "should_accept_fn": None,
                "group_size": 1,
                "reward_normalization": False,
                "drop_incomplete_group": False,
            },
        ),
    ]
    assert [row["task_id"] for row in first[:2]] == ["a", "r"]
    assert [row["task_id"] for row in second[:1]] == ["t"]
    assert [row["task_id"] for row in third[:2]] == ["a", "r"]
    assert len(first) == 4
    assert len(second) == 4
    assert len(third) == 4
    assert trainer._batch_counts == {
        "tau2_batch/real_prompts": 2,
        "tau2_batch/real_episodes": 2,
        "tau2_batch/physical_prompt_groups": 4,
        "tau2_batch/dispatch_replication_factor": 2,
        "tau2_batch/logical_step": 2,
        "tau2_batch/task_id_start": 4,
    }
    assert trainer._tau2_next_train_step == 3


def test_strict_train_batch_resume_task_ids_start_after_committed_steps():
    class FakeDataLoader:
        def __iter__(self):
            return iter([[{"domain": "airline", "task_id": "a"}]])

    class FakeRollout:
        def __init__(self):
            self.task_ids = []

        def submit(self, *, data, task_id, **kwargs):  # noqa: ARG002
            self.task_ids.append(task_id)

        def wait(self, count):  # noqa: ARG002
            return [{"task_id": "a"}]

    trainer = object.__new__(train_critic.Tau2CriticOnlineTrainer)
    trainer.rollout = FakeRollout()
    trainer.actor = SimpleNamespace(parallel_strategy=SimpleNamespace(dp_size=1))
    trainer.config = SimpleNamespace(
        gconfig=SimpleNamespace(n_samples=1),
        train_dataset=SimpleNamespace(batch_size=16),
    )
    trainer._tau2_train_epoch = 0
    trainer._tau2_train_iterator = None
    trainer._tau2_next_train_step = 5

    trainer._strict_prepare_training_batch(FakeDataLoader(), workflow="wf")

    assert trainer.rollout.task_ids == [80]
    assert trainer._batch_counts["tau2_batch/logical_step"] == 5
    assert trainer._batch_counts["tau2_batch/task_id_start"] == 80


def test_strict_train_batch_rejects_result_order_drift():
    class FakeDataLoader:
        def __iter__(self):
            return iter([[{"domain": "airline", "task_id": "a"}]])

    class FakeRollout:
        def __init__(self):
            self.pending = []

        def submit(self, *, data, **kwargs):  # noqa: ARG002
            self.pending.append({"task_id": "different"})

        def wait(self, count):  # noqa: ARG002
            return self.pending

    trainer = object.__new__(train_critic.Tau2CriticOnlineTrainer)
    trainer.rollout = FakeRollout()
    trainer.actor = SimpleNamespace(parallel_strategy=SimpleNamespace(dp_size=1))
    trainer.config = SimpleNamespace(
        gconfig=SimpleNamespace(n_samples=1),
        train_dataset=SimpleNamespace(batch_size=2),
    )
    trainer._tau2_train_epoch = 0
    trainer._tau2_train_iterator = None
    trainer._tau2_next_train_step = 0

    with pytest.raises(RuntimeError, match="order drifted"):
        trainer._strict_prepare_training_batch(FakeDataLoader(), workflow="wf")


def test_metric_and_probe_targets_use_pre_action_value_alignment():
    row = _critic_row(0)
    row["reward"] = 0.0
    values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

    metric = train_critic._metric_row(row, values)
    target_row = train_critic._value_target_row(row)

    assert metric["n_tokens"] == 2
    assert metric["prediction_mean"] == pytest.approx(20.0)
    torch.testing.assert_close(
        target_row["loss_mask"],
        torch.tensor([[True, False, True, False]]),
        rtol=0,
        atol=0,
    )


def test_plain_mse_zero_mask_padding_has_finite_zero_gradient():
    value = torch.tensor([[1.0, 2.0]], requires_grad=True)
    inputs = {
        "values": torch.zeros_like(value),
        "returns": torch.ones_like(value),
        "loss_mask": torch.zeros_like(value, dtype=torch.bool),
    }

    loss = ppo_loss_fn(value, inputs, eps_clip=None, loss_reduction="token_mean")
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)
    torch.testing.assert_close(value.grad, torch.zeros_like(value), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("n_rows", "dp_size", "expected_factor"),
    [(16, 4, 1), (14, 4, 2), (2, 4, 2)],
)
def test_critic_dp4_dispatch_uniformly_replicates_tail_without_dropping_tasks(
    n_rows, dp_size, expected_factor
):
    rows = [{"task_id": str(index)} for index in range(n_rows)]

    physical, replicas = train_critic.repeat_groups_for_dispatch(rows, dp_size=dp_size)

    assert replicas == expected_factor
    assert len(physical) % dp_size == 0
    assert len(physical) == n_rows * expected_factor
    assert [row["task_id"] for row in physical[:n_rows]] == [
        str(index) for index in range(n_rows)
    ]
