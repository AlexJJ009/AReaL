# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for tau2 offline critic fitting helpers."""

from __future__ import annotations

import json

import pytest
import torch

from scripts.tau2 import train_critic

from areal.api.cli_args import PPOConfig, load_expr_config
from areal.trainer.ppo.critic import ppo_loss_fn


def _domain_for_index(index: int) -> str:
    if index < 30:
        return "airline"
    if index < 104:
        return "retail"
    return "telecom"


def _official_train_ids() -> set[tuple[str, str]]:
    return {(_domain_for_index(index), f"train-{index}") for index in range(178)}


def _critic_row(index: int, critic_split: str) -> dict:
    episode_id = 1000 + index
    return {
        "domain": _domain_for_index(index),
        "task_id": f"train-{index}",
        "split": "train",
        "critic_split": critic_split,
        "episode_id": episode_id,
        "episode_tensor_id": episode_id,
        "attempt_id": f"attempt-{index}",
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "loss_mask": [0, 1, 0, 1],
        "action_origin_mask": [0, 1, 0, 1],
        "behavior_logprobs": [0.0, -0.1, 0.0, -0.2],
        "versions": [-1, 3, -1, 3],
        "turn_ids": [-1, 0, -1, 1],
        "token_roles": ["prompt", "assistant", "tool", "assistant"],
        "reward": float(index % 2),
        "official_score": float(index % 2),
        "terminated": True,
        "truncated": False,
        "bootstrap_mask": False,
        "policy_id": "Qwen/Qwen3.5-4B",
        "policy_revision": "policy0-revision",
        "simulator_id": "deepseek-pinned",
    }


def _write_rows(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _production_rows() -> list[dict]:
    task_rows = [
        {"domain": domain, "task_id": task_id, "split": "train"}
        for domain, task_id in sorted(_official_train_ids())
    ]
    split_rows = train_critic.split_critic_tasks(
        task_rows,
        seed=42,
        dev_fraction=0.2,
    )
    split_by_identity = {
        (str(row["domain"]), str(row["task_id"])): str(row["critic_split"])
        for row in split_rows
    }
    rows = []
    for index in range(178):
        domain = _domain_for_index(index)
        split = split_by_identity[(domain, f"train-{index}")]
        rows.append(_critic_row(index, split))
    return rows


def test_load_episode_rows_accepts_contract_derived_full_coverage_split(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(train_critic, "_load_official_train_ids", _official_train_ids)
    data_path = tmp_path / "episodes.jsonl"
    _write_rows(data_path, _production_rows())

    summary, train_rows, dev_rows = train_critic._load_episode_rows(
        data_path,
        require_full_coverage=True,
        split_seed=42,
        dev_fraction=0.2,
    )

    assert summary["require_full_coverage"] is True
    assert len(train_rows) == 142
    assert len(dev_rows) == 36
    assert summary["episodes_by_split_domain"] == {
        "dev:airline": 6,
        "dev:retail": 15,
        "dev:telecom": 15,
        "train:airline": 24,
        "train:retail": 59,
        "train:telecom": 59,
    }


def test_load_episode_rows_rejects_production_count_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(train_critic, "_load_official_train_ids", _official_train_ids)
    rows = _production_rows()
    first_train = next(
        index for index, row in enumerate(rows) if row["critic_split"] == "train"
    )
    rows[first_train]["critic_split"] = "dev"
    data_path = tmp_path / "episodes.jsonl"
    _write_rows(data_path, rows)

    with pytest.raises(ValueError, match="task-stratified"):
        train_critic._load_episode_rows(
            data_path,
            require_full_coverage=True,
            split_seed=42,
            dev_fraction=0.2,
        )


def test_load_episode_rows_generic_keeps_sample_count_out_of_validator(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(train_critic, "_load_official_train_ids", _official_train_ids)
    rows = [
        _critic_row(0, "train"),
        _critic_row(1, "dev"),
        _critic_row(30, "train"),
        _critic_row(31, "dev"),
        _critic_row(104, "train"),
        _critic_row(105, "dev"),
    ]
    data_path = tmp_path / "episodes.jsonl"
    _write_rows(data_path, rows)

    summary, train_rows, dev_rows = train_critic._load_episode_rows(data_path)

    assert summary["require_full_coverage"] is False
    assert len(train_rows) == 3
    assert len(dev_rows) == 3


def test_production_config_resolves_frozen_training_contract(monkeypatch, tmp_path):
    actor = "Qwen/Qwen3.5-4B@" + "a" * 40
    monkeypatch.setenv("TAU2_ACTOR_PATH", actor)
    monkeypatch.setenv("TAU2_CRITIC_INIT_PATH", actor)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))

    config, _ = load_expr_config(
        ["--config", "examples/tau2/config_critic_production.yaml"],
        PPOConfig,
    )
    train_critic.validate_tau2_critic_config(config, train_rows=142)

    assert config.total_train_steps == 18
    assert config.total_train_epochs == 2
    assert config.train_dataset.batch_size == 16
    assert config.train_dataset.drop_last is False
    assert config.critic is not None
    assert config.critic.freeze_critic_attention is True
    assert config.critic.optimizer.lr == pytest.approx(5.0e-6)
    assert config.critic.eps_clip is None
    assert config.critic.loss_reduction == "token_mean"


def test_metric_and_probe_targets_use_pre_action_value_alignment():
    row = _critic_row(0, "dev")
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


def test_check_config_does_not_resolve_model_snapshot(monkeypatch, tmp_path, capsys):
    actor = "Qwen/Qwen3.5-4B@" + "a" * 40
    monkeypatch.setenv("TAU2_ACTOR_PATH", actor)
    monkeypatch.setenv("TAU2_CRITIC_INIT_PATH", actor)
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setattr(train_critic, "_load_official_train_ids", _official_train_ids)

    rows = [
        _critic_row(0, "train"),
        _critic_row(1, "dev"),
        _critic_row(30, "train"),
        _critic_row(31, "dev"),
        _critic_row(104, "train"),
        _critic_row(105, "dev"),
    ]
    data_path = tmp_path / "episodes.jsonl"
    _write_rows(data_path, rows)

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
            "--episodes",
            str(data_path),
            "total_train_steps=1",
            "num_critic_only_steps=1",
            "total_train_epochs=1",
            "train_dataset.batch_size=3",
            "valid_dataset.batch_size=3",
            "--check-config",
        ]
    )

    captured = capsys.readouterr()
    assert "model_snapshot_resolution: 0" in captured.out


def test_critic_cadence_includes_final_step_between_regular_saves():
    assert train_critic._cadence_steps(18, 5) == (5, 10, 15, 18)
    assert train_critic._cadence_steps(20, 5) == (5, 10, 15, 20)


def test_fsdp_worker_exposes_critic_gradient_diagnostic():
    from types import SimpleNamespace

    from areal.engine.fsdp_engine import FSDPPPOCritic

    engine = object.__new__(FSDPPPOCritic)
    calls = []
    engine.critic = SimpleNamespace(
        grad_norm=lambda data: calls.append(data) or {"grad_norm": 3.0}
    )
    batch = [{"input_ids": [1, 2]}]
    assert engine.grad_norm(batch) == {"grad_norm": 3.0}
    assert calls == [batch]
