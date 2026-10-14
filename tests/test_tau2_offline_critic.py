# SPDX-License-Identifier: Apache-2.0
"""CPU replay checks; FSDP sharding itself requires GPU integration hardware."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from examples.tau2.critic_head import Tau2HeadCriticConfig, Tau2HeadFSDPPPOCritic
from scripts.tau2 import offline_critic_fit as replay
from scripts.tau2.train_critic import _value_target_row
from tests.test_sao_qwen35_critic import (
    _force_qwen35_torch_reference_cpu,  # noqa: F401
    _make_tiny_saved_critic,
)
from tests.test_tau2_critic_head import _build, _patch_cpu_engine

from areal.api.cli_args import OptimizerConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_utils.critic_freeze import apply_critic_attention_freeze
from areal.trainer.ppo.update import require_single_update
from areal.utils import stats_tracker


def _row(reward=1.0):
    return dict(
        input_ids=[1, 2, 3, 4],
        attention_mask=[1, 1, 1, 1],
        loss_mask=[0, 1, 0, 1],
        reward=reward,
        domain="airline",
        task_id="train-task",
    )


def test_replay_targets_match_online_shift_and_do_not_mutate_reward():
    """Replay must supervise precisely the preceding action-prefix positions."""
    row = _row()
    actual = replay.target_row(row)
    expected = _value_target_row(row)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["loss_mask"], torch.tensor([[True, False, True, False]]), rtol=0, atol=0
    )
    overridden = replay.target_row(row, lambda _row: 0.25)
    torch.testing.assert_close(
        overridden["returns"], torch.full((1, 4), 0.25), rtol=0, atol=0
    )
    assert row["reward"] == 1.0
    with pytest.raises(ValueError, match="non-finite"):
        replay.target_row(row, lambda _row: float("nan"))


def test_replay_complete_pass_retains_short_batch_and_padding_has_no_loss():
    """Dataset size need not divide the batch size or DP size."""
    rows = [dict(_row(), episode=i) for i in range(5)]
    batches = list(replay.replay_batches(rows, 2))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [row["episode"] for batch in batches for row in batch] == list(range(5))
    padded, count = replay._pad_for_dp(batches[-1], 2)
    assert count == 1
    assert replay.target_row(padded[0])["loss_mask"].count_nonzero() == 2
    assert replay.target_row(padded[1])["loss_mask"].count_nonzero() == 0


def test_replay_tiny_head_engine_uses_actual_optimizer_scheduler_and_loss(
    tmp_path, monkeypatch
):
    """Unsharded CPU model exercises actual critic optimizer and PPO MSE path."""
    _, save_dir, _, _ = _make_tiny_saved_critic(tmp_path)
    _patch_cpu_engine(monkeypatch)
    config = Tau2HeadCriticConfig(
        experiment_name="test",
        trial_name="offline",
        path=str(save_dir),
        is_critic=True,
        backend="fsdp:d1p1t1",
        dtype="float32",
        optimizer_dtype="float32",
        attn_impl="eager",
        head_bias_init=0.5,
        ppo_n_minibatches=1,
        eps_clip=None,
        head_lr=1e-3,
        freeze_critic_attention=True,
        optimizer=OptimizerConfig(
            lr=2e-4, weight_decay=0.0, warmup_steps=1, lr_scheduler_type="constant"
        ),
    )
    engine = _build(Tau2HeadFSDPPPOCritic(config))
    # initialize() applies this same freeze before FSDP; on CPU we deliberately
    # omit sharding/process groups, without mocking their internals.
    manifest = apply_critic_attention_freeze(engine.model, enabled=True, is_critic=True)
    engine.parallel_helper = SimpleNamespace(tp_enabled=False)
    engine.logger = MagicMock()
    engine._create_optimizer(FinetuneSpec(1, 3, 1))
    assert engine.lr_scheduler.get_last_lr() == [0.0, 0.0]
    batch = replay.target_row(_row())
    initial_head = engine.model.score.weight.detach().clone()
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in engine.model.named_parameters()
        if not parameter.requires_grad
    }
    losses = []

    def cpu_train_batch(data, loss_fn, loss_weight_fn):
        # Real ppo_update chooses the loss and masks. CPU executes its forward,
        # backward and optimizer on the unsharded tiny backbone.
        engine.optimizer.zero_grad()
        output = engine.model(
            input_ids=data["input_ids"], attention_mask=data["attention_mask"]
        )
        values = engine._extract_model_forward_output(output)
        assert loss_weight_fn(data).item() == 2
        loss = loss_fn(values, data)
        losses.append(float(loss.detach()))
        loss.backward()
        assert engine.model.score.weight.grad.abs().sum() > 0
        grad_norm = torch.nn.utils.clip_grad_norm_(engine.model.parameters(), 1.0)
        engine.optimizer.step()
        return {
            "update_successful": 1.0,
            "grad_norm": float(grad_norm),
            "lr": engine.optimizer.param_groups[0]["lr"],
        }

    engine.train_batch = cpu_train_batch
    for _ in range(3):
        report = engine.ppo_update([batch])
        require_single_update(report, "cpu offline smoke")
        assert report["nonfinite"] == 0
        json.dumps(report, allow_nan=False)
        engine.lr_scheduler_step()
    assert not torch.equal(initial_head, engine.model.score.weight.detach())
    assert losses[-1] < losses[0]
    assert engine.lr_scheduler.get_last_lr() == pytest.approx([2e-4, 1e-3])
    assert manifest["frozen_numel"] > 0
    for name, parameter in engine.model.named_parameters():
        if name in frozen:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    # The path would fail with an unshifted mask or wrong returns.
    stats_tracker.get().stats.clear()


def test_label_override_import_requires_callable():
    """Import hooks fail before an engine is allocated."""
    with pytest.raises(ValueError, match="module:function"):
        replay.import_label_override("missing-separator")
    with pytest.raises(ValueError, match="callable"):
        replay.import_label_override("scripts.tau2.offline_critic_fit:__doc__")


def test_check_config_does_not_create_engine_or_output(tmp_path, monkeypatch):
    """A check reads tensors/config only and leaves source/output paths alone."""
    from scripts.tau2 import critic_features

    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path / "source-run"))
    train_path = tmp_path / "0.jsonl"
    train_path.write_text(json.dumps(_row()) + "\n")
    validation_path = tmp_path / "fixed.pt"
    monkeypatch.setattr(
        critic_features,
        "load_stored_rollouts",
        lambda _path: [dict(_row(0.0), task_id="test-task")]
        if str(_path) == str(validation_path)
        else [
            dict(_row(), _episode_order=0, _source_path=str(train_path), _source_row=0)
        ],
    )

    def fail_engine(*_args, **_kwargs):
        raise AssertionError("check-config constructed an engine")

    monkeypatch.setattr(replay, "Tau2HeadFSDPPPOCritic", fail_engine)
    output = tmp_path / "replay-output"
    result = replay.main(
        [
            "--config",
            "examples/tau2/config_critic_production.yaml",
            "critic_head_bias_init=0.5",
            "--train-rollouts",
            str(train_path),
            "--validation-rollouts",
            str(validation_path),
            "--output-dir",
            str(output),
            "--check-config",
            "--backbone-lr",
            "0",
            "--head-lr",
            "0.001",
            "--label-override",
            "scripts.tau2.offline_critic_fit:first_token_parity",
            "--validation-label-override",
            "scripts.tau2.offline_critic_fit:first_token_parity",
        ]
    )
    assert result["total_train_steps"] == 1
    assert result["validation_targets"] == "synthetic_diagnostic"
    assert result["critic_config"]["freeze_critic_attention"] is True
    assert result["critic_config"]["optimizer"]["lr"] == 0
    assert not output.exists()
    assert not (tmp_path / "source-run").exists()


def test_probe_head_rejects_logistic_and_mismatched_source(tmp_path):
    """An exported linear head must refer to the same backbone and objective."""
    path = tmp_path / "head.pt"
    artifact = {
        "base_checkpoint": "base",
        "requires_sigmoid": True,
        "score.weight": torch.ones(1, 3),
        "score.bias": torch.zeros(1),
    }
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="Logistic"):
        replay.validate_probe_artifact(str(path), "base")
    artifact["requires_sigmoid"] = False
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="checkpoint"):
        replay.validate_probe_artifact(str(path), "different")
    state = replay.validate_probe_artifact(str(path), "base")
    assert state["score.weight"].shape == (1, 3)


def test_validation_adapter_partitions_and_gathers_original_order(monkeypatch):
    """Each DP worker sees only its share; evaluator receives original row order."""
    observed = []
    engine = SimpleNamespace(
        parallel_helper=SimpleNamespace(dp_size=2),
        dp_group="dp",
        cpu_group="cpu",
        compute_values=lambda data: observed.extend(data)
        or [torch.tensor([2.0]), torch.tensor([3.0])],
        grad_norm=lambda data: {"grad_norm": float(sum(data))},
    )
    monkeypatch.setattr(replay.dist, "get_rank", lambda _group: 1)

    def gather(output, values, group):
        assert group == "cpu"
        output[:] = [[torch.tensor([0.0]), torch.tensor([1.0])], values]

    monkeypatch.setattr(replay.dist, "all_gather_object", gather)
    adapter = replay.LocalEvaluationCritic(engine)
    values = adapter.compute_values([0, 1, 2, 3])
    assert observed == [2, 3]
    torch.testing.assert_close(torch.cat(values), torch.arange(4.0), rtol=0, atol=0)
    assert adapter.grad_norm([0, 1, 2, 3]) == {"grad_norm": 5.0}
    with pytest.raises(ValueError, match="padded"):
        adapter.compute_values([0, 1, 2])
