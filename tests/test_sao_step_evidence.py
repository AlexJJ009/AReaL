# SPDX-License-Identifier: Apache-2.0
"""Supervision rejects skipped updates, stale counters, and invalid gradients."""

from types import SimpleNamespace

import pytest
import torch

from examples.math.sao_ppo import (
    check_step_metrics,
    install_audit_hooks,
    source_id_digest,
    source_key_digest,
    verify_published_policy,
    weight_only_start_step_from_env,
)

from areal.engine.sglang_remote import SGLangBackend


def _metrics(count=3):
    return {
        "grad_norm": 0.2,
        "update_successful": 1,
        "optimizer_steps_since_init": count,
        "critic/grad_norm": 0.4,
        "critic/update_successful": 1,
        "critic/optimizer_steps_since_init": count,
        "ppo_loss": -0.1,
        "critic/value_loss": 0.3,
    }


@pytest.mark.parametrize("role", ["", "critic/"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("grad_norm", 0),
        ("grad_norm", float("nan")),
        ("update_successful", 0),
        ("optimizer_steps_since_init", 2),
    ],
)
def test_joint_update_rejects_invalid_role(role, key, value):
    metrics = _metrics()
    metrics[role + key] = value
    with pytest.raises(RuntimeError):
        check_step_metrics(metrics, 3)


def test_recovered_process_counts_updates_relative_to_init():
    assert set(check_step_metrics(_metrics(1), 21, updates_since_init=1)) == {
        "actor",
        "critic",
    }
    with pytest.raises(RuntimeError):
        check_step_metrics(_metrics(1), 21)


def test_weight_only_start_step_env_validation(monkeypatch):
    monkeypatch.delenv("SAO_WEIGHT_ONLY_START_STEP", raising=False)
    assert weight_only_start_step_from_env() == 0
    monkeypatch.setenv("SAO_WEIGHT_ONLY_START_STEP", "50")
    assert weight_only_start_step_from_env() == 50
    monkeypatch.setenv("SAO_WEIGHT_ONLY_START_STEP", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        weight_only_start_step_from_env()


def test_weight_only_source_digests_are_order_sensitive():
    ids = ["gsm8k/train/2", "math/train/4"]
    assert source_id_digest(ids) != source_id_digest(list(reversed(ids)))
    assert source_key_digest(ids) != source_key_digest(list(reversed(ids)))


def test_weight_only_hook_forces_first_continuation_checkpoint(tmp_path):
    calls = []
    trainer = SimpleNamespace()
    trainer.recover_info = None
    trainer.actor = SimpleNamespace(
        prepare_batch=lambda *a, **k: [{"audit_source_key": torch.tensor([1])}],
        compute_advantages=lambda *a, **k: a[0],
        ppo_update=lambda *a, **k: None,
    )
    trainer.critic = SimpleNamespace(ppo_update=lambda *a, **k: None)
    trainer._save_training_state = lambda **kwargs: calls.append(kwargs)
    trainer.stats_logger = SimpleNamespace(commit=lambda *a, **k: None)
    trainer._save_perf_tracer = lambda step: None
    trainer.rollout = SimpleNamespace(pause=lambda: None, resume=lambda: None)

    install_audit_hooks(trainer, tmp_path, initial_step=50)
    trainer._save_training_state(epoch=0, epoch_step=50, global_step=50)
    trainer._save_training_state(epoch=0, epoch_step=51, global_step=51)

    assert calls[0]["force"] is True
    assert calls[1]["force"] is False


def test_recovered_weight_only_hook_uses_persisted_optimizer_base(tmp_path):
    commits = []
    trainer = SimpleNamespace()
    trainer.recover_info = SimpleNamespace(
        last_step_info=SimpleNamespace(
            next=lambda: SimpleNamespace(global_step=51),
        ),
        trainer_state={"optimizer_steps_base": 50},
    )
    trainer.actor = SimpleNamespace(
        prepare_batch=lambda *a, **k: [],
        compute_advantages=lambda *a, **k: a[0],
        ppo_update=lambda *a, **k: None,
    )
    trainer.critic = SimpleNamespace(ppo_update=lambda *a, **k: None)
    trainer._save_training_state = lambda **kwargs: None
    trainer.stats_logger = SimpleNamespace(
        commit=lambda *args: commits.append(args),
    )
    trainer._save_perf_tracer = lambda step: None
    trainer.rollout = SimpleNamespace(
        pause=lambda: None,
        resume=lambda: None,
        get_version=lambda: 101,
    )

    install_audit_hooks(trainer, tmp_path)
    metrics = _metrics(count=51)
    metrics["ppo_actor/explicit_termination"] = 1
    trainer.stats_logger.commit(0, 100, 100, metrics)

    assert commits


def test_joint_update_rejects_nonfinite_loss():
    metrics = _metrics()
    metrics["critic/value_loss"] = float("inf")
    with pytest.raises(RuntimeError):
        check_step_metrics(metrics, 3)


def test_publication_probe_aligns_next_token_with_scored_token(tmp_path):
    """An off-by-one comparison must not pass even with identical token IDs."""
    ids = list(range(12))

    def actor_logp(probes):
        return [torch.arange(1, 13).float().unsqueeze(0) for _ in probes]

    def rollout_logp(probes):
        for probe in probes:
            assert probe["loss_mask"].sum() == 4
            assert not probe["loss_mask"][0, 0]
        return [torch.arange(12).float().unsqueeze(0) for _ in probes]

    trainer = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda *a, **k: ids),
        actor=SimpleNamespace(compute_logp=actor_logp),
        rollout=SimpleNamespace(compute_logp=rollout_logp, get_version=lambda: 1),
    )
    verify_published_policy(trainer, tmp_path, 1)
    assert (tmp_path / "published-policy-1.json").is_file()


def test_sglang_scoring_ignores_only_unrequested_undefined_prefix():
    backend = SGLangBackend()
    response = {"meta_info": {"input_token_logprobs": [[None, 1], [-0.5, 2]]}}
    assert backend.parse_score_response(response, 1) == [-0.5]
    with pytest.raises(ValueError, match="undefined"):
        backend.parse_score_response(response, 2)
