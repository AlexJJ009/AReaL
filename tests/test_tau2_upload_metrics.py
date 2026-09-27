# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/tau2/upload_metrics.py"
SPEC = importlib.util.spec_from_file_location("tau2_upload_metrics", MODULE_PATH)
upload_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(upload_metrics)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_grpo_log_parser_uses_committed_train_step_boundaries(tmp_path):
    log = tmp_path / "logs/root/tau2-grpo/run/main.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            [
                "│ rollout/reward │ 9.9990e+00 │",
                "StatsLogger INFO: Epoch 1/2 Step 1/23 Train step 1/46 done.",
                "╒════════════════════════════════════╤════════════╕",
                "│ rollout/reward                    │ 5.0000e-01 │",
                "├────────────────────────────────────┼────────────┤",
                "│ ppo_actor/n_seqs                  │ 6.4000e+01 │",
                "├────────────────────────────────────┼────────────┤",
                "│ ppo_actor/update/update_successful│ 1.0000e+00 │",
                "├────────────────────────────────────┼────────────┤",
                "│ ppo_actor/update/grad_norm        │ 2.0000e+00 │",
                "╘════════════════════════════════════╧════════════╛",
                "│ rollout/reward │ 1.0000e+00 │",
                "StatsLogger INFO: Epoch 1/2 Step 2/23 Train step 2/46 done.",
                "╒════════════════════════════════════╤════════════╕",
                "│ rollout/reward                    │ 7.5000e-01 │",
                "├────────────────────────────────────┼────────────┤",
                "│ ppo_actor/n_seqs                  │ 3.2000e+01 │",
                "├────────────────────────────────────┼────────────┤",
                "│ ppo_actor/update/update_successful│ 1.0000e+00 │",
                "├────────────────────────────────────┼────────────┤",
                "│ ppo_actor/update/grad_norm        │ 3.0000e+00 │",
                "╘════════════════════════════════════╧════════════╛",
            ]
        ),
        encoding="utf-8",
    )

    records = upload_metrics.load_grpo_train_metrics(tmp_path)

    assert [r["train/completed_step"] for r in records] == [1, 2]
    assert records[0]["rollout/reward"] == 0.5
    assert records[0]["ppo_actor/n_seqs"] == 64
    assert records[0]["ppo_actor/update/update_successful"] == 1
    assert records[0]["ppo_actor/update/grad_norm"] == 2
    assert records[1]["rollout/reward"] == 0.75
    assert records[1]["ppo_actor/n_seqs"] == 32
    assert records[1]["ppo_actor/update/update_successful"] == 1
    assert records[1]["ppo_actor/update/grad_norm"] == 3


def test_grpo_log_parser_rejects_conflicting_duplicate_steps(tmp_path):
    log = tmp_path / "logs/root/tau2-grpo/run/main.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            [
                "StatsLogger INFO: Train step 1/46 done.",
                "│ rollout/reward │ 5.0000e-01 │",
                "╘════════════════╧════════════╛",
                "StatsLogger INFO: Train step 1/46 done.",
                "│ rollout/reward │ 7.5000e-01 │",
                "╘════════════════╧════════════╛",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        upload_metrics.load_grpo_train_metrics(tmp_path)


def test_grpo_log_parser_rejects_partial_table_and_nonfinite_values(tmp_path):
    log = tmp_path / "logs/root/tau2-grpo/run/main.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            [
                "StatsLogger INFO: Train step 1/46 done.",
                "│ rollout/reward │ nan │",
                "╘════════════════╧═════╛",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Non-finite"):
        upload_metrics.load_grpo_train_metrics(tmp_path)

    log.write_text(
        "\n".join(
            [
                "StatsLogger INFO: Train step 1/46 done.",
                "│ rollout/reward │ 5.0000e-01 │",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no table footer"):
        upload_metrics.load_grpo_train_metrics(tmp_path)


def test_critic_loader_splits_train_and_validation_axes(tmp_path):
    evidence = tmp_path / "tau2-critic-fit-evidence"
    _write_json(
        evidence / "train-metrics/step-000001.json",
        {
            "completed_step": 1,
            "metrics": {
                "critic/ppo_critic/critic_mse": 1.25,
                "tau2_critic_validation/overall_mse": 99.0,
                "critic/ppo_critic/critic_explained_variance": None,
                "critic/ppo_critic/critic_explained_variance_defined": 0.0,
                "critic/ppo_critic/lr__count": 4,
            },
        },
    )
    validation = {
        "completed_step": 0,
        "summary": {
            "mse": 2.0,
            "macro_mse": 2.1,
            "explained_variance": None,
            "explained_variance_defined": False,
        },
        "grad_norm": {
            "overall": 3.0,
            "airline": 3.1,
            "retail": 3.2,
            "telecom": 3.3,
        },
        "by_domain": {
            domain: {
                "mse": 4.0 + index,
                "explained_variance": None,
                "explained_variance_defined": False,
            }
            for index, domain in enumerate(upload_metrics.DOMAINS)
        },
    }
    _write_json(evidence / "validation-step-000000.json", validation)

    train, eval_ = upload_metrics.load_critic_metrics(tmp_path)

    assert train[0]["train/completed_step"] == 1
    assert train[0]["critic/ppo_critic/critic_mse"] == 1.25
    assert "tau2_critic_validation/overall_mse" not in train[0]
    assert "critic/ppo_critic/lr__count" not in train[0]
    assert "critic/ppo_critic/critic_explained_variance" not in train[0]
    assert train[0]["critic/ppo_critic/critic_explained_variance_defined"] == 0.0
    assert eval_[0]["eval/completed_step"] == 0
    assert eval_[0]["critic_validation/overall_mse"] == 2.0
    assert "critic_validation/overall_explained_variance" not in eval_[0]
    assert eval_[0]["critic_validation/overall_explained_variance_defined"] == 0.0


def test_upload_once_resumes_from_remote_summary(tmp_path):
    evidence = tmp_path / "tau2-critic-fit-evidence"
    for step in [1, 2]:
        _write_json(
            evidence / f"train-metrics/step-{step:06d}.json",
            {"completed_step": step, "metrics": {"rollout/reward": float(step)}},
        )
    for step in [0, 2]:
        _write_json(
            evidence / f"validation-step-{step:06d}.json",
            {
                "completed_step": step,
                "summary": {
                    "mse": float(step),
                    "macro_mse": float(step),
                    "explained_variance": None,
                    "explained_variance_defined": False,
                },
                "grad_norm": {
                    "overall": 1.0,
                    "airline": 1.0,
                    "retail": 1.0,
                    "telecom": 1.0,
                },
                "by_domain": {
                    domain: {
                        "mse": 1.0,
                        "explained_variance": None,
                        "explained_variance_defined": False,
                    }
                    for domain in upload_metrics.DOMAINS
                },
            },
        )
    run = SimpleNamespace(
        summary={
            "tau2_upload/critic/last_train_completed_step": 1,
            "tau2_upload/critic/last_eval_completed_step": 0,
        },
        logged=[],
    )
    run.log = lambda payload: run.logged.append(payload)
    args = SimpleNamespace(kind="critic", run_root=tmp_path)

    uploaded, max_train, max_eval, discovered = upload_metrics.upload_once(args, run)

    assert uploaded == 2
    assert max_train == 2
    assert max_eval == 2
    assert discovered == 4
    assert [record.get("train/completed_step") for record in run.logged] == [2, None]
    assert [record.get("eval/completed_step") for record in run.logged] == [None, 2]
    assert run.summary["tau2_upload/critic/last_train_completed_step"] == 2
    assert run.summary["tau2_upload/critic/last_eval_completed_step"] == 2


def test_grpo_recovery_discards_updates_not_in_restored_checkpoint(tmp_path):
    log = tmp_path / "logs/root/tau2-grpo/run/main.log"
    log.parent.mkdir(parents=True)

    def block(step, reward):
        return (
            f"Train step {step}/4 done.\n"
            "╒═════════╤════════╕\n"
            f"│ rollout/reward │ {reward} │\n"
            "╘═════════╧════════╛\n"
        )

    log.write_text(
        block(1, 0.5)
        + block(2, 0.25)
        + "Recovering from StepInfo(epoch=0, epoch_step=1, global_step=1, steps_per_epoch=4).\n"
        + block(2, 0.75)
    )
    rows = upload_metrics.load_grpo_train_metrics(tmp_path)
    assert [row["train/completed_step"] for row in rows] == [1, 2]
    assert rows[1]["rollout/reward"] == 0.75


def test_offline_eval_keeps_all_trials_and_rejects_duplicate_cells(tmp_path):
    _write_json(tmp_path / "manifest.json", {"repeats": 2, "planned_episodes": 200})
    _write_json(tmp_path / "summary.json", {"status": "complete"})
    episodes = [
        {
            "trial": trial,
            "domain": domain,
            "task_id": str(task),
            "status": "completed",
            "official_score": float(trial),
        }
        for trial in range(2)
        for domain, count in (("airline", 20), ("retail", 40), ("telecom", 40))
        for task in range(count)
    ]
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in episodes))
    run = SimpleNamespace(summary={}, log=lambda row: None)
    args = SimpleNamespace(kind="offline_eval", run_root=tmp_path)
    upload_metrics.upload_once(args, run)
    assert run.summary["offline_eval/reward_mean"] == 0.5
    assert run.summary["offline_eval/repeats"] == 2
    episodes[-1] = episodes[0]
    path.write_text("\n".join(json.dumps(row) for row in episodes))
    with pytest.raises(ValueError, match="Duplicate"):
        upload_metrics.load_offline_eval_metrics(tmp_path)


def test_offline_eval_refuses_partial_run(tmp_path):
    _write_json(tmp_path / "manifest.json", {"repeats": 8})
    _write_json(tmp_path / "summary.json", {"status": "partial"})
    with pytest.raises(ValueError, match="incomplete"):
        upload_metrics.load_offline_eval_metrics(tmp_path)
