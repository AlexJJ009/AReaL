# SPDX-License-Identifier: Apache-2.0
"""Independent CPU acceptance checks for the frozen offline preexperiment."""

from __future__ import annotations

import hashlib
import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.tau2 import critic_head_sim as sim
from scripts.tau2 import critic_preexp as preexp
from scripts.tau2.critic_preexp_stats import TaskBootstrap


def test_grouped_folds_repeated_tasks_never_cross_folds():
    """Run and trajectory multiplicity cannot split the same task across folds."""
    tasks = [("airline", str(i)) for i in range(15)]
    repeated = [task for task in tasks for _ in range(6 if int(task[1]) < 5 else 2)]

    folds = preexp.grouped_folds(repeated)

    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
    for task in tasks:
        assert (
            len({int(folds[i]) for i, key in enumerate(repeated) if key == task}) == 1
        )
    torch.testing.assert_close(folds, preexp.grouped_folds(repeated), rtol=0, atol=0)


def test_soft_labels_two_and_six_trajectory_tasks_use_episode_means():
    """Two and six observations have literal means; token counts do not reweight."""
    tasks = [("airline", "two")] * 2 + [("airline", "six")] * 6
    labels = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    actual = preexp.soft_labels(labels, tasks)

    torch.testing.assert_close(
        actual, torch.tensor([0.5] * 2 + [2 / 3] * 6), rtol=1e-6, atol=1e-7
    )
    assert set(preexp.soft_labels(labels[:2], tasks[:2]).tolist()) <= {0.0, 0.5, 1.0}


def test_e3_grid_matches_all_twenty_registered_arms():
    """The frozen protocol requires every lr, bias and warmup combination."""
    expected = set(
        itertools.product((5e-6, 2e-5, 1e-4, 3e-4, 1e-3), (False, True), (0, 10))
    )
    arms = preexp.e3_grid()

    assert len(arms) == 20
    assert {
        (arm["lr"], arm["frozen_bias"], arm["warmup_steps"]) for arm in arms
    } == expected


@pytest.mark.parametrize("batch,steps", [(16, 80), (64, 20), (128, 10)])
def test_e4_budget_equal_samples_and_equal_steps_are_distinct(batch, steps):
    """A consumes 1280 episodes; B spends eighty updates for every batch size."""
    assert preexp.e4_budgets(batch) == {"A": steps, "B": 80}


def test_planted_label_training_prevalence_rejects_outside_registered_range():
    """A readable signal outside 20–60% cannot silently be called protocol E8."""
    assert preexp.validate_planted_prevalence(
        torch.tensor([0.0, 1.0]), torch.ones(2)
    ) == pytest.approx(0.5)
    for labels in (torch.zeros(5), torch.ones(5)):
        with pytest.raises(ValueError, match="20|60|prevalence"):
            preexp.validate_planted_prevalence(labels, torch.ones(5))


def _metric_rows():
    return [
        sim.official_metric_row(
            torch.tensor(predictions),
            {
                "reward": label,
                "task_id": task,
                "domain": domain,
                "run": run,
                "episode_id": f"{run}-{domain}-{task}",
            },
        )
        for run in ("r499", "r510")
        for task, domain, label, predictions in (
            ("a", "airline", 0, [0.1, 0.3]),
            ("b", "airline", 1, [0.7]),
            ("c", "retail", 0, [0.2]),
            ("d", "retail", 1, [0.6, 0.8, 0.9]),
        )
    ]


def test_paired_task_bootstrap_same_predictions_zero_and_shared_twenty_thousand_draws():
    """Two runs of each task share draws; pairing a model with itself is exact."""
    rows = _metric_rows()
    bootstrap = TaskBootstrap(rows, require_60=False)

    report = bootstrap.report(rows, reference=rows)

    assert bootstrap.counts.shape == (20000, 4)
    torch.testing.assert_close(
        bootstrap.counts.sum(1),
        torch.full((20000,), 4.0, dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    # Build the oracle from four literal task clusters, independent of the
    # implementation's row grouping; eight independent episodes is incorrect.
    sampled = torch.randint(
        4, (20000, 4), generator=torch.Generator().manual_seed(20261014)
    )
    expected_counts = torch.zeros(20000, 4, dtype=torch.float64)
    expected_counts.scatter_add_(
        1, sampled, torch.ones_like(sampled, dtype=torch.float64)
    )
    torch.testing.assert_close(bootstrap.counts, expected_counts, rtol=0, atol=0)
    assert report["draws"] == 20000
    assert report["tasks_per_draw"] == 4
    for result in report["by_domain"].values():
        assert result["ev_95ci"] == [0.0, 0.0]


def test_paired_task_bootstrap_changed_targets_fails():
    """Task IDs alone cannot pair a planted-target model with reward targets."""
    rows = _metric_rows()
    bootstrap = TaskBootstrap(rows, require_60=False)
    modified = [dict(row) for row in rows]
    modified[0]["target"] = 1.0

    with pytest.raises(ValueError, match="targets"):
        bootstrap.report(modified)


def test_task_bootstrap_requires_registered_sixty_tasks():
    """A production cache missing tasks must fail, even if every row is valid."""
    with pytest.raises(ValueError, match="60 validation tasks"):
        TaskBootstrap(_metric_rows())


@pytest.mark.parametrize(
    "norm,ev,status",
    [
        (0.0012, 0.1, "成立"),
        (0.005, -0.1, "成立"),
        (0.001199, 0, "不成立"),
        (0.005001, 0, "不成立"),
        (0.0024, 0.10001, "不成立"),
    ],
)
def test_h2_exact_registered_norm_and_ev_boundaries(norm, ev, status):
    """The registered interval is absolute, not a ratio or a generic near-zero EV."""
    assert preexp.h2_status(norm, ev, 0.0)["status"] == status


def test_h2_wrong_reference_step_or_missing_ev_cannot_claim_reproduction():
    """The same eighty-step real value is required for the H2 gate."""
    for kwargs in (
        {"reference_step": 55},
        {"reference_ev": None},
        {"sim_airline_ev": None},
    ):
        arguments = dict(norm=0.0024, sim_airline_ev=0.12, reference_ev=0.1)
        arguments.update(kwargs)
        with pytest.raises(ValueError, match="step 80"):
            preexp.h2_status(**arguments)


def test_gradient_summary_literal_cosines_and_sign_flips():
    """Three literal gradients distinguish sign consistency from majority rate."""
    gradients = torch.tensor([[1.0, 2.0, 0.0], [1.0, -2.0, 0.0], [-1.0, 2.0, 0.0]])
    full = torch.tensor([[1.0, 0.0, 0.0]] * 3)

    report = preexp.gradient_summary(gradients, full)

    assert report["adjacent_cosines"] == pytest.approx([-3 / 5, -1])
    assert report["full_data_cosines"] == pytest.approx(
        [1 / 5**0.5, 1 / 5**0.5, -1 / 5**0.5]
    )
    assert report["coordinate_sign_consistency"] == pytest.approx([1 / 3, 1 / 3, 0])
    assert report["coordinate_majority_sign_fraction"] == pytest.approx(
        [2 / 3, 2 / 3, 0.5]
    )
    assert report["fraction_coordinates_same_sign_all_steps"] == 0


@pytest.mark.parametrize(
    "progress,expected",
    [
        (0, "0-20%"),
        (0.1999, "0-20%"),
        (0.2, "20-40%"),
        (0.4, "40-60%"),
        (0.6, "60-80%"),
        (0.8, "80-100%"),
        (1.0, "80-100%"),
    ],
)
def test_position_group_boundary_prefixes_belong_to_exactly_one_bin(progress, expected):
    """The final token remains in 80–100%; exact interior boundaries move right."""
    assert preexp.position_group(progress) == expected
    assert preexp.position_group(progress, first=True) == "first_action_token"


def _feature_cache(tmp_path, *, validation_reward=0.0):
    records = []
    pool = []
    for split, count in (("train", 10), ("validation", 60)):
        for task in range(count):
            domain = "airline" if task % 2 else "retail"
            for episode in range(2):
                identity = f"{split}-{task}-{episode}"
                reward = float(episode) if split == "train" else validation_reward
                pool.append({"_trajectory_id": identity})
                for token in range(1 + episode):
                    records.append(
                        dict(
                            split=split,
                            domain=domain,
                            task_id=f"{split}-task-{task}",
                            trajectory_id=identity,
                            reward=reward,
                            run="r499" if episode == 0 else "r510",
                            step=task,
                            order=0,
                            original_action_count=1 + episode,
                            progress=float(token) / max(1, episode),
                            turn=token,
                            planted_label=float(token),
                            source_id=identity,
                            sampling_weight=1.0,
                            position=token,
                            features=[float(token), float(task)],
                        )
                    )
    shard = {key: [row[key] for row in records] for key in records[0]}
    shard["features"] = torch.tensor(shard["features"])
    shard["reward"] = torch.tensor(shard["reward"])
    shard["sampling_weight"] = torch.tensor(shard["sampling_weight"])
    torch.save(shard, tmp_path / "shard.pt")
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool))
    manifest = dict(
        synthetic=True,
        layer="final",
        schema="tau2-critic-features-v1",
        hidden_size=2,
        checkpoint=str(tmp_path / "base"),
        feature_semantics="score_input_final_norm_shifted_action_prefix",
        pool_manifest_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        shards=[dict(path="shard.pt", rows=len(records))],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, pool_path, pool


def test_cv_construction_validation_sentinels_never_enter_training(
    tmp_path, monkeypatch
):
    """Actual cache filtering and fitting exclude validation IDs and sentinel labels."""
    from scripts.tau2 import critic_preexp_data, critic_preexp_extract

    path, pool_path, pool = _feature_cache(tmp_path, validation_reward=12345.0)
    # Integrity checks have dedicated tests; this test isolates actual training
    # filtering and grouped CV rather than repeating the source manifest fixture.
    monkeypatch.setattr(
        critic_preexp_data, "load_pool", lambda *a, **k: ({"synthetic": True}, pool)
    )
    monkeypatch.setattr(critic_preexp_extract, "verify_cache", lambda *a, **k: None)
    experiment = preexp.Preexperiment(
        path, tmp_path / "out", pool_manifest=pool_path, synthetic=True
    )

    assert len(experiment.x) == 30
    assert set(experiment.y.tolist()) == {0.0, 1.0}
    assert all(identity.startswith("train-") for identity in experiment.identities)
    assert {task for _, task in experiment.tasks} == {
        f"train-task-{i}" for i in range(10)
    }
    assert set(experiment.folds.tolist()) == {0, 1, 2, 3, 4}
    model, selection = experiment.cv_probe()
    assert len(selection["candidates"]) == 6
    assert all(
        len(candidate["fold_mse"]) == 15 for candidate in selection["candidates"]
    )
    assert all(len(candidate["seed_mse"]) == 3 for candidate in selection["candidates"])
    assert torch.isfinite(model[0]).all()


def test_planted_evaluation_uses_each_validation_prefix_label(tmp_path):
    """All rewards are zero but the perfect prefix predictor has planted EV one."""
    path, _, _ = _feature_cache(tmp_path, validation_reward=0.0)
    store = preexp.PreexpStore(path)
    models = {"shared": (torch.tensor([1.0, 0.0]), torch.tensor(0.0))}

    planted = store.evaluate_targets(models, planted=True)
    reward = store.evaluate_targets(models)

    assert planted["summary"]["explained_variance"] == pytest.approx(1.0)
    assert reward["summary"]["explained_variance"] is None
    assert planted["summary"]["n_tokens"] == 180
    two_prefix_row = next(
        row for row in planted["rows"] if row["episode_id"] == "validation-0-1"
    )
    assert two_prefix_row["target"] == 0.5
    assert two_prefix_row["target_sum"] == 1.0
    assert two_prefix_row["target2_sum"] == 1.0


def test_e9_export_exact_weights_bias_and_real_consumer_readback(tmp_path):
    """An exported nonzero intercept survives both the fit loader and SAO checker."""
    from safetensors.torch import save_file

    from scripts.sao.check_critic_head import check_critic_head
    from scripts.tau2.offline_critic_fit import validate_probe_artifact

    path, _, _ = _feature_cache(tmp_path)
    store = preexp.PreexpStore(path)
    weight, bias = torch.tensor([0.123, -0.456]), torch.tensor(0.789)
    artifact = preexp.export_head(store, weight, bias, tmp_path / "head.pt", 0.01)

    state = validate_probe_artifact(str(artifact), store.manifest["checkpoint"])
    torch.testing.assert_close(state["score.weight"], weight[None], rtol=0, atol=0)
    torch.testing.assert_close(state["score.bias"], bias.reshape(1), rtol=0, atol=0)
    consumer_checkpoint = tmp_path / "consumer"
    consumer_checkpoint.mkdir()
    (consumer_checkpoint / "config.json").write_text(
        json.dumps({"hidden_size": 2, "areal_value_head_bias": True})
    )
    save_file(state, consumer_checkpoint / "model.safetensors")
    report = check_critic_head(consumer_checkpoint)
    assert report["passed"] is True
    assert report["head"]["bias"]["value"] == pytest.approx(0.789)


def _replay_experiment():
    experiment = preexp.Preexperiment.__new__(preexp.Preexperiment)
    experiment.x = (
        torch.arange(1280, dtype=torch.float32).remainder(2).mul(2).sub(1)[:, None]
    )
    experiment.y = torch.arange(1280, dtype=torch.float32).remainder(2)
    experiment.mass = torch.ones(1280)
    experiment.indices = {str(i): [i] for i in range(1280)}
    experiment.store = SimpleNamespace(
        episodes=[
            dict(trajectory_id=str(i), run="r510", step=i // 16, order=i % 16)
            for i in range(1280)
        ]
    )
    return experiment


def test_e3_actual_adam_replay_eighty_steps_frozen_bias_and_warmup():
    """CPU replay exercises Adam, clipping and warmup consumption order."""
    experiment = _replay_experiment()

    result, (weight, bias) = experiment.simulate(
        lr=1e-4,
        frozen_bias=True,
        warmup_steps=10,
        replay=True,
        evaluate=False,
        diagnostics=True,
    )

    assert result["total_steps"] == 80
    assert result["episodes_seen"] == 1280
    assert len(result["steps"]) == 80
    assert all(step["episodes"] == 16 for step in result["steps"])
    assert result["steps"][0]["lr"] == 0
    assert result["steps"][0]["weight_norm"] == 0
    assert result["steps"][1]["lr"] == pytest.approx(1e-5)
    assert result["steps"][10]["lr"] == 1e-4
    assert float(bias) == pytest.approx(0.78)
    assert float(weight.norm()) > 0
    assert len(result["gradient"]["adjacent_cosines"]) == 79
    assert result["gradient"]["coordinate_sign_consistency"] == [1.0]


def test_e5_trainable_bias_is_in_primary_gradient_and_weight_only_is_retained():
    """All eighty gradients include the actual nonzero bias derivative."""
    experiment = _replay_experiment()

    result, _ = experiment.simulate(
        lr=5e-6, replay=True, evaluate=False, diagnostics=True
    )

    assert len(result["gradient"]["coordinate_sign_consistency"]) == 2
    assert len(result["gradient_weight_only"]["coordinate_sign_consistency"]) == 1
    assert len(result["gradient"]["adjacent_cosines"]) == 79
    assert len(result["bias_gradients"]) == 80
    assert result["bias_gradients"][0] == pytest.approx(0.56, abs=1e-6)
    assert result["gradient"]["full_data_cosines"] == pytest.approx(
        [1.0] * 80, abs=1e-6
    )


def test_e8_constant_domain_targets_keep_undefined_ev_in_seed_summary(tmp_path):
    """A constant planted domain stays visible without inventing a numeric EV."""
    path, _, _ = _feature_cache(tmp_path)
    shard_path = tmp_path / "shard.pt"
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    shard["planted_label"] = [
        0.0 if domain == "retail" else label
        for domain, label in zip(shard["domain"], shard["planted_label"], strict=True)
    ]
    torch.save(shard, shard_path)
    store = preexp.PreexpStore(path)

    validation = store.evaluate_targets(
        {"shared": (torch.tensor([1.0, 0.0]), torch.tensor(0.0))}, planted=True
    )
    summary = preexp.summarize_validations([validation] * 3)

    assert validation["by_domain"]["retail"]["explained_variance"] is None
    assert summary["retail"]["mean_ev"] is None
    assert summary["retail"]["range_ev"] is None
    assert summary["retail"]["defined_ev_seeds"] == 0
    assert summary["retail"]["seeds"] == 3
    assert "variance is zero" in summary["retail"]["undefined_reason"]
    assert summary["airline"]["mean_ev"] == pytest.approx(1.0)
    json.dumps(summary, allow_nan=False)


def test_e4_runner_all_budgets_and_budget_b_training_ev():
    """Reporting B computes training EV from predictions rather than validation EV."""
    experiment = preexp.Preexperiment.__new__(preexp.Preexperiment)
    experiment.x = torch.tensor([[0.0], [1.0]])
    experiment.y = torch.tensor([0.0, 1.0])
    experiment.mass = torch.tensor([2.0, 3.0])
    experiment.indices = {"zero": [0], "one": [1]}
    experiment.episode_map = {
        identity: dict(
            trajectory_id=identity,
            reward=reward,
            task_id=identity,
            domain="airline",
            original_action_count=2 if reward == 0 else 3,
        )
        for identity, reward in (("zero", 0.0), ("one", 1.0))
    }
    experiment.store = SimpleNamespace(sampled_training=True)
    experiment.lr_cv = [dict(lr=1e-4), dict(lr=3e-4)]
    cv_calls = []

    def select_lrs(**settings):
        cv_calls.append(settings)
        return experiment.lr_cv

    experiment.select_lrs = select_lrs
    experiment.bootstrap = SimpleNamespace(report=lambda *args: {})
    calls = []

    def simulate(**settings):
        calls.append(settings)
        return {
            "evaluations": [
                {
                    "validation": {
                        "by_domain": {"airline": {"explained_variance": -0.5}},
                        "rows": [],
                    }
                }
            ]
        }, (torch.ones(1), torch.tensor(0.0))

    experiment.simulate = simulate
    experiment.save = lambda name, result: result

    result = experiment.run_E4()

    assert len(calls) == 54
    assert len(cv_calls) == 6
    assert {call["seed"] for call in calls} == {20261014, 20261015, 20261016}
    assert {(call["batch"], call["steps"]) for call in calls} == {
        (16, 80),
        (64, 20),
        (64, 80),
        (128, 10),
        (128, 80),
    }
    for arm in result["arms"]:
        assert arm["summary"]["airline"]["mean_ev"] == -0.5
        for seed in arm["seeds"]:
            if arm["budget"] == "B":
                assert seed["training_ev"] == pytest.approx(1.0)
                assert seed["training_mse"] == 0
                assert seed["training_ev_approximate"] is True
            else:
                assert "training_ev" not in seed


def _probe_result(value, lower=0.01):
    return {
        "validation": {"by_domain": {"airline": {"explained_variance": value}}},
        "bootstrap": {"by_domain": {"airline": {"ev_95ci": [lower, 0.9]}}},
    }


@pytest.mark.parametrize(
    "value,lower,status",
    [
        (0.15, 0.001, "成立"),
        (0.15, 0.0, "不确定"),
        (0.049, 0.001, "不成立"),
        (0.05, 0.001, "不确定"),
    ],
)
def test_h1_table_registered_ev_and_interval_boundaries(value, lower, status):
    """H1 needs both EV>=.15 and a strictly positive lower interval bound."""
    from scripts.tau2.critic_preexp_delivery import assess_hypotheses

    result = assess_hypotheses(
        {
            "E2": {
                "probes": {
                    "ridge_shared": _probe_result(value, lower),
                    "ridge_domain": _probe_result(0.8),
                }
            }
        }
    )
    assert result["H1"]["status"] == status


@pytest.mark.parametrize(
    "ev,jump,h2,status",
    [
        (0.14, 0.2, "成立", "成立"),
        (0.14, 0.2001, "成立", "不确定"),
        (0.14, 0.2, "不成立", "不确定"),
        (0.079, 0.2, "成立", "不成立"),
        (0.08, 0.2, "成立", "不确定"),
    ],
)
def test_h3_table_exact_ratio_jump_and_h2_gate(ev, jump, h2, status):
    """Reaching 70% differs from every point being strictly below 40%."""
    from scripts.tau2.critic_preexp_delivery import assess_hypotheses

    result = assess_hypotheses(
        {
            "E2": {
                "probes": {
                    "ridge_shared": _probe_result(0.2),
                    "ridge_domain": _probe_result(0.2),
                }
            },
            "E3": {
                "H2": {"status": h2},
                "arms": [
                    dict(
                        lr=5e-6,
                        frozen_bias=False,
                        warmup_steps=0,
                        maximum_adjacent_bias_jump=jump,
                        evaluations=[
                            dict(step=0, validation=_probe_result(0.0)["validation"]),
                            dict(step=80, validation=_probe_result(ev)["validation"]),
                        ],
                    )
                ],
            },
        }
    )
    assert result["H3"]["status"] == status


@pytest.mark.parametrize(
    "gap,status",
    [(0.05, "成立"), (0.049, "不确定"), (0.02, "不确定"), (0.019, "不成立")],
)
def test_h4_h7_table_gap_cutoffs_use_training_cv_selected_arms(gap, status):
    """All four registered .05/.02 gaps use prespecified training selections."""
    from scripts.tau2.critic_preexp_delivery import assess_hypotheses

    result = assess_hypotheses(
        {
            "E2": {
                "probes": {
                    "ridge_shared": _probe_result(0.0),
                    "ridge_domain": _probe_result(0.0),
                }
            },
            "E4": {
                "batch_budget_training_cv": {
                    "16/B": [{"lr": 1e-4}],
                    "128/B": [{"lr": 1e-4}],
                },
                "arms": [
                    dict(
                        batch=batch,
                        budget="B",
                        lr=1e-4,
                        summary={"airline": {"mean_ev": value}},
                    )
                    for batch, value in ((16, 0.0), (128, gap))
                ],
            },
            "E6": {"probes": {"ridge_shared": _probe_result(gap)}},
            "E7": {
                "selected_intermediate_training_cv": "base-layer-1",
                "layer_backbone_probes": {
                    "base-layer-1": _probe_result(gap),
                    "base-layer-2": _probe_result(0.99),
                    "r510-final": _probe_result(gap),
                },
            },
        }
    )
    assert {result[f"H{i}"]["status"] for i in (4, 5, 6, 7)} == {status}
    assert result["H6"]["selected_cache"] == "base-layer-1"


@pytest.mark.parametrize(
    "ridge,replay,status",
    [
        (0.8, 0.299, "成立"),
        (0.799, 0.299, "不确定"),
        (0.8, 0.3, "不确定"),
        (0.8, 0.6, "不成立"),
    ],
)
def test_h8_table_planted_probe_and_replay_boundaries(ridge, replay, status):
    """The planted signal gate retains its asymmetric .3/.6 thresholds."""
    from scripts.tau2.critic_preexp_delivery import assess_hypotheses

    result = assess_hypotheses(
        {
            "E8": {
                "probes": {
                    "ridge_shared": {
                        "validation": {"summary": {"explained_variance": ridge}}
                    }
                },
                "fixed_r510_replay": [
                    {
                        "evaluations": [
                            {"validation": {"summary": {"explained_variance": replay}}}
                        ]
                    }
                ],
            }
        }
    )
    assert result["H8"]["status"] == status


def test_hypothesis_table_synthetic_inputs_cannot_claim_scientific_support():
    """Synthetic data disclose cutoff behavior while withholding real conclusions."""
    from scripts.tau2.critic_preexp_delivery import assess_hypotheses

    result = assess_hypotheses(
        {
            "E2": {
                "synthetic": True,
                "probes": {
                    "ridge_shared": _probe_result(0.8),
                    "ridge_domain": _probe_result(0.8),
                },
            }
        }
    )
    assert result["H1"]["synthetic_cutoff_check"] == "成立"
    assert {decision["status"] for decision in result.values()} == {"不确定"}
