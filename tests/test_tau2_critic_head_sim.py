# SPDX-License-Identifier: Apache-2.0
"""CPU checks for official EV parity, causal alignment, and recorded replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.tau2 import critic_head_sim as sim
from scripts.tau2 import critic_preexp as pre
from scripts.tau2 import train_critic


@pytest.fixture
def feature_cache(tmp_path):
    """Mixed splits and dtypes; validation episodes span both shards."""
    records, expected = [], []
    for index, dtype in enumerate((torch.bfloat16, torch.float32)):
        features = (torch.arange(8).reshape(4, 2) + index).to(dtype)
        shard = {
            "features": features,
            "split": ["train", "validation", "train", "validation"],
            "trajectory_id": ["t0", "v0", "t1", "v1"],
            "reward": torch.tensor([0.0, 0.0, 1.0, 1.0]),
            "sampling_weight": torch.ones(4),
            "original_action_count": [2] * 4,
            "domain": ["retail"] * 4,
            "task_id": ["t0", "v0", "t1", "v1"],
            "run": ["fixture"] * 4,
        }
        path = tmp_path / f"shard-{index}.pt"
        torch.save(shard, path)
        records.append(
            {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        expected.append(features.float())
    manifest = {
        "schema": "tau2-critic-features-v1",
        "hidden_size": 2,
        "feature_semantics": "score_input_final_norm_shifted_action_prefix",
        "shards": records,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, expected


@pytest.mark.parametrize("store_type", [sim.FeatureStore, pre.PreexpStore])
def test_feature_store_repeated_passes_hash_and_load_each_shard_once(
    feature_cache, monkeypatch, store_type
):
    """Partial iteration, split changes, training and evaluation never reread."""
    path, expected = feature_cache
    loads, hashes = [], []
    original_load, original_hash = torch.load, hashlib.sha256

    def load(path, **kwargs):
        loads.append(Path(path).name)
        return original_load(path, **kwargs)

    def sha256(*args, **kwargs):
        hashes.append(True)
        return original_hash(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(sim.hashlib, "sha256", sha256)
    store = store_type(path)
    next(store.shards("validation"))
    parts = list(store.shards())
    for (actual, _, indices), wanted in zip(parts, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        assert indices.tolist() == [0, 1, 2, 3]
    assert loads == ["shard-0.pt", "shard-1.pt"]
    assert len(hashes) == 2

    def no_io(*args, **kwargs):
        pytest.fail("Feature shards were read after being cached")

    monkeypatch.setattr(Path, "open", no_io)
    training = store.training()
    torch.testing.assert_close(
        training[0], torch.cat([x[[0, 2]] for x in expected]), rtol=0, atol=0
    )
    assert training[3] == ["t0", "t1", "t0", "t1"]
    assert store.training() is training
    assert list(store.shards("absent")) == []
    weight, bias = torch.tensor([0.125, -0.25]), torch.tensor(0.5)
    first = store.evaluate(weight, bias)
    assert first == store.evaluate(weight, bias)
    assert [row["n_tokens"] for row in first["rows"]] == [2, 2]
    if isinstance(store, pre.PreexpStore):
        targets = store.evaluate_targets({"shared": (weight, bias)})
        assert targets == store.evaluate_targets({"shared": (weight, bias)})
        assert [row["n_tokens"] for row in targets["rows"]] == [2, 2]
    assert len(loads) == len(hashes) == 2


@pytest.mark.parametrize("store_type", [sim.FeatureStore, pre.PreexpStore])
def test_feature_store_bad_checksum_rejected_before_loading(
    feature_cache, monkeypatch, store_type
):
    """A resident reader must still reject corruption before deserializing."""
    path, _ = feature_cache
    manifest = json.loads(path.read_text())
    manifest["shards"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))

    def no_load(*args, **kwargs):
        pytest.fail("Corrupt feature shard reached torch.load")

    monkeypatch.setattr(torch, "load", no_load)
    store = store_type(path)
    for _ in range(2):
        with pytest.raises(ValueError, match="Feature shard hash mismatch"):
            list(store.shards())


@pytest.mark.parametrize("store_type", [sim.FeatureStore, pre.PreexpStore])
def test_feature_store_wrong_hidden_size_rejected(feature_cache, store_type):
    """Caching must not bypass the feature dimension guard."""
    path, _ = feature_cache
    manifest = json.loads(path.read_text())
    manifest["hidden_size"] = 3
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incorrect feature shape"):
        list(store_type(path).shards())


def test_official_ev_unequal_trajectory_lengths_matches_online_evaluator(
    monkeypatch, tmp_path
):
    """Use the real online evaluator as oracle; unequal counts catch macro EV."""
    rows = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "loss_mask": [0, 1, 1],
            "reward": 0.0,
            "domain": "retail",
            "task_id": "a",
            "episode_id": "a",
        },
        {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1] * 5,
            "loss_mask": [0, 1, 1, 1, 1],
            "reward": 1.0,
            "domain": "retail",
            "task_id": "b",
            "episode_id": "b",
        },
    ]
    predictions = [
        torch.tensor([0.4, 0.2, 99.0]),
        torch.tensor([0.5, 0.6, 0.9, 0.8, 99.0]),
    ]
    critic = SimpleNamespace(
        parallel_strategy=SimpleNamespace(dp_size=1),
        compute_values=lambda _: predictions,
        clear_batches=lambda *args: None,
        grad_norm=lambda _: {"grad_norm": 0.0},
    )
    monkeypatch.setattr(train_critic, "_localize_remote_values", lambda values: values)
    expected = train_critic.evaluate_tau2_critic(
        SimpleNamespace(critic=critic), rows, evidence_dir=tmp_path, completed_step=0
    )

    actual_rows = [
        sim.official_metric_row(value[:-1], row)
        for value, row in zip(predictions, rows, strict=True)
    ]
    actual = sim.aggregate_metrics(actual_rows)

    assert actual["summary"] == expected["summary"]
    assert actual["by_domain"] == expected["by_domain"]
    assert actual["summary"]["n_tokens"] == 6
    assert actual["summary"]["explained_variance"] == pytest.approx(0.55)


def test_replay_logged_retry_ids_restore_original_submission_order():
    """Expected IDs are literal original submissions, independently of sorting."""
    # The final submission order is 0,1,2,3 then 4,5,6,7; 2 and 5
    # were replayed after all first attempts (the real trainer retry offset).
    offset = 1 << 40
    logged = [
        {"rollout_task_id": value, "episode_id": str(value)}
        for value in [7, offset + 2, 0, 6, 3, offset + 5, 1, 4]
    ]
    rows = sim.recorded_replay_rows(logged, batch_size=4)

    batches = sim.replay_batches(rows, batch_size=4, total_steps=2)

    assert [
        [row["canonical_rollout_task_id"] for row in batch] for batch in batches
    ] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_replay_missing_step_raises_instead_of_silently_shortening():
    """A missing data step must fail the replay contract."""
    with pytest.raises(ValueError, match="Incomplete replay"):
        sim.replay_batches(
            [{"trajectory_id": "a", "step": 0, "order": 0}], total_steps=2
        )


def test_paired_bootstrap_different_targets_raises():
    """Matching task IDs alone are insufficient for model pairing."""
    row = sim.official_metric_row(
        torch.tensor([0.2]),
        {
            "reward": 0,
            "domain": "retail",
            "task_id": "a",
            "episode_id": "a",
            "run": "r1",
        },
    )
    with pytest.raises(ValueError, match="targets or token counts differ"):
        sim.cluster_bootstrap([row], reference=[{**row, "target": 1}], draws=5)


def test_replay_missing_episode_in_full_batch_raises():
    """Recorded steps alone do not prove all sixteen episode gradients exist."""
    rows = [{"trajectory_id": str(i), "step": 0, "order": i} for i in range(15)]
    with pytest.raises(ValueError, match="Incomplete replay batch"):
        sim.replay_batches(rows, total_steps=1)


def test_replay_global_canonical_feature_order_is_supported():
    """Extraction stores global episode order, not just order within a step."""
    rows = [
        {"trajectory_id": str(i), "step": i // 4, "order": i}
        for i in [5, 1, 7, 0, 3, 4, 6, 2]
    ]
    batches = sim.replay_batches(rows, batch_size=4, total_steps=2)
    assert [[row["order"] for row in batch] for batch in batches] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_paired_bootstrap_same_model_task_clusters_zero_difference():
    """All runs of a repeated task stay together and identical models pair to zero."""
    rows = [
        sim.official_metric_row(
            torch.tensor([prediction]),
            {
                "reward": target,
                "domain": "retail",
                "task_id": task,
                "episode_id": f"{run}-{task}",
                "run": run,
            },
        )
        for run in ["r1", "r2"]
        for task, target, prediction in [("a", 0, 0.3), ("b", 1, 0.7), ("c", 0, 0.2)]
    ]

    report = sim.cluster_bootstrap(rows, reference=rows, draws=50)

    assert report["by_domain"]["retail"]["task_clusters"] == 3
    assert report["by_domain"]["retail"]["ev_95ci"] == [0.0, 0.0]
    assert report["by_domain"]["retail"]["auc_95ci"] == [0.0, 0.0]


def test_token_weighted_mse_matches_replicated_full_tokens():
    """Long trajectories have proportionate token mass, and MSE has no half."""
    x = torch.tensor([[1.0], [3.0]])
    weight = torch.tensor([0.1], requires_grad=True)
    bias = torch.tensor(0.78, requires_grad=True)
    y = torch.tensor([0.0, 1.0])
    mass = torch.tensor([3.0, 1.0])

    loss = sim.weighted_mse(x, y, mass, weight, bias)
    expected = (
        torch.tensor([0.88, 0.88, 0.88, 1.08])
        .sub(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        .square()
        .mean()
    )

    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    loss.backward()
    torch.testing.assert_close(weight.grad, torch.tensor([1.44]), rtol=1e-6, atol=1e-7)


def test_feature_store_sampled_validation_rejected(tmp_path):
    """Official token EV must never silently accept a sampled validation set."""
    shard = {
        "features": torch.ones(1, 2),
        "split": ["validation"],
        "sampling_weight": torch.tensor([2.0]),
    }
    torch.save(shard, tmp_path / "shard.pt")
    manifest = {
        "schema": "tau2-critic-features-v1",
        "hidden_size": 2,
        "feature_semantics": "score_input_final_norm_shifted_action_prefix",
        "shards": [{"path": "shard.pt", "rows": 1}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    store = sim.FeatureStore(tmp_path / "manifest.json")

    with pytest.raises(ValueError, match="Validation requires every action prefix"):
        store.evaluate(torch.zeros(2), 0.78)


def test_ridge_bias_is_unpenalized():
    """Constant labels fit through an intercept even with strong regularisation."""
    x = torch.tensor([[-1.0], [1.0]])
    weight, bias = sim.fit_ridge(x, torch.tensor([0.78, 0.78]), torch.ones(2), 1000)

    torch.testing.assert_close(weight, torch.zeros(1), rtol=0, atol=1e-7)
    torch.testing.assert_close(bias, torch.tensor(0.78), rtol=0, atol=1e-7)


def test_adam_all_run_store_replays_only_requested_run_and_warmup_zero_first(tmp_path):
    """Other probe runs must not enter replay; HF warmup consumes lr=0 first."""
    shard = {
        "features": torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [1.0], [-1.0]]),
        "trajectory_id": ["510-a", "510-b", "499-a", "499-b", "val-a", "val-b"],
        "position": torch.zeros(6, dtype=torch.long),
        "step": torch.tensor([0, 1, 0, 1, 0, 0]),
        "order": torch.zeros(6, dtype=torch.long),
        "domain": ["retail"] * 6,
        "task_id": ["train-a", "train-b", "train-a", "train-b", "test-a", "test-b"],
        "run": ["r510", "r510", "r499", "r499", "r510", "r510"],
        "reward": torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        "original_action_count": torch.ones(6, dtype=torch.long),
        "sampling_weight": torch.ones(6),
        "split": ["train"] * 4 + ["validation"] * 2,
    }
    torch.save(shard, tmp_path / "shard.pt")
    manifest = {
        "schema": "tau2-critic-features-v1",
        "hidden_size": 1,
        "feature_semantics": "score_input_final_norm_shifted_action_prefix",
        "shards": [{"path": "shard.pt", "rows": 6}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    result = sim.simulate_adam(
        sim.FeatureStore(tmp_path / "manifest.json"),
        lr=1e-4,
        warmup_steps=10,
        total_steps=2,
        batch_size=1,
    )

    assert [step["trajectory_ids"] for step in result["steps"]] == [
        ["510-a"],
        ["510-b"],
    ]
    assert result["steps"][0]["lr"] == 0
    assert result["steps"][0]["weight_norm"] == 0
    assert result["steps"][1]["lr"] == pytest.approx(1e-5)
    assert result["steps"][1]["weight_norm"] > 0
