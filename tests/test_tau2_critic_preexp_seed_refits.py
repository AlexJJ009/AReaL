# SPDX-License-Identifier: Apache-2.0
"""Seed-specific probe refits must honor training-CV choices and strata."""

from types import SimpleNamespace

import pytest
import torch

from scripts.tau2.critic_preexp import SEEDS, Preexperiment, summarize_validations


@pytest.mark.parametrize(
    "scope,soft", [("shared", False), ("shared", True), ("airline", False)]
)
def test_seed_probe_refits_training_cv_winners_preserve_stratum_and_labels(scope, soft):
    """Each seed uses its own training winner and excludes other strata/domains."""
    # The excluded trajectory has extreme features and reward. Including it in a
    # stratum fit changes the literal two-point ridge solution substantially.
    experiment = Preexperiment.__new__(Preexperiment)
    experiment.x = torch.tensor([[-1.0], [1.0], [10.0]])
    experiment.y = torch.tensor([0.0, 1.0, 99.0])
    experiment.mass = torch.ones(3)
    experiment.fields = {"domain": ["airline", "airline", "retail"]}
    captured = []

    def evaluate(models, *, logistic, planted):
        assert not logistic and not planted
        weight, bias = models[scope]
        captured.append((weight.clone(), bias.clone()))
        return {
            "by_domain": {"airline": {"explained_variance": float(weight[0])}},
            "rows": [{"prediction": float(weight[0])}],
        }

    experiment.store = SimpleNamespace(evaluate_targets=evaluate)
    experiment.bootstrap = SimpleNamespace(
        report=lambda rows: {"rows_checked": len(rows)}
    )
    penalties = (0.1, 1.0, 10.0)
    choices = {
        scope: {
            "candidates": [
                {
                    "penalty": penalty,
                    "seed_mse": {
                        str(seed): 0.0 if index == rank else 1.0
                        for rank, seed in enumerate(SEEDS)
                    },
                    # The validation winner is deliberately a different penalty.
                    "validation_mse": 0.0 if index == 2 else 100.0,
                }
                for index, penalty in enumerate(penalties)
            ]
        }
    }
    labels = torch.tensor([0.25, 0.75, 99.0]) if soft else None
    subset = torch.tensor([True, True, False]) if scope == "shared" else None

    result = experiment.seed_probe_refits(choices, y=labels, subset=subset)

    assert [row["seed"] for row in result] == list(SEEDS)
    assert [row["selected_penalties"][scope] for row in result] == list(penalties)
    for (weight, bias), penalty, row in zip(captured, penalties, result, strict=True):
        slope = (0.25 if soft else 0.5) / (1 + penalty)
        torch.testing.assert_close(weight, torch.tensor([slope]), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(bias, torch.tensor(0.5), rtol=1e-6, atol=1e-7)
        assert row["bootstrap"] == {"rows_checked": 1}
        assert row["heads"][scope]["weight_norm"] == pytest.approx(slope)
    summary = summarize_validations([row["validation"] for row in result])["airline"]
    assert summary["defined_ev_seeds"] == summary["seeds"] == 3
    assert summary["range_ev"] == pytest.approx(
        [float(captured[2][0][0]), float(captured[0][0][0])]
    )


def _base_cache_metadata(tmp_path, *, synthetic=True):
    return {
        "synthetic": synthetic,
        "layer": "final",
        "depth": 32,
        "hidden_size": 8,
        "checkpoint": str(tmp_path / "base"),
        "feature_semantics": "score_input_final_norm_shifted_action_prefix",
    }


def test_e7_base_final_manifest_under_intermediate_alias_rejected(tmp_path):
    """A self-consistent final cache cannot impersonate an intermediate arm."""
    from scripts.tau2.critic_preexp import validate_e7_cache_alias

    base = _base_cache_metadata(tmp_path)
    with pytest.raises(ValueError, match="alias.*layer"):
        validate_e7_cache_alias("base-layer-8", base, base, synthetic=True)


def test_e7_intermediate_layer_number_and_alias_mismatch_rejected(tmp_path):
    """Even valid intermediate metadata must match its advertised layer number."""
    from scripts.tau2.critic_preexp import validate_e7_cache_alias

    base = _base_cache_metadata(tmp_path)
    middle = {
        **base,
        "layer": "16",
        "feature_semantics": "intermediate_layer_output_shifted_action_prefix",
    }
    with pytest.raises(ValueError, match="alias.*layer"):
        validate_e7_cache_alias("base-layer-8", middle, base, synthetic=True)


def test_e7_base_checkpoint_under_trained_backbone_alias_rejected(tmp_path):
    """H7 cannot compare a checkpoint with itself under a different cache name."""
    from scripts.tau2.critic_preexp import validate_e7_cache_alias

    base = _base_cache_metadata(tmp_path)
    with pytest.raises(ValueError, match="differ.*base checkpoint"):
        validate_e7_cache_alias("r510-final", base, base, synthetic=True)


def test_e7_intermediate_config_fingerprint_mismatch_rejected(tmp_path):
    """Matching paths and layer labels do not replace checkpoint byte binding."""
    from scripts.tau2.critic_preexp import validate_e7_cache_alias

    base = {
        **_base_cache_metadata(tmp_path),
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_files_sha256": {"model.safetensors": "b" * 64},
    }
    middle = {
        **base,
        "layer": "8",
        "feature_semantics": "intermediate_layer_output_shifted_action_prefix",
        "checkpoint_config_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="fingerprints differ"):
        validate_e7_cache_alias("base-layer-8", middle, base, synthetic=True)


def test_primary_intermediate_and_unpinned_production_checkpoints_rejected(tmp_path):
    """E2 cannot silently use intermediate or previously trained base features."""
    from scripts.tau2.critic_preexp import validate_primary_cache

    base = _base_cache_metadata(tmp_path)
    with pytest.raises(ValueError, match="final score-input"):
        validate_primary_cache({**base, "layer": "8"}, synthetic=True)
    production = {
        **base,
        "synthetic": False,
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_files_sha256": {"model.safetensors": "b" * 64},
    }
    with pytest.raises(ValueError, match="registered pinned"):
        validate_primary_cache(production)
    with pytest.raises(ValueError, match="nonempty.*fingerprints"):
        validate_primary_cache({**base, "synthetic": False})


def test_e7_valid_intermediate_and_registered_trained_checkpoint_provenance(tmp_path):
    """Valid aliases retain exact layer, config and weight fingerprints."""
    from scripts.tau2.critic_preexp import PINNED_BASE_REVISION, validate_e7_cache_alias

    base = {
        **_base_cache_metadata(tmp_path, synthetic=False),
        "checkpoint": str(tmp_path / "snapshots" / PINNED_BASE_REVISION),
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_files_sha256": {"model.safetensors": "b" * 64},
    }
    middle = {
        **base,
        "layer": "8",
        "feature_semantics": "intermediate_layer_output_shifted_action_prefix",
    }
    trained = {
        **base,
        "checkpoint": str(
            tmp_path
            / "critic-head-qwen35-mix-r2"
            / "checkpoints"
            / "critic"
            / "epoch2epochstep14globalstep54"
        ),
        "checkpoint_files_sha256": {"model.safetensors": "c" * 64},
    }

    observed_middle = validate_e7_cache_alias("base-layer-8", middle, base)
    observed_trained = validate_e7_cache_alias("r510-final", trained, base)

    assert observed_middle["layer"] == "8"
    assert observed_middle["checkpoint_config_sha256"] == "a" * 64
    assert observed_middle["checkpoint_files_sha256"] == {"model.safetensors": "b" * 64}
    assert observed_trained["layer"] == "final"
    assert observed_trained["checkpoint"] != observed_middle["checkpoint"]
    assert observed_trained["checkpoint_files_sha256"] == {
        "model.safetensors": "c" * 64
    }
    with pytest.raises(ValueError, match="registered r510 step55"):
        validate_e7_cache_alias(
            "r510-final", {**trained, "checkpoint": str(tmp_path / "wrong-step")}, base
        )

    with pytest.raises(ValueError, match="registered r510 step55"):
        validate_e7_cache_alias(
            "r510-final",
            {
                **trained,
                "checkpoint": str(
                    tmp_path
                    / "critic-head-qwen35-mix-r2"
                    / "checkpoints"
                    / "default"
                    / "epoch2epochstep14globalstep54"
                ),
            },
            base,
        )
    with pytest.raises(ValueError, match="weight fingerprints must differ"):
        validate_e7_cache_alias(
            "r510-final",
            {**trained, "checkpoint_files_sha256": base["checkpoint_files_sha256"]},
            base,
        )
