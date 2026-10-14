# SPDX-License-Identifier: Apache-2.0

"""CPU checks of final-normalized causal token critic features and cache provenance."""

import json

import pytest
import torch
from transformers import Qwen3_5Config

from scripts.tau2.critic_features import (
    action_positions,
    estimate,
    extract_score_inputs,
    iter_rollouts,
    load_stored_rollouts,
    selected_positions,
    validate_splits,
    write_feature_cache,
)

from areal.models.transformers.token_critic import Qwen35TokenCriticForCausalBackbone


def tiny_critic():
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "layer_types": ["full_attention", "full_attention"],
            "dtype": "float32",
            "use_cache": False,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        areal_value_head_bias=True,
    )
    torch.manual_seed(17)
    return Qwen35TokenCriticForCausalBackbone.from_config(
        config, attn_implementation="eager"
    ).eval()


def rollout():
    return {
        "input_ids": [1, 2, 3, 4, 5, 6],
        "attention_mask": [1] * 6,
        "loss_mask": [0, 0, 1, 1, 0, 1],
        "reward": 1.0,
        "domain": "airline",
        "task_id": "train-task",
        "split": "train",
        "_split": "train",
        "_run": "r510",
        "_trajectory_id": "r510:train:0",
        "_episode_order": 0,
        "_trainer_step": 0,
    }


def test_score_input_actual_critic_hook_matches_final_normalized_prefixes():
    """Independent score prehook confirms final norm and exact t-1 alignment."""
    model, row = tiny_critic(), rollout()
    captured = []
    hook = model.score.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([row["input_ids"]]),
            attention_mask=torch.ones(1, 6, dtype=torch.bool),
            use_cache=False,
        )
    hook.remove()
    actual = extract_score_inputs(model, row)
    expected = captured[0][0, torch.tensor([1, 2, 4])]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["shift", "pre_norm"])
def test_score_input_intentional_corruption_is_detected(corruption):
    """Both alignment and final-norm corruption must make the oracle red."""
    model, row = tiny_critic(), rollout()
    expected = extract_score_inputs(model, row)
    if corruption == "shift":
        broken = extract_score_inputs(model, row, torch.tensor([2, 3, 5]))
    else:
        captured = []
        hook = model.model.language_model.norm.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone())
        )
        extract_score_inputs(model, row)
        hook.remove()
        broken = captured[0][0, torch.tensor([1, 2, 4])]
    with pytest.raises(AssertionError):
        torch.testing.assert_close(broken, expected, rtol=0, atol=0)


def test_subsample_deterministic_and_validation_retains_all():
    """Uniform train sampling is stable; validation cannot be subsampled."""
    row = rollout()
    first, count = selected_positions(row, train_k=2, seed=7)
    second, _ = selected_positions(row, train_k=2, seed=7)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert count == 3 and len(first) == 2
    row["_split"] = "validation"
    positions, _ = selected_positions(row, train_k=1, seed=7)
    torch.testing.assert_close(positions, torch.tensor([1, 2, 4]), rtol=0, atol=0)


def test_train_task_overlap_and_test_label_are_rejected(tmp_path):
    """Held-out identity or explicit split contamination is refused."""
    row = rollout()
    with pytest.raises(ValueError, match="overlap"):
        validate_splits([row], [row])
    row["split"] = "test"
    path = tmp_path / "0.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="held-out"):
        iter_rollouts([path], split="train", run="r510")


def test_numeric_episode_order_and_estimate_without_model(tmp_path):
    """File 10 follows file 2 and estimate accounts for original forward cost."""
    paths = []
    for order in (10, 2):
        path = tmp_path / f"{order}.jsonl"
        path.write_text(json.dumps(rollout()) + "\n")
        paths.append(path)
    rows = iter_rollouts(paths, split="train", run="r510", batch_size=4)
    assert [r["_episode_order"] for r in rows] == [2, 10]
    assert [r["_trainer_step"] for r in rows] == [0, 2]
    result = estimate(
        rows, hidden_size=32, train_k=2, seed=7, tokens_per_second=12, gpu_count=2
    )
    assert result["selected_feature_rows"] == 4
    assert result["feature_tensor_bytes"] == 256
    assert result["input_tokens"] == 12 and result["estimated_wall_seconds"] == 1


def test_cache_shards_preserve_sampling_and_head(tmp_path):
    """Half-precision shards carry identities, weight A/K and initial head."""
    source = tmp_path / "0.jsonl"
    source.write_text(json.dumps(rollout()) + "\n")
    rows = iter_rollouts([source], split="train", run="r510")
    output = tmp_path / "cache"
    manifest = write_feature_cache(
        tiny_critic(), rows, output, train_k=2, seed=7, dtype="float16", shard_rows=1
    )
    assert not manifest["train_features_exact"]
    assert manifest["validation_features_exact"]
    assert len(manifest["shards"]) == 2
    shard = torch.load(output / manifest["shards"][0]["path"], weights_only=True)
    assert shard["features"].dtype == torch.float16
    torch.testing.assert_close(
        shard["sampling_weight"], torch.tensor([1.5]), rtol=0, atol=0
    )
    assert shard["run"] == ["r510"] and shard["original_action_count"].tolist() == [3]
    head = torch.load(output / manifest["head_path"], weights_only=True)
    assert set(head) == {"weight", "bias"}


def test_validation_tensor_requires_matching_metadata(tmp_path):
    """Validation task provenance must accompany the tensor cache."""
    path = tmp_path / "fixed-validation-rollouts.pt"
    torch.save([rollout()], path)
    with pytest.raises(ValueError, match="require"):
        load_stored_rollouts(path)
    (tmp_path / "fixed-validation-manifest.json").write_text(
        json.dumps(
            {"payload": {"task_ids": [{"domain": "retail", "task_id": "different"}]}}
        )
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_stored_rollouts(path)


def test_first_token_action_without_prefix_is_rejected():
    """Position zero has no preceding state and must not silently disappear."""
    row = rollout()
    row["loss_mask"][0] = 1
    with pytest.raises(ValueError, match="First token"):
        action_positions(row)


def test_replay_id_canonicalization_and_metadata_split_guard(tmp_path):
    """Successful retry IDs belong to the original training step."""
    path = tmp_path / f"{(1 << 40) + 35}.jsonl"
    row = rollout()
    row["rollout_task_id"] = (1 << 40) + 35
    path.write_text(json.dumps(row) + "\n")
    rows = iter_rollouts([path], split="train", run="r510", batch_size=16)
    assert rows[0]["_episode_order"] == 35
    assert rows[0]["_trainer_step"] == 2
    row["metadata"] = {"critic_split": "test"}
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="held-out"):
        iter_rollouts([path], split="train", run="r510")


def test_multiple_jsonl_tensor_rows_have_distinct_trajectory_ids(tmp_path):
    """Multiple stored tensor records must not be silently truncated."""
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(rollout()) + "\n" + json.dumps(rollout()) + "\n")
    rows = iter_rollouts([path], split="train", run="r510")
    assert len(rows) == 2
    assert rows[0]["_trajectory_id"] != rows[1]["_trajectory_id"]


def test_numeric_ids_in_distinct_runs_do_not_conflict(tmp_path):
    """Different run directories can reuse numeric training episode IDs."""
    paths = []
    for name in ("r499", "r510"):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "0.jsonl"
        path.write_text(json.dumps(rollout()) + "\n")
        paths.append(path)
    rows = iter_rollouts(paths, split="train", run="pooled")
    assert len(rows) == 2
    assert rows[0]["_trajectory_id"] != rows[1]["_trajectory_id"]


def test_non_tensor_jsonl_record_is_rejected(tmp_path):
    """Malformed stored rollout records cannot silently disappear from replay."""
    path = tmp_path / "0.jsonl"
    path.write_text(json.dumps(rollout()) + '\n{"unexpected": true}\n')
    with pytest.raises(ValueError, match="Malformed"):
        load_stored_rollouts(path)
