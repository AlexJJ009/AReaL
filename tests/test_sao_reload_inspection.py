# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the SAO final reload probe helpers."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

from tests.torchrun.run_sao_reload import (
    _compare_tensors,
    _resolve_selected_pairs,
    _sha256_small,
    aggregate_rank_statuses,
    inspect_hf_metadata,
    resolve_required_suffixes,
)


def _write_safetensors_header(path: Path) -> None:
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        },
        separators=(",", ":"),
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")


def test_selected_pairs_include_text_and_critic_score():
    model_keys = {
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.norm.weight",
        "score.weight",
    }
    hf_keys = set(model_keys)

    actor_pairs = _resolve_selected_pairs(
        role="actor", model_param_names=model_keys, hf_weight_keys=hf_keys
    )
    critic_pairs = _resolve_selected_pairs(
        role="critic", model_param_names=model_keys, hf_weight_keys=hf_keys
    )

    assert [pair[0] for pair in actor_pairs] == [
        "language_model.layers.0.linear_attn.A_log",
        "language_model.layers.0.linear_attn.dt_bias",
        "language_model.layers.0.linear_attn.norm.weight",
    ]
    assert critic_pairs[-1] == ("score.weight", "score.weight", "score.weight")


def test_resolve_selected_pairs_rejects_old_wrong_text_keys():
    old_keys = {
        "layers.0.linear_attn.A_log",
        "layers.0.linear_attn.dt_bias",
        "layers.0.linear_attn.norm.weight",
    }
    hf_keys = {
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.norm.weight",
    }

    try:
        _resolve_selected_pairs(
            role="actor", model_param_names=old_keys, hf_weight_keys=hf_keys
        )
    except KeyError as exc:
        assert "language_model.layers.0.linear_attn.A_log" in str(exc)
    else:
        raise AssertionError("old text keys must not be silently accepted")


def test_resolve_selected_pairs_requires_critic_score():
    keys_without_score = {
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.norm.weight",
    }

    try:
        _resolve_selected_pairs(
            role="critic",
            model_param_names=keys_without_score,
            hf_weight_keys=keys_without_score,
        )
    except KeyError as exc:
        assert "score.weight" in str(exc)
    else:
        raise AssertionError("critic score.weight must be required")


def test_resolve_required_suffixes_rejects_ambiguous_suffix():
    keys = {
        "a.model.language_model.layers.0.linear_attn.A_log",
        "b.model.language_model.layers.0.linear_attn.A_log",
    }

    try:
        resolve_required_suffixes(keys, ("language_model.layers.0.linear_attn.A_log",))
    except KeyError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous suffix must fail")


def test_aggregate_rank_statuses_keeps_missing_optimizer_step_unknown():
    status = aggregate_rank_statuses(
        [
            {"status": "ok", "passed": True},
            {
                "status": "unknown",
                "passed": False,
                "optimizer_step_status": "unknown_absent",
            },
        ]
    )

    assert status == "unknown"


def test_compare_tensors_casts_expected_dtype_before_compare():
    actual = torch.tensor([1.0, 2.0], dtype=torch.float32)
    expected = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    report = _compare_tensors(actual, expected)

    assert report["allclose"]
    assert report["exact_equal_after_dtype_cast"]
    assert report["max_abs_diff"] == 0.0


def test_compare_tensors_rejects_bf16_mismatch_exactly():
    actual = torch.tensor([1.0, 2.125], dtype=torch.float32)
    expected = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    report = _compare_tensors(actual, expected)

    assert not report["allclose"]
    assert not report["exact_equal_after_dtype_cast"]


def test_inspect_hf_metadata_reads_index_without_weight_hashing(tmp_path):
    _write_safetensors_header(tmp_path / "model.safetensors")
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {"weight": "model.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )

    report = inspect_hf_metadata(str(tmp_path))

    assert report["weight_map_count"] == 1
    assert report["metadata"] == {"total_size": 4}
    assert report["files"]["model.safetensors"]["sha256"] is not None


def test_sha256_small_skips_large_files(tmp_path):
    path = tmp_path / "large.bin"
    path.write_bytes(b"0" * 32)

    assert _sha256_small(path, max_bytes=16) is None


def test_every_selected_parameter_needs_an_exact_optimizer_step():
    from tests.torchrun.run_sao_reload import optimizer_step_status

    assert optimizer_step_status({"text": 135, "score": 135.0}, 135, 2) == "ok"
    assert (
        optimizer_step_status({"text": 135, "score": None}, 135, 2) == "unknown_absent"
    )
    assert optimizer_step_status({"text": 135}, 135, 2) == "unknown_absent"
    assert optimizer_step_status({"text": 135.5}, 135, 1) == "mismatch"
