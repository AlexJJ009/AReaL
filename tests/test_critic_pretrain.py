# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the critic pretraining qualification helpers."""

from __future__ import annotations

import json

import pytest
import torch

from scripts.qualification.critic_pretrain import (
    aggregate_validation,
    load_validation_records,
    make_validation_item,
    pad_for_dp,
    response_values,
)


def test_load_validation_records_requires_fixed_schema(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps({"source_id": "x"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        load_validation_records(path)


def test_make_validation_item_uses_response_token_mask():
    record = {
        "source_id": "s1",
        "benchmark": "math",
        "input_tokens": [10, 11, 12],
        "output_tokens": [20, 21],
        "reward": 1.0,
        "truncated": False,
    }

    item = make_validation_item(record)

    torch.testing.assert_close(
        item["input_ids"], torch.tensor([[10, 11, 12, 20, 21]]), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        item["loss_mask"],
        torch.tensor([[False, False, False, True, True]]),
        rtol=0.0,
        atol=0.0,
    )


def test_response_values_uses_shifted_training_alignment():
    values = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4]])

    selected = response_values(values, prompt_len=3, token_count=5)

    torch.testing.assert_close(selected, torch.tensor([0.2, 0.3]), rtol=0.0, atol=0.0)


def test_pad_for_dp_duplicates_and_preserves_real_count():
    items = [{"id": 0}, {"id": 1}, {"id": 2}]

    padded, real_count = pad_for_dp(items, 4)

    assert real_count == 3
    assert len(padded) == 4
    assert padded[-1]["_padding_duplicate"] is True


def test_aggregate_validation_computes_token_weighted_mse_and_ev():
    rows = [
        {
            "n_tokens": 2,
            "sse": 0.5,
            "error_sum": 0.0,
            "target_sum": 0.0,
            "target2_sum": 0.0,
            "target": 0.0,
            "truncated": False,
        },
        {
            "n_tokens": 2,
            "sse": 0.25,
            "error_sum": 0.0,
            "target_sum": 2.0,
            "target2_sum": 2.0,
            "target": 1.0,
            "truncated": True,
        },
    ]

    summary = aggregate_validation(rows)

    assert summary["n_tokens"] == 4
    assert summary["mse"] == pytest.approx(0.1875)
    assert summary["target_var"] == pytest.approx(0.25)
    assert summary["explained_variance"] == pytest.approx(0.25)
    assert summary["truncated_rate"] == pytest.approx(0.5)


def test_aggregate_validation_marks_ev_undefined_for_constant_targets():
    rows = [
        {
            "n_tokens": 3,
            "sse": 0.75,
            "error_sum": 0.0,
            "target_sum": 3.0,
            "target2_sum": 3.0,
            "target": 1.0,
            "truncated": False,
        }
    ]

    summary = aggregate_validation(rows)

    assert summary["explained_variance"] is None
    assert summary["explained_variance_defined"] is False


def test_validation_ev_is_invariant_to_constant_prediction_bias():
    rows = [
        {
            "n_tokens": 1,
            "sse": 4.0,
            "error_sum": 2.0,
            "target_sum": y,
            "target2_sum": y,
            "target": y,
            "truncated": False,
        }
        for y in (0.0, 1.0)
    ]
    summary = aggregate_validation(rows)
    assert summary["explained_variance"] == pytest.approx(1.0)
    assert summary["mse"] == pytest.approx(4.0)
    assert summary["bias"] == pytest.approx(2.0)


def test_response_values_rejects_truncated_prediction_tensor():
    with pytest.raises(ValueError, match="shorter"):
        response_values(torch.zeros(3), prompt_len=2, token_count=4)
