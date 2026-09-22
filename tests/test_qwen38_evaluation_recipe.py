# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.swe.qwen38_flash_next.train_rl import (
    select_arena_dataset,
    select_evaluation_rows,
    validate_evaluation_only,
)


def test_arena_training_selection_replaces_latest_diagnostic_version():
    from datasets import Dataset

    dataset = Dataset.from_list(
        [
            {"data_id": "env:issue@cancel-repro", "stream_id": "rl"},
            {"data_id": "env:other@v1", "stream_id": "rl"},
        ]
    )

    selected = select_arena_dataset(dataset, ["env:other@v1", "env:issue@benchmark"])

    assert list(selected["data_id"]) == ["env:other@v1", "env:issue@benchmark"]
    assert list(selected["stream_id"]) == ["rl", "rl"]
    assert dataset[0]["data_id"] == "env:issue@cancel-repro"


def evaluation_config():
    return SimpleNamespace(
        total_train_steps=0,
        evaluator=SimpleNamespace(eval_before_train=True),
        recover=SimpleNamespace(mode="disabled"),
        eval_gconfig=SimpleNamespace(n_samples=1),
        valid_dataset=SimpleNamespace(shuffle=False, drop_last=False, batch_size=3),
    )


def test_evaluation_complete_selection_preserves_config():
    config = evaluation_config()
    before = repr(config)

    validate_evaluation_only(config, ["task-a", "task-b", "task-c"])

    assert repr(config) == before


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "total_train_steps", 1),
        ("evaluator", "eval_before_train", False),
        ("recover", "mode", "auto"),
        ("eval_gconfig", "n_samples", 8),
        (None, "valid_dataset", None),
        ("valid_dataset", "shuffle", True),
        ("valid_dataset", "drop_last", True),
        ("valid_dataset", "batch_size", 2),
    ],
)
def test_evaluation_unsafe_config_rejected_before_initialization(section, field, value):
    config = evaluation_config()
    setattr(config if section is None else getattr(config, section), field, value)

    with pytest.raises(ValueError, match="swe-eval"):
        validate_evaluation_only(config, ["task-a", "task-b", "task-c"])


def test_evaluation_empty_selection_rejected():
    config = evaluation_config()
    config.valid_dataset.batch_size = 0

    with pytest.raises(ValueError, match="selected task count"):
        validate_evaluation_only(config, [])


def test_evaluation_pins_historical_version_and_preserves_stream_routing():
    rows = [
        {"data_id": "env:a@new", "stream_id": "stream", "arena_task_type": "swe"},
        {"data_id": "env:b@v1", "stream_id": "stream", "arena_task_type": "swe"},
    ]

    selected = select_evaluation_rows(rows, ["env:b@v1", "env:a@old"])

    assert [row["data_id"] for row in selected] == ["env:b@v1", "env:a@old"]
    assert all(row["stream_id"] == "stream" for row in selected)
    assert all(row["arena_task_type"] == "swe" for row in selected)
    assert rows[0]["data_id"] == "env:a@new"


@pytest.mark.parametrize(
    "selected",
    [[], ["env:missing@v1"], ["env:a@v1", "env:a@v2"], ["env:a"], ["env:a@"]],
)
def test_evaluation_invalid_or_unrelated_selection_rejected(selected):
    with pytest.raises(ValueError):
        select_evaluation_rows([{"data_id": "env:a@new"}], selected)


def test_evaluation_ambiguous_source_rejected():
    with pytest.raises(ValueError, match="unique environment keys"):
        select_evaluation_rows(
            [{"data_id": "env:a@v1"}, {"data_id": "env:a@v2"}], ["env:a@old"]
        )


def test_reference_stream_recipe_loads_with_runtime_loader():
    from examples.swe.arena_config import load_arena_stream_configs

    path = (
        Path(__file__).resolve().parents[1]
        / "examples/swe/qwen38_flash_next/reference_mm_theta/streams.yaml"
    )
    streams = load_arena_stream_configs({"arena_streams_file": str(path)})

    assert len(streams) == 1
    assert streams[0].llm_protocol == "chat_completions"
    assert streams[0].expected_reward_ref.key
    assert streams[0].expected_reward_ref.version
