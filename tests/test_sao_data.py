# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest
from datasets import load_from_disk

from examples.math.sao_data import build_dataset, canonicalize_ground_truth

SOURCE_ENV = "SAO_MATH_DAPO_SOURCE"
EVAL_ROOT_ENV = "SAO_MATH_EVAL_ROOT"
MODEL_PATH_ENV = "SAO_MATH_MODEL_PATH"


def _real_data_paths() -> tuple[Path, Path, str]:
    missing = [
        name
        for name in (SOURCE_ENV, EVAL_ROOT_ENV, MODEL_PATH_ENV)
        if not os.environ.get(name)
    ]
    if missing:
        pytest.skip(f"Set {', '.join(missing)} to run real SAO math data tests")
    return (
        Path(os.environ[SOURCE_ENV]),
        Path(os.environ[EVAL_ROOT_ENV]),
        os.environ[MODEL_PATH_ENV],
    )


def test_canonicalize_ground_truth_only_for_numeric_contract_benchmarks():
    assert canonicalize_ground_truth("033", "aime24") == "33"
    assert canonicalize_ground_truth("043", "aime25") == "43"
    assert canonicalize_ground_truth("080", "beyond_aime") == "80"
    assert canonicalize_ground_truth("00101", "math500") == "00101"
    assert canonicalize_ground_truth("00101", "amc23") == "00101"


def test_sao_data_builds_full_dapo_and_corrected_eval(tmp_path: Path):
    source, eval_root, model_path = _real_data_paths()
    output = tmp_path / "ppo-math-v1"
    manifest = build_dataset(
        source=source,
        eval_root=eval_root,
        output=output,
        seed=42,
        model_path=model_path,
        max_prompt_length=1024,
    )

    dataset = load_from_disk(str(output))
    assert set(dataset) == {"train", "test"}
    assert len(dataset["test"]) == 700
    assert len(dataset["train"]) == manifest["splits"]["train"]
    assert len(dataset["train"]) > 8192
    assert manifest["exclusions"] == {
        "bare_prompt_over_cap": 9,
        "conflicting_answers": 12,
        "evaluation_exact_normalized_overlap": 9,
    }
    assert manifest["loss_bound"]["max_prompt_length"] == 1024
    assert manifest["loss_bound"]["train_max_prompt_tokens"] <= 1024
    assert manifest["loss_bound"]["test_max_prompt_tokens"] <= 1024
    assert manifest["scorer"]["path"] == "areal.reward.math_prd.math_prd_reward_fn"
    assert manifest["eval_selection"] == {
        "source_candidate_rows_with_hmmt": 730,
        "selected_rows": 700,
        "excluded_rows": 30,
        "excluded_benchmark_rows": {"hmmt25": 30},
    }
    assert manifest["test_counts_by_benchmark"] == {
        "aime24": 30,
        "aime25": 30,
        "amc23": 40,
        "beyond_aime": 100,
        "math500": 500,
    }

    correction_map = json.loads((output / "correction_map.json").read_text())
    assert manifest["corrections"]["leading_zero_scan"] == {
        "aime24": {"scanned": 30, "leading_zero_ground_truth": 7, "corrected": 7},
        "aime25": {"scanned": 30, "leading_zero_ground_truth": 0, "corrected": 0},
        "amc23": {"scanned": 40, "leading_zero_ground_truth": 0, "corrected": 0},
        "beyond_aime": {"scanned": 100, "leading_zero_ground_truth": 0, "corrected": 0},
        "math500": {"scanned": 500, "leading_zero_ground_truth": 0, "corrected": 0},
    }
    assert {
        "train/data-00000-of-00001.arrow",
        "train/state.json",
        "test/data-00000-of-00001.arrow",
        "test/state.json",
    }.issubset(manifest["output_files"])
    observed_corrections = [
        (
            row["benchmark"],
            row["row"],
            row["source_id"],
            row["original_answer"],
            row["answer"],
        )
        for row in correction_map
    ]
    assert observed_corrections == [
        ("aime24", 7, "aime24:67", "025", "25"),
        ("aime24", 15, "aime24:75", "073", "73"),
        ("aime24", 18, "aime24:78", "023", "23"),
        ("aime24", 23, "aime24:83", "045", "45"),
        ("aime24", 24, "aime24:84", "033", "33"),
        ("aime24", 25, "aime24:85", "080", "80"),
        ("aime24", 26, "aime24:86", "055", "55"),
    ]

    for item in dataset["train"].select(range(32)):
        assert len(item["messages"]) == 1
        assert item["messages"][0]["role"] == "user"
        assert "Answer:" not in item["messages"][0]["content"]
        assert "ground_truth" not in item["messages"][0]["content"]
    assert {"messages", "answer", "source_id", "benchmark", "data_source"}.issubset(
        dataset["train"].column_names
    )
    assert {"messages", "answer", "source_id", "benchmark", "data_source"}.issubset(
        dataset["test"].column_names
    )
    assert dataset["test"][7]["source_id"] == "aime24:67"
    assert dataset["test"][30]["benchmark"] == "aime25"
    assert dataset["test"][30]["source_dataset_id"] == "yentinglin/aime_2025"


def test_sao_data_seed_controls_train_order(tmp_path: Path):
    source, eval_root, model_path = _real_data_paths()
    build_dataset(
        source=source,
        eval_root=eval_root,
        output=tmp_path / "a",
        seed=42,
        model_path=model_path,
        max_prompt_length=1024,
    )
    build_dataset(
        source=source,
        eval_root=eval_root,
        output=tmp_path / "b",
        seed=42,
        model_path=model_path,
        max_prompt_length=1024,
    )
    build_dataset(
        source=source,
        eval_root=eval_root,
        output=tmp_path / "c",
        seed=43,
        model_path=model_path,
        max_prompt_length=1024,
    )

    a = load_from_disk(str(tmp_path / "a"))["train"]
    b = load_from_disk(str(tmp_path / "b"))["train"]
    c = load_from_disk(str(tmp_path / "c"))["train"]
    assert a[0]["source_id"] == b[0]["source_id"]
    assert a[0]["source_id"] != c[0]["source_id"]
