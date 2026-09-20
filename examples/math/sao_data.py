# SPDX-License-Identifier: Apache-2.0

"""Build the SAO/PPO math DatasetDict from audited DAPO/eval sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict

DAPO_SOURCE_REV = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DAPO_SOURCE_SHA256 = "134671de0cd455477e3bbca80f125ea32094084ef1ae62e2fa3e1b066414ba4c"
EXPECTED_EVAL_ROWS = {
    "aime24": 30,
    "aime25": 30,
    "amc23": 40,
    "beyond_aime": 100,
    "math500": 500,
}
SOURCE_EVAL_CANDIDATE_ROWS_WITH_HMMT = 730
EXCLUDED_EVAL_BENCHMARK_ROWS = {"hmmt25": 30}
EVAL_FILES = {
    "aime24": "aime-2024.parquet",
    "aime25": "aime-2025.parquet",
    "amc23": "amc23.parquet",
    "beyond_aime": "beyond-aime.parquet",
    "math500": "math-500.parquet",
}
ZERO_PADDED_GT_RE = re.compile(r"^0+[0-9]+$")
ZERO_PADDED_GT_NUMERIC_CONTRACT_BENCHMARKS = frozenset(
    {"aime24", "aime25", "beyond_aime"}
)
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be "
    "of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
SCORER_PATH = "areal.reward.math_prd.math_prd_reward_fn"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_file_hashes(output: Path) -> dict[str, dict[str, str | int]]:
    hashes: dict[str, dict[str, str | int]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            hashes[str(path.relative_to(output))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return hashes


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _source_id(value: Any) -> str:
    return str(value)


def problem_from_dapo_row(row: dict[str, Any]) -> str:
    messages = row["prompt"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Expected one DAPO user message")
    content = messages[0]["content"]
    if not content.startswith(DAPO_PREFIX) or not content.endswith(DAPO_SUFFIX):
        raise ValueError(f"Unexpected DAPO wrapper for {row['extra_info']['index']}")
    return content[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)]


def problem_from_eval_row(row: dict[str, Any]) -> str:
    content = row["prompt"][-1]["content"]
    content = content.removesuffix("\n\n" + BOXED_INSTRUCTION)
    return content.removeprefix("Problem: ")


def make_messages(problem: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"Problem: {problem}\n\n{BOXED_INSTRUCTION}"}]


def canonicalize_ground_truth(answer: str, benchmark: str) -> str:
    if (
        benchmark in ZERO_PADDED_GT_NUMERIC_CONTRACT_BENCHMARKS
        and ZERO_PADDED_GT_RE.fullmatch(answer)
    ):
        return str(int(answer))
    return answer


def rendered_prompt_len(tokenizer: Any | None, messages: list[dict[str, str]]) -> int:
    if tokenizer is None:
        return 0
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_attention_mask=False)
    return len(encoded["input_ids"])


def load_tokenizer(model_path: str | None):
    if model_path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )


def convert_train(
    source: Path,
    eval_keys: dict[str, list[str]],
    seed: int,
    *,
    tokenizer: Any | None,
    max_prompt_length: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = pq.read_table(source).to_pylist()
    if len({_source_id(row["extra_info"]["index"]) for row in source_rows}) != len(
        source_rows
    ):
        raise ValueError(
            "DAPO source must be the unique-ID parquet, not expanded copies"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[normalize_text(problem_from_dapo_row(row))].append(row)

    converted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for key, group in groups.items():
        row = sorted(group, key=lambda item: _source_id(item["extra_info"]["index"]))[0]
        source_ids = sorted(_source_id(item["extra_info"]["index"]) for item in group)
        answers = {item["reward_model"]["ground_truth"] for item in group}
        if len(answers) != 1:
            exclusions.append(
                {
                    "source_id": _source_id(row["extra_info"]["index"]),
                    "source_ids": source_ids,
                    "reason": "conflicting_answers",
                    "answers": sorted(str(answer) for answer in answers),
                }
            )
            continue
        if key in eval_keys:
            exclusions.append(
                {
                    "source_id": _source_id(row["extra_info"]["index"]),
                    "source_ids": source_ids,
                    "reason": "evaluation_exact_normalized_overlap",
                    "evaluation_sets": eval_keys[key],
                }
            )
            continue
        problem = problem_from_dapo_row(row)
        answer = row["reward_model"]["ground_truth"]
        messages = make_messages(problem)
        prompt_tokens = rendered_prompt_len(tokenizer, messages)
        if max_prompt_length is not None and prompt_tokens > max_prompt_length:
            exclusions.append(
                {
                    "source_id": _source_id(row["extra_info"]["index"]),
                    "source_ids": source_ids,
                    "reason": "bare_prompt_over_cap",
                    "bare_prompt_tokens": prompt_tokens,
                    "max_prompt_length": max_prompt_length,
                }
            )
            continue
        converted.append(
            {
                "messages": messages,
                "answer": answer,
                "source_id": _source_id(row["extra_info"]["index"]),
                "benchmark": "dapo_math",
                "data_source": row["data_source"],
                "source_revision": DAPO_SOURCE_REV,
                "source_dataset_id": "BytedTsinghua-SIA/DAPO-Math-17k",
                "source_row_index": _source_id(row["extra_info"]["index"]),
                "source_raw_id": _source_id(row["extra_info"]["index"]),
                "original_answer": answer,
                "prompt_tokens": prompt_tokens,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(converted)
    return converted, exclusions


def convert_eval(
    eval_root: Path,
    *,
    tokenizer: Any | None,
    max_prompt_length: int | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[str]],
    dict[str, dict[str, int]],
]:
    rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    leading_zero_scan: dict[str, dict[str, int]] = {
        benchmark: {"scanned": 0, "leading_zero_ground_truth": 0, "corrected": 0}
        for benchmark in EVAL_FILES
    }
    eval_keys: dict[str, list[str]] = defaultdict(list)
    for benchmark, filename in EVAL_FILES.items():
        path = eval_root / filename
        table_rows = pq.read_table(path).to_pylist()
        expected_rows = EXPECTED_EVAL_ROWS[benchmark]
        if len(table_rows) != expected_rows:
            raise ValueError(
                f"{path} has {len(table_rows)} rows, expected {expected_rows}"
            )
        for local_index, row in enumerate(table_rows):
            problem = problem_from_eval_row(row)
            eval_keys[normalize_text(problem)].append(benchmark)
            original_answer = row["reward_model"]["ground_truth"]
            answer = canonicalize_ground_truth(original_answer, benchmark)
            leading_zero_scan[benchmark]["scanned"] += 1
            if ZERO_PADDED_GT_RE.fullmatch(original_answer):
                leading_zero_scan[benchmark]["leading_zero_ground_truth"] += 1
            extra_info = row.get("extra_info") or {}
            metadata = json.loads(extra_info.get("source_metadata_json") or "{}")
            source_raw_id = metadata.get("id", extra_info.get("index", local_index))
            source_id = f"{benchmark}:{source_raw_id}"
            messages = make_messages(problem)
            prompt_tokens = rendered_prompt_len(tokenizer, messages)
            if max_prompt_length is not None and prompt_tokens > max_prompt_length:
                raise ValueError(
                    f"Eval prompt exceeds max_prompt_length: source_id={source_id} "
                    f"benchmark={benchmark} prompt_tokens={prompt_tokens} "
                    f"max_prompt_length={max_prompt_length}"
                )
            if answer != original_answer:
                leading_zero_scan[benchmark]["corrected"] += 1
                corrections.append(
                    {
                        "benchmark": benchmark,
                        "row": local_index,
                        "source_id": source_id,
                        "original_answer": original_answer,
                        "answer": answer,
                        "rule": "decimal_integer_leading_zero_to_int_string",
                    }
                )
            rows.append(
                {
                    "messages": messages,
                    "answer": answer,
                    "source_id": source_id,
                    "benchmark": benchmark,
                    "data_source": row["data_source"],
                    "source_revision": str(extra_info.get("source_revision", "")),
                    "source_dataset_id": str(extra_info.get("source_id", "")),
                    "source_row_index": _source_id(
                        extra_info.get("index", local_index)
                    ),
                    "source_raw_id": _source_id(source_raw_id),
                    "original_answer": original_answer,
                    "prompt_tokens": prompt_tokens,
                }
            )
    if len(rows) != sum(EXPECTED_EVAL_ROWS.values()):
        raise ValueError(f"Converted {len(rows)} eval rows, expected 700")
    return rows, corrections, exclusions, dict(eval_keys), leading_zero_scan


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def build_dataset(
    source: Path,
    eval_root: Path,
    output: Path,
    seed: int,
    *,
    model_path: str | None = None,
    max_prompt_length: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if sha256_file(source) != DAPO_SOURCE_SHA256:
        raise ValueError(f"DAPO source hash mismatch: {source}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"Refusing to overwrite non-empty output directory: {output}"
            )
        shutil.rmtree(output)

    tokenizer = load_tokenizer(model_path)
    eval_rows, corrections, eval_exclusions, eval_keys, leading_zero_scan = (
        convert_eval(
            eval_root,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
        )
    )
    train_rows, train_exclusions = convert_train(
        source,
        eval_keys,
        seed,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )
    exclusions = [*train_exclusions, *eval_exclusions]
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(eval_rows),
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output))

    correction_map_path = output / "correction_map.json"
    exclusions_path = output / "exclusions.jsonl"
    write_json(correction_map_path, corrections)
    exclusions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in exclusions
        )
    )

    manifest = {
        "schema_version": 1,
        "dataset": "ppo-math-v1",
        "seed": seed,
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "revision": DAPO_SOURCE_REV,
            "rows": pq.read_metadata(source).num_rows,
        },
        "eval_root": str(eval_root),
        "eval_files": {
            benchmark: {
                "path": str(eval_root / filename),
                "sha256": sha256_file(eval_root / filename),
                "rows": EXPECTED_EVAL_ROWS[benchmark],
            }
            for benchmark, filename in EVAL_FILES.items()
        },
        "splits": {"train": len(train_rows), "test": len(eval_rows)},
        "eval_selection": {
            "source_candidate_rows_with_hmmt": SOURCE_EVAL_CANDIDATE_ROWS_WITH_HMMT,
            "selected_rows": len(eval_rows),
            "excluded_rows": SOURCE_EVAL_CANDIDATE_ROWS_WITH_HMMT - len(eval_rows),
            "excluded_benchmark_rows": EXCLUDED_EVAL_BENCHMARK_ROWS,
        },
        "test_counts_by_benchmark": dict(
            Counter(row["benchmark"] for row in eval_rows)
        ),
        "train_counts_by_benchmark": dict(
            Counter(row["benchmark"] for row in train_rows)
        ),
        "exclusions": dict(Counter(row["reason"] for row in exclusions)),
        "loss_bound": {
            "max_prompt_length": max_prompt_length,
            "tokenizer_model_path": model_path,
            "train_max_prompt_tokens": max(
                (row["prompt_tokens"] for row in train_rows), default=0
            ),
            "test_max_prompt_tokens": max(
                (row["prompt_tokens"] for row in eval_rows), default=0
            ),
        },
        "scorer": {
            "path": SCORER_PATH,
            "reward_version": "math-prd-semantic-boxed-v2-standalone",
        },
        "corrections": {
            "count": len(corrections),
            "by_benchmark": dict(Counter(row["benchmark"] for row in corrections)),
            "path": str(correction_map_path),
            "sha256": sha256_file(correction_map_path),
            "leading_zero_scan": leading_zero_scan,
        },
        "exclusion_path": str(exclusions_path),
        "exclusion_sha256": sha256_file(exclusions_path),
        "output_files": output_file_hashes(output),
        "prompt_contract": (
            "Problem-only user prompt plus boxed-answer instruction; "
            "answers are not included in messages."
        ),
        "fields": [
            "messages",
            "answer",
            "source_id",
            "benchmark",
            "data_source",
            "source_revision",
            "source_dataset_id",
            "source_row_index",
            "source_raw_id",
            "original_answer",
            "prompt_tokens",
        ],
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=os.environ.get("SAO_MATH_DAPO_SOURCE")
    )
    parser.add_argument(
        "--eval-root", type=Path, default=os.environ.get("SAO_MATH_EVAL_ROOT")
    )
    parser.add_argument(
        "--output", type=Path, default=os.environ.get("SAO_MATH_OUTPUT")
    )
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=os.environ.get("SAO_MATH_MODEL_PATH"),
    )
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    missing = [
        name
        for name in ("source", "eval_root", "output", "model_path")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Missing required path arguments or env vars: "
            + ", ".join(missing)
            + ". Use --source/--eval-root/--output/--model-path or SAO_MATH_* env vars."
        )
    return args


def main() -> None:
    args = parse_args()
    manifest = build_dataset(
        args.source,
        args.eval_root,
        args.output,
        args.seed,
        model_path=args.model_path,
        max_prompt_length=args.max_prompt_length,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
