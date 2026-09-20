# SPDX-License-Identifier: Apache-2.0
"""Verify frozen data bytes and scorer gold roundtrips in the selected runtime."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from areal.reward.math_prd import score_math_prd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    hashes = {}
    for name, expected in manifest["output_files"].items():
        actual = sha256(args.dataset / name)
        assert actual == expected["sha256"], name
        hashes[name] = actual
    dataset = load_from_disk(str(args.dataset))
    assert len(dataset["train"]) == 17157 and len(dataset["test"]) == 700
    eval_counts = Counter(dataset["test"]["benchmark"])
    assert eval_counts == {
        "aime24": 30,
        "aime25": 30,
        "amc23": 40,
        "beyond_aime": 100,
        "math500": 500,
    }
    rows = 0
    for split in ("train", "test"):
        source_ids = list(dataset[split]["source_id"])
        assert len(set(source_ids)) == len(source_ids)
        for row in dataset[split]:
            assert row["answer"].strip()
            assert row["prompt_tokens"] <= 1024
            assert len(row["messages"]) == 1 and row["messages"][0]["role"] == "user"
            assert score_math_prd("\\boxed{" + row["answer"] + "}", row["answer"]) == 1
            rows += 1
    corrections = json.loads((args.dataset / "correction_map.json").read_text())
    assert len(corrections) == 7
    assert all(row["answer"] == str(int(row["original_answer"])) for row in corrections)
    result = {
        "passed": True,
        "gold_roundtrips": rows,
        "train_rows": len(dataset["train"]),
        "eval_counts": dict(eval_counts),
        "corrections": len(corrections),
        "manifest_sha256": sha256(args.dataset / "manifest.json"),
        "output_hashes": hashes,
        "scorer_sha256": sha256(Path("areal/reward/math_prd.py")),
        "worker_sha256": sha256(Path("areal/reward/math_prd_worker.py")),
        "lock_sha256": sha256(Path("uv.lock")),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
