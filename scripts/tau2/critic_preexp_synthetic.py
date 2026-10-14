# SPDX-License-Identifier: Apache-2.0
"""Generate tiny-dimensional synthetic caches with the registered pool topology."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.tau2.critic_features import SCHEMA, selected_positions
from scripts.tau2.critic_head_sim import aggregate_metrics, official_metric_row
from scripts.tau2.critic_preexp_data import (
    SEED,
    array_hash,
    load_pool,
    prefix_labels,
    select_planted_rule,
    sha256,
)
from scripts.tau2.critic_preexp_extract import token_metadata, verify_cache


class SyntheticTokenizer:
    """Only synthetic token IDs; never loads model weights or contacts a service."""

    def decode(self, ids, **_kwargs):
        tokens = {
            9: "Error",
            10: "<tool_call><function=",
            11: "cancel_reservation",
            12: "cancel_pending_order",
            13: "></function></tool_call>",
            14: "get_user_details",
        }
        return "".join(tokens.get(token, "word") for token in ids)


def make_row(task: int, run: str, index: int, *, validation: bool) -> dict:
    domain = "airline" if task < (20 if validation else 410) else "retail"
    key = f"{'test' if validation else 'train'}-{task}"
    label = float((task + index // (60 if validation else 320)) % 4 != 0)
    planted = index % 2 == 0
    return {
        # Three tokens form the generated call, followed by three action tokens.
        # Half the rows call a WRITE tool; the rest call READ get_user_details.
        "input_ids": [
            9 if domain == "retail" else 1,
            10,
            (11 if domain == "airline" else 12) if planted else 14,
            13,
            2,
            2,
            2,
        ],
        "loss_mask": [0, *([1] * 6)],
        "token_roles": ["prompt", *(["assistant"] * 6)],
        "attention_mask": [1] * 7,
        "reward": label,
        "domain": domain,
        "task_id": key,
        "source_id": f"{domain}:{key}",
        "split": "test" if validation else "train",
        "rollout_task_id": index,
        "episode_id": f"{run}:{key}:{index}",
    }


def make_fixture(out: Path) -> dict:
    """Create exact 2774/120 trajectories, but six tokens and eight feature dimensions."""
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Synthetic destination must be empty")
    out.mkdir(parents=True, exist_ok=True)
    pool = {
        "schema_version": "offline-critic-pool/1",
        "synthetic": True,
        "training": [],
        "validation": [],
        "counts": {},
    }
    # r510 reuses the first320 tasks: six trajectories total; others have two.
    for run, tasks in [
        ("r499", list(range(747)) * 2),
        ("r510", (list(range(180)) + list(range(410, 550))) * 4),
    ]:
        rows = [
            make_row(task, run, i, validation=False) for i, task in enumerate(tasks)
        ]
        source = out / f"{run}-train.jsonl"
        lines = [json.dumps(row, sort_keys=True) for row in rows]
        source.write_text("\n".join(lines) + "\n")
        file_hash = sha256(source)
        for i, row in enumerate(rows):
            pool["training"].append(
                {
                    "run": run,
                    "rollout_file": str(source),
                    "line": i + 1,
                    "file_sha256": file_hash,
                    "line_sha256": hashlib.sha256(lines[i].encode()).hexdigest(),
                    **{
                        key: row[key]
                        for key in ("domain", "task_id", "source_id", "reward")
                    },
                    "input_ids_sha256": array_hash(row["input_ids"]),
                    "loss_mask_sha256": array_hash(row["loss_mask"]),
                    "seqlen": 7,
                    "loss_tokens": 6,
                }
            )
        rows = [
            make_row(task, run, task + (60 if run == "r510" else 0), validation=True)
            for task in range(60)
        ]
        source = out / f"{run}-validation.pt"
        torch.save(rows, source)
        file_hash = sha256(source)
        for i, row in enumerate(rows):
            pool["validation"].append(
                {
                    "run": run,
                    "pt_file": str(source),
                    "row_index": i,
                    "pt_sha256": file_hash,
                    **{
                        key: row[key]
                        for key in ("domain", "task_id", "source_id", "reward")
                    },
                    "input_ids_sha256": array_hash(row["input_ids"]),
                    "loss_mask_sha256": array_hash(row["loss_mask"]),
                    "seqlen": 7,
                    "loss_tokens": 6,
                }
            )
    for section in ("training", "validation"):
        pool["counts"][section] = {"rows": len(pool[section])}
    manifest_path = out / "pool-manifest.json"
    manifest_path.write_text(json.dumps(pool, indent=2) + "\n")
    manifest_path.with_name(manifest_path.name + ".sha256").write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n"
    )
    pool, rows = load_pool(manifest_path, synthetic=True)
    tokenizer = SyntheticTokenizer()
    rule = {
        **select_planted_rule(rows, tokenizer),
        "pool_manifest_sha256": pool["pool_manifest_sha256"],
        "synthetic": True,
    }
    (out / "planted-rule.json").write_text(json.dumps(rule, indent=2) + "\n")
    generator = torch.Generator().manual_seed(SEED)
    # Identical token ordering across all layers; label-correlated synthetic signal.
    signal = torch.randn(len(rows) * 6, 8, generator=generator) * 0.25
    for index, row in enumerate(rows):
        signal[index * 6 : (index + 1) * 6, 0] += 2 * row["reward"] - 1
        positions, _ = selected_positions(row, train_k=128, seed=SEED)
        signal[index * 6 : (index + 1) * 6, 1] += (
            2 * prefix_labels(row, positions, tokenizer, rule["rule"]) - 1
        )
    cache_paths = {}
    for name, layer in [
        ("base-final", "final"),
        ("base-layer-8", "8"),
        ("base-layer-16", "16"),
        ("base-layer-24", "24"),
        ("r510-final", "final"),
    ]:
        cache = out / "caches" / name
        cache.mkdir(parents=True)
        checkpoint = out / (
            "synthetic-r510" if name.startswith("r510") else "synthetic-base"
        )
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "config.json").write_text(
            json.dumps({"hidden_size": 8, "synthetic": True}) + "\n"
        )
        shards = []
        for split in ("train", "validation"):
            features, metadata = [], {}
            for index, row in enumerate(rows):
                if row["_split"] != split:
                    continue
                positions, _ = selected_positions(row, train_k=128, seed=SEED)
                fields = token_metadata(row, positions, tokenizer, rule["rule"], layer)
                for key, values in fields.items():
                    metadata.setdefault(key, []).extend(values)
                features.append(signal[index * 6 : (index + 1) * 6])
            payload = {"features": torch.cat(features).to(torch.bfloat16), **metadata}
            filename = f"features-{split}.pt"
            torch.save(payload, cache / filename)
            shards.append(
                {
                    "path": filename,
                    "rows": len(payload["features"]),
                    "sha256": sha256(cache / filename),
                }
            )
        torch.save(
            {"weight": torch.zeros(1, 8), "bias": torch.tensor([0.78])},
            cache / "initial-head.pt",
        )
        manifest = {
            "schema": SCHEMA,
            "synthetic": True,
            "hidden_size": 8,
            "dtype": "bfloat16",
            "checkpoint": str(checkpoint),
            "layer": layer,
            "depth": 32,
            "feature_semantics": "score_input_final_norm_shifted_action_prefix"
            if layer == "final"
            else "intermediate_layer_output_shifted_action_prefix",
            "position_semantics": "prefix_position=t-1_for_action_token_t",
            "pool_manifest_sha256": pool["pool_manifest_sha256"],
            "seed": SEED,
            "train_k": 128,
            "validation_features_exact": True,
            "train_features_exact": True,
            "planted_rule": rule,
            "head_path": "initial-head.pt",
            "shards": shards,
        }
        path = cache / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        verify_cache(path, rows, pool, synthetic=True)
        cache_paths[name] = str(path)
    metric_rows = []
    for row in rows:
        if row["_split"] == "validation" and row["_run"] == "r510":
            metric_rows.append(
                official_metric_row(
                    torch.full((6,), 0.78),
                    {**row, "run": "r510", "trajectory_id": row["_trajectory_id"]},
                )
            )
    reference = {
        "completed_step": 80,
        "synthetic": True,
        **aggregate_metrics(metric_rows),
    }
    reference_path = out / "synthetic-r510-step80.json"
    reference_path.write_text(json.dumps(reference, indent=2) + "\n")
    result = {
        "synthetic": True,
        "pool_manifest": str(manifest_path),
        "caches": cache_paths,
        "r510_reference_json": str(reference_path),
        "train_rows": 2774,
        "validation_rows": 120,
        "hidden_size": 8,
        "action_tokens_per_row": 6,
    }
    (out / "fixture.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_fixture(args.out), indent=2))


if __name__ == "__main__":
    main()
