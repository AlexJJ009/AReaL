# SPDX-License-Identifier: Apache-2.0
"""Manifest-bound E1 extraction, independent trajectory shards, and exact merge."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from scripts.tau2.critic_features import (
    SCHEMA,
    _reward,
    action_positions,
    estimate,
    identity,
    selected_positions,
)
from scripts.tau2.critic_preexp_data import (
    LABEL_SEMANTICS,
    PLANTED_RULES,
    SEED,
    WRITE_TOOL_INVENTORY,
    PlantedRuleSelectionError,
    load_pool,
    prefix_labels,
    select_planted_rule,
    sha256,
)


def layer_indices(depth: int) -> list[int]:
    return [round(depth * fraction) for fraction in (0.25, 0.5, 0.75)]


def extract_layers(
    model: Any, row: dict[str, Any], positions: Any, layers: list[str]
) -> dict[str, Any]:
    """Capture selected layer outputs and verify final input against score hook."""
    import torch

    device = next(model.parameters()).device
    selected, hooks = {}, []
    backbone = model.model.language_model
    for layer in layers:
        if layer != "final":
            index = int(layer)
            if not 1 <= index < len(backbone.layers):
                raise ValueError("Intermediate layer index outside backbone")

            def capture(module, inputs, output, key=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                selected[key] = hidden[0, positions.to(device)].detach().cpu()

            hooks.append(backbone.layers[index - 1].register_forward_hook(capture))
    score_inputs = []
    hooks.append(
        model.score.register_forward_pre_hook(
            lambda module, args: score_inputs.append(
                args[0][0, positions.to(device)].detach().cpu()
            )
        )
    )
    model.eval()
    try:
        with torch.inference_mode():
            output = model(
                input_ids=torch.as_tensor(
                    row["input_ids"], dtype=torch.long, device=device
                ).reshape(1, -1),
                attention_mask=torch.as_tensor(
                    row["attention_mask"], dtype=torch.bool, device=device
                ).reshape(1, -1),
                use_cache=False,
            )
            del output
        # Hook records the exact input used by the actual score module.
        selected["final"] = score_inputs[0]
    finally:
        for hook in hooks:
            hook.remove()
    for value in selected.values():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite extracted features")
    return selected


def token_metadata(
    row: dict[str, Any], positions: Any, tokenizer: Any, rule: str, layer: str
) -> dict[str, Any]:
    import torch

    all_positions = action_positions(row)
    ranks = torch.searchsorted(all_positions, positions)
    if "turn_ids" in row:
        turns = torch.as_tensor(row["turn_ids"]).reshape(-1)
    else:
        mask = torch.as_tensor(row["loss_mask"], dtype=torch.bool).reshape(-1)
        starts = mask & ~torch.cat([torch.zeros(1, dtype=torch.bool), mask[:-1]])
        turns = starts.long().cumsum(0) - 1
    domain, task = identity(row)
    n = len(positions)
    metadata = {
        key: [value] * n
        for key, value in {
            "trajectory_id": row["_trajectory_id"],
            "step": row["_trainer_step"],
            "order": row["_episode_order"] % 16,
            "domain": domain,
            "task_id": task,
            "source_id": row["source_id"],
            "run": row["_run"],
            "reward": _reward(row),
            "original_action_count": len(all_positions),
            "sampling_weight": len(all_positions) / n,
            "split": row["_split"],
            "layer": str(layer),
            "pool_index": row["_pool_index"],
        }.items()
    }
    metadata.update(
        position=positions.tolist(),
        turn=turns[positions + 1].tolist(),
        progress=(ranks.float() / max(1, len(all_positions) - 1)).tolist(),
        planted_label=prefix_labels(row, positions, tokenizer, rule).tolist(),
    )
    return metadata


def write_part(
    model,
    rows,
    output,
    *,
    pool_manifest,
    tokenizer,
    rule,
    layers,
    checkpoint,
    shard_index,
    shard_count,
):
    """Write one selected-position tensor per trajectory/layer, bounded CPU memory."""
    import torch

    if output.exists():
        raise ValueError("Extraction part destination already exists")
    output.mkdir(parents=True)
    depth = len(model.model.language_model.layers)
    checkpoint_files = {
        p.name: sha256(p) for p in sorted(Path(checkpoint).glob("*.safetensors"))
    }
    common = {
        "schema": SCHEMA,
        "hidden_size": model.score.in_features,
        "dtype": "bfloat16",
        "position_semantics": "prefix_position=t-1_for_action_token_t",
        "pool_manifest_sha256": pool_manifest["pool_manifest_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_files_sha256": checkpoint_files,
        "checkpoint_config_sha256": sha256(Path(checkpoint) / "config.json")
        if (Path(checkpoint) / "config.json").exists()
        else None,
        "seed": SEED,
        "train_k": 128,
        "validation_features_exact": True,
        "planted_rule": rule,
        "depth": depth,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "run_ids": ["r499", "r510"],
        "head_path": "initial-head.pt",
        "train_features_exact": False,
    }
    records = {str(layer): [] for layer in layers}
    for row in rows:
        if row["_pool_index"] % shard_count != shard_index:
            continue
        positions, _ = selected_positions(row, train_k=128, seed=SEED)
        features = extract_layers(model, row, positions, layers)
        metadata = token_metadata(row, positions, tokenizer, rule["rule"], "final")
        for layer in layers:
            layer = str(layer)
            directory = output / layer
            directory.mkdir(exist_ok=True)
            payload = {
                "features": features[layer].to(torch.bfloat16),
                **metadata,
                "layer": [layer] * len(positions),
            }
            filename = f"features-{row['_pool_index']:05d}.pt"
            torch.save(payload, directory / filename)
            records[layer].append(
                {
                    "path": filename,
                    "rows": len(positions),
                    "sha256": sha256(directory / filename),
                }
            )
    for layer, shards in records.items():
        directory = output / layer
        directory.mkdir(exist_ok=True)
        torch.save(
            {k: v.detach().cpu().float() for k, v in model.score.state_dict().items()},
            directory / "initial-head.pt",
        )
        manifest = {
            **common,
            "layer": layer,
            "feature_semantics": "score_input_final_norm_shifted_action_prefix"
            if layer == "final"
            else "intermediate_layer_output_shifted_action_prefix",
            "shards": shards,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify_cache(
    manifest_path, rows, pool_manifest, *, synthetic=False, tokenizer=None
):
    """Reject missing/duplicate/misaligned positions or identities after merge."""
    import torch

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["pool_manifest_sha256"] != pool_manifest["pool_manifest_sha256"]:
        raise ValueError("Cache pool manifest SHA mismatch")
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("position_semantics")
        != "prefix_position=t-1_for_action_token_t"
    ):
        raise ValueError("Cache schema/prefix semantics mismatch")
    if not synthetic and (
        manifest.get("seed") != SEED or manifest.get("train_k") != 128
    ):
        raise ValueError("Cache registered seed/sample count mismatch")
    if manifest.get("layer") != "final" and not 1 <= int(manifest["layer"]) < int(
        manifest["depth"]
    ):
        raise ValueError("Cache intermediate layer depth mismatch")
    rule = manifest.get("planted_rule", {})
    if (
        rule.get("label_semantics") != LABEL_SEMANTICS
        or rule.get("status") != "selected"
        or rule.get("rule") not in PLANTED_RULES
        or rule.get("write_tool_inventory") != WRITE_TOOL_INVENTORY
    ):
        raise ValueError(
            "Cache planted rule semantics/status mismatch; regenerate labels"
        )
    expected = {r["_trajectory_id"]: r for r in rows}
    observed = {}
    numeric_expected = {}
    label_expected = {}
    for trajectory, row in expected.items():
        selected, count = selected_positions(
            row, train_k=manifest["train_k"], seed=manifest["seed"]
        )
        all_positions = action_positions(row)
        ranks = torch.searchsorted(all_positions, selected).tolist()
        if "turn_ids" in row:
            turns = torch.as_tensor(row["turn_ids"]).reshape(-1)
        else:
            mask = torch.as_tensor(row["loss_mask"], dtype=torch.bool).reshape(-1)
            turns = (
                mask & ~torch.cat([torch.zeros(1, dtype=torch.bool), mask[:-1]])
            ).long().cumsum(0) - 1
        numeric_expected[trajectory] = {
            int(position): {
                "step": row["_trainer_step"],
                "order": row["_episode_order"] % 16,
                "original_action_count": count,
                "sampling_weight": count / len(selected),
                "progress": rank / max(1, count - 1),
                "turn": int(turns[position + 1]),
                "pool_index": row["_pool_index"],
            }
            for position, rank in zip(selected.tolist(), ranks, strict=True)
        }
        if tokenizer is not None:
            label_expected[trajectory] = dict(
                zip(
                    selected.tolist(),
                    prefix_labels(
                        row, selected, tokenizer, manifest["planted_rule"]["rule"]
                    ).tolist(),
                    strict=True,
                )
            )
    for record in manifest["shards"]:
        path = manifest_path.parent / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Cache shard SHA mismatch")
        part = torch.load(path, map_location="cpu", weights_only=True)
        n = len(part["features"])
        if (
            n != record["rows"]
            or part["features"].shape != (n, manifest["hidden_size"])
            or part["features"].dtype != torch.bfloat16
        ):
            raise ValueError("Cache shard shape/dtype/count mismatch")
        if not torch.isfinite(part["features"]).all():
            raise ValueError("Cache features nonfinite")
        if any(len(value) != n for key, value in part.items() if key != "features"):
            raise ValueError("Cache metadata count mismatch")
        for i, trajectory in enumerate(part["trajectory_id"]):
            if trajectory not in expected:
                raise ValueError("Cache contains unselected trajectory")
            row = expected[trajectory]
            for key, value in [
                ("domain", row["domain"]),
                ("task_id", row["task_id"]),
                ("source_id", row["source_id"]),
                ("run", row["_run"]),
                ("split", row["_split"]),
                ("layer", manifest["layer"]),
            ]:
                if str(part[key][i]) != str(value):
                    raise ValueError(f"Cache source metadata mismatch: {key}")
            if float(part["reward"][i]) != _reward(row):
                raise ValueError("Cache reward mismatch")
            position = int(part["position"][i])
            if position not in numeric_expected[trajectory]:
                raise ValueError("Cache sampled position not selected")
            numerical = numeric_expected[trajectory][position]
            for key, value in numerical.items():
                if abs(float(part[key][i]) - value) > 1e-6:
                    raise ValueError(f"Cache numeric metadata mismatch: {key}")
            if float(part["planted_label"][i]) not in (0.0, 1.0):
                raise ValueError("Cache planted label is nonbinary")
            if (
                tokenizer is not None
                and float(part["planted_label"][i])
                != label_expected[trajectory][position]
            ):
                raise ValueError("Cache prefix-visible planted label mismatch")
            observed.setdefault(trajectory, []).append(position)
    if set(observed) != set(expected):
        raise ValueError("Cache trajectory count mismatch")
    for trajectory, row in expected.items():
        positions, _ = selected_positions(
            row, train_k=manifest["train_k"], seed=manifest["seed"]
        )
        if sorted(observed[trajectory]) != positions.tolist():
            raise ValueError("Cache sampled positions mismatch or duplicate")
    return {
        "trajectories": len(expected),
        "feature_rows": sum(map(len, observed.values())),
        "status": "pass",
    }


def merge_parts(parts, output, rows, pool_manifest, *, tokenizer=None):
    """Merge a full set of disjoint process caches and verify every source position."""
    parts = [Path(p) for p in parts]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in parts]
    common_keys = (
        "pool_manifest_sha256",
        "checkpoint",
        "checkpoint_files_sha256",
        "checkpoint_config_sha256",
        "layer",
        "depth",
        "hidden_size",
        "seed",
        "train_k",
        "planted_rule",
        "shard_count",
    )
    if any(any(m[k] != manifests[0][k] for k in common_keys) for m in manifests):
        raise ValueError("Extraction part provenance mismatch")
    if sorted(m["shard_index"] for m in manifests) != list(
        range(manifests[0]["shard_count"])
    ):
        raise ValueError("Incomplete or duplicate extraction process shards")
    import torch

    heads = [
        torch.load(part / "initial-head.pt", map_location="cpu", weights_only=True)
        for part in parts
    ]
    for head in heads[1:]:
        if set(head) != set(heads[0]) or any(
            not torch.equal(head[key], heads[0][key]) for key in head
        ):
            raise ValueError("Extraction initial head mismatch")
    output = Path(output)
    if output.exists():
        raise ValueError("Merged cache destination already exists")
    output.mkdir(parents=True)
    records = []
    for part, manifest in zip(parts, manifests, strict=True):
        for record in manifest["shards"]:
            destination = output / record["path"]
            if destination.exists():
                raise ValueError("Duplicate merged shard file")
            if sha256(part / record["path"]) != record["sha256"]:
                raise ValueError("Extraction part SHA mismatch")
            shutil.copyfile(part / record["path"], destination)
            records.append(record)
    shutil.copyfile(parts[0] / "initial-head.pt", output / "initial-head.pt")
    manifest = {**manifests[0], "shards": sorted(records, key=lambda r: r["path"])}
    manifest.pop("shard_index")
    manifest["sources"] = [
        {"path": source, "sha256": sha256(source)}
        for source in sorted({r["_source_path"] for r in rows})
    ]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify_cache(
        output / "manifest.json", rows, pool_manifest, tokenizer=tokenizer
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["estimate", "rule", "extract", "merge"], required=True
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--rule-file", type=Path)
    parser.add_argument("--model-name", choices=["base", "r510"], default="base")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--tokens-per-second", type=float, default=4000)
    args = parser.parse_args()
    pool, rows = load_pool(args.pool)
    if args.mode == "estimate":
        payload = estimate(
            rows,
            hidden_size=2560,
            train_k=128,
            seed=SEED,
            tokens_per_second=args.tokens_per_second,
            gpu_count=args.shard_count,
        )
        if args.checkpoint is not None:
            payload["base_checkpoint"] = str(args.checkpoint.resolve())
            payload["base_config_sha256"] = sha256(args.checkpoint / "config.json")
        payload["five_caches_feature_bytes"] = 5 * payload["feature_tensor_bytes"]
        payload["peak_feature_disk_bytes_parts_plus_merge"] = (
            10 * payload["feature_tensor_bytes"]
        )
        payload["total_two_model_wall_seconds"] = 2 * payload["estimated_wall_seconds"]
        payload["total_two_model_gpu_hours"] = 2 * payload["estimated_gpu_hours"]
        payload["planted_rule_cpu_decode_calls"] = sum(
            len(selected_positions(r, train_k=128, seed=SEED)[0])
            for r in rows
            if r["_split"] == "train"
        )
        payload["planted_rule_cpu_decoder_roundtrips"] = sum(
            r["_split"] == "train" for r in rows
        )
        payload["estimated_cpu_rule_seconds"] = (
            0.08 * payload["planted_rule_cpu_decoder_roundtrips"]
        )
        payload["planted_rule_cpu_time_assumption"] = (
            "0.08 seconds/trajectory for ByteLevel roundtrip plus128byte prefixes (local23177-token example measured0.076s); source readback/model initialization excluded; fallback generic tokenizer slower"
        )

        payload["base_plus_trained_forward_tokens"] = 2 * payload["input_tokens"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        return
    if args.mode == "merge":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        layers = (
            ["final"]
            if args.model_name == "r510"
            else [
                "final",
                *map(
                    str,
                    layer_indices(
                        json.loads(
                            (
                                args.parts / "part-0" / "final" / "manifest.json"
                            ).read_text()
                        )["depth"]
                    ),
                ),
            ]
        )
        for layer in layers:
            name = (
                f"{args.model_name}-final"
                if layer == "final"
                else f"base-layer-{layer}"
            )
            merge_parts(
                [args.parts / f"part-{i}" / layer for i in range(args.shard_count)],
                args.output / name,
                rows,
                pool,
                tokenizer=tokenizer,
            )
        return
    from transformers import AutoTokenizer

    if args.tokenizer is None or not args.tokenizer.is_dir():
        parser.error("Requires an existing local tokenizer directory")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.mode == "rule":
        started = time.perf_counter()
        failure = None
        try:
            rule = select_planted_rule(rows, tokenizer)
        except PlantedRuleSelectionError as error:
            rule, failure = error.report, error
        rule["cpu_wall_seconds"] = time.perf_counter() - started
        rule["cpu_timing_excludes_source_readback"] = True
        rule["tokenizer_path"] = str(args.tokenizer.resolve())
        rule["pool_manifest_sha256"] = pool["pool_manifest_sha256"]
        args.output.write_text(json.dumps(rule, indent=2) + "\n")
        if failure is not None:
            raise failure
        return
    if args.checkpoint is None or not args.checkpoint.is_dir():
        parser.error("Requires an existing local checkpoint directory")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid process shard index/count")
    rule = json.loads(args.rule_file.read_text())
    if (
        rule["pool_manifest_sha256"] != pool["pool_manifest_sha256"]
        or rule["selection_split"] != "train"
        or rule.get("label_semantics") != LABEL_SEMANTICS
        or rule.get("status") != "selected"
        or rule.get("rule") not in PLANTED_RULES
        or rule.get("write_tool_inventory") != WRITE_TOOL_INVENTORY
    ):
        raise ValueError("Planted rule provenance mismatch")
    import torch

    from areal.models.transformers.token_critic import (
        Qwen35TokenCriticForCausalBackbone,
    )

    torch.manual_seed(SEED)
    model = Qwen35TokenCriticForCausalBackbone.from_pretrained(
        str(args.checkpoint),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    layers = (
        ["final"]
        if args.model_name == "r510"
        else ["final", *map(str, layer_indices(len(model.model.language_model.layers)))]
    )
    write_part(
        model,
        rows,
        args.output,
        pool_manifest=pool,
        tokenizer=tokenizer,
        rule=rule,
        layers=layers,
        checkpoint=args.checkpoint,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )


if __name__ == "__main__":
    main()
