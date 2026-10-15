# SPDX-License-Identifier: Apache-2.0

"""Extract frozen, final-normalized critic features at causal action prefixes.

Estimate mode never constructs a model or initializes CUDA. Training positions are
sampled without replacement per trajectory; validation always retains all positions.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "tau2-critic-features-v1"
_REPLAY_TASK_ID_OFFSET = 1 << 40


def canonical_episode_order(path: Path) -> int:
    return int(path.stem) % _REPLAY_TASK_ID_OFFSET if path.stem.isdigit() else -1


def identity(row: dict[str, Any]) -> tuple[str, str]:
    """Return domain/task identity, refusing unidentified training examples."""
    metadata = row.get("metadata", {})
    domain = row.get("domain", metadata.get("domain"))
    task = row.get("task_id", metadata.get("task_id"))
    if domain is None or task is None:
        raise ValueError("Rollout lacks domain/task_id identity")
    return str(domain), str(task)


def load_stored_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Read rollout tensors, preserving source provenance and numeric episode order."""
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        rows = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(rows, list):
            raise ValueError("Stored validation tensor file must contain a list")
        metadata_path = path.with_name("fixed-validation-manifest.json")
        if not metadata_path.exists():
            raise ValueError(
                "Validation tensors require fixed-validation-manifest.json"
            )
        manifest = json.loads(metadata_path.read_text())
        expected_hash = manifest.get("rollouts_sha256")
        if (
            expected_hash
            and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError("Validation rollout SHA256 mismatch")
        task_rows = manifest["payload"]["task_ids"]
        if len(task_rows) != len(rows):
            raise ValueError("Validation metadata row count mismatch")
        for row, task in zip(rows, task_rows, strict=True):
            for key in ("domain", "task_id", "source_id"):
                if key in row and key in task and str(row[key]) != str(task[key]):
                    raise ValueError("Validation metadata identity mismatch")
                if key in task:
                    row[key] = task[key]
    else:
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if any(not isinstance(row, dict) or "input_ids" not in row for row in rows):
            raise ValueError(f"Malformed non-tensor JSONL rollout record: {path}")
        if not rows:
            raise ValueError(f"No rollout tensor rows: {path}")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "input_ids" not in row:
            raise ValueError(f"Not a rollout tensor row: {path}")
        row = dict(row)
        row["_source_path"] = str(path.resolve())
        row["_source_row"] = index
        row["_episode_order"] = (
            int(
                row.get(
                    "rollout_task_id",
                    canonical_episode_order(path) if path.stem.isdigit() else index,
                )
            )
            % _REPLAY_TASK_ID_OFFSET
        )
        identity(row)
        result.append(row)
    return result


def iter_rollouts(
    paths: list[str | Path], *, split: str, run: str, batch_size: int = 16
) -> list[dict[str, Any]]:
    """Load numerically ordered episode files with explicit run/split provenance."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ordered = sorted(
        (Path(p) for p in paths),
        key=lambda p: (str(p.parent.resolve()), canonical_episode_order(p), str(p)),
    )
    result = []
    seen_orders = set()
    for path in ordered:
        episode_key = canonical_episode_order(path)
        source_key = (str(path.parent.resolve()), episode_key)
        if episode_key >= 0 and source_key in seen_orders:
            raise ValueError(f"Duplicate original/replay episode id: {episode_key}")
        seen_orders.add(source_key)
        for row in load_stored_rollouts(path):
            metadata = row.get("metadata", {})
            declared_splits = [
                container.get(key)
                for container in (row, metadata)
                for key in ("split", "critic_split")
            ]
            if split == "train" and any(
                value in ("test", "dev", "validation", "eval")
                for value in declared_splits
            ):
                raise ValueError("Refusing to train on held-out task rollout")
            row.update(_split=split, _run=run)
            row["_trainer_step"] = int(
                row.get("trainer_step", row["_episode_order"] // batch_size)
            )
            row["_trajectory_id"] = (
                f"{run}:{split}:{row['_source_path']}:{row['_episode_order']}:{row['_source_row']}"
            )
            result.append(row)
    return sorted(
        result,
        key=lambda row: (
            str(Path(row["_source_path"]).parent),
            row["_trainer_step"],
            row["_episode_order"],
            row["_source_row"],
        ),
    )


def validate_splits(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> None:
    """Prevent every held-out domain/task identity from entering training."""
    overlap = {identity(row) for row in train} & {identity(row) for row in validation}
    if overlap:
        raise ValueError(f"Train/validation task overlap: {sorted(overlap)[:5]}")


def action_positions(row: dict[str, Any]):
    """Return t-1 prefix positions for action tokens t, matching official loss mask."""
    import torch

    ids = torch.as_tensor(row["input_ids"])
    if (
        ids.ndim not in (1, 2)
        or (ids.ndim == 2 and ids.shape[0] != 1)
        or ids.numel() == 0
    ):
        raise ValueError("Each rollout must contain one nonempty trajectory")
    mask = torch.as_tensor(row["loss_mask"], dtype=torch.bool).reshape(-1)
    attention = torch.as_tensor(row["attention_mask"], dtype=torch.bool).reshape(-1)
    if mask.shape != attention.shape or mask.numel() != ids.numel():
        raise ValueError("Mask shape mismatch")
    if bool(mask[0]):
        raise ValueError("First token cannot be an action with a preceding prefix")
    shifted = torch.roll(mask, -1)
    shifted[-1] = False
    if bool((shifted & ~attention).any()):
        raise ValueError("Action prefix is outside attention mask")
    return torch.nonzero(shifted, as_tuple=False).flatten()


def selected_positions(row: dict[str, Any], *, train_k: int, seed: int):
    """Stable per-trajectory uniform subsampling independent of traversal order."""
    import torch

    positions = action_positions(row)
    if positions.numel() == 0:
        raise ValueError("Rollout has no action tokens")
    if train_k < 0:
        raise ValueError("train_k must be nonnegative; zero means all")
    if row.get("_split") != "train" or train_k == 0 or len(positions) <= train_k:
        return positions, len(positions)
    digest = hashlib.sha256(f"{seed}:{row['_trajectory_id']}".encode()).digest()
    generator = torch.Generator().manual_seed(
        int.from_bytes(digest[:8], "big") % (2**63)
    )
    indices = (
        torch.randperm(len(positions), generator=generator)[:train_k].sort().values
    )
    return positions[indices], len(positions)


def critic_backbone(model):
    """Return the decoder shared by Qwen3 token classification and Qwen3.5."""
    return getattr(model.model, "language_model", model.model)


def extract_score_inputs(model, row: dict[str, Any], positions=None):
    """Get the exact token critic score input without evaluating a vocabulary head."""
    import torch

    if positions is None:
        positions = action_positions(row)
    parameter = next(model.parameters())
    input_ids = torch.as_tensor(
        row["input_ids"], dtype=torch.long, device=parameter.device
    ).reshape(1, -1)
    attention = torch.as_tensor(
        row["attention_mask"], dtype=torch.bool, device=parameter.device
    ).reshape(1, -1)
    model.eval()
    with torch.inference_mode():
        outputs = critic_backbone(model)(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        features = (
            outputs.last_hidden_state[0, positions.to(parameter.device)].detach().cpu()
        )
    if not torch.isfinite(features).all():
        raise ValueError("Nonfinite critic features")
    return features


def _reward(row):
    import torch

    value = row.get("reward", row.get("official_score", row.get("rewards")))
    if value is None:
        value = row.get("metadata", {}).get("official_score")
    value = float(torch.as_tensor(value).reshape(-1)[-1])
    if value not in (0.0, 1.0):
        raise ValueError("Expected official binary terminal reward")
    return value


def estimate(
    rows,
    *,
    hidden_size: int,
    train_k: int,
    seed: int,
    tokens_per_second: float,
    gpu_count: int,
):
    """Report explicit throughput assumptions; sampling saves disk, not forward work."""
    if hidden_size <= 0 or tokens_per_second <= 0 or gpu_count <= 0:
        raise ValueError(
            "Estimate dimensions, throughput and GPU count must be positive"
        )
    tokens = selected = actions = 0
    counts = {}
    for row in rows:
        positions, count = selected_positions(row, train_k=train_k, seed=seed)
        tokens += (
            len(row["input_ids"][0])
            if isinstance(row["input_ids"], list)
            and isinstance(row["input_ids"][0], list)
            else int(
                math.prod(getattr(row["input_ids"], "shape", [len(row["input_ids"])]))
            )
        )
        selected += len(positions)
        actions += count
        key = f"{row['_run']}:{row['_split']}"
        counts[key] = counts.get(key, 0) + 1
    wall = tokens / tokens_per_second
    return {
        "schema": SCHEMA,
        "estimate_only": True,
        "episodes": len(rows),
        "episodes_by_run_split": counts,
        "input_tokens": tokens,
        "original_action_tokens": actions,
        "selected_feature_rows": selected,
        "feature_tensor_bytes": selected * hidden_size * 2,
        "feature_tensor_gib": selected * hidden_size * 2 / 2**30,
        "metadata_bytes_not_included": True,
        "estimated_wall_seconds": wall,
        "estimated_gpu_hours": wall * gpu_count / 3600,
        "assumptions": {
            "hidden_size": hidden_size,
            "feature_bytes_per_element": 2,
            "aggregate_input_tokens_per_second": tokens_per_second,
            "gpu_count": gpu_count,
            "forward_processes_all_input_tokens": True,
            "model_initialization_and_io_seconds_not_included": True,
        },
    }


def write_feature_cache(
    model,
    rows,
    output: Path,
    *,
    train_k: int,
    seed: int,
    dtype: str,
    shard_rows: int = 65536,
    checkpoint: str = "unknown",
):
    """Write bounded half-precision shards and row-level sampling provenance."""
    import torch

    if dtype not in ("float16", "bfloat16") or shard_rows <= 0:
        raise ValueError("Invalid shard dtype or row limit")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or list(output.glob("features-*.pt")):
        raise ValueError("Feature destination already contains a cache")
    storage_dtype = getattr(torch, dtype)
    shards = []
    pending = []
    pending_rows = 0
    hidden_size = model.score.in_features
    fields = (
        "trajectory_id",
        "position",
        "step",
        "order",
        "domain",
        "task_id",
        "run",
        "reward",
        "original_action_count",
        "sampling_weight",
        "split",
    )

    def flush():
        nonlocal pending, pending_rows
        if not pending:
            return
        payload = {"features": torch.cat([p["features"] for p in pending])}
        for key in fields:
            values = [value for part in pending for value in part[key]]
            if key in ("position", "step", "order", "original_action_count"):
                values = torch.tensor(values, dtype=torch.int64)
            elif key in ("reward", "sampling_weight"):
                values = torch.tensor(values, dtype=torch.float32)
            payload[key] = values
        name = f"features-{len(shards):05d}.pt"
        torch.save(payload, output / name)
        shards.append(
            {
                "path": name,
                "rows": pending_rows,
                "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
            }
        )
        pending, pending_rows = [], 0

    for row in rows:
        positions, count = selected_positions(row, train_k=train_k, seed=seed)
        features = extract_score_inputs(model, row, positions).to(storage_dtype)
        domain, task = identity(row)
        for start in range(0, len(positions), shard_rows):
            end = min(start + shard_rows, len(positions))
            n = end - start
            if pending_rows + n > shard_rows:
                flush()
            part = {
                "features": features[start:end],
                "position": positions[start:end].tolist(),
            }
            scalar = {
                "trajectory_id": row["_trajectory_id"],
                "step": row["_trainer_step"],
                "order": row["_episode_order"],
                "domain": domain,
                "task_id": task,
                "run": row["_run"],
                "reward": _reward(row),
                "original_action_count": count,
                "sampling_weight": count / len(positions),
                "split": row["_split"],
            }
            part.update({key: [value] * n for key, value in scalar.items()})
            pending.append(part)
            pending_rows += n
    flush()
    torch.save(
        {
            key: value.detach().cpu().float()
            for key, value in model.score.state_dict().items()
        },
        output / "initial-head.pt",
    )
    manifest = {
        "schema": SCHEMA,
        "hidden_size": hidden_size,
        "dtype": dtype,
        "feature_semantics": "score_input_final_norm_shifted_action_prefix",
        "position_semantics": "prefix_position=t-1_for_action_token_t",
        "checkpoint": checkpoint,
        "model_config": model.config.to_dict(),
        "model_config_sha256": hashlib.sha256(
            model.config.to_json_string().encode()
        ).hexdigest(),
        "model_architecture": type(model).__name__,
        "head_path": "initial-head.pt",
        "train_k": train_k,
        "seed": seed,
        "train_features_exact": train_k == 0
        or all(
            len(action_positions(row)) <= train_k
            for row in rows
            if row["_split"] == "train"
        ),
        "validation_features_exact": True,
        "run_ids": sorted({row["_run"] for row in rows}),
        "shards": shards,
        "sources": [
            {
                "path": path,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in sorted({row["_source_path"] for row in rows})
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", action="append", default=[], help="run_id=JSONL glob (repeatable)"
    )
    parser.add_argument(
        "--validation",
        action="append",
        default=[],
        help="run_id=fixed-validation-rollouts.pt (repeatable)",
    )
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", help="Local pinned base or trained token critic checkpoint"
    )
    parser.add_argument("--train-k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--tokens-per-second", type=float, default=4000)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--shard-rows", type=int, default=65536)
    args = parser.parse_args()
    train, validation = [], []
    for specs, split, destination in (
        (args.train, "train", train),
        (args.validation, "validation", validation),
    ):
        for spec in specs:
            run, pattern = spec.split("=", 1)
            paths = glob.glob(pattern)
            if not paths:
                raise ValueError(f"No rollout paths matched {pattern}")
            destination.extend(
                iter_rollouts(paths, split=split, run=run, batch_size=args.batch_size)
            )
    validate_splits(train, validation)
    rows = train + validation
    if not rows:
        parser.error("Provide train or validation rollout paths")
    if args.estimate_only:
        payload = estimate(
            rows,
            hidden_size=args.hidden_size,
            train_k=args.train_k,
            seed=args.seed,
            tokens_per_second=args.tokens_per_second,
            gpu_count=args.gpu_count,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        return
    if not args.checkpoint:
        parser.error("Extraction requires --checkpoint")
    import torch

    from areal.models.transformers.token_critic import (
        Qwen35TokenCriticForCausalBackbone,
    )

    if not Path(args.checkpoint).is_dir():
        parser.error(
            "Checkpoint must be an existing local directory (no API downloads)"
        )
    model = Qwen35TokenCriticForCausalBackbone.from_pretrained(
        args.checkpoint,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    write_feature_cache(
        model,
        rows,
        args.output,
        train_k=args.train_k,
        seed=args.seed,
        dtype=args.dtype,
        shard_rows=args.shard_rows,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
