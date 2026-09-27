# SPDX-License-Identifier: Apache-2.0
"""CPU-only preflight for Qwen3.5 token critic scalar heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


class CriticHeadPreflightError(ValueError):
    """Raised when a critic checkpoint cannot safely initialize SAO."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hidden_size(config: dict[str, Any]) -> int:
    text_config = config.get("text_config")
    source = text_config if isinstance(text_config, dict) else config
    hidden = source.get("hidden_size")
    if not isinstance(hidden, int) or hidden <= 0:
        raise CriticHeadPreflightError(
            "config.json must provide a positive hidden_size"
        )
    return hidden


def _scalar_head_keys(keys: list[str]) -> list[str]:
    candidates = {
        "score.weight",
        "classifier.weight",
        "value_head.weight",
        "v_head.weight",
    }
    return sorted(key for key in keys if key in candidates)


def check_critic_head(checkpoint: str | Path) -> dict[str, Any]:
    root = Path(checkpoint).resolve()
    if not root.is_dir():
        raise CriticHeadPreflightError(f"checkpoint is not a directory: {root}")

    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    if not config_path.is_file():
        raise CriticHeadPreflightError(f"missing config.json: {config_path}")
    if not weights_path.is_file():
        raise CriticHeadPreflightError(f"missing model.safetensors: {weights_path}")

    config = _load_json(config_path)
    hidden_size = _hidden_size(config)
    has_value_manifest = (root / "value_manifest.json").is_file()
    sealed_by_config = bool(config.get("areal_scalar_value_artifact"))

    with safe_open(weights_path, framework="pt", device="cpu") as shard:
        keys = sorted(shard.keys())
        scalar_keys = _scalar_head_keys(keys)
        if scalar_keys != ["score.weight"]:
            raise CriticHeadPreflightError(
                "critic checkpoint must contain exactly one scalar head key: "
                f"score.weight; observed={scalar_keys or 'none'}"
            )
        shape = tuple(shard.get_slice("score.weight").get_shape())
        expected_shape = (1, hidden_size)
        if shape != expected_shape:
            raise CriticHeadPreflightError(
                f"score.weight shape mismatch: expected={expected_shape} observed={shape}"
            )
        head = shard.get_tensor("score.weight")

    finite = bool(torch.isfinite(head).all().item())
    nonzero_count = int(torch.count_nonzero(head).item())
    if not finite:
        raise CriticHeadPreflightError("score.weight contains non-finite values")
    if nonzero_count == 0:
        raise CriticHeadPreflightError("score.weight is all zeros")

    tokenizer_hashes = {
        name: _json_sha256(root / name)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
    }
    artifact_hashes = {
        "model.safetensors": _sha256_file(weights_path),
        "config.json": _sha256_file(config_path),
        **{
            name: digest
            for name, digest in tokenizer_hashes.items()
            if digest is not None
        },
    }
    loader_status = (
        "sealed_scalar_value_artifact"
        if has_value_manifest or sealed_by_config
        else "compat_qwen35_token_critic_unsealed"
    )
    loader_note = (
        "value_manifest/config seal present; scalar artifact strict path may apply"
        if loader_status == "sealed_scalar_value_artifact"
        else "no value_manifest/config seal; this preflight supplies the fail-closed "
        "score.weight guard for the Qwen3.5 compatibility loader"
    )
    return {
        "passed": True,
        "checkpoint": str(root),
        "schema": "areal.sao.critic-head-preflight/1",
        "artifact_hashes": artifact_hashes,
        "config": {
            "model_type": config.get("model_type"),
            "hidden_size": hidden_size,
            "architectures": config.get("architectures"),
            "areal_scalar_value_artifact": sealed_by_config,
            "has_value_manifest": has_value_manifest,
        },
        "head": {
            "key": "score.weight",
            "shape": list(head.shape),
            "dtype": str(head.dtype).removeprefix("torch."),
            "sha256": _sha256_tensor(head),
            "finite": finite,
            "nonzero_count": nonzero_count,
            "numel": int(head.numel()),
        },
        "loader_status": loader_status,
        "loader_note": loader_note,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="HF critic checkpoint path")
    parser.add_argument("--output", required=True, help="JSON report path")
    args = parser.parse_args(argv)

    output = Path(args.output)
    try:
        report = check_critic_head(args.checkpoint)
    except Exception as exc:
        _write_report(
            output,
            {
                "passed": False,
                "schema": "areal.sao.critic-head-preflight/1",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "errors": [str(exc)],
            },
        )
        return 1
    _write_report(output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
