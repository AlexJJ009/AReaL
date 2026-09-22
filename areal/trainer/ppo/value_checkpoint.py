# SPDX-License-Identifier: Apache-2.0

"""Integrity and protocol checks for independently trained scalar value artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForTokenClassification

from areal.models.transformers.scalar_value import Qwen35ScalarValueModel

MANIFEST = "value_manifest.json"
PROTOCOL_KEYS = frozenset(
    {
        "discount",
        "target_horizon",
        "thinking",
        "reward",
        "termination",
        "scorer_digest",
        "split_digest",
        "freeze_policy",
        "template_digest",
    }
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_value_manifest(
    path: str | Path,
    *,
    identity: dict[str, str],
    protocol: dict[str, Any],
    qualification: dict[str, Any],
) -> dict[str, Any]:
    """Seal an already saved artifact. This records, not grants, qualification.

    Qualification must reference an included report file. Synthetic fixtures
    never qualify a real pretrained critic, even if their reload tests pass.
    """
    root = Path(path)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["areal_scalar_value_artifact"] = True
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    files = {
        p.name: file_digest(p)
        for p in root.iterdir()
        if p.is_file()
        and p.name != MANIFEST
        and p.suffix in {".json", ".safetensors", ".jinja", ".txt"}
    }
    manifest = {
        "schema_version": "areal.scalar-value/1",
        "identity": identity,
        "protocol": protocol,
        "qualification": qualification,
        "files": files,
    }
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return validate_value_artifact(root)


def validate_value_artifact(
    path: str | Path,
    *,
    expected_identity: dict[str, str] | None = None,
    expected_protocol: dict[str, Any] | None = None,
    require_pretrained: bool = False,
) -> dict[str, Any]:
    """Validate exact artifact bytes and compare against the consuming run."""
    root = Path(path)
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("schema_version") != "areal.scalar-value/1":
        raise ValueError("Unsupported scalar value artifact schema")
    files = manifest.get("files", {})
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if name not in files:
            raise ValueError(f"Value artifact missing {name}")
    weights = {p.name for p in root.glob("*.safetensors")}
    if not json.loads((root / "config.json").read_text()).get(
        "areal_scalar_value_artifact"
    ):
        raise ValueError("Checkpoint config is not a sealed scalar value model")
    if not weights or weights != {
        name for name in files if name.endswith(".safetensors")
    }:
        raise ValueError("Value artifact weights must exactly match the manifest")
    for name, digest in files.items():
        if Path(name).name != name or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid value artifact file entry")
        if file_digest(root / name) != digest:
            raise ValueError(f"Value artifact hash mismatch: {name}")
    identity, protocol = manifest.get("identity", {}), manifest.get("protocol", {})
    if not all(
        isinstance(identity.get(key), str) and identity[key]
        for key in ("backbone_id", "tokenizer_id")
    ):
        raise ValueError("Value artifact requires backbone and tokenizer identities")
    if not PROTOCOL_KEYS <= protocol.keys():
        raise ValueError("Incomplete value target/termination/provenance protocol")
    for actual, expected, kind in (
        (identity, expected_identity, "identity"),
        (protocol, expected_protocol, "protocol"),
    ):
        if expected is not None and any(
            actual.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"Value artifact {kind} does not match the consuming run")
    qualification = manifest.get("qualification", {})
    if (
        qualification.get("kind") not in {"synthetic", "pretrained", "online"}
        or qualification.get("report") not in files
    ):
        raise ValueError("Value artifact requires a hashed qualification report")
    if require_pretrained and (
        qualification.get("kind") != "pretrained"
        or qualification.get("passed") is not True
    ):
        raise ValueError("A qualified pretrained value artifact is required")
    return manifest


def scalar_value_state(path: str | Path) -> dict:
    """Load only sealed safetensor shards, rejecting duplicate or missing heads."""
    root = Path(path)
    manifest = validate_value_artifact(root)
    state = {}
    for name in sorted(manifest["files"]):
        if name.endswith(".safetensors"):
            shard = load_file(root / name)
            if state.keys() & shard.keys():
                raise ValueError("Duplicate keys across value checkpoint shards")
            state.update(shard)
    head = state.get("score.weight", state.get("classifier.weight"))
    if head is None or head.ndim != 2 or head.shape[0] != 1:
        raise ValueError("Value checkpoint must contain a scalar score.weight head")
    return state


def validate_scalar_state_layout(path: str | Path, model) -> None:
    """Check every rank's expected keys/shapes without materializing weights."""
    root = Path(path)
    manifest = validate_value_artifact(root)
    shapes = {}
    for name in sorted(manifest["files"]):
        if name.endswith(".safetensors"):
            with safe_open(root / name, framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key in shapes:
                        raise ValueError(
                            "Duplicate keys across value checkpoint shards"
                        )
                    shapes[key] = tuple(shard.get_slice(key).get_shape())
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    if expected != shapes:
        raise ValueError(
            "Value checkpoint requires complete matching backbone and scalar head"
        )


def create_scalar_value_model(
    path: str, *, config=None, initialize_only: bool = False, **kwargs
):
    """Actual model-loader path for both tiny and dense Qwen3.5 value models."""
    config = config or AutoConfig.from_pretrained(path)
    model_class = (
        Qwen35ScalarValueModel
        if config.model_type == "qwen3_5"
        else AutoModelForTokenClassification
    )
    if model_class is AutoModelForTokenClassification:
        config.num_labels = 1
    kwargs.pop("trust_remote_code", None)
    model = model_class.from_config(config, **kwargs)
    if kwargs.get("dtype") is not None:
        model.to(dtype=kwargs["dtype"])
    validate_scalar_state_layout(path, model)
    if not initialize_only:
        model.load_state_dict(scalar_value_state(path), strict=True)
    return model
