# SPDX-License-Identifier: Apache-2.0
"""Validate checkpoint evidence for AWEX's Qwen4Exp frozen-state declaration."""

import hashlib
import json
import re
from pathlib import Path

from awex.models.qwen4_exp_contract import Qwen4ExpFrozenContract


def load_frozen_contract(
    manifest_path: Path, model_directory: Path
) -> Qwen4ExpFrozenContract:
    """Check declared identity and exact names against checkpoint/evidence metadata.

    This validates config/index bytes and the frozen-state evidence manifest.
    It does not read all live tensor values or authorize a runtime dependency.
    """
    manifest = json.loads(manifest_path.read_text())
    contract = Qwen4ExpFrozenContract.from_dict(manifest["contract"])
    basis = manifest["identity_basis"]
    encoded = json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded).hexdigest() != contract.checkpoint_manifest_sha256:
        raise ValueError("Frozen contract evidence identity changed")
    config = (model_directory / "config.json").read_bytes()
    index = (model_directory / "model.safetensors.index.json").read_bytes()
    if hashlib.sha256(config).hexdigest() != basis["config_sha256"]:
        raise ValueError("Checkpoint config differs from the frozen contract")
    if hashlib.sha256(index).hexdigest() != basis["weight_index_sha256"]:
        raise ValueError("Checkpoint weight index differs from the frozen contract")
    if json.loads(config).get("architectures") != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("Checkpoint is not the declared Qwen4Exp architecture")
    table_names = set()
    source_names = set()
    for shard in basis["ple_source_shards"]:
        name = shard["name"]
        match = re.fullmatch(
            r"model\.language_model\.layers\.(\d+)\.ple\.ple_embedding"
            r"\.ngram_embedding\.shard_\d+\.weight",
            name,
        )
        if match is None or name in source_names:
            raise ValueError("Invalid or duplicate PLE source shard name")
        source_names.add(name)
        table_names.add(
            f"model.layers.{match[1]}.ple.ple_embedding.ngram_embedding.weight"
        )
    if table_names != contract.ple_table_names:
        raise ValueError("PLE exclusions differ from the source evidence")
    if not basis["visual_reference"]:
        raise ValueError("Missing visual preservation evidence")
    for reference in basis["visual_reference"]:
        names = [parameter["name"] for parameter in reference["parameters"]]
        if (
            len(names) != len(set(names))
            or set(names) != contract.visual_parameter_names
        ):
            raise ValueError("Visual exclusions differ from the preservation evidence")
    return contract
