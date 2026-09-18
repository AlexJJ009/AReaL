# SPDX-License-Identifier: Apache-2.0

"""Complete Qwen4-Exp text-training exports with fixed HF vision assets."""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Collection
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

_VISION_PREFIX = "model.visual."
_MTP_PREFIXES = ("mtp.",)
_HF_AUXILIARY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "generation_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _preserve_hf_auxiliary_files(
    source_path: str, output_path: str
) -> dict[str, list[str]]:
    """Fill missing runtime assets after tokenizer/processor exports take priority."""
    source = Path(source_path)
    output = Path(output_path)
    copied = []
    preserved = []
    for filename in _HF_AUXILIARY_FILES:
        source_file = source / filename
        output_file = output / filename
        if output_file.is_symlink():
            raise ValueError(
                f"Exported HF auxiliary file must not be a symlink: {output_file}"
            )
        if output_file.exists() and not output_file.is_file():
            raise ValueError(
                f"Exported HF auxiliary path must be a regular file: {output_file}"
            )
        if output_file.is_file():
            preserved.append(filename)
        elif source_file.is_file():
            shutil.copy2(source_file, output_file)
            copied.append(filename)
    return {"copied": copied, "preserved_existing": preserved}


def qwen4_exp_export_config(hf_config: Any, *, mtp_enabled: bool) -> Any:
    """Return an export-only config; keep the running model's config untouched."""
    exported = deepcopy(hf_config)
    text_config = getattr(exported, "text_config", exported)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        # Newer Transformers normalizes this checkpoint spelling to QSA at
        # load time. Preserve the original portable spelling when exporting:
        # the pinned SGLang Qwen4-Exp loader selects its QSA implementation
        # through "full_attention", not "qwen_sparse_attention".
        text_config.layer_types = [
            "full_attention" if kind == "qwen_sparse_attention" else kind
            for kind in layer_types
        ]
    if not mtp_enabled:
        text_config.mtp = None
        text_config.mtp_num_hidden_layers = 0
        for config in (exported, text_config):
            if hasattr(config, "mtp"):
                config.mtp = None
            for field in (
                "mtp_num_hidden_layers",
                "mtp_num_layers",
                "num_nextn_predict_layers",
            ):
                if hasattr(config, field):
                    setattr(config, field, 0)
    return exported


def finalize_mcore_bridge_checkpoint(
    source_path: str,
    output_path: str,
    *,
    hf_config: Any,
    language_model_only: bool,
    mtp_enabled: bool,
    cpu_group: dist.ProcessGroup,
    tokenizer: Any | None = None,
    processor: Any | None = None,
) -> dict[str, Any]:
    """Publish fixed weights and metadata on rank zero; broadcast errors to peers.

    All ranks must call after the bridge's tensor-save collectives have returned.
    This cannot recover a failed rank inside the upstream tensor-save collectives.
    """
    result: list[Any] = [None, None]
    if dist.get_rank(group=cpu_group) == 0:
        try:
            config = hf_config
            report = {}
            if hf_config.model_type == "qwen4_exp":
                report = restore_qwen4_exp_fixed_assets(
                    source_path,
                    output_path,
                    language_model_only=language_model_only,
                    mtp_enabled=mtp_enabled,
                )
                config = qwen4_exp_export_config(hf_config, mtp_enabled=mtp_enabled)
            config.save_pretrained(output_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_path)
            if processor is not None:
                processor.save_pretrained(output_path)
            report["auxiliary_files"] = _preserve_hf_auxiliary_files(
                source_path, output_path
            )
            result[0] = report
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(
        result,
        src=dist.get_global_rank(cpu_group, 0),
        group=cpu_group,
    )
    if result[1] is not None:
        raise RuntimeError(f"mcore-bridge checkpoint finalization failed: {result[1]}")
    return result[0]


def _checkpoint_inventory(
    directory: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Read tensor headers and verify the index against files without loading weights."""
    index_path = directory / "model.safetensors.index.json"
    index = None
    if index_path.is_file():
        if not index_path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Checkpoint index escapes its directory: {index_path}")
        with index_path.open() as stream:
            index = json.load(stream)
        if not isinstance(index.get("weight_map"), dict):
            raise ValueError(f"Invalid safetensors weight_map: {index_path}")
        for filename in index["weight_map"].values():
            if (
                not isinstance(filename, str)
                or Path(filename).is_absolute()
                or ".." in Path(filename).parts
                or Path(filename).suffix != ".safetensors"
            ):
                raise ValueError(f"Invalid checkpoint shard path: {filename!r}")
    filenames = {path.name for path in directory.glob("*.safetensors")}
    if index is not None:
        filenames.update(index["weight_map"].values())
    if not filenames:
        raise ValueError(f"No safetensors weight files in {directory}")
    weight_map = {}
    tensors = {}
    for filename in sorted(filenames):
        path = directory / filename
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Checkpoint shard escapes its directory: {path}")
        # safe_open validates the file format before offsets are used for the
        # same byte-accounting convention as MegatronEngine's index rebuild.
        with safe_open(path, framework="pt", device="cpu") as handle:
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            for key in handle.keys():
                if key in weight_map:
                    raise ValueError(
                        f"Duplicate checkpoint tensor {key} in {directory}"
                    )
                begin, end = header[key]["data_offsets"]
                weight_map[key] = filename
                tensors[key] = {
                    "shape": header[key]["shape"],
                    "dtype": header[key]["dtype"],
                    "nbytes": end - begin,
                }
    if index is not None and index["weight_map"] != weight_map:
        raise ValueError(
            f"Safetensors index disagrees with actual tensor files in {directory}"
        )
    return weight_map, tensors, {} if index is None else index.get("metadata", {})


def restore_qwen4_exp_fixed_assets(
    source_path: str,
    output_path: str,
    *,
    language_model_only: bool,
    mtp_enabled: bool = False,
    omitted_mtp_keys: Collection[str] | None = None,
    max_shard_size_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Validate an HF export and restore only missing ``model.visual.*`` tensors.

    Call on the saving rank after the bridge finished writing all tensor shards.
    Callers own distributed error propagation and config/tokenizer preservation.
    Each bucket is bounded by ``max_shard_size_bytes``; an indivisible larger
    tensor gets its own shard. Existing tensors are never copied over. With MTP
    disabled, only top-level ``mtp.*`` keys are allowed to be omitted; callers
    may narrow that whitelist by providing exact ``omitted_mtp_keys``.
    """
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    if source == output:
        raise ValueError("Source checkpoint and output checkpoint must differ.")
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be positive.")
    with (source / "config.json").open() as stream:
        config = json.load(stream)
    if config.get("model_type") != "qwen4_exp":
        raise ValueError(
            "Fixed-asset restoration requires a qwen4_exp source checkpoint."
        )
    source_map, source_tensors, _ = _checkpoint_inventory(source)
    output_map, output_tensors, output_metadata = _checkpoint_inventory(output)
    source_keys = set(source_map)
    output_keys = set(output_map)
    if omitted_mtp_keys is None:
        omitted_mtp_keys = (
            set()
            if mtp_enabled
            else {key for key in source_keys if key.startswith(_MTP_PREFIXES)}
        )
    else:
        omitted_mtp_keys = set(omitted_mtp_keys)
    if mtp_enabled and omitted_mtp_keys:
        raise ValueError("Enabled MTP cannot have omitted checkpoint keys.")
    if any(not key.startswith(_MTP_PREFIXES) for key in omitted_mtp_keys):
        raise ValueError("The omission whitelist accepts exact MTP tensor keys only.")
    if omitted_mtp_keys - source_keys:
        raise ValueError(
            f"Omitted MTP keys are absent from the source: {sorted(omitted_mtp_keys - source_keys)}"
        )
    if omitted_mtp_keys & output_keys:
        raise ValueError("The output contains MTP tensors explicitly declared omitted.")
    unknown = output_keys - source_keys
    if unknown:
        raise ValueError(
            f"Export contains unknown checkpoint tensors: {sorted(unknown)}"
        )
    # The bridge exports dequantized PLE tables as current BF16 parameters.
    # Their original FP8 scale must disappear, but only after every source
    # shard has been exported with the same shape in the new representation.
    ple_groups: dict[str, dict[int, str]] = {}
    for key in source_keys:
        match = re.fullmatch(
            r"(model\.language_model\.layers\.\d+\.ple\.ple_embedding"
            r"\.ngram_embedding)\.shard_(\d+)\.weight",
            key,
        )
        if match:
            ple_groups.setdefault(match[1], {})[int(match[2])] = key
    converted_ple_keys: set[str] = set()
    omitted_ple_scales: set[str] = set()
    for prefix, shards in ple_groups.items():
        if not any(
            source_tensors[key]["dtype"] == "F8_E4M3"
            and output_tensors.get(key, {}).get("dtype") == "BF16"
            for key in shards.values()
        ):
            continue
        if set(shards) != set(range(len(shards))) or any(
            source_tensors[key]["dtype"] != "F8_E4M3"
            or output_tensors.get(key, {}).get("dtype") != "BF16"
            or output_tensors[key]["shape"] != source_tensors[key]["shape"]
            for key in shards.values()
        ):
            raise ValueError(f"Incomplete or invalid BF16 PLE conversion: {prefix}")
        scale_key = f"{prefix}.weight_scale"
        if scale_key not in source_keys or scale_key in output_keys:
            raise ValueError(
                f"BF16 PLE conversion requires removing the source scale: {scale_key}"
            )
        converted_ple_keys.update(shards.values())
        omitted_ple_scales.add(scale_key)
    vision_keys = {key for key in source_keys if key.startswith(_VISION_PREFIX)}
    missing = source_keys - output_keys - omitted_mtp_keys - omitted_ple_scales
    copy_keys = missing & vision_keys if language_model_only else set()
    missing_required = missing - copy_keys
    if missing_required:
        raise ValueError(
            f"Export is missing required non-restorable tensors: {sorted(missing_required)}"
        )
    for key in output_keys:
        if output_tensors[key]["shape"] != source_tensors[key]["shape"]:
            raise ValueError(f"Export tensor shape differs from source: {key}")
        if (
            output_tensors[key]["dtype"] != source_tensors[key]["dtype"]
            and key not in converted_ple_keys
        ):
            raise ValueError(
                f"Export tensor dtype differs from source: {key}: "
                f"{source_tensors[key]['dtype']} -> {output_tensors[key]['dtype']}"
            )
    if language_model_only and not vision_keys:
        raise ValueError(
            "The source contains no recognized model.visual.* fixed assets."
        )

    bucket: dict[str, torch.Tensor] = {}
    bucket_size = 0
    next_shard = 1
    new_shards = []
    pending_shards = []

    def flush_bucket(cleanup: ExitStack) -> None:
        nonlocal bucket_size, next_shard
        if not bucket:
            return
        filename = f"model-fixed-visual-{next_shard:05d}.safetensors"
        while (output / filename).exists():
            next_shard += 1
            filename = f"model-fixed-visual-{next_shard:05d}.safetensors"
        with tempfile.NamedTemporaryFile(
            dir=output, prefix=".fixed-visual-", suffix=".pending", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        cleanup.callback(temporary_path.unlink, missing_ok=True)
        save_file(bucket, temporary_path, metadata={"format": "pt"})
        pending_shards.append((temporary_path, output / filename))
        for key in bucket:
            output_map[key] = filename
            output_tensors[key] = source_tensors[key]
        new_shards.append(filename)
        bucket.clear()
        bucket_size = 0
        next_shard += 1

    with ExitStack() as cleanup:
        # Group reads by source shard. get_tensor touches only selected vision
        # tensors, even when that file also stores trainable text weights.
        for filename in sorted({source_map[key] for key in copy_keys}):
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                for key in sorted(
                    key for key in copy_keys if source_map[key] == filename
                ):
                    nbytes = source_tensors[key]["nbytes"]
                    if bucket and bucket_size + nbytes > max_shard_size_bytes:
                        flush_bucket(cleanup)
                    bucket[key] = handle.get_tensor(key)
                    bucket_size += nbytes
                    if bucket_size >= max_shard_size_bytes:
                        flush_bucket(cleanup)
        flush_bucket(cleanup)
        for temporary_path, shard_path in pending_shards:
            os.replace(temporary_path, shard_path)
            cleanup.callback(shard_path.unlink, missing_ok=True)

        # HF prefers model.safetensors over an index, so it must become a shard
        # when extra files are added. The trained tensor bytes remain unchanged.
        if (
            "model.safetensors" in output_map.values()
            and len(set(output_map.values())) > 1
        ):
            shard_number = 1
            filename = f"model-exported-{shard_number:05d}.safetensors"
            while (output / filename).exists():
                shard_number += 1
                filename = f"model-exported-{shard_number:05d}.safetensors"
            os.replace(output / "model.safetensors", output / filename)
            cleanup.callback(
                os.replace, output / filename, output / "model.safetensors"
            )
            output_map = {
                key: filename if value == "model.safetensors" else value
                for key, value in output_map.items()
            }

        total_size = sum(tensor["nbytes"] for tensor in output_tensors.values())
        output_metadata = dict(output_metadata, total_size=total_size)
        index = {
            "metadata": output_metadata,
            "weight_map": dict(sorted(output_map.items())),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output,
            prefix=".model-index-",
            suffix=".pending",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            cleanup.callback(temporary_path.unlink, missing_ok=True)
            json.dump(index, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output / "model.safetensors.index.json")
        cleanup.pop_all()
    return {
        "restored_keys": sorted(copy_keys),
        "omitted_mtp_keys": sorted(omitted_mtp_keys),
        "new_shards": new_shards,
        "total_size": total_size,
    }
