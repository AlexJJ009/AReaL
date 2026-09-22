# SPDX-License-Identifier: Apache-2.0
"""Bind frozen exclusions to live MCore Parameters and actual PP ownership."""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from awex.models.qwen4_exp_contract import (
    Qwen4ExpFrozenContract,
    mcore_visual_parameter_name,
)
from torch import nn


class McoreFrozenBinder:
    """Process-local callback; fetch model objects again on every invocation.

    This validates frozen exclusions only, not trainable weight coverage. Native
    metadata conversion retains local layer numbering. The currently validated
    PLE placement has matching local/global IDs; other placements fail explicitly
    until the converter supports a separate global identity for exclusions.
    """

    def __init__(self, engine: Any, contract: Qwen4ExpFrozenContract) -> None:
        self.engine = engine
        self.contract = contract
        self._visual_shapes: dict[str, tuple[int, ...]] | None = None

    def _validate_visual_shapes(self, parameters: dict[str, nn.Parameter]) -> None:
        """The HF vision tower is replicated, so checkpoint shapes must match."""
        if self._visual_shapes is None:
            from safetensors import safe_open

            directory = Path(self.engine.config.path)
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            shapes = {}
            by_shard: dict[str, list[str]] = {}
            for name, shard in index["weight_map"].items():
                if name.startswith("model.visual."):
                    by_shard.setdefault(shard, []).append(name)
            for shard, names in by_shard.items():
                with safe_open(
                    directory / shard, framework="pt", device="cpu"
                ) as source:
                    for name in names:
                        actor_name = "visual.visual." + name[len("model.visual.") :]
                        canonical = mcore_visual_parameter_name(
                            actor_name, self.contract
                        )
                        if canonical in shapes:
                            raise ValueError(
                                f"Duplicate checkpoint visual identity: {canonical}"
                            )
                        shapes[canonical] = tuple(source.get_slice(name).get_shape())
                        parameter = parameters[canonical]
                        if tuple(parameter.shape) != shapes[canonical]:
                            raise ValueError(
                                f"Frozen actor visual shape differs from checkpoint: {canonical}"
                            )
                        checkpoint = source.get_tensor(name).to(dtype=parameter.dtype)
                        if not torch.equal(parameter.detach().cpu(), checkpoint):
                            raise ValueError(
                                f"Frozen actor visual values differ from checkpoint: {canonical}"
                            )
            if shapes.keys() != self.contract.visual_parameter_names:
                raise ValueError("Checkpoint visual keys differ from frozen contract")
            self._visual_shapes = shapes
        for name, parameter in parameters.items():
            if (
                name.startswith("model.visual.")
                and tuple(parameter.shape) != self._visual_shapes[name]
            ):
                raise ValueError(
                    f"Frozen actor visual shape differs from checkpoint: {name}"
                )

    def __call__(self, converter: Any) -> None:
        config = self.engine.mcore_config
        if (
            config.language_model_only is not self.contract.language_model_only
            or config.freeze_ple_table is not True
        ):
            raise ValueError(
                "Frozen binding requires matching actor mode and frozen PLE"
            )
        if self.engine.hf_config.architectures != ["Qwen4ExpForConditionalGeneration"]:
            raise ValueError("Frozen binding requires the Qwen4Exp architecture")
        models = self.engine.model
        if not isinstance(models, (list, tuple)) or not models:
            raise ValueError("Expected initialized MCore model chunks")
        parameters: dict[str, nn.Parameter] = {}
        global_layers: set[int] = set()
        visual_owners = 0
        local_visual_names: set[str] = set()
        stage_map = converter._pp_stage_layer_id_map
        for vp_stage, model in enumerate(models):
            unwrapped = model
            while hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            owns_visual = False
            if not self.contract.language_model_only:
                if type(getattr(unwrapped, "pre_process", None)) is not bool:
                    raise ValueError(
                        "Vision binding requires explicit chunk pre_process ownership"
                    )
                owns_visual = unwrapped.pre_process
                if owns_visual and (converter.rank_info.pp_rank != 0 or vp_stage != 0):
                    raise ValueError(
                        "Frozen visual owner must be the first PP/VP stage"
                    )
                visual_owners += int(owns_visual)
            chunk_visual_names: set[str] = set()
            layers: dict[str, tuple[int, int]] = {}
            for path, layer in model.named_modules():
                clean = _clean_name(path)
                match = re.fullmatch(r"decoder\.layers\.(\d+)", clean)
                if match is None:
                    continue
                number = getattr(layer, "layer_number", None)
                if type(number) is not int or number < 1:
                    raise ValueError(f"Missing actual global layer identity: {path}")
                local_id, global_id = int(match[1]), number - 1
                if global_id in global_layers:
                    raise ValueError(
                        f"Duplicate global layer across model chunks: {global_id}"
                    )
                global_layers.add(global_id)
                layers[path] = (local_id, global_id)
                if stage_map:
                    mapped = stage_map.get(
                        (converter.rank_info.pp_rank, vp_stage), {}
                    ).get(local_id)
                    if mapped != global_id:
                        raise ValueError(
                            f"AWEX PP map disagrees with actual layer: {path}"
                        )
            if not layers:
                raise ValueError("MCore chunk has no identifiable decoder layers")
            for name, parameter in model.named_parameters():
                clean = _clean_name(name)
                if clean.startswith("visual."):
                    if self.contract.language_model_only or not owns_visual:
                        raise ValueError(
                            "Unexpected actor visual parameters on this chunk"
                        )
                    canonical = mcore_visual_parameter_name(name, self.contract)
                    if canonical in parameters:
                        raise ValueError(
                            f"Duplicate frozen parameter identity: {canonical}"
                        )
                    parameters[canonical] = parameter
                    chunk_visual_names.add(canonical)
                    continue
                if ".ple_embedding." not in clean:
                    continue
                match = re.fullmatch(r"(.+\.layers\.\d+)\.(.+)", name)
                if match is None or match[1] not in layers:
                    raise ValueError(f"PLE parameter has no actual layer owner: {name}")
                local_id, global_id = layers[match[1]]
                canonical = f"model.layers.{global_id}.{match[2]}"
                if not stage_map and local_id != global_id:
                    raise ValueError(
                        "Metadata PLE local/global identity differs; unsupported placement"
                    )
                if canonical in parameters:
                    raise ValueError(
                        f"Duplicate frozen parameter identity: {canonical}"
                    )
                parameters[canonical] = parameter
            if (
                owns_visual
                and chunk_visual_names != self.contract.visual_parameter_names
            ):
                raise ValueError(
                    "Owning actor chunk must contain the complete visual tower"
                )
            local_visual_names.update(chunk_visual_names)
        if not self.contract.language_model_only:
            if visual_owners != int(converter.rank_info.pp_rank == 0):
                raise ValueError("Missing or duplicate frozen visual PP owner")
            if local_visual_names:
                self._validate_visual_shapes(parameters)
        expected = frozenset(
            name
            for name in self.contract.ple_table_names
            if int(name.split(".")[2]) in global_layers
        )
        converter.bind_frozen_contract(
            self.contract, parameters, expected, frozenset(local_visual_names)
        )


def _clean_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    if name.startswith("language_model."):
        name = name[len("language_model.") :]
    return name


def load_actor_frozen_contract(engine: Any) -> Qwen4ExpFrozenContract:
    """Load explicit local evidence without enabling the engine AWEX guard."""
    from areal.models.mcore.qwen4_exp_awex_contract import load_frozen_contract

    if engine.bridge_cls != "mcore-bridge":
        raise ValueError("Qwen4Exp frozen contract requires mcore-bridge")
    if engine.mcore_config.freeze_ple_table is not True:
        raise ValueError("Frozen binding requires frozen PLE")
    manifest = os.environ.get("QWEN_AWEX_FROZEN_CONTRACT")
    if not manifest:
        raise ValueError("Qwen4Exp AWEX requires QWEN_AWEX_FROZEN_CONTRACT")
    contract = load_frozen_contract(Path(manifest), Path(engine.config.path))
    if engine.mcore_config.language_model_only is not contract.language_model_only:
        raise ValueError("Actor model mode differs from frozen contract")
    return contract


def build_awex_train_info(engine: Any, world_size: int) -> dict[str, Any]:
    """Use one payload for eager publication and adapter initialization."""
    info: dict[str, Any] = {"train_world_size": world_size}
    if engine.hf_config.architectures == ["Qwen4ExpForConditionalGeneration"]:
        info["qwen4_exp_frozen_contract"] = load_actor_frozen_contract(engine).to_dict()
    return info


class SglangFrozenBinder:
    """Bind current original inference Parameters only with preservation installed."""

    def __init__(
        self,
        get_model: Callable[[], nn.Module],
        contract: Qwen4ExpFrozenContract,
        weight_updater: ModuleType,
    ) -> None:
        self.get_model = get_model
        self.contract = contract
        self.weight_updater = weight_updater

    def __call__(self, converter: Any) -> None:
        model = self.get_model()
        if type(model).__name__ != "Qwen4ExpForConditionalGeneration":
            raise ValueError("Frozen inference binding requires Qwen4Exp")
        if not getattr(self.weight_updater, "_areal_qwen4_exp_static_hooks", False):
            raise ValueError(
                "Qwen4Exp visual preservation must be installed before release"
            )
        parameters = {}
        visual_names = set()
        for name, parameter in model.named_parameters():
            if name.startswith("visual."):
                canonical = "model." + name
                visual_names.add(canonical)
            elif ".ple_embedding." in name:
                canonical = name.replace("model.language_model.", "model.", 1)
            else:
                continue
            if canonical in parameters:
                raise ValueError(f"Duplicate frozen inference identity: {canonical}")
            parameters[canonical] = parameter
        converter.bind_frozen_contract(
            self.contract, parameters, frozenset(visual_names)
        )
