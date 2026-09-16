# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch
from torch import nn

from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract

TABLE = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
VISUAL = "model.visual.patch_embed.proj.weight"
QSA = "model.layers.3.self_attn.indexer.index_qk_proj.weight"


@pytest.fixture
def contract():
    return Qwen4ExpFrozenContract(
        "a" * 64, frozenset({TABLE}), frozenset({VISUAL}), True, True
    )


def table():
    return nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16), requires_grad=False)


def test_wire_roundtrip_and_exact_exclusion_keep_frozen_qsa(contract):
    assert Qwen4ExpFrozenContract.from_dict(contract.to_dict()) == contract
    assert contract.excludes(TABLE, "actor")
    assert contract.excludes(VISUAL, "inference")
    assert not contract.excludes(VISUAL, "actor")
    assert not contract.excludes(QSA, "actor")
    assert not contract.excludes("model.visual.unexpected.weight", "inference")
    with pytest.raises(ValueError):
        contract.excludes(TABLE, "typo")


@pytest.mark.parametrize(
    "change",
    [
        {"freeze_ple_table": False},
        {"language_model_only": False},
        {"schema_version": 2},
        {"schema_version": True},
        {"checkpoint_manifest_sha256": "not-a-hash"},
        {"ple_table_names": frozenset({QSA})},
        {"visual_parameter_names": frozenset({"model.visual.*"})},
    ],
)
def test_invalid_declarations_rejected(contract, change):
    with pytest.raises(ValueError):
        replace(contract, **change)


@pytest.mark.parametrize("fault", ["extra", "missing", "duplicate", "string"])
def test_invalid_wire_contract_rejected(contract, fault):
    data = contract.to_dict()
    if fault == "extra":
        data["skip_unknown_parameters"] = True
    elif fault == "missing":
        del data["schema_version"]
    elif fault == "duplicate":
        data["ple_table_names"] *= 2
    else:
        data["ple_table_names"] = TABLE
    with pytest.raises(ValueError):
        Qwen4ExpFrozenContract.from_dict(data)


def test_actor_validates_original_parameters_and_local_pipeline_ownership(contract):
    parameter = table()
    contract.validate_actor_parameters({TABLE: parameter}, frozenset({TABLE}))
    contract.validate_actor_parameters({QSA: table()}, frozenset())
    with pytest.raises(TypeError, match="detached"):
        contract.validate_actor_parameters(
            {TABLE: parameter.detach()}, frozenset({TABLE})
        )
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        contract.validate_actor_parameters({TABLE: parameter}, frozenset({TABLE}))
    with pytest.raises(ValueError, match="ownership"):
        contract.validate_actor_parameters({}, frozenset({TABLE}))
    with pytest.raises(ValueError, match="visual"):
        contract.validate_actor_parameters({VISUAL: table()}, frozenset())


def test_inference_requires_exact_backup_and_exclusion_keys(contract):
    parameters = {TABLE: table(), VISUAL: table(), QSA: table()}
    contract.validate_inference_parameters(parameters, frozenset({VISUAL}))
    with pytest.raises(ValueError, match="preservation"):
        contract.validate_inference_parameters(parameters, frozenset())
    parameters["model.visual.extra.weight"] = table()
    with pytest.raises(ValueError, match="visual parameters"):
        contract.validate_inference_parameters(parameters, frozenset({VISUAL}))


@pytest.fixture
def manifest_files(tmp_path, contract):
    import hashlib
    import json

    model = tmp_path / "model"
    model.mkdir()
    config = b'{"architectures":["Qwen4ExpForConditionalGeneration"]}'
    index = b"{}"
    (model / "config.json").write_bytes(config)
    (model / "model.safetensors.index.json").write_bytes(index)
    basis = {
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "weight_index_sha256": hashlib.sha256(index).hexdigest(),
        "ple_source_shards": [
            {
                "name": "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                "sha256": "b" * 64,
            }
        ],
        "visual_reference": [{"tp_rank": 0, "parameters": [{"name": VISUAL}]}],
    }
    digest = hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    declared = replace(contract, checkpoint_manifest_sha256=digest)
    manifest = {"contract": declared.to_dict(), "identity_basis": basis}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, model, declared


def test_manifest_load_checks_checkpoint_identity(manifest_files):
    from areal.models.mcore.qwen4_exp_awex_contract import load_frozen_contract

    path, model, declared = manifest_files
    assert load_frozen_contract(path, model) == declared
    (model / "model.safetensors.index.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="weight index"):
        load_frozen_contract(path, model)


@pytest.mark.parametrize("fault", ["evidence", "ple", "visual", "config"])
def test_manifest_rejects_identity_or_exclusion_drift(manifest_files, fault):
    import json

    from areal.models.mcore.qwen4_exp_awex_contract import load_frozen_contract

    path, model, _ = manifest_files
    data = json.loads(path.read_text())
    if fault == "evidence":
        data["identity_basis"]["ple_source_shards"][0]["sha256"] = "c" * 64
    elif fault == "ple":
        data["contract"]["ple_table_names"] = [TABLE.replace("layers.1", "layers.2")]
    elif fault == "visual":
        data["contract"]["visual_parameter_names"] = ["model.visual.other.weight"]
    else:
        (model / "config.json").write_text("{}")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_frozen_contract(path, model)
