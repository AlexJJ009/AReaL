# SPDX-License-Identifier: Apache-2.0
"""Compatibility checks for the existing PPO HF critic export format."""

import json

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from areal.trainer.ppo.value_checkpoint import file_digest, validate_ppo_value_export


@pytest.fixture
def exported(tmp_path):
    actor, critic = tmp_path / "actor", tmp_path / "critic"
    actor.mkdir()
    critic.mkdir()
    for root in (actor, critic):
        (root / "config.json").write_text(
            json.dumps({"model_type": "test", "hidden_size": 2})
        )
        Tokenizer(WordLevel({"test": 0})).save(str(root / "tokenizer.json"))
        (root / "tokenizer_config.json").write_text(
            '{"tokenizer_class": "PreTrainedTokenizerFast"}'
        )
    save_file(
        {"score.weight": torch.ones(1, 2), "model.embedding.weight": torch.ones(2, 2)},
        critic / "model.safetensors",
    )

    def seal():
        files = [
            {"name": p.name, "sha256": file_digest(p)}
            for p in critic.iterdir()
            if p.name != "export-manifest.json"
        ]
        (critic / "export-manifest.json").write_text(
            json.dumps(
                {
                    "source": "unit-test-fixture",
                    "source_step": 1,
                    "selection_metrics": {"mse": 0.1},
                    "files": files,
                }
            )
        )

    seal()
    return actor, critic, seal


def test_existing_export_validates_without_rewriting_it(exported):
    actor, critic, _ = exported
    before = {p.name: file_digest(p) for p in critic.iterdir()}
    validate_ppo_value_export(critic, actor)
    assert before == {p.name: file_digest(p) for p in critic.iterdir()}
    assert not (critic / "value_manifest.json").exists()


def test_export_rejects_modified_weights(exported):
    actor, critic, _ = exported
    with (critic / "model.safetensors").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_ppo_value_export(critic, actor)


def test_export_rejects_missing_scalar_head_even_with_valid_hash(exported):
    actor, critic, seal = exported
    save_file(
        {"model.embedding.weight": torch.ones(2, 2)}, critic / "model.safetensors"
    )
    seal()
    with pytest.raises(ValueError, match="score.weight"):
        validate_ppo_value_export(critic, actor)


def test_export_rejects_different_tokenizer(exported):
    actor, critic, _ = exported
    Tokenizer(WordLevel({"different": 0})).save(str(actor / "tokenizer.json"))
    with pytest.raises(ValueError, match="tokenizer differs"):
        validate_ppo_value_export(critic, actor)
