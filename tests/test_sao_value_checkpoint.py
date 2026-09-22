"""M5: independent scalar value artifact sealing and strict reload."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForTokenClassification,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen3_5Config,
)

from areal.api.cli_args import PPOCriticConfig
from areal.engine.fsdp_engine import FSDPEngine
from areal.models.transformers.scalar_value import Qwen35ScalarValueModel
from areal.trainer.ppo.value_checkpoint import (
    create_scalar_value_model,
    scalar_value_state,
    validate_value_artifact,
    write_value_manifest,
)


def qwen35_config(*, linear=False):
    return Qwen3_5Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 512,
            "dtype": "float32",
            "layer_types": [
                "linear_attention" if linear else "full_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "use_cache": False,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=30,
        video_token_id=31,
        tie_word_embeddings=False,
    )


def make_artifact(tmp_path, *, qwen35=False, linear=False):
    root = tmp_path / "value"
    root.mkdir(parents=True, exist_ok=True)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_labels=1,
    )
    model = (
        Qwen35ScalarValueModel.from_config(
            qwen35_config(linear=linear),
            attn_implementation="eager",
            dtype=torch.float32,
        )
        if qwen35
        else AutoModelForTokenClassification.from_config(
            config, attn_implementation="eager"
        )
    )
    model.eval()
    model.save_pretrained(root, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0")
        ),
        unk_token="t0",
        pad_token="t0",
        eos_token="t1",
    )
    tokenizer.save_pretrained(root)
    report = {"kind": "synthetic", "passed": True, "model": "tiny-qwen2"}
    (root / "qualification.json").write_text(json.dumps(report))
    write_value_manifest(
        root,
        identity={"backbone_id": "tiny-qwen2", "tokenizer_id": "tiny-tokenizer"},
        protocol={
            "discount": 1.0,
            "target_horizon": 32,
            "thinking": False,
            "reward": "binary",
            "termination": "stop-v1",
            "scorer_digest": "score-v1",
            "split_digest": "split-v1",
            "freeze_policy": "backbone-initialized",
            "template_digest": "template-v1",
        },
        qualification={
            "kind": "synthetic",
            "passed": True,
            "report": "qualification.json",
        },
    )
    return root, model


def test_independent_value_checkpoint_reloads_scalar_predictions(tmp_path):
    root, original = make_artifact(tmp_path)
    manifest = validate_value_artifact(
        root, expected_identity={"backbone_id": "tiny-qwen2"}
    )
    assert manifest["qualification"]["kind"] == "synthetic"
    engine = FSDPEngine(
        PPOCriticConfig(
            path=str(root), is_critic=True, optimizer_dtype="float32", attn_impl="eager"
        )
    )
    restored = engine._create_llm_actor_or_critic()
    restored.eval()
    ids = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(
            restored(input_ids=ids, attention_mask=torch.ones_like(ids)).logits,
            original(input_ids=ids, attention_mask=torch.ones_like(ids)).logits,
            rtol=0,
            atol=0,
        )
    assert scalar_value_state(root)


def test_artifact_integrity_missing_manifest_and_identity_rejected(tmp_path):
    root, _ = make_artifact(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        validate_value_artifact(root, expected_identity={"tokenizer_id": "other"})
    with (root / "tokenizer.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_value_artifact(root)
    (root / "value_manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        FSDPEngine(PPOCriticConfig(path=str(root), is_critic=True))


def test_changed_backbone_and_head_shape_rejected_even_when_resealed(tmp_path):
    for name in ("missing_backbone", "head_shape"):
        root, _ = make_artifact(tmp_path / name)
        manifest = validate_value_artifact(root)
        state = scalar_value_state(root)
        if name == "missing_backbone":
            del state[next(key for key in state if key.startswith("model."))]
        else:
            key = "score.weight" if "score.weight" in state else "classifier.weight"
            state[key] = torch.ones(1, 99)
        save_file(state, root / "model.safetensors")
        write_value_manifest(
            root,
            identity=manifest["identity"],
            protocol=manifest["protocol"],
            qualification=manifest["qualification"],
        )
        with pytest.raises(ValueError, match="complete matching"):
            create_scalar_value_model(str(root), dtype=torch.float32)


def test_missing_or_wrong_protocol_is_rejected(tmp_path):
    root, _ = make_artifact(tmp_path)
    weights = next(root.glob("*.safetensors"))
    manifest = validate_value_artifact(root)
    state = {
        k: v
        for k, v in load_file(weights).items()
        if k not in {"score.weight", "classifier.weight"}
    }
    save_file(state, weights)
    write_value_manifest(
        root,
        identity=manifest["identity"],
        protocol=manifest["protocol"],
        qualification=manifest["qualification"],
    )
    with pytest.raises(ValueError, match="scalar score"):
        scalar_value_state(root)
    root, _ = make_artifact(tmp_path / "second")
    with pytest.raises(ValueError, match="does not match"):
        validate_value_artifact(root, expected_protocol={"target_horizon": 8192})


def test_pretrained_gate_does_not_accept_synthetic_fixture(tmp_path):
    root, _ = make_artifact(tmp_path)
    with pytest.raises(ValueError, match="qualified pretrained"):
        validate_value_artifact(root, require_pretrained=True)


def test_real_fsdp_loader_enforces_consuming_value_contract(tmp_path):
    root, _ = make_artifact(tmp_path)
    manifest = validate_value_artifact(root)
    contract = {
        "identity": manifest["identity"],
        "protocol": manifest["protocol"],
        "require_pretrained": False,
    }
    config = PPOCriticConfig(
        path=str(root), is_critic=True, value_contract=contract, attn_impl="eager"
    )
    engine = FSDPEngine(config)
    assert engine._value_manifest["identity"] == contract["identity"]
    contract["protocol"] = {**contract["protocol"], "target_horizon": 8192}
    with pytest.raises(ValueError, match="protocol"):
        FSDPEngine(config)

    contract["protocol"] = manifest["protocol"]
    contract["identity"] = {**contract["identity"], "tokenizer_id": "wrong"}
    with pytest.raises(ValueError, match="identity"):
        FSDPEngine(config)
    contract["identity"] = manifest["identity"]
    contract["require_pretrained"] = True
    with pytest.raises(ValueError, match="qualified pretrained"):
        FSDPEngine(config)


def test_qwen35_scalar_checkpoint_is_causal_and_reloads_exactly(tmp_path):
    root, original = make_artifact(tmp_path, qwen35=True)
    config = PPOCriticConfig(
        path=str(root), is_critic=True, attn_impl="eager", optimizer_dtype="float32"
    )
    model = FSDPEngine(config)._create_llm_actor_or_critic().eval()
    ids = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        expected = original(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        actual = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        assert actual.shape == (1, 4, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ids[0, -1] = 9
        changed = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        torch.testing.assert_close(actual[:, :-1], changed[:, :-1], rtol=0, atol=0)
