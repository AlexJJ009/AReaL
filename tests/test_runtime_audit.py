# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from areal.api.cli_args import OptimizerConfig
from areal.engine.fsdp_engine import FSDPEngine, _checkpoint_score_weight_evidence
from areal.utils.runtime_audit import (
    sanitize_runtime_value,
    tensor_evidence,
    write_runtime_audit,
)


def test_write_runtime_audit_default_off_returns_none(monkeypatch):
    """Runtime evidence is opt-in through AREAL_RUNTIME_AUDIT_DIR."""
    monkeypatch.delenv("AREAL_RUNTIME_AUDIT_DIR", raising=False)

    assert write_runtime_audit("stage", {"value": 1}) is None


def test_write_runtime_audit_redacts_credentials_but_keeps_token_counts(
    monkeypatch, tmp_path
):
    """Sanitization keeps budget evidence while dropping credential-like values."""
    monkeypatch.setenv("AREAL_RUNTIME_AUDIT_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "3")

    path = write_runtime_audit(
        "tau2/controller",
        {
            "max_tokens": 32768,
            "max_new_tokens": 4096,
            "user_llm_api_key_env": "DEEPSEEK_API_KEY",
            "admin_api_key": "secret-value",
            "refresh_token": "secret-token",
        },
    )

    assert path is not None
    assert path.name.startswith("tau2_controller.rank3.pid")
    assert path.name.endswith(".json")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = record["payload"]
    assert record["env"] == {}
    assert payload["max_tokens"] == 32768
    assert payload["max_new_tokens"] == 4096
    assert payload["user_llm_api_key_env"] == "DEEPSEEK_API_KEY"
    assert payload["admin_api_key"] == "<redacted>"
    assert payload["refresh_token"] == "<redacted>"


def test_sanitize_runtime_value_avoids_unknown_object_repr_secret_leak():
    """Unknown objects serialize by type, not repr, to avoid accidental leaks."""

    class SecretObject:
        def __repr__(self):
            return "SecretObject(password='leaked')"

    assert sanitize_runtime_value({"object": SecretObject()}) == {
        "object": "<SecretObject>"
    }


def test_tensor_evidence_hashes_bfloat16_without_numpy_dtype_support():
    """Tensor digests work for bfloat16 score heads."""
    evidence = tensor_evidence(torch.ones((1, 4), dtype=torch.bfloat16))

    assert evidence["shape"] == [1, 4]
    assert evidence["dtype"] == "bfloat16"
    assert len(evidence["sha256"]) == 64


def test_fsdp_runtime_audit_records_mocked_post_init_fields(monkeypatch, tmp_path):
    """Post-init audit fields are valid on a small initialized engine shell."""

    class TinyCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.score = torch.nn.Linear(2560, 1, bias=False)
            self.body = torch.nn.Linear(2, 2, bias=False)

    model = TinyCritic()
    with torch.no_grad():
        model.score.weight.copy_(
            torch.arange(2560, dtype=torch.float32).reshape(1, 2560)
        )
        model.body.weight.fill_(1.0)
    model.body.weight.requires_grad_(False)

    shard = "model.safetensors"
    save_file({"score.weight": model.score.weight.detach()}, tmp_path / shard)

    engine = FSDPEngine.__new__(FSDPEngine)
    engine.config = SimpleNamespace(
        path=str(tmp_path),
        is_critic=True,
        backend="fsdp:d4p1t1",
        offload=True,
        fsdp=SimpleNamespace(offload_params=False),
    )
    engine.optimizer_config = OptimizerConfig(lr=5e-6)
    engine.model = model
    engine.model_config = SimpleNamespace(
        model_type="tiny_qwen35_critic",
        architectures=["TinyCritic"],
    )
    engine._scalar_value_artifact = False
    engine._critic_freeze_manifest = {
        "enabled": True,
        "frozen_parameters": ["body.weight"],
        "frozen_numel": 4,
        "trainable_numel": 2560,
    }
    engine.optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=5e-6,
        weight_decay=0.01,
    )
    engine._initialized = True
    engine.world_size = 4
    engine.rank = 0
    engine.dp_rank = 0
    engine.dp_head = 0
    engine.parallel_helper = SimpleNamespace(__str__=lambda _self: "d4p1t1")
    engine.world_mesh = "mesh(d4p1t1)"
    engine.device = torch.device("cpu")
    engine.is_offload = False

    monkeypatch.setenv("AREAL_RUNTIME_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    engine._write_post_init_runtime_audit()

    [path] = (tmp_path / "audit").glob("fsdp-post-init-critic.rank0.pid*.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    payload = record["payload"]
    assert record["env"] == {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}
    assert payload["engine"]["initialized"] is True
    assert payload["engine"]["backend"] == "fsdp:d4p1t1"
    assert payload["model"]["parameter_counts"] == {
        "trainable_numel": 2560,
        "frozen_numel": 4,
        "total_numel": 2564,
    }
    assert payload["model"]["critic_freeze"]["enabled"] is True
    assert payload["model"]["score_weight"]["shape"] == [1, 2560]
    assert payload["model"]["score_weight"]["checkpoint"]["shape"] == [1, 2560]
    assert (
        payload["model"]["score_weight"]["checkpoint_comparison"]["status"]
        == "exact_match"
    )
    assert payload["optimizer"]["param_groups"][0]["lr"] == 5e-6
    assert payload["optimizer"]["param_groups"][0]["parameter_numel"] == 2560
    assert payload["parallel"]["world_size"] == 4
    assert payload["parallel"]["rank"] == 0
    assert payload["parallel"]["backend"]
    assert payload["device"]["tms_enabled"] in {True, False}


def test_checkpoint_score_weight_evidence_reads_safetensors_index(tmp_path):
    """Checkpoint evidence binds the scalar head shard and digest."""
    shard = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.embed_tokens.weight": torch.zeros((4, 4)),
            "score.weight": torch.arange(4, dtype=torch.float32).reshape(1, 4),
        },
        tmp_path / shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": shard,
                    "score.weight": shard,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = _checkpoint_score_weight_evidence(str(tmp_path))

    assert evidence is not None
    assert evidence["status"] == "loaded"
    assert evidence["shape"] == [1, 4]
    assert evidence["dtype"] == "float32"
    assert evidence["shard"].endswith(shard)
    assert len(evidence["sha256"]) == 64
