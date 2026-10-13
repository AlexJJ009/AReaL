# SPDX-License-Identifier: Apache-2.0
"""CPU-only guard tests for SAO critic head preflight."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.sao.check_critic_head import (
    CriticHeadPreflightError,
    check_critic_head,
    main,
)


def _write_checkpoint(
    tmp_path,
    *,
    state: dict[str, torch.Tensor],
    hidden_size=4,
    nested_text_config=True,
    head_bias=False,
):
    checkpoint = tmp_path / "critic"
    checkpoint.mkdir(parents=True)
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
    }
    if head_bias:
        config["areal_value_head_bias"] = True
    if nested_text_config:
        config["text_config"] = {"hidden_size": hidden_size, "vocab_size": 8}
    else:
        config["hidden_size"] = hidden_size
        config["vocab_size"] = 8
    (checkpoint / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    (checkpoint / "tokenizer.json").write_text(
        '{"model":{"vocab":{}}}', encoding="utf-8"
    )
    (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    save_file(state, checkpoint / "model.safetensors")
    return checkpoint


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def test_check_critic_head_accepts_qwen35_score_weight(tmp_path):
    head = torch.tensor([[1.0, -2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    checkpoint = _write_checkpoint(
        tmp_path,
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "score.weight": head,
        },
    )

    report = check_critic_head(checkpoint)

    assert report["passed"] is True
    assert report["config"]["hidden_size"] == 4
    assert report["head"]["shape"] == [1, 4]
    assert report["head"]["dtype"] == "bfloat16"
    assert report["head"]["sha256"] == _tensor_digest(head)
    assert report["head"]["nonzero_count"] == 4
    assert report["loader_status"] == "compat_qwen35_token_critic_unsealed"
    assert set(report["artifact_hashes"]) == {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }


def test_check_critic_head_rejects_missing_score_weight(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path,
        state={"model.language_model.embed_tokens.weight": torch.ones(8, 4)},
    )

    with pytest.raises(CriticHeadPreflightError, match="score.weight"):
        check_critic_head(checkpoint)


def test_check_critic_head_accepts_top_level_hidden_size(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path,
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "score.weight": torch.ones(1, 4),
        },
        nested_text_config=False,
    )

    report = check_critic_head(checkpoint)

    assert report["config"]["hidden_size"] == 4


def test_check_critic_head_rejects_wrong_scalar_head_schema(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path,
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "classifier.weight": torch.ones(1, 4),
            "score.weight": torch.ones(1, 4),
        },
    )

    with pytest.raises(CriticHeadPreflightError, match="exactly one scalar head"):
        check_critic_head(checkpoint)


def test_check_critic_head_rejects_shape_mismatch(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path,
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "score.weight": torch.ones(1, 3),
        },
    )

    with pytest.raises(CriticHeadPreflightError, match="shape mismatch"):
        check_critic_head(checkpoint)


def test_check_critic_head_rejects_nonfinite_or_zero(tmp_path):
    nonfinite = _write_checkpoint(
        tmp_path / "nonfinite",
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "score.weight": torch.tensor([[1.0, float("nan"), 0.0, 0.0]]),
        },
    )
    zeros = _write_checkpoint(
        tmp_path / "zeros",
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "score.weight": torch.zeros(1, 4),
        },
    )

    with pytest.raises(CriticHeadPreflightError, match="non-finite"):
        check_critic_head(nonfinite)
    with pytest.raises(CriticHeadPreflightError, match="all zeros"):
        check_critic_head(zeros)


def _biased_state(bias):
    return {
        "model.language_model.embed_tokens.weight": torch.ones(8, 4),
        "score.weight": torch.ones(1, 4),
        "score.bias": bias,
    }


def test_check_critic_head_accepts_bias_with_config_flag(tmp_path):
    bias = torch.tensor([0.9], dtype=torch.bfloat16)
    checkpoint = _write_checkpoint(tmp_path, state=_biased_state(bias), head_bias=True)

    report = check_critic_head(checkpoint)

    assert report["config"]["areal_value_head_bias"] is True
    assert report["head"]["bias"]["value"] == pytest.approx(0.9, abs=1e-2)
    assert report["head"]["bias"]["sha256"] == _tensor_digest(bias)


def test_check_critic_head_reports_no_bias_for_plain_head(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path,
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "score.weight": torch.ones(1, 4),
        },
    )

    report = check_critic_head(checkpoint)

    assert report["head"]["bias"] is None
    assert report["config"]["areal_value_head_bias"] is False


def test_check_critic_head_rejects_bias_without_config_flag(tmp_path):
    checkpoint = _write_checkpoint(tmp_path, state=_biased_state(torch.zeros(1)))

    with pytest.raises(CriticHeadPreflightError, match="areal_value_head_bias=False"):
        check_critic_head(checkpoint)


def test_check_critic_head_rejects_config_flag_without_bias(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path,
        state={
            "model.language_model.embed_tokens.weight": torch.ones(8, 4),
            "score.weight": torch.ones(1, 4),
        },
        head_bias=True,
    )

    with pytest.raises(CriticHeadPreflightError, match="areal_value_head_bias=True"):
        check_critic_head(checkpoint)


def test_check_critic_head_rejects_foreign_head_bias(tmp_path):
    state = _biased_state(torch.zeros(1))
    state["classifier.bias"] = torch.zeros(1)
    checkpoint = _write_checkpoint(tmp_path, state=state, head_bias=True)

    with pytest.raises(CriticHeadPreflightError, match="classifier.bias"):
        check_critic_head(checkpoint)


@pytest.mark.parametrize(
    ("bias", "message"),
    [
        (torch.zeros(2), "score.bias shape"),
        (torch.tensor([float("nan")]), "non-finite"),
    ],
)
def test_check_critic_head_rejects_bad_bias(tmp_path, bias, message):
    checkpoint = _write_checkpoint(tmp_path, state=_biased_state(bias), head_bias=True)

    with pytest.raises(CriticHeadPreflightError, match=message):
        check_critic_head(checkpoint)


def test_check_critic_head_cli_writes_failure_report(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path,
        state={"model.language_model.embed_tokens.weight": torch.ones(8, 4)},
    )
    output = tmp_path / "report.json"

    assert main(["--checkpoint", str(checkpoint), "--output", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert "score.weight" in report["errors"][0]
