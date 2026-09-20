# SPDX-License-Identifier: Apache-2.0
# Optional model packages must be checked before dependent imports.
# ruff: noqa: E402

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from examples.math.sao_ppo import split_trajectory_groups
from tests.test_sao_qwen35_critic import _require_qwen35, _write_tiny_qwen35_config

from areal.api.cli_args import MicroBatchSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.infra.controller.train_controller import _dispatch_tensors
from areal.models.transformers.token_critic import (
    Qwen35TokenCriticForCausalBackbone,
)


@pytest.fixture(autouse=True)
def _force_qwen35_torch_reference_cpu(monkeypatch):
    _require_qwen35()
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35

    monkeypatch.setattr(qwen35, "FusedRMSNormGated", None)
    monkeypatch.setattr(qwen35, "causal_conv1d_fn", None)
    monkeypatch.setattr(
        qwen35,
        "causal_conv1d_update",
        qwen35.torch_causal_conv1d_update,
    )
    monkeypatch.setattr(
        qwen35,
        "chunk_gated_delta_rule",
        qwen35.torch_chunk_gated_delta_rule,
    )
    monkeypatch.setattr(
        qwen35,
        "fused_recurrent_gated_delta_rule",
        qwen35.torch_recurrent_gated_delta_rule,
    )
    monkeypatch.setattr(qwen35, "is_fast_path_available", False)


def _make_prompt_group(prompt_idx: int, *, n_samples: int = 4) -> dict:
    seq_len = 3
    row_ids = torch.arange(
        prompt_idx * n_samples, (prompt_idx + 1) * n_samples, dtype=torch.long
    )
    input_ids = row_ids[:, None] * 10 + torch.arange(seq_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones((n_samples, seq_len), dtype=torch.bool),
        "loss_mask": torch.ones((n_samples, seq_len), dtype=torch.bool),
        "audit_source_key": torch.full((n_samples,), prompt_idx, dtype=torch.long),
        "audit_task_id": row_ids.clone(),
        "audit_sample_idx": torch.arange(n_samples, dtype=torch.long),
        "prompt_metadata": {"prompt_idx": prompt_idx},
    }


def _trajectory_ids(items: list[dict]) -> list[int]:
    return [int(item["audit_task_id"].reshape(-1)[0].item()) for item in items]


def test_split_tail_groups_then_dispatches_twenty_trajectories_without_loss_or_dup():
    grouped_tail = [_make_prompt_group(prompt_idx) for prompt_idx in range(5)]

    individual = split_trajectory_groups(grouped_tail)
    assert len(individual) == 20
    assert [item["input_ids"].shape for item in individual] == [(1, 3)] * 20
    assert [item["audit_task_id"].shape for item in individual] == [(1,)] * 20
    assert sorted(_trajectory_ids(individual)) == list(range(20))

    splits, group_indices = _dispatch_tensors(individual, dp_size=4)

    assert [len(split) for split in splits] == [5, 5, 5, 5]
    assert sorted(
        idx for rank_indices in group_indices for idx in rank_indices
    ) == list(range(20))
    assert sorted(
        _trajectory_ids([item for split in splits for item in split])
    ) == list(range(20))


def test_full_grouped_batch_dispatch_keeps_prompt_groups_atomic():
    grouped_full = [_make_prompt_group(prompt_idx) for prompt_idx in range(128)]

    splits, group_indices = _dispatch_tensors(grouped_full, dp_size=4)

    assert [len(split) for split in splits] == [32, 32, 32, 32]
    assert sorted(
        idx for rank_indices in group_indices for idx in rank_indices
    ) == list(range(128))
    for split in splits:
        assert all(item["input_ids"].shape == (4, 3) for item in split)


def test_qwen35_prepare_mb_list_isolates_each_real_sequence(monkeypatch):
    monkeypatch.setattr("areal.engine.fsdp_engine.dist.get_rank", lambda: 0)
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.enable_tree_training = False
    engine.config = SimpleNamespace(
        mb_spec=MicroBatchSpec(n_mbs=1),
        pad_to_maximum=False,
    )
    engine.model_config = SimpleNamespace(model_type="qwen3_5")
    engine.logger = MagicMock()
    batch = {
        "input_ids": torch.tensor(
            [
                [11, 12, 13, 0],
                [21, 22, 23, 24],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "loss_mask": torch.tensor(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    }

    mb_list = FSDPEngine._prepare_mb_list(engine, batch)

    assert len(mb_list.mbs) == 2
    assert mb_list.group_lens == [3, 4]
    assert [mb["cu_seqlens"].tolist() for mb in mb_list.mbs] == [[0, 3], [0, 4]]
    assert [mb["input_ids"].tolist() for mb in mb_list.mbs] == [
        [11, 12, 13],
        [21, 22, 23, 24],
    ]
    assert all(mb["attention_mask"] is None for mb in mb_list.mbs)


@torch.no_grad()
def test_tiny_qwen35_forward_matches_batched_and_separate_isolated_sequences(tmp_path):
    config = _write_tiny_qwen35_config(tmp_path)
    model = Qwen35TokenCriticForCausalBackbone.from_config(config).eval()
    input_ids = torch.tensor(
        [
            [5, 6, 7, 0],
            [9, 10, 11, 12],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    batched = model(input_ids=input_ids, attention_mask=attention_mask).logits
    first = model(
        input_ids=input_ids[:1, :3],
        attention_mask=attention_mask[:1, :3],
    ).logits
    second = model(
        input_ids=input_ids[1:2],
        attention_mask=attention_mask[1:2],
    ).logits

    torch.testing.assert_close(batched[:1, :3], first, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(batched[1:2], second, rtol=0.0, atol=1e-7)
