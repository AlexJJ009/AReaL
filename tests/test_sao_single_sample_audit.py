# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for single-sample audit metadata in the SAO math workflow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areal import workflow_context
from areal.infra.workflow_context import WorkflowContext
from areal.workflow.rlvr import RLVRWorkflow
from areal.workflow.sao_math import AuditedMathWorkflow


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [TimeoutError("deadline"), RuntimeError("scorer crashed")]
)
async def test_reward_failure_returns_audited_zero_without_regeneration(error):
    workflow_context.set(WorkflowContext(is_eval=False, task_id=17, sample_idx=4))
    workflow = AuditedMathWorkflow.__new__(AuditedMathWorkflow)
    workflow.tokenizer = SimpleNamespace(
        decode=lambda tokens: r"Reasoning: \boxed{768}"
    )
    records = []
    calls = []
    response = SimpleNamespace(
        input_tokens=[9],
        output_tokens=[1, 2, 3],
        output_logprobs=[-0.1] * 3,
        output_versions=[51] * 3,
        stop_reason="stop",
    )

    async def generate(_request):
        calls.append("generate")
        return response

    async def fail_reward(*_args):
        calls.append("score")
        raise error

    async def write(record, _is_eval):
        records.append(record)

    workflow._compute_rewards = fail_reward
    workflow._write_record = write
    result, reward = await workflow._collect_samples(
        SimpleNamespace(agenerate=generate),
        SimpleNamespace(
            rid="request", input_ids=[9], gconfig=SimpleNamespace(max_new_tokens=10)
        ),
        "prompt",
        {"source_id": "sample", "answer": "768"},
    )
    assert result is response
    assert reward == 0.0
    assert calls == ["generate", "score"]
    assert len(records) == 1
    assert records[0]["completion"] == r"Reasoning: \boxed{768}"
    assert records[0]["answer"] == "768"
    assert records[0]["reward_fallback_zero"] is True
    assert records[0]["scoring_error"]["type"] == type(error).__name__


@pytest.mark.asyncio
async def test_generation_failure_is_not_converted_to_zero():
    workflow_context.set(WorkflowContext(is_eval=False, task_id=17, sample_idx=4))
    workflow = AuditedMathWorkflow.__new__(AuditedMathWorkflow)
    records = []

    async def generate(_request):
        raise RuntimeError("generation failed")

    async def write(record, _is_eval):
        records.append(record)

    workflow._write_record = write
    with pytest.raises(RuntimeError, match="generation failed"):
        await workflow._collect_samples(
            SimpleNamespace(agenerate=generate),
            SimpleNamespace(rid="request"),
            "prompt",
            {"source_id": "sample", "answer": "768"},
        )
    assert records[0]["reward"] is None
    assert records[0]["completion"] is None


async def _legal_parent_episode(_self, _engine, _data):
    return {
        "terminated": torch.tensor([True]),
        "truncated": torch.tensor([False]),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sample_idx", "expected_audit_sample_idx"),
    [(None, 0), (3, 3)],
)
async def test_audited_math_workflow_normalizes_single_sample_index(
    monkeypatch, sample_idx, expected_audit_sample_idx
):
    monkeypatch.setattr(RLVRWorkflow, "arun_episode", _legal_parent_episode)
    workflow_context.set(
        WorkflowContext(is_eval=False, task_id=17, sample_idx=sample_idx)
    )
    workflow = AuditedMathWorkflow.__new__(AuditedMathWorkflow)

    result = await workflow.arun_episode(
        engine=None,
        data={"source_id": "gsm8k-source"},
    )

    assert result["terminated"].tolist() == [True]
    assert result["truncated"].tolist() == [False]
    assert result["audit_task_id"].dtype == torch.int64
    assert result["audit_task_id"].tolist() == [17]
    assert result["audit_sample_idx"].dtype == torch.int64
    assert result["audit_sample_idx"].tolist() == [expected_audit_sample_idx]
