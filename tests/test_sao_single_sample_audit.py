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
@pytest.mark.parametrize("include_answer", [True, False])
async def test_reward_failure_retains_exact_completion_and_answer(include_answer):
    workflow_context.set(WorkflowContext(is_eval=False, task_id=17, sample_idx=4))
    workflow = AuditedMathWorkflow.__new__(AuditedMathWorkflow)
    workflow.tokenizer = SimpleNamespace(
        decode=lambda tokens: r"Reasoning: \boxed{768}"
    )
    records = []

    async def generate(_request):
        return SimpleNamespace(output_tokens=[1, 2, 3])

    async def fail_reward(*_args):
        raise TimeoutError("semantic scoring deadline")

    async def write(record, _is_eval):
        records.append(record)

    workflow._compute_rewards = fail_reward
    workflow._write_record = write
    with pytest.raises(TimeoutError, match="semantic scoring deadline"):
        await workflow._collect_samples(
            SimpleNamespace(agenerate=generate),
            SimpleNamespace(rid="request"),
            "prompt",
            {"source_id": "sample", **({"answer": "768"} if include_answer else {})},
        )

    assert records[0]["completion"] == r"Reasoning: \boxed{768}"
    assert records[0]["answer"] == ("768" if include_answer else None)
    assert records[0]["sample_idx"] == 4
    assert records[0]["reward"] is None


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
