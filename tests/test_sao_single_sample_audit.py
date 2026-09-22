# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for single-sample audit metadata in the SAO math workflow."""

from __future__ import annotations

import pytest
import torch

from areal import workflow_context
from areal.infra.workflow_context import WorkflowContext
from areal.workflow.rlvr import RLVRWorkflow
from areal.workflow.sao_math import AuditedMathWorkflow


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
