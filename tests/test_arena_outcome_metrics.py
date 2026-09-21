from types import SimpleNamespace

import pytest
import torch

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaTaskResult

from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.utils import stats_tracker


def test_turn_metrics_mixed_outcomes_have_separate_denominators():
    stats_tracker.export_all(reset=True)
    for turns, failed, code in [
        (2, False, None),
        (4, True, "LLM_RESPONSE_FAILED"),
        (3, True, "UNBOUNDED"),
    ]:
        interaction = SimpleNamespace(
            reward=0.0 if failed else 1.0,
            has_tensor_data=True,
            to_tensor_dict=lambda n=turns: {
                "turn_ids": torch.tensor([[-1, *range(n), n - 1]])
            },
        )
        OpenAIProxyWorkflow._record_interaction_stats(
            {"last": interaction},
            is_harness_error=failed,
            harness_outcome_code=code,
        )
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/num_turns_count"] == 3
    assert stats["rollout/num_turns/avg"] == 3
    assert stats["rollout/num_turns/min"] == 2
    assert stats["rollout/num_turns/max"] == 4
    assert stats["rollout/num_turns_no_harness_err/avg"] == 2
    assert stats["rollout/num_turns_harness_err/avg"] == 3.5
    assert stats["rollout/num_turns_harness_err/LLM_RESPONSE_FAILED/avg"] == 4
    assert stats["rollout/num_turns_harness_err/OTHER/avg"] == 3
    assert not any("UNBOUNDED" in key for key in stats)


@pytest.mark.parametrize(
    "status,code,bucket",
    [
        ("OK", None, None),
        ("HARNESS_FAILED", "LLM_RESPONSE_FAILED", "LLM_RESPONSE_FAILED"),
        ("HARNESS_FAILED", "UNBOUNDED", "OTHER"),
    ],
)
def test_harness_metrics_terminal_outcomes_are_bounded(
    monkeypatch, status, code, bucket
):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    stats_tracker.export_all(reset=True)
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example", "stream_id": "stream-a"}
    )
    workflow._task_result.set(
        ArenaTaskResult(
            task_id="task-a",
            status=status,
            score=1.0 if status == "OK" else None,
            raw={"outcome_code": code},
        )
    )
    workflow.record_episode_metrics({"stream_id": "stream-a"}, 0.0)
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/harness_success"] == (status == "OK")
    assert stats["rollout/harness_error"] == (status == "HARNESS_FAILED")
    if bucket:
        assert stats[f"rollout/harness_error/{bucket}"] == 1.0
    assert not any("UNBOUNDED" in key for key in stats)


@pytest.mark.parametrize("missing", ["reward", "turns", "tensor_error", "empty"])
def test_missing_metric_data_does_not_invent_zero_or_raise(missing):
    stats_tracker.export_all(reset=True)

    def tensors():
        if missing == "tensor_error":
            raise ValueError("unavailable tensor data")
        return {} if missing == "turns" else {"turn_ids": torch.tensor([[0, 0]])}

    interaction = SimpleNamespace(
        reward=None if missing == "reward" else 1.0,
        has_tensor_data=True,
        to_tensor_dict=tensors,
    )
    OpenAIProxyWorkflow._record_interaction_stats(
        {} if missing == "empty" else {"a": interaction}
    )
    stats = stats_tracker.export_all(reset=True)
    if missing in {"reward", "empty"}:
        assert "rollout/reward" not in stats
    if missing in {"turns", "tensor_error", "empty"}:
        assert "rollout/num_turns_count" not in stats


def test_group_quality_tracks_attempts_by_stream_and_task_type(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    stats_tracker.export_all(reset=True)
    workflow = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example", "stream_id": "stream-a"}
    )
    for rewards in [[1.0, 0.0, None], [None, None, None]]:
        workflow.record_group_metrics({"arena_task_type": "terrain"}, rewards, 3)
    stats = stats_tracker.export_all(reset=True)
    for prefix in ["rollout/", "rollout/domain/terrain/", "rollout/stream/default/"]:
        assert stats[prefix + "all_correct"] == 0.0
        assert stats[prefix + "all_wrong"] == 0.5
        assert stats[prefix + "group_pass_count_1_ratio"] == 0.5
        assert stats[prefix + "group_pass_count_0_ratio"] == 0.5
    assert stats["rollout/group_pass_1"] == stats["rollout/group_pass_count_1_ratio"]


@pytest.mark.asyncio
async def test_arena_result_audit_keeps_session_and_failure_identity(
    monkeypatch, tmp_path
):
    import json

    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    workflow = ArenaStreamAgentWorkflow(
        econfig={
            "arena_base_url": "https://arena.example",
            "stream_id": "stream-a",
            "arena_result_dump_dir": str(tmp_path),
        }
    )
    workflow._proxy_session_id.set("session-a")
    result = ArenaTaskResult(
        task_id="task-a",
        status="HARNESS_FAILED",
        score=None,
        raw={"outcome_code": "LLM_RESPONSE_FAILED"},
    )
    workflow._task_result.set(result)
    metadata = workflow.get_episode_metadata()
    await workflow._dump_task_result(
        {}, workflow.stream_configs["default"], result, training_score=0.0
    )
    record = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    for key in ["session_id", "arena_task_id", "harness_outcome_code"]:
        assert record[key] == metadata[key]
    assert record["status"] == metadata["arena_status"]
