# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmark.swe_sample_refill import RecordedArenaAgent, validate_canary_events
from examples.swe.arena_agent import ArenaStreamAgentWorkflow

from areal.infra import workflow_context


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_benchmark_agent_records_terminal_identity_and_propagates_error(
    tmp_path, monkeypatch, failed
):
    agent = object.__new__(RecordedArenaAgent)
    agent.event_dir = tmp_path
    agent._event_lock = asyncio.Lock()
    result = (
        None if failed else SimpleNamespace(task_id="arena-1", status="DONE", score=0.0)
    )
    agent._task_result = ContextVar("benchmark_test_result", default=result)
    mock = (
        AsyncMock(side_effect=RuntimeError("test failure"))
        if failed
        else AsyncMock(return_value=0.0)
    )
    monkeypatch.setattr(ArenaStreamAgentWorkflow, "run", mock)
    previous = workflow_context.get()
    workflow_context.set(
        workflow_context.WorkflowContext(task_id=7, sample_idx=2, group_size=4)
    )
    try:
        if failed:
            with pytest.raises(RuntimeError, match="test failure"):
                await agent.run({"data_id": "fixed-task"})
        else:
            assert await agent.run({"data_id": "fixed-task"}) == 0.0
    finally:
        workflow_context.set(previous)
    records = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [item["event"] for item in records] == ["agent_start", "agent_end"]
    assert all(
        item["task_id"] == 7
        and item["sample_idx"] == 2
        and item["data_id"] == "fixed-task"
        for item in records
    )
    assert records[1]["monotonic"] >= records[0]["monotonic"]
    assert records[1]["outcome"] == ("failed" if failed else "completed")
    assert records[1]["arena_task_id"] == (None if failed else "arena-1")


@pytest.mark.parametrize("status", ["DONE", "OK"])
def test_canary_accepts_arena_success_status(status):
    validate_canary_events(
        [{"event": "agent_end", "arena_status": status, "outcome": "completed"}]
    )


@pytest.mark.parametrize(
    "status,outcome",
    [
        ("HARNESS_FAILED", "failed"),
        ("TIMEOUT", "failed"),
        ("AGENT_RUNNING", "completed"),
        ("OK", "failed"),
    ],
)
def test_canary_rejects_unsuccessful_or_nonterminal_status(status, outcome):
    with pytest.raises(RuntimeError, match="Canary did not complete"):
        validate_canary_events(
            [{"event": "agent_end", "arena_status": status, "outcome": outcome}]
        )


def test_rolling_collector_reports_batches_before_slow_initial_groups(tmp_path):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from benchmark.swe_sample_refill import collect_rolling_batches

    release = threading.Event()
    submitted = []

    class Dispatcher:
        def wait_for_task(self, task_id, timeout):
            if task_id in [0, 1]:
                assert release.wait(5)
            return SimpleNamespace(
                trajectory={"rollout_group": SimpleNamespace(row_counts=(1, 1))}
            )

    class Controller:
        dispatcher = Dispatcher()

        def submit(self, row, **kwargs):
            submitted.append(kwargs["task_id"])

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            collect_rolling_batches,
            Controller(),
            [{"data_id": str(i)} for i in range(6)],
            {},
            tmp_path,
            time.monotonic(),
            2,
            2,
            2,
            2,
            5,
        )
        try:
            deadline = time.monotonic() + 5
            batches = []
            while time.monotonic() < deadline:
                path = tmp_path / "batches.jsonl"
                if path.exists():
                    batches = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                if len(batches) == 2:
                    break
                threading.Event().wait(0.01)
            assert len(batches) == 2
            assert all(
                task_id not in [0, 1]
                for batch in batches
                for task_id in batch["task_ids"]
            )
            assert submitted == list(range(6))
            assert not future.done()
        finally:
            release.set()
        records, batches = future.result(timeout=5)
    assert len(records) == 6
    assert len(batches) == 2


def test_rolling_collector_replaces_rejected_prompt_and_drains_all(tmp_path):
    import time

    from benchmark.swe_sample_refill import collect_rolling_batches

    class Dispatcher:
        def wait_for_task(self, task_id, timeout):
            if task_id == 0:
                return None
            return SimpleNamespace(
                trajectory={"rollout_group": SimpleNamespace(row_counts=(1, 1))}
            )

    class Controller:
        dispatcher = Dispatcher()

        def submit(self, row, **kwargs):
            pass

    records, batches = collect_rolling_batches(
        Controller(),
        [{"data_id": str(i)} for i in range(20)],
        {},
        tmp_path,
        time.monotonic(),
        2,
        2,
        2,
        2,
        5,
    )
    assert len(batches) == 2
    assert all(0 not in batch["task_ids"] for batch in batches)
    assert len(records) == 7
    assert sum(not record["accepted"] for record in records) == 1


@pytest.mark.parametrize("status", ["ARCHIVED", "DELETED", None])
def test_metadata_lookup_rejects_inactive_stream(status):
    from benchmark.swe_sample_refill import MetadataArenaClient
    from examples.swe.arena_client import ArenaAPIError

    with pytest.raises(ArenaAPIError):
        MetadataArenaClient.select_stream(
            [{"stream_id": "wanted", "status": status}], "wanted"
        )


def test_metadata_lookup_selects_exact_identity_and_preserves_reward():
    from benchmark.swe_sample_refill import MetadataArenaClient
    from examples.swe.arena_client import ArenaAPIError

    item = {
        "stream_id": "wanted",
        "status": "ACTIVE",
        "default_reward_ref": {"key": "grader", "version": "1.1.1"},
    }
    assert (
        MetadataArenaClient.select_stream(
            [{"stream_id": "other", "status": "ACTIVE"}, item], "wanted"
        )
        == item
    )
    with pytest.raises(ArenaAPIError):
        MetadataArenaClient.select_stream([item, item], "wanted")


@pytest.mark.parametrize("free_mib,expected_pass", [(20000, True), (8000, False)])
def test_memory_preflight_checks_actual_headroom(
    tmp_path, monkeypatch, free_mib, expected_pass
):
    import benchmark.swe_sample_refill as benchmark

    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=f"0, 143771, {free_mib}\n1, 143771, {free_mib}\n"
        ),
    )
    if expected_pass:
        benchmark.check_gpu_memory_headroom(tmp_path, 12, 2)
    else:
        with pytest.raises(RuntimeError, match="headroom"):
            benchmark.check_gpu_memory_headroom(tmp_path, 12, 2)
    assert (tmp_path / "startup-gpu-memory.json").exists()
