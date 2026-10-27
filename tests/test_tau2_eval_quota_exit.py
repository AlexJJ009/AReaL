# SPDX-License-Identifier: Apache-2.0
"""CPU-only quota delivery across worker processes and blocked eval callbacks."""

import asyncio
import json
import multiprocessing
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from examples.tau2.api_reliability import (
    bounded_call,
    quota_exit_guard,
)
from examples.tau2.official_runner import expand_trials
from examples.tau2.utils import Tau2EnvConfig
from tests.test_tau2_evaluate import tau2_evaluate as evaluate
from tests.test_tau2_ppo import _config

from areal.infra.workflow_executor import BatchTaskDispatcher
from areal.utils.rollout_error import QUOTA_EXIT_CODE, FatalRolloutError


def _trip_worker(config, entered, provider):
    assert entered.wait(5)
    try:
        error = (
            RuntimeError(
                "Account quota exceeded; quota will reset at 2026-10-27 01:00 UTC"
            )
            if provider == "volcano-plan"
            else openai.APIStatusError(
                "Insufficient Balance",
                response=httpx.Response(
                    402, request=httpx.Request("POST", "https://mock.invalid")
                ),
                body={"error": {"message": "Insufficient Balance"}},
            )
        )
        bounded_call(Mock(side_effect=error), config=config, num_retries=120)
    except FatalRolloutError:
        pass


@pytest.mark.parametrize("provider", ["volcano-plan", "deepseek-official"])
def test_eval_worker_quota_wakes_wait_cancels_inflight_and_exits(tmp_path, provider):
    """A missing remote callback cannot postpone fatal until request_timeout."""
    context = multiprocessing.get_context("fork")
    entered = context.Event()
    canceled = threading.Event()
    config = Tau2EnvConfig(
        _quota_breaker_state_path=str(tmp_path / "quota-breaker.json"),
        quota_breaker_threshold=1,
    )
    dispatcher = BatchTaskDispatcher(8, Mock(), Mock())
    dispatcher.runner.initialize(logger=Mock())
    submissions = []

    async def missing_callback():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()

    def submit(row, **kwargs):
        task_id = len(submissions)
        dispatcher.submit_task_input(SimpleNamespace(task_id=task_id))
        submissions.append(row)
        dispatcher.runner.submit(missing_callback, task_id=task_id)
        return task_id

    # The production caller passes timeout=None; a test-only ceiling makes
    # reverting the monitor fail promptly instead of hanging the test process.
    wait = dispatcher.wait_results
    dispatcher.wait_results = lambda *a, **kw: wait(*a, **{**kw, "timeout": 3})
    controller = SimpleNamespace(dispatcher=dispatcher, submit=submit)
    rows = expand_trials([{"domain": "airline", "task_id": "fake"}], 5)
    worker = context.Process(target=_trip_worker, args=(config, entered, provider))
    worker.start()
    started = time.monotonic()
    try:
        with pytest.raises(SystemExit) as caught:
            with quota_exit_guard(config):
                evaluate.collect_eval_records(
                    controller,
                    rows,
                    workflow_kwargs={"econfig": asdict(config)},
                    max_inflight=2,
                    output_root=tmp_path,
                    repeats=5,
                )
        assert caught.value.code == QUOTA_EXIT_CODE
        assert time.monotonic() - started < 2.5
        assert len(submissions) == 2
        assert canceled.wait(1)
        with pytest.raises(FatalRolloutError):
            submit(rows[-1])
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["status"] == "failed"
        reset = (
            "2026-10-27 01:00 UTC"
            if provider == "volcano-plan"
            else "not supplied by provider"
        )
        assert reset in summary["quota_breaker"]["fatal"]
        assert summary["quota_breaker"]["provider_reset_time"] == reset
        assert "FatalRolloutError" in summary["error"]
        assert summary["completed_episodes"] == summary["infra_excluded_episodes"] == 0
    finally:
        worker.join(timeout=5)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)
        dispatcher.runner.destroy(timeout=2)
    assert worker.exitcode == 0


def test_eval_pretripped_breaker_does_not_submit(tmp_path):
    """Already fatal state blocks even the initial rolling window."""
    config = Tau2EnvConfig(_quota_breaker_state_path=str(tmp_path / "quota.json"))
    (tmp_path / "quota.json").write_text(
        json.dumps({"fatal": "provider reset tomorrow"})
    )
    controller = Mock()
    with pytest.raises(FatalRolloutError, match="reset tomorrow"):
        evaluate.collect_eval_records(
            controller,
            [],
            workflow_kwargs={"econfig": asdict(config)},
            max_inflight=2,
            output_root=tmp_path,
            repeats=1,
        )
    controller.submit.assert_not_called()
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"


def test_eval_binds_attempt_local_breaker_before_workflow_kwargs(monkeypatch, tmp_path):
    """Remote worker kwargs cannot inherit a training run's breaker path."""
    config = _config(monkeypatch, tmp_path)
    config.econfig._quota_breaker_state_path = str(tmp_path / "training.json")
    first = evaluate.bind_eval_config(
        config, model_path="fake", output_root=tmp_path / "a"
    )
    second = evaluate.bind_eval_config(
        config, model_path="fake", output_root=tmp_path / "b"
    )
    path = evaluate.build_workflow_kwargs(first)["econfig"]["_quota_breaker_state_path"]
    assert path == str(tmp_path / "a/quota-breaker.json")
    assert second.econfig._quota_breaker_state_path != path
    assert config.econfig._quota_breaker_state_path == str(tmp_path / "training.json")


@pytest.mark.parametrize("shared_fatal", [True, False])
def test_eval_fatal_controller_cleanup_has_total_wait_bound(
    monkeypatch, tmp_path, shared_fatal
):
    """A hung destroy RPC cannot hold fatal teardown beyond its outer budget."""
    config = Tau2EnvConfig(_quota_breaker_state_path=str(tmp_path / "quota.json"))
    if shared_fatal:
        (tmp_path / "quota.json").write_text(json.dumps({"fatal": "reset tomorrow"}))
    release = threading.Event()
    entered = threading.Event()
    finished = threading.Event()

    def destroy():
        entered.set()
        release.wait(3)
        finished.set()

    join = threading.Thread.join
    budgets = []

    def short_join(thread, timeout=None):
        budgets.append(timeout)
        return join(thread, timeout=0.05)

    monkeypatch.setattr(threading.Thread, "join", short_join)
    started = time.monotonic()
    try:
        evaluate.destroy_eval_controller(
            SimpleNamespace(destroy=destroy), config, fatal=not shared_fatal
        )
        assert entered.is_set()
        assert time.monotonic() - started < 0.5
        assert budgets == [30]
    finally:
        release.set()
        assert finished.wait(1)


@pytest.mark.parametrize("trip_stage", ["initialize", "initialize_error", "cleanup"])
def test_eval_main_quota_has_dedicated_exit_despite_cleanup_error(
    monkeypatch, tmp_path, trip_stage
):
    """Exercise the real entrypoint guard, without starting any engine or client."""
    config = _config(monkeypatch, tmp_path)
    monkeypatch.setattr(evaluate, "load_eval_config", lambda _: (config, "fake.yaml"))
    monkeypatch.setattr(evaluate, "build_eval_rows", lambda *a: [])
    monkeypatch.setattr(
        evaluate.benchmark_variant, "validate_task_files", lambda *a: {}
    )
    monkeypatch.setattr(evaluate, "LocalScheduler", Mock())
    controller = Mock()
    monkeypatch.setattr(
        evaluate.RemoteSGLangEngine, "as_controller", Mock(return_value=controller)
    )

    def trip(**kwargs):
        path = tmp_path / "eval/quota-breaker.json"
        path.write_text(json.dumps({"fatal": "provider reset tomorrow"}))

    if trip_stage == "initialize_error":
        controller.initialize.side_effect = FatalRolloutError("provider reset tomorrow")
        controller.destroy.side_effect = RuntimeError("cleanup RPC failed")
    elif trip_stage == "initialize":
        controller.initialize.side_effect = trip
        controller.destroy.side_effect = RuntimeError("cleanup RPC failed")
    else:
        controller.destroy.side_effect = trip
    with pytest.raises(SystemExit) as caught:
        evaluate.main(
            [
                "--config",
                "fake.yaml",
                "--model-path",
                "fake",
                "--output-root",
                str(tmp_path / "eval"),
            ]
        )
    assert caught.value.code == QUOTA_EXIT_CODE
    controller.initialize.assert_called_once()
    assert (
        json.loads((tmp_path / "eval/summary.json").read_text())["status"] == "failed"
    )
