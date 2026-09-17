# SPDX-License-Identifier: Apache-2.0

"""CPU-only A/B benchmark of the real workflow dispatcher with fixed episode work.

Run from the repository root with ``python -m benchmark.sample_level_refill``.
Async sleeps model episode service times; measured speedups are NOT GPU speedups.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from areal.api import RolloutWorkflow
from areal.api.cli_args import InferenceEngineConfig
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_executor import WorkflowExecutor
from areal.utils import logging

logger = logging.getLogger("WorkflowExecutor")


class TimedEpisodes(RolloutWorkflow):
    """Identical episode work in both arms, with no rejection or early stopping."""

    def __init__(self, durations: list[float]):
        self.durations = durations
        self.origin = 0.0
        self.events: list[tuple[float, int]] = []
        self.starts: dict[int, float] = {}
        self.finished_members: dict[int, int] = {}
        self.group_ends: list[float] = []

    async def arun_episode(self, engine: Any, data: dict[str, Any]) -> dict[str, Any]:
        index = workflow_context.get().sample_idx
        assert index is not None
        group = data["group"]
        started = time.perf_counter() - self.origin
        self.starts.setdefault(group, started)
        self.events.append((started, 1))
        try:
            await asyncio.sleep(self.durations[index])
            return {
                "input_ids": torch.tensor(
                    [[group, index]], dtype=torch.long, device="cpu"
                )
            }
        finally:
            self.events.append((time.perf_counter() - self.origin, -1))
            finished = self.finished_members.get(group, 0) + 1
            self.finished_members[group] = finished
            if finished == len(self.durations):
                self.group_ends.append(time.perf_counter() - self.origin)


def run_once(
    mode: str, durations: list[float], groups: int, slots: int
) -> dict[str, Any]:
    group_size = len(durations)
    group_limit = slots // group_size
    config = InferenceEngineConfig(
        backend="sglang:d1",
        max_concurrent_rollouts=group_limit,
        max_concurrent_samples=slots if mode == "sample" else None,
        consumer_batch_size=groups,
        max_head_offpolicyness=0,
        queue_size=max(256, groups * 2),
        dump_to_file=False,
        check_trajectory_format=False,
    )
    executor = WorkflowExecutor(config, SimpleNamespace(get_version=lambda: 0))
    executor.initialize(train_data_parallel_size=1)
    episodes = TimedEpisodes(durations)
    workflow = GroupedRolloutWorkflow(episodes, group_size, logger)
    try:
        episodes.origin = time.perf_counter()
        for group in range(groups):
            executor.submit({"group": group}, workflow, task_id=group)
        results = executor.wait(groups, timeout=max(30.0, groups * max(durations) * 4))
        elapsed = time.perf_counter() - episodes.origin
        # Validate all episodes, including long ones, and their group membership.
        actual = sorted(
            (row[0], row[1])
            for result in results
            for row in result["input_ids"].tolist()
        )
        assert actual == [
            (group, index) for group in range(groups) for index in range(group_size)
        ]
        assert all(
            result["rollout_group"].row_counts == (1,) * group_size
            for result in results
        )
        active = peak = 0
        area = previous = 0.0
        for timestamp, change in sorted(episodes.events):
            area += active * (timestamp - previous)
            active += change
            peak = max(peak, active)
            previous = timestamp
        assert active == 0 and peak <= slots
        stats = executor.staleness_manager.get_stats()
        assert stats.accepted == groups and stats.rejected == 0
        first_refill = episodes.starts.get(group_limit)
        first_group_end = min(episodes.group_ends)
        return {
            "mode": mode,
            "elapsed_s": elapsed,
            "episodes_per_s": groups * group_size / elapsed,
            "groups_per_s": groups / elapsed,
            "mean_active_samples": area / elapsed,
            "peak_active_samples": peak,
            "first_refill_s": first_refill,
            "first_group_members_finished_s": first_group_end,
            "refilled_before_first_group_finished": first_refill is not None
            and first_refill < first_group_end,
            "accepted_groups": stats.accepted,
            "rejected_groups": stats.rejected,
        }
    finally:
        executor.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--slots", type=int, default=48)
    parser.add_argument("--short-members", type=int, default=9)
    parser.add_argument("--short-seconds", type=float, default=0.05)
    parser.add_argument("--long-seconds", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.group_size < 1
        or args.slots < args.group_size
        or args.slots % args.group_size
    ):
        parser.error("slots must be a positive multiple of group-size")
    if args.groups <= args.slots // args.group_size or args.repeats < 1:
        parser.error(
            "groups must exceed the initial group count; repeats must be positive"
        )
    if (
        not 0 < args.short_members < args.group_size
        or not 0 < args.short_seconds <= args.long_seconds
    ):
        parser.error(
            "use 0 < short-members < group-size and 0 < short-seconds <= long-seconds"
        )

    # Warm both paths before measuring; alternate A/B ordering across repeats.
    for mode in ("group", "sample"):
        run_once(
            mode, [0.01] * args.group_size, args.slots // args.group_size, args.slots
        )
    profiles = {
        "uniform": [args.long_seconds] * args.group_size,
        "long_tail": [args.short_seconds] * args.short_members
        + [args.long_seconds] * (args.group_size - args.short_members),
    }
    report: dict[str, Any] = {
        "kind": "synthetic_cpu_dispatcher_benchmark_not_gpu_throughput",
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "profiles": {},
    }
    for profile, durations in profiles.items():
        runs = []
        for repeat in range(args.repeats):
            modes = ("group", "sample") if repeat % 2 == 0 else ("sample", "group")
            for mode in modes:
                result = run_once(mode, durations, args.groups, args.slots)
                result["repeat"] = repeat
                runs.append(result)
                logger.info(
                    "%s repeat=%d mode=%s elapsed=%.3fs rate=%.1f episodes/s",
                    profile,
                    repeat,
                    mode,
                    result["elapsed_s"],
                    result["episodes_per_s"],
                )
        medians = {
            mode: statistics.median(
                run["elapsed_s"] for run in runs if run["mode"] == mode
            )
            for mode in ("group", "sample")
        }
        report["profiles"][profile] = {
            "runs": runs,
            "median_elapsed_s": medians,
            "speedup": medians["group"] / medians["sample"],
        }
    serialized = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    logger.info("Synthetic dispatcher benchmark:\n%s", serialized)


if __name__ == "__main__":
    main()
