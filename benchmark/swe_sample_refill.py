# SPDX-License-Identifier: Apache-2.0

"""Frozen-weight Arena SWE cohort through AReaL's actual grouped dispatcher."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, build_opener

import aiofiles
from datasets import Dataset

from benchmark.rolling_batch import RollingBatchWindow
from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaAPIError, ArenaOpenAPIClient
from examples.swe.arena_config import (
    build_weighted_arena_rows,
    load_arena_stream_configs,
)
from examples.swe.train_swe_rl import _resolve_arena_stream
from examples.swe.utils import SWEPPOConfig

from areal.api.cli_args import SGLangConfig, load_expr_config
from areal.engine import RemoteSGLangEngine
from areal.infra import LocalScheduler, workflow_context
from areal.utils import logging
from areal.utils.network import format_hostport

logger = logging.getLogger("WorkflowExecutor")


class MetadataArenaClient(ArenaOpenAPIClient):
    """Read live stream metadata without computing historical task statistics."""

    @staticmethod
    def select_stream(items: list[dict[str, Any]], stream_id: str) -> dict[str, Any]:
        matches = [item for item in items if item.get("stream_id") == stream_id]
        if len(matches) != 1 or matches[0].get("status") != "ACTIVE":
            raise ArenaAPIError(f"Expected one active stream {stream_id!r}")
        return matches[0]

    def resolve_stream(
        self, stream_id: str = "", *, client: Any = None
    ) -> dict[str, Any]:
        return self.select_stream(self.list_streams(client=client), stream_id)

    async def resolve_stream_async(
        self, stream_id: str = "", *, client: Any, timeout: float
    ) -> dict[str, Any]:
        items = await self.list_streams_async(client=client, timeout=timeout)
        return self.select_stream(items, stream_id)


def get_benchmark_dataset(econfig: Any) -> tuple[Any, list[Any]]:
    client = MetadataArenaClient(
        base_url=econfig.arena_base_url,
        timeout=econfig.arena_request_timeout,
        request_retries=econfig.arena_request_retries,
    )
    streams, rows_by_stream = [], {}
    for configured in load_arena_stream_configs(econfig):
        resolved, rows = _resolve_arena_stream(client, configured)
        streams.append(resolved)
        rows_by_stream[resolved.name] = rows
    rows = build_weighted_arena_rows(
        rows_by_stream,
        streams,
        epoch_size=econfig.arena_mixture_epoch_size,
        size_multiple=1,
    )
    return Dataset.from_list(rows), streams


class RecordedArenaAgent(ArenaStreamAgentWorkflow):
    """Record agent execution, separately from group export and delivery latency."""

    def __init__(self, event_dir: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = MetadataArenaClient(
            base_url=self.client.base_url,
            timeout=self.request_timeout,
            request_retries=self.econfig.get("arena_request_retries", 3),
        )
        self.event_dir = Path(event_dir)
        self.event_dir.mkdir(parents=True, exist_ok=True)
        self._event_lock = asyncio.Lock()

    async def _record(self, data: dict[str, Any], event: str, **values: Any) -> None:
        context = workflow_context.get()
        record = {
            "event": event,
            "time": time.time(),
            "monotonic": time.monotonic(),
            "task_id": context.task_id,
            "sample_idx": context.sample_idx,
            "data_id": data["data_id"],
            **values,
        }
        async with self._event_lock:
            async with aiofiles.open(
                self.event_dir / f"agents-{os.getpid()}.jsonl", "a"
            ) as stream:
                await stream.write(json.dumps(record) + "\n")

    async def run(self, data: dict[str, Any], **kwargs: Any) -> float:
        await self._record(data, "agent_start")
        status = "failed"
        try:
            reward = await super().run(data, **kwargs)
            status = "completed"
            return reward
        finally:
            result = self._task_result.get()
            await self._record(
                data,
                "agent_end",
                outcome=status,
                arena_task_id=result.task_id if result else None,
                arena_status=result.status if result else None,
                arena_score=result.score if result else None,
            )


def validate_canary_events(events: list[dict[str, Any]]) -> None:
    """Accept the same successful terminal statuses as ArenaOpenAPIClient."""
    ends = [event for event in events if event["event"] == "agent_end"]
    if (
        len(ends) != 1
        or ends[0]["arena_status"] not in {"DONE", "OK"}
        or ends[0]["outcome"] != "completed"
    ):
        raise RuntimeError("Canary did not complete the Arena lifecycle normally")


def collect_rolling_batches(
    controller: Any,
    rows: list[dict[str, Any]],
    workflow_kwargs: dict[str, Any],
    output: Path,
    start: float,
    group_size: int,
    batch_size: int,
    prefetch_batches: int,
    measure_batches: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    window = RollingBatchWindow(batch_size, prefetch_batches, measure_batches)
    if len(rows) < window.total_groups:
        raise ValueError(
            "Candidate list must cover measured batches and their lookahead tail"
        )
    records, batches = [], []
    previous_ready = 0.0

    def wait_group(task_id: int) -> tuple[float, Any]:
        result = controller.dispatcher.wait_for_task(task_id, timeout)
        return time.monotonic(), result

    with ThreadPoolExecutor(max_workers=window.window_size) as pool:
        futures = {}

        def fill_window() -> None:
            while (
                window.submitted < len(rows)
                and (task_id := window.take_submission()) is not None
            ):
                controller.submit(
                    rows[task_id],
                    workflow="benchmark.swe_sample_refill.RecordedArenaAgent",
                    workflow_kwargs=workflow_kwargs,
                    task_id=task_id,
                    is_eval=True,
                    group_size=group_size,
                    min_usable_group_size=group_size,
                    reward_normalization=True,
                    drop_incomplete_group=False,
                )
                futures[pool.submit(wait_group, task_id)] = task_id

        fill_window()
        with (
            (output / "groups.jsonl").open("a") as group_stream,
            (output / "batches.jsonl").open("a") as batch_stream,
        ):
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                # Preserve completion order even if multiple futures wake the collector.
                for future in sorted(
                    completed, key=lambda item: (item.result()[0], futures[item])
                ):
                    task_id = futures.pop(future)
                    completed_at, result = future.result()
                    record = {
                        "task_id": task_id,
                        "data_id": rows[task_id]["data_id"],
                        "elapsed": completed_at - start,
                        "accepted": result is not None,
                        "usable_samples": len(
                            result.trajectory["rollout_group"].row_counts
                        )
                        if result is not None
                        else 0,
                    }
                    records.append(record)
                    group_stream.write(json.dumps(record) + "\n")
                    group_stream.flush()
                    ready = window.complete(task_id, accepted=result is not None)
                    if result is None:
                        # Match prepare_batch: rejected prompts do not count toward
                        # training batches, but their slot can admit a new prompt.
                        fill_window()
                    if ready is not None:
                        batch = {
                            "batch_index": window.batches,
                            "task_ids": ready,
                            "elapsed": completed_at - start,
                            "interval_s": completed_at - start - previous_ready,
                            "consumed_groups": window.consumed,
                            "wall_time": time.time(),
                        }
                        previous_ready = batch["elapsed"]
                        batches.append(batch)
                        batch_stream.write(json.dumps(batch) + "\n")
                        batch_stream.flush()
                        logger.info(
                            "BATCH_READY batch=%d elapsed=%.2fs interval=%.2fs consumed=%d",
                            window.batches,
                            batch["elapsed"],
                            batch["interval_s"],
                            window.consumed,
                        )
                        fill_window()
                    logger.info(
                        "Rolling progress: completed=%d submitted=%d measured_batches=%d draining=%s",
                        len(records),
                        window.submitted,
                        window.batches,
                        window.batches == measure_batches,
                    )
    if window.batches != measure_batches or len(records) != window.submitted:
        raise RuntimeError("Incomplete rolling cohort")
    return records, batches


def check_gpu_memory_headroom(
    output: Path, minimum_free_gib: float, expected_gpus: int
) -> None:
    """Fail before Arena submission if model/graph initialization leaves too little memory."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    devices = []
    for line in result.stdout.splitlines():
        index, total, free = [value.strip() for value in line.split(",")]
        devices.append(
            {"index": int(index), "total_mib": float(total), "free_mib": float(free)}
        )
    (output / "startup-gpu-memory.json").write_text(
        json.dumps({"minimum_free_gib": minimum_free_gib, "devices": devices}, indent=2)
        + "\n"
    )
    if len(devices) != expected_gpus or any(
        device["free_mib"] < minimum_free_gib * 1024 for device in devices
    ):
        raise RuntimeError(
            "GPU count or post-initialization memory headroom failed resource preflight"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["canary", "group", "sample"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--sample-slots", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=20000)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=3)
    parser.add_argument("--measure-batches", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--min-free-gpu-gib", type=float, default=0)
    args, config_args = parser.parse_known_args()
    if (
        args.group_size < 1
        or args.sample_slots < args.group_size
        or args.sample_slots % args.group_size
    ):
        parser.error("sample-slots must be a positive multiple of group-size")
    if args.batch_size < 0 or min(args.prefetch_batches, args.measure_batches) < 1:
        parser.error(
            "batch-size must be nonnegative; prefetch/measure counts must be positive"
        )
    if args.mode == "canary" and args.batch_size:
        parser.error("canary and rolling batch mode are separate trials")
    if args.min_free_gpu_gib < 0:
        parser.error("min-free-gpu-gib must be nonnegative")
    config, _ = load_expr_config(config_args, SWEPPOConfig)
    args.output.mkdir(parents=True, exist_ok=False)
    logging.setup_file_logging(str(args.output))

    dataset, streams = get_benchmark_dataset(config.econfig)
    rows = sorted(dataset.to_list(), key=lambda row: (row["stream_id"], row["data_id"]))
    if args.batch_size:
        window = RollingBatchWindow(
            args.batch_size, args.prefetch_batches, args.measure_batches
        )
        if len(rows) < window.total_groups:
            raise ValueError(
                f"Need {window.total_groups} unique prompts, but stream has {len(rows)}"
            )
        identities = {(row["stream_id"], row["data_id"]) for row in rows}
        if len(identities) != len(rows):
            raise ValueError("Rolling benchmark requires unique prompt identities")
        random.Random(args.data_seed).shuffle(rows)
    manifest = {"rows": rows, "streams": [asdict(stream) for stream in streams]}
    serialized = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    if args.manifest.exists():
        if json.loads(args.manifest.read_text()) != manifest:
            raise RuntimeError(
                "Arena cohort or stream contract changed since manifest creation"
            )
    else:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(serialized + "\n")
    config.econfig.arena_streams = streams
    # Resolve agent hooks/transforms before reserving GPU runtime resources.
    RecordedArenaAgent(
        event_dir=str(args.output / "events"),
        econfig=asdict(config.econfig),
        gen_args={"temperature": config.gconfig.temperature},
        timeout=config.econfig.timeout,
    )
    if args.prepare_only:
        logger.info("Prepared fixed cohort: rows=%d sha256=%s", len(rows), digest)
        return
    if args.mode == "canary":
        rows = rows[:1]
    group_size = 1 if args.mode == "canary" else args.group_size
    config.rollout.max_concurrent_rollouts = args.sample_slots // group_size
    config.rollout.max_concurrent_samples = (
        args.sample_slots if args.mode == "sample" else None
    )
    # A frozen checkpoint cannot advance its version to replenish this budget.
    config.rollout.consumer_batch_size = len(rows)
    config.rollout.max_head_offpolicyness = 0
    config.rollout.dump_to_file = True
    config.econfig.arena_result_dump_dir = str(args.output / "arena_results")
    scheduler = LocalScheduler(exp_config=config, log_dir=args.output / "workers")
    controller = RemoteSGLangEngine.as_controller(config.rollout, scheduler)
    stop_monitor = threading.Event()
    monitor = None
    inference_monitor = None
    records = []
    try:
        server_args = SGLangConfig.build_args(config.sglang, tp_size=1, base_gpu_id=0)
        controller.initialize(role="eval-rollout", server_args=server_args)
        controller.start_proxy()
        controller.start_proxy_gateway()
        controller.set_version(0)
        if args.min_free_gpu_gib:
            check_gpu_memory_headroom(
                args.output, args.min_free_gpu_gib, len(controller.server_infos)
            )
        targets = [
            f"http://{format_hostport(info.host, info.port)}/metrics"
            for info in controller.server_infos
        ]
        (args.output / "inference_targets.json").write_text(
            json.dumps(targets, indent=2) + "\n"
        )

        def monitor_inference_metrics() -> None:
            def scrape(url: str) -> dict[str, Any]:
                try:
                    with build_opener(ProxyHandler({})).open(
                        url, timeout=3
                    ) as response:
                        return {"target": url, "text": response.read().decode()}
                except Exception as exc:
                    return {"target": url, "error": str(exc)}

            with (
                gzip.open(args.output / "sglang-metrics.jsonl.gz", "at") as stream,
                ThreadPoolExecutor(max_workers=len(targets)) as pool,
            ):
                while not stop_monitor.is_set():
                    captured = {
                        "wall_time": time.time(),
                        "metrics": list(pool.map(scrape, targets)),
                    }
                    stream.write(json.dumps(captured) + "\n")
                    stream.flush()
                    stop_monitor.wait(5)

        inference_monitor = threading.Thread(
            target=monitor_inference_metrics, daemon=True
        )
        inference_monitor.start()
        kwargs = {
            "event_dir": str(args.output / "events"),
            "econfig": asdict(config.econfig),
            "gen_args": {
                "temperature": config.gconfig.temperature,
                "max_completion_tokens": config.gconfig.max_new_tokens,
            },
            "timeout": config.econfig.timeout,
        }
        start = time.monotonic()

        def monitor_capacity() -> None:
            with (args.output / "capacity.jsonl").open("a") as stream:
                while not stop_monitor.is_set():
                    sample_stats = controller.dispatcher.sample_stats()
                    group_stats = asdict(controller.staleness_manager.get_stats())
                    stream.write(
                        json.dumps(
                            {
                                "elapsed": time.monotonic() - start,
                                **sample_stats,
                                **group_stats,
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                    stop_monitor.wait(1)

        monitor = threading.Thread(target=monitor_capacity, daemon=True)
        monitor.start()
        (args.output / "measurement.json").write_text(
            json.dumps(
                {
                    "start_monotonic": start,
                    "start_wall_time": time.time(),
                    "batch_size": args.batch_size,
                    "prefetch_batches": args.prefetch_batches,
                    "measure_batches": args.measure_batches,
                    "planned_groups": len(rows),
                },
                indent=2,
            )
            + "\n"
        )
        batches = []
        if args.batch_size:
            records, batches = collect_rolling_batches(
                controller,
                rows,
                kwargs,
                args.output,
                start,
                group_size,
                args.batch_size,
                args.prefetch_batches,
                args.measure_batches,
                args.timeout,
            )
        else:
            pending = {}
            for index, row in enumerate(rows):
                task_id = controller.submit(
                    row,
                    workflow="benchmark.swe_sample_refill.RecordedArenaAgent",
                    workflow_kwargs=kwargs,
                    task_id=index,
                    is_eval=True,
                    group_size=group_size,
                    min_usable_group_size=group_size,
                    reward_normalization=True,
                    drop_incomplete_group=False,
                )
                pending[task_id] = row["data_id"]

            with ThreadPoolExecutor(max_workers=len(rows)) as pool:
                futures = {
                    pool.submit(
                        controller.dispatcher.wait_for_task, task_id, args.timeout
                    ): task_id
                    for task_id in pending
                }
                with (args.output / "groups.jsonl").open("a") as stream:
                    for future in as_completed(futures):
                        task_id = futures[future]
                        result = future.result()
                        record = {
                            "task_id": task_id,
                            "data_id": pending[task_id],
                            "elapsed": time.monotonic() - start,
                            "accepted": result is not None,
                            "usable_samples": len(
                                result.trajectory["rollout_group"].row_counts
                            )
                            if result is not None
                            else 0,
                        }
                        records.append(record)
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        logger.info(
                            "Cohort progress: %d/%d groups, task=%d accepted=%s",
                            len(records),
                            len(rows),
                            task_id,
                            record["accepted"],
                        )
        elapsed = time.monotonic() - start
        stats = controller.export_stats()
        summary = {
            "mode": args.mode,
            "model": config.actor.path,
            "manifest_sha256": digest,
            "groups": len(records),
            "candidate_groups": len(rows),
            "group_size": group_size,
            "sample_slots": args.sample_slots,
            "elapsed_s": elapsed,
            "accepted_groups": sum(item["accepted"] for item in records),
            "usable_samples": sum(item["usable_samples"] for item in records),
            "stats": stats,
            "batch_size": args.batch_size,
            "prefetch_batches": args.prefetch_batches,
            "batch_ready": batches,
            "measurement_elapsed_s": batches[-1]["elapsed"] if batches else elapsed,
            "drain_elapsed_s": elapsed - batches[-1]["elapsed"] if batches else 0.0,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logger.info(
            "Cohort completed: mode=%s elapsed=%.1fs accepted=%d/%d",
            args.mode,
            elapsed,
            summary["accepted_groups"],
            len(rows),
        )
        if args.mode == "canary":
            events = [
                json.loads(line)
                for path in (args.output / "events").glob("*.jsonl")
                for line in path.read_text().splitlines()
            ]
            validate_canary_events(events)
        if not args.batch_size and summary["accepted_groups"] != len(rows):
            raise RuntimeError(
                "Cohort has rejected groups; inspect Arena failures before proceeding"
            )
    finally:
        stop_monitor.set()
        if monitor is not None:
            monitor.join(timeout=5)
        if inference_monitor is not None:
            inference_monitor.join(timeout=5)
        controller.destroy()


if __name__ == "__main__":
    main()
