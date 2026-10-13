# SPDX-License-Identifier: Apache-2.0
"""SAO policy wrapper: task-pack tasks plus the official fixed train split."""

from __future__ import annotations

import random
import sys
from collections import Counter
from typing import Any

from datasets import Dataset

from examples.tau2.contracts import build_sampling_schedule
from examples.tau2.task_pack import _task_pack_rows
from examples.tau2.train import get_tau2_dataset, logger
from examples.tau2.train import main as policy_main
from examples.tau2.utils import Tau2PPOConfig


def build_train_dataset(
    config: Tau2PPOConfig, dataset_kwargs: dict[str, Any]
) -> Dataset:
    """Combine the unscheduled pools, then require equal positive row coverage."""

    if not config.train_task_pack:
        raise ValueError("SAO pack entry requires train_task_pack")
    if config.algorithm != "sao":
        raise ValueError("SAO pack entry requires algorithm=sao")
    if config.econfig.benchmark_variant != "fixed":
        raise ValueError("SAO pack training requires benchmark_variant=fixed")
    domains = tuple(dataset_kwargs["domain"])
    if config.domain_effective_episodes and set(
        config.domain_effective_episodes
    ) != set(domains):
        raise ValueError(
            "domain_effective_episodes must cover selected domains exactly"
        )
    rows = _task_pack_rows(
        config.train_task_pack,
        domains,
        provenance=dataset_kwargs["provenance"],
        seed=config.seed,
    )
    official = get_tau2_dataset(**dataset_kwargs, split="train")
    pack_counts = Counter(row["domain"] for row in rows)
    official_counts = Counter(row["domain"] for row in official)
    rows += [{**row, "task_pack": None} for row in official]
    scheduled = (
        build_sampling_schedule(
            rows,
            domain_effective_episodes=config.domain_effective_episodes,
            algorithm=config.algorithm,
            seed=config.seed,
        )
        if config.domain_effective_episodes
        else rows.copy()
    )

    def identity(row: dict[str, Any]) -> tuple[str, str, str | None]:
        return row["domain"], row["task_id"], row["task_pack"]

    counts = Counter(identity(row) for row in scheduled)
    repetitions = {counts[identity(row)] for row in rows}
    if len(repetitions) != 1 or 0 in repetitions:
        raise ValueError(
            "SAO pack schedule must repeat every row the same positive number of "
            "times; domain_effective_episodes must be a multiple of each combined "
            "domain pool size with the same multiplier across domains"
        )
    if config.task_limit is not None and config.task_limit < len(scheduled):
        raise ValueError("task_limit must not truncate the SAO pack schedule")
    logger.info(
        "SAO pack pool: pack=%s, official=%s, repeats=%d",
        dict(pack_counts),
        dict(official_counts),
        next(iter(repetitions)),
    )
    random.Random(config.seed).shuffle(scheduled)
    return Dataset.from_list(scheduled)


def main(args: list[str]) -> None:
    policy_main(args, train_rows_hook=build_train_dataset)


if __name__ == "__main__":
    main(sys.argv[1:])
