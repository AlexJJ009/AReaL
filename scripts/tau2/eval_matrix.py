# SPDX-License-Identifier: Apache-2.0

"""Build and verify the independent τ² policy0 plus 4-by-3 evaluation ledger."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from examples.tau2.contracts import (
    OFFICIAL_TAU2_REVISION,
    SUPPORTED_DOMAINS,
    build_task_rows,
    validate_installed_tau2_revision,
)

BRANCHES = ("policy0", "mixed", "airline", "retail", "telecom")


def _parse_checkpoints(values: list[str]) -> dict[str, str]:
    checkpoints: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Expected BRANCH=CHECKPOINT, got {raw!r}")
        branch, checkpoint = raw.split("=", 1)
        if branch not in BRANCHES or branch in checkpoints or not checkpoint:
            raise ValueError(f"Invalid or duplicate checkpoint binding: {raw!r}")
        checkpoints[branch] = checkpoint
    if set(checkpoints) != set(BRANCHES):
        raise ValueError(
            f"Checkpoint bindings must cover {list(BRANCHES)}, got {sorted(checkpoints)}"
        )
    return checkpoints


def load_official_test_rows() -> list[dict[str, str]]:
    """Read the pinned package's official test split without a generated manifest."""

    from tau2.registry import registry

    validate_installed_tau2_revision()
    splits_by_domain: dict[str, dict[str, list[str]]] = {}
    for domain in SUPPORTED_DOMAINS:
        splits_loader = registry.get_task_splits_loader(domain)
        if splits_loader is None:
            raise ValueError(f"No task splits loader found for domain {domain}")
        splits_by_domain[domain] = splits_loader()
    return build_task_rows(
        splits_by_domain,
        domains=SUPPORTED_DOMAINS,
        split="test",
    )


def _test_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    identities = {
        (str(row["domain"]), str(row["task_id"]))
        for row in rows
        if row.get("split") == "test"
    }
    expected_counts = {"airline": 20, "retail": 40, "telecom": 40}
    observed_counts = {
        domain: sum(1 for row_domain, _ in identities if row_domain == domain)
        for domain in SUPPORTED_DOMAINS
    }
    if observed_counts != expected_counts or len(identities) != 100:
        raise ValueError(
            f"Official test identity mismatch: observed={observed_counts}, "
            f"expected={expected_counts}"
        )
    return [
        {"domain": domain, "task_id": task_id} for domain, task_id in sorted(identities)
    ]


def build_eval_plan(
    test_rows: list[dict[str, str]],
    *,
    checkpoints: dict[str, str],
    repeats: int,
    simulator_id: str,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("Evaluation repeats must be positive")
    if set(checkpoints) != set(BRANCHES):
        raise ValueError("Evaluation requires policy0 and all four actor branches")
    rows = _test_rows(test_rows)
    cells = []
    for branch in BRANCHES:
        for row in rows:
            for trial in range(repeats):
                cell_id = f"{branch}:{row['domain']}:{row['task_id']}:{trial}"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "branch": branch,
                        "checkpoint": checkpoints[branch],
                        "domain": row["domain"],
                        "task_id": row["task_id"],
                        "trial": trial,
                        "simulator_id": simulator_id,
                    }
                )
    return {
        "schema_version": 1,
        "tau2_revision": OFFICIAL_TAU2_REVISION,
        "repeats": repeats,
        "simulator_id": simulator_id,
        "checkpoints": checkpoints,
        "planned_cells": len(cells),
        "actor_matrix_units": 4 * 3,
        "policy0_baseline_domains": 3,
        "cells": cells,
        "side_effects": {"api_calls": 0, "gpu_processes": 0, "queue_tasks": 0},
    }


def verify_eval_ledger(
    plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = {cell["cell_id"]: cell for cell in plan["cells"]}
    observed: dict[str, dict[str, Any]] = {}
    for record in records:
        cell_id = str(record.get("cell_id", ""))
        if cell_id not in expected:
            raise ValueError(f"Unexpected evaluation cell: {cell_id}")
        if cell_id in observed:
            raise ValueError(f"Duplicate evaluation cell: {cell_id}")
        target = expected[cell_id]
        for key in (
            "branch",
            "checkpoint",
            "domain",
            "task_id",
            "trial",
            "simulator_id",
        ):
            if record.get(key) != target[key]:
                raise ValueError(
                    f"Evaluation cell {cell_id} changed {key}: "
                    f"{record.get(key)!r} != {target[key]!r}"
                )
        status = record.get("status")
        if status not in ("completed", "infra_failed"):
            raise ValueError(f"Evaluation cell {cell_id} has invalid status {status}")
        if status == "completed" and float(record.get("official_score", -1)) not in (
            0.0,
            1.0,
        ):
            raise ValueError(f"Evaluation cell {cell_id} lacks a binary official score")
        observed[cell_id] = record

    completed = [
        record for record in observed.values() if record["status"] == "completed"
    ]
    aggregates: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in completed:
        grouped[f"{record['branch']}:{record['domain']}"].append(
            float(record["official_score"])
        )
    for key, scores in sorted(grouped.items()):
        aggregates[key] = {
            "completed": len(scores),
            "mean_official_score": sum(scores) / len(scores),
        }
    missing = sorted(set(expected) - set(observed))
    infra_failed = sorted(
        cell_id
        for cell_id, record in observed.items()
        if record["status"] == "infra_failed"
    )
    return {
        "status": "complete" if not missing and not infra_failed else "partial",
        "planned": len(expected),
        "reported": len(observed),
        "completed": len(completed),
        "missing": missing,
        "infra_failed": infra_failed,
        "coverage": len(completed) / len(expected),
        "aggregates": aggregates,
    }


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[], required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--simulator-id", required=True)
    parser.add_argument("--ledger")
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args(argv)
    plan = build_eval_plan(
        load_official_test_rows(),
        checkpoints=_parse_checkpoints(args.checkpoint),
        repeats=args.repeats,
        simulator_id=args.simulator_id,
    )
    output: dict[str, Any] = {"plan": plan}
    if args.ledger:
        records = [
            json.loads(line)
            for line in Path(args.ledger).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        output["verification"] = verify_eval_ledger(plan, records)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])
