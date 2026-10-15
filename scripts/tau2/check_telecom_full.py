# SPDX-License-Identifier: Apache-2.0
"""CPU-only audit and split/overlap manifest for the unchanged official full pool.

TAU2_DATA_DIR must point to the pinned official data. No LLM is used. Example:
  python scripts/tau2/check_telecom_full.py --areal-pack <v1> --output-root <art>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def id_fault_key(task_id: str) -> str:
    return re.sub(r"\[PERSONA:[^\]]*\]$", "", task_id)


def fault_key(task: dict[str, Any]) -> str:
    """Exact initial-state identity, preserving initialization action order.

    The ID also encodes the reported symptom: different symptom IDs can set up
    identical faults. Hash the complete initial_state instead of the ID so these
    overlaps are counted. No effective-state or semantic equivalence is inferred.
    """
    state = canonical(task["initial_state"])
    return "initial_state:" + hashlib.sha256(state.encode()).hexdigest()


def overlap_rows(
    tasks: list[dict[str, Any]],
    splits: dict[str, list[str]],
    pack_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    test_ids = set(splits["test"])
    test_by_key: dict[str, list[str]] = defaultdict(list)
    pack_by_key: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        if task["id"] in test_ids:
            test_by_key[fault_key(task)].append(task["id"])
    for task in pack_tasks:
        pack_by_key[fault_key(task)].append(task["id"])
    membership = {
        name: set(splits[name]) for name in ("small", "train", "test", "base")
    }
    return [
        {
            "task_id": task["id"],
            "official_splits": {
                name: task["id"] in ids for name, ids in membership.items()
            },
            "id_fault_combination_key": id_fault_key(task["id"]),
            "fault_combination_key": fault_key(task),
            "shares_fault_combination_with_test40": fault_key(task) in test_by_key,
            "test40_same_initial_state_ids": test_by_key.get(fault_key(task), []),
            "overlaps_areal_pack_v1": fault_key(task) in pack_by_key,
            "areal_pack_v1_same_initial_state_ids": pack_by_key.get(
                fault_key(task), []
            ),
        }
        for task in tasks
    ]


def _init() -> None:
    import socket

    from loguru import logger

    from examples.tau2 import task_pack

    logger.remove()
    # replay() accepts task_db; None delegates to the unmodified official DBs.
    task_pack.install("telecom")

    def no_network(*args, **kwargs):
        raise RuntimeError("Network disabled during CPU Telecom audit")

    socket.socket.connect = no_network
    socket.create_connection = no_network


def check_task(raw: dict[str, Any]) -> dict[str, Any]:
    from tau2.data_model.tasks import Task

    from scripts.tau2.check_task_pack import evaluate, replay

    task = Task.model_validate(raw)
    results: dict[str, Any] = {"task_id": task.id, "reasons": []}
    for label, actions, expected in (
        ("gold", task.evaluation_criteria.actions or [], 1.0),
        ("empty", [], 0.0),
    ):
        try:
            messages, errors = replay("telecom", task, actions, None)
            action_names = {str(i): action.name for i, action in enumerate(actions)}
            tool_errors = [
                {
                    "name": action_names.get(m.id),
                    "requestor": m.requestor,
                    "content": m.content,
                }
                for m in messages
                if m.role == "tool" and m.error
            ]
            # Same scorer as the reference pack run, including its fixed transfer
            # rule; default_db=True selects official db.toml and user_db.toml.
            info = evaluate("telecom", task, messages, None, default_db=True)
            failed_assertions = [
                c.env_assertion.model_dump(mode="json")
                for c in info.env_assertions or []
                if not c.met
            ]
            results[label] = {
                "score": info.reward,
                "passed": info.reward == expected and errors == 0,
                "tool_errors": tool_errors,
                "failed_env_assertions": failed_assertions,
                "db_match": info.db_check.db_match if info.db_check else None,
            }
            if info.reward != expected:
                results["reasons"].append(
                    f"{label}_score={info.reward}, expected={expected}"
                )
            if errors:
                results["reasons"].append(f"{label}_tool_errors={errors}")
        except Exception as exc:  # noqa: BLE001 - record every task failure
            error = f"{type(exc).__name__}: {exc}"
            results[label] = {"score": None, "passed": False, "error": error}
            results["reasons"].append(f"{label}: {error}")
    return results


def audit(areal_pack: Path, output_root: Path, workers: int) -> dict[str, Any]:
    from tau2.domains.telecom import utils
    from tau2.domains.telecom.data_model import TelecomDB
    from tau2.domains.telecom.user_data_model import TelecomUserDB
    from tau2.registry import registry

    from examples.tau2 import benchmark_variant, task_pack
    from examples.tau2.agent import _get_task
    from examples.tau2.contracts import validate_installed_tau2_revision
    from scripts.tau2.evaluate import build_eval_rows

    target = output_root / "telecom-full-manifest.json"
    failures_path = output_root / "telecom-full-failures.json"
    for path in (target, failures_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    provenance = validate_installed_tau2_revision()
    benchmark_variant.activate("official", ["telecom"])
    source = Path(utils.TELECOM_TASK_SET_PATH).parent
    names = (
        "tasks.json",
        "tasks_full.json",
        "split_tasks.json",
        "db.toml",
        "user_db.toml",
    )
    provenance["files"] = {
        name: {
            "path": str(source / name),
            "sha256": benchmark_variant.file_sha256(source / name),
        }
        for name in names
    }
    assert (source / "tasks.json").read_bytes() == (
        source / "tasks_full.json"
    ).read_bytes()
    tasks = json.loads((source / "tasks_full.json").read_text())
    splits = registry.get_task_splits_loader("telecom")()
    assert len(tasks) == len({t["id"] for t in tasks}) == 2285
    assert set(splits["full"]) == {t["id"] for t in tasks}
    assert all(t.get("user_tools") is None for t in tasks)
    assert all(not t["evaluation_criteria"].get("nl_assertions") for t in tasks)
    loaded = {t.id: t for t in registry.get_tasks_loader("telecom")(None)}
    # exclude_unset preserves absent fields; this checks every raw field, including
    # reward_basis, rather than comparing only IDs or the number of tasks.
    assert all(
        loaded[t["id"]].model_dump(mode="json", exclude_unset=True) == t for t in tasks
    )
    for raw in (tasks[0], tasks[-1]):
        assert _get_task("telecom", raw["id"]) == loaded[raw["id"]]
    env = registry.get_env_constructor("telecom")()
    assert env.tools.db == TelecomDB.load(utils.TELECOM_DB_PATH)
    assert env.user_tools.db == TelecomUserDB.load(utils.TELECOM_USER_DB_PATH)
    from types import SimpleNamespace

    rows = build_eval_rows(
        SimpleNamespace(domains=["telecom"], seed=42, experiment_mode="formal"),
        4,
        "full",
    )
    assert len(rows) == 9140
    provenance.update(
        task_file_byte_identity=True,
        registry_all_fields_identity=True,
        default_agent_and_user_db_identity=True,
        evaluator_rows=len(rows),
        user_tools_none_count=len(tasks),
        scorer="check_task_pack.evaluate(default_db=True): official ALL + fixed transfer rule",
        reward_basis_counts=dict(
            Counter("+".join(t["evaluation_criteria"]["reward_basis"]) for t in tasks)
        ),
    )
    areal_pack = areal_pack.resolve()
    task_pack.load_manifest(str(areal_pack))
    pack_tasks = json.loads((areal_pack / "telecom/tasks.json").read_text())
    assert len(pack_tasks) == 261
    metadata = overlap_rows(tasks, splits, pack_tasks)
    by_state: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        by_state[fault_key(task)].add(id_fault_key(task["id"]))
    collisions = [sorted(ids) for ids in by_state.values() if len(ids) > 1]
    with Pool(workers, initializer=_init) as pool:
        results = pool.map(check_task, tasks, chunksize=4)
    for row, result in zip(metadata, results, strict=True):
        assert row["task_id"] == result["task_id"]
        row.update(result)
    failures = [row for row in metadata if row["reasons"]]
    summary = {
        "tasks": len(metadata),
        "planned_episodes_x4": len(rows),
        "official_split_counts": {name: len(ids) for name, ids in splits.items()},
        "id_fault_keys": len({id_fault_key(t["id"]) for t in tasks}),
        "initial_state_fault_keys": len(by_state),
        "initial_states_with_multiple_id_keys": len(collisions),
        "full_tasks_sharing_test40_fault_key": sum(
            r["shares_fault_combination_with_test40"] for r in metadata
        ),
        "full_tasks_overlapping_areal_v1": sum(
            r["overlaps_areal_pack_v1"] for r in metadata
        ),
        "areal_v1_tasks_matching_full": len(
            {tid for r in metadata for tid in r["areal_pack_v1_same_initial_state_ids"]}
        ),
        "gold_1_passed": sum(r["gold"]["passed"] for r in metadata),
        "empty_0_passed": sum(r["empty"]["passed"] for r in metadata),
        "failed_tasks": len(failures),
        "score_counts": dict(
            Counter(
                f"gold={r['gold']['score']} empty={r['empty']['score']}"
                for r in metadata
            )
        ),
    }
    payload = {
        "schema_version": 1,
        "summary": summary,
        "provenance": provenance,
        "areal_pack_v1": {
            "path": str(areal_pack),
            "manifest_sha256": benchmark_variant.file_sha256(
                areal_pack / "manifest.json"
            ),
        },
        "fault_key_definition": "SHA256 of canonical complete initial_state (JSON keys sorted; array order preserved). ID without persona over-separates identical states across symptom families.",
        "id_key_collision_groups": collisions,
        "tasks": metadata,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for path, value in ((target, payload), (failures_path, failures)):
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areal-pack", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _init()
    result = audit(args.areal_pack, args.output_root, args.workers)
    print(json.dumps(result["summary"], indent=2))
    return int(result["summary"]["failed_tasks"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
