# SPDX-License-Identifier: Apache-2.0
"""Build a τ² task pack from screened AReaL-tau2-data tasks.

Usage:
  python scripts/tau2/build_areal_task_pack.py <areal data dir> <decisions.json> <out dir>

``<areal data dir>`` holds ``tau2_rl_train.jsonl`` and ``tau2_rl_database/``
from ``inclusionAI/AReaL-tau2-data``. ``<decisions.json>`` maps a task id to
``{"verdict": "keep"|"drop", "fixes": [...], "why": [...]}``; the screening work
directory composes it from every review round (``tools/decisions.py``). Only
``keep`` tasks enter the pack, with their fixes applied in order. A fix edits the
user's ``task_instructions`` (``instruction_replace``, then
``instruction_append``), may replace the gold actions and may set the
``reward_basis``, exactly as the
screening replay (``tools/score.py``) did. The decisions file is copied into the
pack, so every pack version records how it was made. Only domains with kept
tasks get a directory, so a Telecom pack can be built on its own.

The source files are parsed as data, never executed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

DOMAINS = ("airline", "retail", "telecom")


def _parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def task_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """The τ² task of one ``tau2_rl_train.jsonl`` row, before any fix."""

    task = {
        "id": row["id"],
        "description": _parse(row.get("description")),
        "user_scenario": _parse(row["user_scenario"]),
        "evaluation_criteria": _parse(row["evaluation_criteria"]),
    }
    # Telecom rows also set up the user's device and carry a plain-text ticket.
    if "initial_state" in row:
        task["initial_state"] = _parse(row["initial_state"])
    if "ticket" in row:
        task["ticket"] = row["ticket"]
    return task


def apply_fix(task: dict[str, Any], fix: dict[str, Any]) -> dict[str, Any]:
    instructions = task["user_scenario"]["instructions"]
    text = instructions["task_instructions"]
    for old, new in fix.get("instruction_replace") or []:
        if old not in text:
            raise ValueError(f"{task['id']}: instruction_replace {old!r} not found")
        text = text.replace(old, new)
    if fix.get("instruction_append"):
        text = text.rstrip() + "\n" + fix["instruction_append"]
    instructions["task_instructions"] = text
    if "gold_actions" in fix:
        task["evaluation_criteria"]["actions"] = [
            {
                "action_id": f"{task['id']}_fix_{i}",
                "requestor": action.get("requestor", "assistant"),
                "name": action["name"],
                "arguments": action["arguments"],
            }
            for i, action in enumerate(fix["gold_actions"])
        ]
    if "reward_basis" in fix:
        task["evaluation_criteria"]["reward_basis"] = fix["reward_basis"]
    return task


def build(source: Path, decisions_path: Path, out: Path) -> dict[str, Any]:
    from tau2.data_model.tasks import Task

    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    raw = {}
    for line in (
        (source / "tau2_rl_train.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        row = json.loads(line)
        raw[row["id"]] = row
    if out.exists():
        raise FileExistsError(out)
    tasks: dict[str, list[dict[str, Any]]] = {d: [] for d in DOMAINS}
    task_db: dict[str, dict[str, str]] = {d: {} for d in DOMAINS}
    splits = {}
    for tid in sorted(decisions, key=lambda t: (t.split("_")[0], int(t.split("_")[1]))):
        decision = decisions[tid]
        if decision["verdict"] != "keep":
            continue
        domain = tid.split("_")[0]
        row = raw[tid]
        task = task_from_row(row)
        for fix in decision["fixes"]:
            task = apply_fix(task, fix)
        Task.model_validate(task)
        tasks[domain].append(task)
        task_db[domain][tid] = row["db_path"]
        splits[f"{domain}:{tid}"] = "fix" if decision["fixes"] else "keep"
    files = {}
    present = [d for d in DOMAINS if tasks[d]]
    for domain in present:
        (out / domain).mkdir(parents=True)
        for name, payload in (
            ("tasks.json", tasks[domain]),
            ("task_db.json", task_db[domain]),
        ):
            path = out / domain / name
            path.write_text(
                json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            files[f"{domain}/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copyfile(decisions_path, out / "decisions.json")
    files["decisions.json"] = hashlib.sha256(
        (out / "decisions.json").read_bytes()
    ).hexdigest()
    for rel in sorted({p for d in task_db.values() for p in d.values()}):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, out / rel)
        files[rel] = hashlib.sha256((out / rel).read_bytes()).hexdigest()
    manifest = {
        "name": out.name,
        "source": "inclusionAI/AReaL-tau2-data tau2_rl_train.jsonl, screened",
        "counts": {d: len(tasks[d]) for d in present},
        "task_splits": splits,
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str]) -> None:
    source, decisions, out = (Path(a) for a in argv)
    manifest = build(source, decisions, out)
    print(json.dumps({"name": manifest["name"], "counts": manifest["counts"]}))


if __name__ == "__main__":
    main(sys.argv[1:])
