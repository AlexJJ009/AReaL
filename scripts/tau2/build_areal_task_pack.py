# SPDX-License-Identifier: Apache-2.0
"""Build a τ² task pack from screened AReaL-tau2-data tasks.

Usage:
  python scripts/tau2/build_areal_task_pack.py <areal data dir> <screen dir> <out dir>

``<areal data dir>`` holds ``tau2_rl_train.jsonl`` and ``tau2_rl_database/``
from ``inclusionAI/AReaL-tau2-data``. ``<screen dir>`` is the screening work
directory: ``out/candidates_after_review.json`` names the tasks to keep
(``keep*`` or ``fix*``), and ``out/s*.jsonl`` holds the first-round rows whose
``fix`` field is applied here. A fix edits the user's ``task_instructions``
(``instruction_replace``, then ``instruction_append``) and may replace the gold
actions, exactly as the screening replay (``tools/score.py``) did.

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

DOMAINS = ("airline", "retail")


def _parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


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
                "requestor": "assistant",
                "name": action["name"],
                "arguments": action["arguments"],
            }
            for i, action in enumerate(fix["gold_actions"])
        ]
    return task


def first_round_rows(screen: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted((screen / "out").glob("s*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def build(source: Path, screen: Path, out: Path) -> dict[str, Any]:
    from tau2.data_model.tasks import Task

    candidates = json.loads(
        (screen / "out" / "candidates_after_review.json").read_text(encoding="utf-8")
    )
    verdicts = first_round_rows(screen)
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
    for tid in sorted(
        candidates, key=lambda t: (t.split("_")[0], int(t.split("_")[1]))
    ):
        label = candidates[tid]
        if not label.startswith(("keep", "fix")):
            continue
        domain = tid.split("_")[0]
        row = raw[tid]
        task = {
            "id": tid,
            "description": _parse(row.get("description")),
            "user_scenario": _parse(row["user_scenario"]),
            "evaluation_criteria": _parse(row["evaluation_criteria"]),
        }
        if label.startswith("fix"):
            task = apply_fix(task, verdicts[tid]["fix"])
        Task.model_validate(task)
        tasks[domain].append(task)
        task_db[domain][tid] = row["db_path"]
        splits[f"{domain}:{tid}"] = "fix" if label.startswith("fix") else "keep"
    files = {}
    for domain in DOMAINS:
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
    for rel in sorted({p for d in task_db.values() for p in d.values()}):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, out / rel)
        files[rel] = hashlib.sha256((out / rel).read_bytes()).hexdigest()
    manifest = {
        "name": out.name,
        "source": "inclusionAI/AReaL-tau2-data tau2_rl_train.jsonl, screened",
        "counts": {d: len(tasks[d]) for d in DOMAINS},
        "task_splits": splits,
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str]) -> None:
    source, screen, out = (Path(a) for a in argv)
    manifest = build(source, screen, out)
    print(json.dumps({"name": manifest["name"], "counts": manifest["counts"]}))


if __name__ == "__main__":
    main(sys.argv[1:])
