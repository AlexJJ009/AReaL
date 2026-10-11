# SPDX-License-Identifier: Apache-2.0
"""τ² tasks that bring their own database: a task pack.

The official domains load one database per domain. Task sets built outside
tau2-bench, such as the screened AReaL-tau2-data subset, start each task from
its own database. A task pack is a directory:

- ``manifest.json``: ``{"name", "source", "files": {relative path: sha256}}``;
- ``<domain>/tasks.json``: τ² ``Task`` objects;
- ``<domain>/task_db.json``: task id -> database path relative to the pack;
- the database files themselves.

Every file is checked against the manifest when the pack is first loaded. The
agent and the evaluator both build environments through the registry, so
:func:`install` wraps each domain's constructor to accept ``task_db``: every
call gets a fresh copy of that database, and the evaluator's gold replay and
the episode never share state.
"""

from __future__ import annotations

import functools
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

MANIFEST = "manifest.json"
_lock = threading.Lock()
_installed: set[str] = set()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.cache
def load_manifest(pack: str) -> dict[str, Any]:
    """Read the manifest and verify every file it lists."""

    root = Path(pack)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        actual = _sha256(root / rel)
        if actual != expected:
            raise ValueError(f"Task pack {pack}: {rel} sha256 {actual} != {expected}")
    return manifest


@functools.cache
def _domain(pack: str, domain: str) -> tuple[dict[str, Any], dict[str, str]]:
    from tau2.data_model.tasks import Task

    load_manifest(pack)
    root = Path(pack) / domain
    tasks = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
    task_db = json.loads((root / "task_db.json").read_text(encoding="utf-8"))
    by_id = {str(t["id"]): Task.model_validate(t) for t in tasks}
    if set(by_id) != set(task_db):
        raise ValueError(f"Task pack {pack}: {domain} tasks and task_db disagree")
    return by_id, {tid: str(Path(pack) / rel) for tid, rel in task_db.items()}


def domains(pack: str) -> list[str]:
    return sorted(
        p.parent.name for p in Path(pack).glob("*/tasks.json") if p.parent.is_dir()
    )


def task_ids(pack: str, domain: str) -> list[str]:
    return list(_domain(pack, domain)[0])


def get_task(pack: str, domain: str, task_id: str) -> Any:
    tasks, _ = _domain(pack, domain)
    if task_id not in tasks:
        raise ValueError(f"No task {task_id} for {domain} in task pack {pack}")
    return tasks[task_id]


def task_db(pack: str, domain: str, task_id: str) -> str:
    return _domain(pack, domain)[1][task_id]


@functools.cache
def _raw_db(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fresh_db(domain: str, path: str) -> Any:
    from tau2.domains.airline.data_model import FlightDB
    from tau2.domains.retail.data_model import RetailDB

    # Validation builds new objects, so each environment gets its own state;
    # it is several times faster than deep-copying a parsed database.
    model = {"airline": FlightDB, "retail": RetailDB}[domain]
    return model.model_validate(_raw_db(path))


def install(domain: str) -> None:
    """Let the domain's registered constructor take a ``task_db`` path."""

    from tau2.registry import registry

    with _lock:
        if domain in _installed:
            return
        original = registry.get_env_constructor(domain)

        @functools.wraps(original)
        def get_environment(*args, task_db: str | None = None, **kwargs):
            if task_db is not None:
                kwargs["db"] = _fresh_db(domain, task_db)
            return original(*args, **kwargs)

        registry._domains[domain] = get_environment
        _installed.add(domain)
