# SPDX-License-Identifier: Apache-2.0
"""Score gold and empty trajectories on every task of a task pack.

Each trajectory is produced on the task's own database through the registry
constructor that ``task_pack.install`` wraps, then scored by the official
``evaluate_simulation`` and the fixed-variant transfer rules, as an evaluation
episode is. Every task must score gold 1 and empty 0.

``--default-db`` scores against the domain's default database instead. Most
tasks must then fail, by a wrong score or a replay error; a run where they do
not means the evaluator is not reading the task's database.

Usage: TAU2_DATA_DIR=<fixed data dir> python scripts/tau2/check_task_pack.py <pack> [--default-db]
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

from examples.tau2 import benchmark_variant, task_pack

_pack = ""
_default_db = False


def _score(domain, task, actions, db_path):
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
    from tau2.data_model.simulation import SimulationRun
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)(task_db=db_path)
    messages = []
    for i, action in enumerate(actions):
        call = ToolCall(id=str(i), name=action.name, arguments=action.arguments)
        out = env.get_response(call)
        messages += [
            AssistantMessage(role="assistant", content=None, tool_calls=[call]),
            ToolMessage(id=str(i), role="tool", content=out.content),
        ]
    run = SimulationRun(
        id="x",
        task_id=task.id,
        start_time="",
        end_time="",
        duration=0.0,
        termination_reason="user_stop",
        messages=messages,
    )
    try:
        info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=run,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            env_kwargs={} if _default_db else {"task_db": db_path},
        )
    except Exception as e:  # noqa: BLE001 - a replay error is a result here
        return f"error:{type(e).__name__}"
    return benchmark_variant.apply_transfer_rule(task, messages, info, domain).reward


def _init(pack: str, default_db: bool) -> None:
    from loguru import logger

    global _pack, _default_db
    _pack, _default_db = pack, default_db
    logger.remove()
    benchmark_variant.activate("fixed", task_pack.domains(pack))
    for domain in task_pack.domains(pack):
        task_pack.install(domain)


def _check(item: tuple[str, str]) -> tuple[str, str, object, object]:
    domain, task_id = item
    task = task_pack.get_task(_pack, domain, task_id)
    db_path = task_pack.task_db(_pack, domain, task_id)
    gold = _score(domain, task, task.evaluation_criteria.actions or [], db_path)
    empty = _score(domain, task, [], db_path)
    return domain, task_id, gold, empty


def main(argv: list[str]) -> int:
    pack, default_db = argv[0], "--default-db" in argv[1:]
    items = [
        (domain, task_id)
        for domain in task_pack.domains(pack)
        for task_id in task_pack.task_ids(pack, domain)
    ]
    with Pool(
        int(os.environ.get("NPROC", "32")),
        initializer=_init,
        initargs=(pack, default_db),
    ) as pool:
        results = pool.map(_check, items, chunksize=4)
    counts = Counter((d, str(g), str(e)) for d, _, g, e in results)
    bad = [
        f"{d}:{t} gold={g} empty={e}" for d, t, g, e in results if (g, e) != (1.0, 0.0)
    ]
    print(
        json.dumps(
            {f"{d} gold={g} empty={e}": n for (d, g, e), n in sorted(counts.items())},
            indent=1,
        )
    )
    print(f"tasks failing gold=1/empty=0: {len(bad)} of {len(results)}")
    for line in bad[:20]:
        print(" ", line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
