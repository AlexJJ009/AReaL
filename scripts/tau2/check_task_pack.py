# SPDX-License-Identifier: Apache-2.0
"""Score gold and empty trajectories on every task of a task pack.

Each trajectory is produced on the task's own database through the registry
constructor that ``task_pack.install`` wraps, then scored by the official
``evaluate_simulation`` and the fixed-variant transfer rules, as an evaluation
episode is. Every task must score gold 1 and empty 0.

The gold trajectory is built as tau2's Telecom ``verify_task`` replays gold:
the environment first takes the task's ``initial_state``, and each gold action
runs on the side of its ``requestor`` (a user action is a user tool call on the
user's device), followed by ``sync_tools``. A gold call that returns an error is
counted, since the official gold replay ignores such errors.

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


def replay(domain, task, actions, db_path):
    """The trajectory of ``actions`` and how many of them returned an error."""

    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)(task_db=db_path)
    state = task.initial_state
    history = list((state.message_history if state else None) or [])
    if state is not None:
        env.set_state(
            initialization_data=state.initialization_data,
            initialization_actions=state.initialization_actions,
            message_history=history,
        )
    messages, errors = history, 0
    for i, action in enumerate(actions):
        call = ToolCall(
            id=str(i),
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        # get_response runs the call on the requestor's side, then sync_tools.
        out = env.get_response(call)
        errors += out.error
        message = AssistantMessage if action.requestor == "assistant" else UserMessage
        messages += [
            message(role=action.requestor, content=None, tool_calls=[call]),
            out,
        ]
    return messages, errors


def evaluate(domain, task, messages, db_path, default_db=False):
    """The official reward info of ``messages``, with the transfer rule."""

    from tau2.data_model.simulation import SimulationRun
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    run = SimulationRun(
        id="x",
        task_id=task.id,
        start_time="",
        end_time="",
        duration=0.0,
        termination_reason="user_stop",
        messages=messages,
    )
    info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=run,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        env_kwargs={} if default_db else {"task_db": db_path},
    )
    return benchmark_variant.apply_transfer_rule(task, messages, info, domain)


def _score(domain, task, messages, db_path):
    try:
        return evaluate(domain, task, messages, db_path, _default_db).reward
    except Exception as e:  # noqa: BLE001 - a replay error is a result here
        return f"error:{type(e).__name__}"


def _init(pack: str, default_db: bool) -> None:
    from loguru import logger

    global _pack, _default_db
    _pack, _default_db = pack, default_db
    logger.remove()
    benchmark_variant.activate("fixed", task_pack.domains(pack))
    for domain in task_pack.domains(pack):
        task_pack.install(domain)


def _check(item: tuple[str, str]) -> tuple[str, str, object, object, int]:
    domain, task_id = item
    task = task_pack.get_task(_pack, domain, task_id)
    db_path = task_pack.task_db(_pack, domain, task_id)
    gold_actions = task.evaluation_criteria.actions or []
    gold_messages, errors = replay(domain, task, gold_actions, db_path)
    gold = _score(domain, task, gold_messages, db_path)
    empty = _score(domain, task, replay(domain, task, [], db_path)[0], db_path)
    return domain, task_id, gold, empty, errors


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
    counts = Counter((d, str(g), str(e)) for d, _, g, e, _ in results)
    bad = [
        f"{d}:{t} gold={g} empty={e}"
        for d, t, g, e, _ in results
        if (g, e) != (1.0, 0.0)
    ]
    errors = [f"{d}:{t}" for d, t, _, _, n in results if n]
    print(
        json.dumps(
            {f"{d} gold={g} empty={e}": n for (d, g, e), n in sorted(counts.items())},
            indent=1,
        )
    )
    print(f"tasks failing gold=1/empty=0: {len(bad)} of {len(results)}")
    print(f"tasks with a gold call that returned an error: {len(errors)}")
    for line in errors[:20]:
        print(" ", line)
    for line in bad[:20]:
        print(" ", line)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
