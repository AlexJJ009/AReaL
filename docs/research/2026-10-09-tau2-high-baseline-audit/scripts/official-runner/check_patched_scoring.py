# SPDX-License-Identifier: Apache-2.0
"""Score synthetic trajectories on the patched fixed Retail tasks.

For each patched task: an empty trajectory, the gold trajectory (gold tool calls
plus one assistant message that states every communicate_info string), and the
gold trajectory without its transfer. NL assertions are stubbed to pass, so
this checks the DB, ACTION and COMMUNICATE parts and the transfer rules only.

Usage: TAU2_DATA_DIR=<fixed data dir> python check_patched_scoring.py
"""

import json
import sys

from examples.tau2 import benchmark_variant


def main() -> int:
    benchmark_variant.activate("fixed", ["retail"])
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
    from tau2.data_model.simulation import RewardInfo, SimulationRun
    from tau2.evaluator import evaluator
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.registry import registry

    evaluator.NLAssertionsEvaluator.calculate_reward = staticmethod(
        lambda **_: RewardInfo(reward=1.0)
    )
    patched = benchmark_variant.load_fixes()["domains"]["retail"]["patches"]
    tasks = {t.id: t for t in registry.get_tasks_loader("retail")("base")}

    def score(task, actions, say):
        env = registry.get_env_constructor("retail")()
        messages = []
        for i, action in enumerate(actions):
            call = ToolCall(id=str(i), name=action.name, arguments=action.arguments)
            out = env.get_response(call)
            messages += [
                AssistantMessage(role="assistant", content=None, tool_calls=[call]),
                ToolMessage(id=str(i), role="tool", content=out.content),
            ]
        if say:
            messages.append(AssistantMessage(role="assistant", content=" ".join(say)))
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
            domain="retail",
            task=task,
            simulation=run,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )
        return benchmark_variant.apply_transfer_rule(
            task, messages, info, "retail"
        ).reward

    rows = {}
    for task_id in patched:
        task = tasks[task_id]
        gold = task.evaluation_criteria.actions or []
        say = task.evaluation_criteria.communicate_info or []
        no_transfer = [a for a in gold if a.name != benchmark_variant.TRANSFER_TOOL]
        rows[task_id] = {
            "empty": score(task, [], []),
            "gold": score(task, gold, say),
            "gold_without_transfer": score(task, no_transfer, say),
        }
    print(json.dumps(rows, indent=1))
    bad = [t for t, r in rows.items() if r["gold"] != 1.0 or r["empty"] != 0.0]
    print("gold<1 or empty>0:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
