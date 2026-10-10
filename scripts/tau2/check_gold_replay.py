# SPDX-License-Identifier: Apache-2.0
"""Replay every gold action of a τ² variant and report the ones that raise.

A state-mutating gold action that raises leaves the gold database incomplete
(upstream PR #580 makes the evaluator refuse such a match). TAU2_DATA_DIR must
point at the variant's data directory.

Usage: python scripts/tau2/check_gold_replay.py {official|fixed}
"""

import json
import sys

from examples.tau2 import benchmark_variant


def main(variant: str) -> int:
    domains = benchmark_variant.FIXED_DOMAINS
    benchmark_variant.activate(variant, domains)
    from tau2.registry import registry

    failures = []
    for domain in domains:
        for task in registry.get_tasks_loader(domain)("base"):
            env = registry.get_env_constructor(domain)()
            for action in task.evaluation_criteria.actions or []:
                try:
                    env.make_tool_call(
                        tool_name=action.name,
                        requestor=action.requestor,
                        **action.arguments,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "domain": domain,
                            "task_id": task.id,
                            "action": action.name,
                            "mutating": env.tools.tool_mutates_state(action.name)
                            if action.requestor == "assistant"
                            else None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    print(json.dumps({"variant": variant, "failures": failures}, indent=1))
    return int(any(f["mutating"] for f in failures))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
