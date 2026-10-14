import collections
import json
import sys
from datetime import datetime

import tau2.evaluator.evaluator_nl_assertions as nl
import tau2.utils.llm_utils as lu
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry

NL_CALLED = []


def fake_eval(traj, assertions):
    NL_CALLED.append(assertions)
    from tau2.data_model.simulation import NLAssertionCheck

    return [
        NLAssertionCheck(nl_assertion=x, met=True, justification="mock")
        for x in assertions
    ]


nl.NLAssertionsEvaluator.evaluate_nl_assertions = classmethod(
    lambda cls, t, a: fake_eval(t, a)
)


def no_llm(*a, **k):
    raise RuntimeError("LLM call attempted")


lu.generate = no_llm
nl.generate = no_llm

out = {}
for domain, split in [
    ("airline", "test"),
    ("airline", "train"),
    ("retail", "test"),
    ("retail", "train"),
    ("telecom", "test"),
    ("telecom", "train"),
]:
    try:
        tasks = registry.get_tasks_loader(domain)(split)
    except Exception as e:
        print(domain, split, "ERR", e)
        continue
    basis = collections.Counter()
    comps = collections.Counter()
    nonempty = collections.Counter()
    passes, nl_flag, errs = [], [], []
    for t in tasks:
        ec = t.evaluation_criteria
        rb = tuple(sorted(r.value for r in ec.reward_basis)) if ec else ("<none>",)
        basis[rb] += 1
        for r in ec.reward_basis if ec else []:
            comps[r.value] += 1
        if ec:
            nonempty["nl_assertions"] += bool(ec.nl_assertions)
            nonempty["communicate_info"] += bool(ec.communicate_info)
            nonempty["actions"] += bool(ec.actions)
            nonempty["env_assertions"] += bool(ec.env_assertions)
        now = datetime.now().isoformat()
        msgs = [
            AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
            UserMessage(role="user", content="Hi, I need help."),
            AssistantMessage(role="assistant", content="Done."),
        ]
        sim = SimulationRun(
            id="x",
            task_id=t.id,
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason=TerminationReason.AGENT_STOP,
            messages=msgs,
        )
        try:
            ri = evaluate_simulation(
                simulation=sim,
                task=t,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=domain,
            )
            if ri.reward == 1.0:
                passes.append(t.id)
        except RuntimeError as e:
            if "NL judge" in str(e):
                # recompute with non-NL components only
                nl_flag.append(t.id)
            else:
                errs.append((t.id, str(e)[:100]))
        except Exception as e:
            errs.append((t.id, repr(e)[:150]))
    key = f"{domain}/{split}"
    out[key] = dict(
        n=len(tasks),
        reward_basis={"+".join(k): v for k, v in basis.items()},
        component_counts=dict(comps),
        nonempty=dict(nonempty),
        do_nothing_pass=passes,
        nl_needed=nl_flag,
        errors=errs,
    )
    print(key, json.dumps(out[key]))
json.dump(out, open(sys.argv[1], "w"), indent=1)
