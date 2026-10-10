# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from examples.tau2 import benchmark_variant


def _reward(reward=1.0):
    from tau2.data_model.simulation import RewardInfo

    return RewardInfo(reward=reward)


def _task(*gold):
    actions = [SimpleNamespace(name=name) for name in gold]
    return SimpleNamespace(evaluation_criteria=SimpleNamespace(actions=actions))


def _assistant(*tools):
    calls = [SimpleNamespace(name=name) for name in tools]
    return SimpleNamespace(role="assistant", tool_calls=calls)


def test_transfer_rule_zeroes_only_transfers_absent_from_gold():
    transfer = _assistant("get_user_details", benchmark_variant.TRANSFER_TOOL)

    scored = benchmark_variant.apply_transfer_rule(_task(), [transfer], _reward())
    assert scored.reward == 0.0
    assert scored.info == {"unexpected_transfer": True, "official_reward": 1.0}

    expected = _task(benchmark_variant.TRANSFER_TOOL)
    assert (
        benchmark_variant.apply_transfer_rule(expected, [transfer], _reward()).reward
        == 1.0
    )
    # A transfer requested by the user simulator is not the agent's hand-off.
    user = SimpleNamespace(role="user", tool_calls=transfer.tool_calls)
    assert (
        benchmark_variant.apply_transfer_rule(_task(), [user], _reward()).reward == 1.0
    )


def test_task_fixes_change_only_listed_tasks_and_reject_unknown_ids():
    tasks = [
        {"id": "0", "evaluation_criteria": {"reward_basis": ["DB", "COMMUNICATE"]}},
        {"id": "1", "evaluation_criteria": {"reward_basis": ["DB"]}},
        {"id": "2", "evaluation_criteria": {"reward_basis": ["DB"]}},
    ]
    replacement = {"id": "2", "evaluation_criteria": {"reward_basis": ["DB"]}, "new": 1}
    fixes = {
        "domains": {
            "airline": {
                "reward_basis": {"0": ["DB", "NL_ASSERTION"]},
                "replace_tasks": {"2": replacement},
            }
        }
    }

    fixed = benchmark_variant.apply_task_fixes(tasks, "airline", fixes)

    assert fixed[0]["evaluation_criteria"]["reward_basis"] == ["DB", "NL_ASSERTION"]
    assert tasks[0]["evaluation_criteria"]["reward_basis"] == ["DB", "COMMUNICATE"]
    assert fixed[1] is tasks[1] and fixed[2] is replacement
    fixes["domains"]["airline"]["reward_basis"]["9"] = ["DB"]
    with pytest.raises(ValueError, match="unknown tasks"):
        benchmark_variant.apply_task_fixes(tasks, "airline", fixes)


def test_committed_fixes_match_the_pinned_fixed_digests(tmp_path):
    from tau2.utils.utils import DATA_DIR

    digests = benchmark_variant.build_fixed_data(DATA_DIR, tmp_path / "data")

    assert digests == benchmark_variant.TASK_FILE_SHA256["fixed"]
    db = tmp_path / "data/tau2/domains/retail/db.json"
    assert db.is_symlink()
    fixes = benchmark_variant.load_fixes()
    assert {d: len(f["reward_basis"]) for d, f in fixes["domains"].items()} == {
        "airline": 44,
        "retail": 105,
    }
    assert sorted(fixes["domains"]["retail"]["replace_tasks"]) == ["105", "64"]


def test_official_data_is_rejected_by_the_fixed_variant():
    benchmark_variant.validate_task_files("official", ["airline", "retail"])
    with pytest.raises(ValueError, match="does not hold the fixed airline tasks"):
        benchmark_variant.validate_task_files("fixed", ["airline"])


def test_fixed_modify_prices_each_item_from_its_own_variant(monkeypatch):
    from tau2.domains.retail.tools import RetailTools
    from tau2.registry import registry

    monkeypatch.setattr(
        RetailTools,
        "modify_pending_order_items",
        RetailTools.modify_pending_order_items,
    )
    task = next(t for t in registry.get_tasks_loader("retail")("base") if t.id == "100")
    action = next(
        a
        for a in task.evaluation_criteria.actions
        if a.name == "modify_pending_order_items"
    )
    args = action.arguments
    assert len(set(args["new_item_ids"])) > 1

    def prices(swap):
        env = registry.get_env_constructor("retail")()
        call = dict(args)
        if swap:
            call["item_ids"] = args["item_ids"][::-1]
            call["new_item_ids"] = args["new_item_ids"][::-1]
        order = json.loads(
            env.get_response(
                SimpleNamespace(
                    id="x", name=action.name, arguments=call, requestor="assistant"
                )
            ).content
        )
        return {i["item_id"]: i["price"] for i in order["items"]}, env.get_db_hash()

    (official, official_hash), (swapped, swapped_hash) = prices(False), prices(True)
    assert official != swapped and official_hash != swapped_hash

    benchmark_variant._patch_modify_pending_order_items()
    (fixed, fixed_hash), (fixed_swapped, fixed_swapped_hash) = (
        prices(False),
        prices(True),
    )
    assert fixed == fixed_swapped and fixed_hash == fixed_swapped_hash
    tools = registry.get_env_constructor("retail")().tools
    for item_id, price in fixed.items():
        if item_id in args["new_item_ids"]:
            product = next(
                p for p in tools.db.products.values() if item_id in p.variants
            )
            assert price == product.variants[item_id].price
