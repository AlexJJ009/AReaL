# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from scripts.tau2.check_telecom_full import check_task, fault_key, overlap_rows


def test_fault_key_collapses_symptom_and_persona_but_preserves_setup():
    a = {"id": "[mobile]fault[PERSONA:Easy]", "initial_state": {"x": 1, "y": 2}}
    b = {"id": "[mms]fault[PERSONA:Hard]", "initial_state": {"y": 2, "x": 1}}
    assert fault_key(a) == fault_key(b)
    b["initial_state"]["x"] = 3
    assert fault_key(a) != fault_key(b)


def test_overlap_metadata_counts_same_state_and_keeps_splits_separate():
    a = {"id": "[mobile]fault[PERSONA:Easy]", "initial_state": {"x": 1}}
    b = {"id": "[mms]fault[PERSONA:Hard]", "initial_state": {"x": 1}}
    c = {"id": "unseen", "initial_state": {"x": 2}}
    d = {"id": "also-unseen", "initial_state": {"x": 2}}
    pack = [{"id": "telecom_1", "initial_state": {"x": 1}}]
    splits = {"test": [a["id"]], "train": [], "base": [a["id"]], "small": []}
    rows = overlap_rows([a, b, c, d], splits, pack)
    assert [r["shares_fault_combination_with_test40"] for r in rows] == [
        True,
        True,
        False,
        False,
    ]
    assert [r["overlaps_areal_pack_v1"] for r in rows] == [True, True, False, False]
    assert rows[1]["official_splits"] == dict(
        small=False, train=False, test=False, base=False
    )
    assert rows[1]["areal_pack_v1_same_initial_state_ids"] == ["telecom_1"]
    assert rows[1]["id_fault_combination_key"] == "[mms]fault"


def test_mechanical_check_detects_broken_gold_and_already_solved_initial_state():
    from tau2.registry import registry

    from examples.tau2 import task_pack

    task_pack.install("telecom")
    tid = "[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]"
    task = next(t for t in registry.get_tasks_loader("telecom")("base") if t.id == tid)
    raw = task.model_dump(mode="json")
    good = check_task(raw)
    assert good["gold"]["score"] == 1.0
    assert good["empty"]["score"] == 0.0
    assert good["reasons"] == []

    broken_gold = deepcopy(raw)
    broken_gold["evaluation_criteria"]["actions"] = []
    result = check_task(broken_gold)
    assert result["gold"]["passed"] is False
    assert any("gold_score=0" in reason for reason in result["reasons"])

    solved = deepcopy(raw)
    solved["initial_state"]["initialization_actions"].extend(
        [
            {
                "env_type": a["requestor"],
                "func_name": a["name"],
                "arguments": a["arguments"],
            }
            for a in raw["evaluation_criteria"]["actions"]
        ]
    )
    result = check_task(solved)
    assert result["empty"]["passed"] is False
    assert any("empty_score=1" in reason for reason in result["reasons"])


def test_mechanical_check_records_replay_exception_and_still_checks_empty(monkeypatch):
    from tau2.registry import registry

    from scripts.tau2 import check_task_pack

    raw = registry.get_tasks_loader("telecom")("base")[0].model_dump(mode="json")
    original = check_task_pack.replay

    def fail_gold(domain, task, actions, db_path):
        if actions:
            raise ValueError("injected broken initialization")
        return original(domain, task, actions, db_path)

    monkeypatch.setattr(check_task_pack, "replay", fail_gold)
    result = check_task(raw)
    assert result["gold"]["passed"] is False
    assert result["gold"]["error"] == "ValueError: injected broken initialization"
    assert result["empty"]["score"] == 0.0


def test_mechanical_check_records_failed_tool_name_and_response():
    from tau2.registry import registry

    from examples.tau2 import task_pack

    task_pack.install("telecom")
    raw = registry.get_tasks_loader("telecom")("base")[0].model_dump(mode="json")
    raw["evaluation_criteria"]["actions"][0]["name"] = "missing_gold_tool"
    result = check_task(raw)
    assert result["gold"]["tool_errors"][0]["name"] == "missing_gold_tool"
    assert "missing_gold_tool" in result["gold"]["tool_errors"][0]["content"]
    assert "gold_tool_errors=1" in result["reasons"]
