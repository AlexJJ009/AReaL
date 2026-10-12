# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from examples.tau2 import task_pack


def _official_retail_db():
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    return json.loads(open(RETAIL_DB_PATH, encoding="utf-8").read())


def _write_pack(root, db, task_id="t1"):
    """A one-task Retail pack whose task changes a user's address on ``db``."""

    from tau2.registry import registry

    official = registry.get_tasks_loader("retail")("base")[0].model_dump(mode="json")
    user_id = next(iter(db["users"]))
    task = {
        **official,
        "id": task_id,
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "a0",
                    "requestor": "assistant",
                    "name": "modify_user_address",
                    "arguments": {
                        "user_id": user_id,
                        "address1": "1 Pack Street",
                        "address2": "",
                        "city": "Austin",
                        "state": "TX",
                        "country": "USA",
                        "zip": "78701",
                    },
                }
            ],
            "reward_basis": ["DB"],
        },
    }
    (root / "retail").mkdir(parents=True)
    (root / "db.json").write_text(json.dumps(db))
    (root / "retail" / "tasks.json").write_text(json.dumps([task]))
    (root / "retail" / "task_db.json").write_text(json.dumps({task_id: "db.json"}))
    files = {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for rel in ("db.json", "retail/tasks.json", "retail/task_db.json")
    }
    (root / "manifest.json").write_text(json.dumps({"name": "p", "files": files}))
    return user_id


@pytest.fixture(autouse=True)
def _clear_caches():
    for fn in (task_pack.load_manifest, task_pack._domain, task_pack._raw_db):
        fn.cache_clear()
    yield


def test_environment_and_evaluator_both_use_the_task_database(tmp_path):
    from tau2.data_model.simulation import SimulationRun
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.registry import registry

    db = _official_retail_db()
    # A user that exists only in the pack database.
    user_id = "pack_only_user_1"
    db["users"] = {user_id: next(iter(db["users"].values())) | {"user_id": user_id}}
    _write_pack(tmp_path, db)
    pack = str(tmp_path)
    task_pack.install("retail")
    task = task_pack.get_task(pack, "retail", "t1")
    path = task_pack.task_db(pack, "retail", "t1")

    env = registry.get_env_constructor("retail")(task_db=path)
    assert user_id in env.tools.db.users
    # Each call starts from an unmodified copy.
    env.tools.db.users.pop(user_id)
    assert (
        user_id in registry.get_env_constructor("retail")(task_db=path).tools.db.users
    )

    def score(env_kwargs):
        run = SimulationRun(
            id="x",
            task_id="t1",
            start_time="",
            end_time="",
            duration=0.0,
            termination_reason="user_stop",
            messages=[],
        )
        return evaluate_simulation(
            domain="retail",
            task=task,
            simulation=run,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            env_kwargs=env_kwargs,
        ).reward

    # Against the task's own gold replay the empty trajectory scores 0. On the
    # default database the gold write fails for the unknown user, both states
    # stay unchanged, and the empty trajectory would wrongly score 1.
    assert score({"task_db": path}) == 0.0
    assert score({}) == 1.0


def test_modified_pack_files_are_rejected(tmp_path):
    _write_pack(tmp_path, _official_retail_db())
    (tmp_path / "db.json").write_text("{}")
    with pytest.raises(ValueError, match="db.json sha256"):
        task_pack.load_manifest(str(tmp_path))


def test_unknown_task_ids_are_rejected(tmp_path):
    _write_pack(tmp_path, _official_retail_db())
    with pytest.raises(ValueError, match="No task missing"):
        task_pack.get_task(str(tmp_path), "retail", "missing")


def test_pack_rows_carry_the_pack_and_the_screening_split(tmp_path):
    from types import SimpleNamespace

    from scripts.tau2.evaluate import build_pack_rows

    _write_pack(tmp_path, _official_retail_db())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["task_splits"] = {"retail:t1": "fix"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = build_pack_rows(SimpleNamespace(domains=["retail"]), 4, str(tmp_path))
    assert [r["source_id"] for r in rows] == [f"retail:t1:trial{i}" for i in range(4)]
    assert {r["split"] for r in rows} == {"fix"}
    assert {r["task_pack"] for r in rows} == {str(tmp_path.absolute())}


def _write_telecom_pack(root):
    """A one-task Telecom pack whose customer exists only in the pack's TOML DB.

    The task is an official one with a user-side and an assistant-side gold
    action: the user turns mobile data on, the agent refuels data.
    """

    from tau2.domains.telecom.utils import TELECOM_DB_PATH
    from tau2.registry import registry

    name = "[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]"
    official = next(
        t for t in registry.get_tasks_loader("telecom")("base") if t.id == name
    )
    task = json.loads(
        json.dumps(official.model_dump(mode="json")).replace("C1001", "C_PACK")
    )
    task["id"] = "t1"
    db = open(TELECOM_DB_PATH, encoding="utf-8").read().replace("C1001", "C_PACK")
    (root / "telecom").mkdir(parents=True)
    (root / "db.toml").write_text(db)
    (root / "telecom" / "tasks.json").write_text(json.dumps([task]))
    (root / "telecom" / "task_db.json").write_text(json.dumps({"t1": "db.toml"}))
    files = {
        rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
        for rel in ("db.toml", "telecom/tasks.json", "telecom/task_db.json")
    }
    (root / "manifest.json").write_text(json.dumps({"name": "p", "files": files}))


def test_telecom_gold_replays_user_actions_on_the_task_database(tmp_path):
    from tau2.registry import registry

    from scripts.tau2 import check_task_pack

    _write_telecom_pack(tmp_path)
    pack = str(tmp_path)
    task_pack.install("telecom")
    task = task_pack.get_task(pack, "telecom", "t1")
    path = task_pack.task_db(pack, "telecom", "t1")
    gold = task.evaluation_criteria.actions
    assert {a.requestor for a in gold} == {"user", "assistant"}

    env = registry.get_env_constructor("telecom")(task_db=path)
    assert [c.customer_id for c in env.tools.db.customers][0] == "C_PACK"

    def reward(actions):
        messages, errors = check_task_pack.replay("telecom", task, actions, path)
        assert errors == 0
        return check_task_pack.evaluate("telecom", task, messages, path).reward

    assert reward(gold) == 1.0
    assert reward([]) == 0.0
    # The user's own fix is part of the gold: without it the task stays broken.
    assert reward([a for a in gold if a.requestor == "assistant"]) == 0.0
    # The initial state names the pack-only customer, so the default database
    # cannot even set the task up.
    state = task.initial_state
    with pytest.raises(ValueError, match="C_PACK not found"):
        registry.get_env_constructor("telecom")().set_state(
            state.initialization_data, state.initialization_actions, []
        )
