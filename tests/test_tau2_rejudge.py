# SPDX-License-Identifier: Apache-2.0
"""Saved NL scoring is replayed without policy, environment, or API calls."""

import copy
import csv
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from scripts.tau2 import rejudge


def record(nl=0.0, db=1.0):
    """A failed NL check can mask a passing deterministic component."""
    return {
        "source_id": "retail:1:trial0",
        "domain": "retail",
        "task_id": "1",
        "status": "completed",
        "official_score": nl * db,
        "task_budget_failure": False,
        "nl_judge": {"model": "old-model"},
        "messages": [{"role": "assistant", "content": "refund $10"}],
        "reward_info": {
            "reward": nl * db,
            "reward_basis": ["DB", "NL_ASSERTION"],
            "reward_breakdown": {"DB": db, "NL_ASSERTION": nl},
            "nl_assertions": [
                {"nl_assertion": "Mention $10", "met": bool(nl), "justification": "old"}
            ],
            "db_check": {"db_reward": db},
            "action_checks": [{"action_match": False}],
            "info": {},
        },
    }


def pass_judge(item, assertions):
    """Return one fake result per assertion; never import the API client."""
    assert item["messages"][0]["content"] == "refund $10"
    return [
        {"nl_assertion": text, "met": True, "justification": "fake"}
        for text in assertions
    ]


@pytest.mark.parametrize("db,expected", [(1.0, 1.0), (0.0, 0.0)])
def test_nl_rejudge_preserves_other_components_and_recomputes_product(db, expected):
    """A changed NL factor changes the score only when other factors pass."""
    item = record(db=db)
    before = copy.deepcopy(item)
    updated = rejudge.rejudge_record(item, pass_judge)
    assert item == before
    assert updated["official_score"] == expected
    assert updated["reward_info"]["reward"] == expected
    assert updated["reward_info"]["reward_breakdown"]["NL_ASSERTION"] == 1
    for field in ("db_check", "action_checks"):
        assert updated["reward_info"][field] == before["reward_info"][field]


@pytest.mark.parametrize("flag", ["missing_transfer", "unexpected_transfer", "budget"])
def test_nl_rejudge_preserves_non_nl_zero_penalties(flag):
    """Rejudging cannot lift fixed benchmark or token budget penalties."""
    item = record()
    if flag == "budget":
        item["task_budget_failure"] = True
    else:
        item["reward_info"]["info"] = {flag: True, "official_reward": 0.0}
    updated = rejudge.rejudge_record(item, pass_judge)
    assert updated["official_score"] == 0
    if flag != "budget":
        assert updated["reward_info"]["info"]["official_reward"] == 1


@pytest.mark.parametrize("kind", ["infra", "no_nl", "empty"])
def test_records_without_required_assertions_make_no_judge_call(kind):
    """Only a completed component with nonempty assertions spends a call."""
    item = record()
    if kind == "infra":
        item.update(status="infra", reward_info=None)
    elif kind == "no_nl":
        item["reward_info"]["reward_basis"] = ["DB"]
        item["reward_info"]["reward_breakdown"] = {"DB": 1.0}
    else:
        item["reward_info"]["nl_assertions"] = []
    updated = rejudge.rejudge_record(item, lambda *_: pytest.fail("judge called"))
    if kind == "empty":
        assert updated["official_score"] == 1
    else:
        assert updated == item


@pytest.mark.parametrize(
    "checks",
    [
        [],
        [{"nl_assertion": "wrong", "met": True}],
        [{"nl_assertion": "Mention $10", "met": "true"}],
    ],
)
def test_missing_mismatched_or_nonboolean_checks_rejected(checks):
    """Malformed model output cannot turn a failed assertion into a pass."""
    with pytest.raises(ValueError, match="every saved assertion"):
        rejudge.rejudge_record(record(), lambda *_: checks)


def save(attempt, item):
    """Create the same two-level trajectory layout as offline eval."""
    target = attempt / "trajectories" / item["source_id"] / "uuid.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(item))
    return target


def test_attempt_writes_siblings_agreement_and_refuses_overwrite(tmp_path):
    """Original byte content survives, including failed non-NL records."""
    original = save(tmp_path, record())
    before = original.read_bytes()
    output = rejudge.rejudge_attempt(tmp_path, "fake", "deepseek-official", pass_judge)
    assert original.read_bytes() == before
    updated = json.loads((output / original.relative_to(tmp_path)).read_text())
    assert updated["official_score"] == 1
    assert updated["rejudge"]["nl_recomputed"] is True
    rows = list(csv.DictReader((output / "agreement.csv").open()))
    assert len(rows) == 1
    assert rows[0]["old_nl"] == "0.0" and rows[0]["new_nl"] == "1.0"
    assert rows[0]["nl_agrees"] == "False" and rows[0]["score_agrees"] == "False"
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    assert rejudge.call_plan(tmp_path)["judge_calls"] == 1  # excludes its own output
    with pytest.raises(FileExistsError):
        rejudge.rejudge_attempt(tmp_path, "fake", "deepseek-official", pass_judge)


def test_provider_error_stops_attempt_and_records_failure(tmp_path):
    """No second call or retry follows any provider failure, including quota."""
    save(tmp_path, record())
    second = record()
    second["source_id"] = "retail:2:trial0"
    save(tmp_path, second)
    calls = []

    def fail(*args):
        calls.append(args)
        raise RuntimeError("AccountQuotaExceeded secret")

    with pytest.raises(RuntimeError):
        rejudge.rejudge_attempt(tmp_path, "fail", "volcano-plan", fail)
    assert len(calls) == 1
    status = json.loads((tmp_path / "rejudge-fail/status.json").read_text())
    assert status["status"] == "failed" and status["error_type"] == "RuntimeError"
    assert "secret" not in json.dumps(status)


def test_plan_counts_one_call_for_multiple_assertions_without_client(tmp_path, capsys):
    """Planning runs without runtime/key files and counts episodes, not checks."""
    item = record()
    item["reward_info"]["nl_assertions"] *= 2
    save(tmp_path, item)
    tasks_path = tmp_path / "data/tau2/domains/retail/tasks.json"
    tasks_path.parent.mkdir(parents=True)
    tasks_path.write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "evaluation_criteria": {
                        "reward_basis": ["NL_ASSERTION"],
                        "nl_assertions": ["Mention $10", "Mention $10"],
                    },
                }
            ]
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "task_file_sha256": {
                    "retail": hashlib.sha256(tasks_path.read_bytes()).hexdigest()
                }
            }
        )
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"tau2_data_dir": str(tmp_path / "data")}))
    assert (
        rejudge.main(
            [
                str(tmp_path),
                "--judge-backend",
                "deepseek-official",
                "--runtime",
                str(runtime),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["judge_calls"] == 1
    assert plan["attempts"][0]["assertions"] == 2


@pytest.mark.parametrize("backend", list(rejudge.ROUTES))
def test_official_adapter_uses_official_evaluator_and_pins_returned_model(
    monkeypatch, backend
):
    """A fake tau2 module checks transport, T=0, retry bounds and global restore."""
    monkeypatch.setattr(rejudge.subprocess, "check_output", lambda *a, **kw: "fake-key")
    calls = []
    models = iter(["served-v1", "served-v1", "served-v2"])

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(raw_data={"model": next(models)})

    evaluator = SimpleNamespace(
        DEFAULT_LLM_NL_ASSERTIONS="original",
        DEFAULT_LLM_NL_ASSERTIONS_ARGS={},
        generate=generate,
    )

    def evaluate(messages, assertions):
        assert messages[0].content == "refund $10"
        evaluator.generate(
            model=evaluator.DEFAULT_LLM_NL_ASSERTIONS,
            **evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        )
        return [
            SimpleNamespace(
                model_dump=lambda: {
                    "nl_assertion": assertions[0],
                    "met": True,
                    "justification": "fake",
                }
            )
        ]

    evaluator.NLAssertionsEvaluator = SimpleNamespace(evaluate_nl_assertions=evaluate)
    monkeypatch.setitem(
        sys.modules,
        "tau2.evaluator",
        SimpleNamespace(evaluator_nl_assertions=evaluator),
    )
    model, url, _, key_field = rejudge.ROUTES[backend]
    judge = rejudge.OfficialJudge(backend, {key_field: "unused"})
    for _ in range(2):
        assert judge(record(), ["Mention $10"])[0]["returned_model"] == "served-v1"
    with pytest.raises(ValueError, match="model changed"):
        judge(record(), ["Mention $10"])
    assert calls[0] == {
        "model": f"openai/{model}",
        "api_base": url,
        "api_key": "fake-key",
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout": 120.0,
        "num_retries": 0,
    }
    assert evaluator.DEFAULT_LLM_NL_ASSERTIONS == "original"
    assert evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS == {}
    assert evaluator.generate is generate


def test_canonical_assertions_recover_empty_old_judge_results():
    """An old malformed empty results array cannot suppress the new judge call."""
    item = record(nl=1.0)
    item["reward_info"]["nl_assertions"] = []
    tasks = {
        ("retail", "1"): {
            "evaluation_criteria": {
                "reward_basis": ["NL_ASSERTION"],
                "nl_assertions": ["Mention $10"],
            }
        }
    }
    calls = []

    def fail_judge(item, assertions):
        calls.append(assertions)
        return [{"nl_assertion": assertions[0], "met": False, "justification": "fake"}]

    updated = rejudge.rejudge_record(item, fail_judge, tasks)
    assert calls == [["Mention $10"]]
    assert updated["official_score"] == 0.0


def test_canonical_task_hash_mismatch_rejected_before_rejudge(tmp_path):
    """A runtime pointing at another benchmark cannot silently alter assertions."""
    path = tmp_path / "data/tau2/domains/retail/tasks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"id": "1"}]))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    save(attempt, record())
    (attempt / "manifest.json").write_text(
        json.dumps({"task_file_sha256": {"retail": "wrong-sha256"}})
    )
    with pytest.raises(ValueError, match="task hash differs"):
        rejudge.load_canonical_tasks(tmp_path / "data", [attempt])


@pytest.mark.parametrize("manifest", [None, {}, {"task_file_sha256": {}}])
def test_scored_nl_domain_without_original_task_hash_rejected(tmp_path, manifest):
    """Canonical assertions require original provenance even if files exist."""
    path = tmp_path / "data/tau2/domains/retail/tasks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"id": "1"}]))
    attempt = tmp_path / "attempt"
    save(attempt, record())
    if manifest is not None:
        (attempt / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original.*(required|hashes)"):
        rejudge.load_canonical_tasks(tmp_path / "data", [attempt])


def test_real_official_prompt_and_parser_with_fake_generate(monkeypatch, tmp_path):
    """Exercise the real tau2 evaluator and Pydantic parser without transport."""
    if not os.getenv("TAU2_DATA_DIR"):
        pytest.skip("TAU2_DATA_DIR is required for the pinned official tau2 package")
    evaluator = pytest.importorskip("tau2.evaluator.evaluator_nl_assertions")
    monkeypatch.setenv("TAU2_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(rejudge.subprocess, "check_output", lambda *a, **kw: "fake-key")
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        messages = kwargs["messages"]
        assert messages[0].role == "system"
        assert "Grade each expected outcome individually" in messages[0].content
        assert messages[1].role == "user"
        assert "assistant: refund $10" in messages[1].content
        assert "Mention $10" in messages[1].content
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": "Mention $10",
                            "metExpectation": True,
                            "reasoning": "fake official parser result",
                        }
                    ]
                }
            ),
            raw_data={"model": "served-v1"},
        )

    monkeypatch.setattr(evaluator, "generate", fake_generate)
    judge = rejudge.OfficialJudge("deepseek-official", {"deepseek_env_file": "unused"})
    updated = rejudge.rejudge_record(record(), judge)
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/deepseek-flash"
    assert calls[0]["num_retries"] == 0
    assert judge.returned_model == "served-v1"
    assert updated["reward_info"]["nl_assertions"] == [
        {
            "nl_assertion": "Mention $10",
            "met": True,
            "justification": "fake official parser result",
            "returned_model": "served-v1",
        }
    ]
    assert updated["official_score"] == 1.0
