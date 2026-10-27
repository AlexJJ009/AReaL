# SPDX-License-Identifier: Apache-2.0
"""Rejudge saved NL assertions on CPU; default is a read-only call-count plan.

Example (after approval of the API run card)::

    CUDA_VISIBLE_DEVICES= python scripts/tau2/rejudge.py ATTEMPT ... \
        --judge-backend deepseek-official --runtime RUNTIME.json \
        --label official-20261026 --execute

Each attempt gets an exclusive rejudge-LABEL directory. Original trajectories,
ledger, manifest and summary are never modified. Infra failures are copied without
scoring; all completed NL components are rejudged, even if another component fails.
The CLI reads canonical assertion text from runtime tau2_data_dir tasks.json. No policy, environment,
user simulator, DB check, action check or communicate check runs again.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROUTES = {
    "deepseek-official": (
        "deepseek-flash",
        "https://api.deepseek.com",
        "DEEPSEEK_API_KEY",
        "deepseek_env_file",
    ),
    "volcano-plan": (
        "deepseek-v4-1-flash-261017",
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        "ARK_PLAN_API_KEY",
        "ark_plan_env_file",
    ),
}
Tasks = dict[tuple[str, str], dict[str, Any]]
Judge = Callable[[dict[str, Any], list[str]], list[dict[str, Any]]]


def load_records(attempt: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Read only original saved records, never prior rejudge outputs."""
    files = sorted((attempt / "trajectories").glob("*/*.json"))
    if not files:
        raise ValueError(f"No saved trajectories in {attempt}")
    return [(path, json.loads(path.read_text())) for path in files]


def assertions_for(
    record: dict[str, Any], tasks: Tasks | None = None
) -> list[str] | None:
    """None means no scored NL component; [] means no API call is needed."""
    info = record.get("reward_info") or {}
    if record.get("status") != "completed" or "NL_ASSERTION" not in (
        info.get("reward_basis") or []
    ):
        return None
    if "NL_ASSERTION" not in (info.get("reward_breakdown") or {}):
        raise ValueError("NL reward basis has no saved breakdown")
    if tasks is not None:
        task = tasks[(record["domain"], str(record["task_id"]))]
        criteria = task.get("evaluation_criteria") or {}
        if "NL_ASSERTION" not in (criteria.get("reward_basis") or []):
            raise ValueError("Canonical task and saved NL reward basis disagree")
        return criteria.get("nl_assertions") or []
    checks = info.get("nl_assertions")
    if checks is None:
        raise ValueError("NL reward basis has no saved assertion checks")
    return [check["nl_assertion"] for check in checks]


def call_plan(attempt: Path, tasks: Tasks | None = None) -> dict[str, Any]:
    """One official generate call per episode, regardless of assertion count."""
    records = load_records(attempt)
    selected = [assertions_for(record, tasks) for _, record in records]
    return {
        "attempt": str(attempt.resolve()),
        "saved_records": len(records),
        "nl_episodes": sum(items is not None for items in selected),
        "judge_calls": sum(bool(items) for items in selected),
        "assertions": sum(len(items or []) for items in selected),
    }


def rejudge_record(
    record: dict[str, Any], judge: Judge, tasks: Tasks | None = None
) -> dict[str, Any]:
    """Replace NL checks and their product factor, keeping all other evidence."""
    result = copy.deepcopy(record)
    assertions = assertions_for(record, tasks)
    if assertions is None:
        return result
    checks = judge(record, assertions) if assertions else []
    if [check.get("nl_assertion") for check in checks] != assertions or any(
        type(check.get("met")) is not bool for check in checks
    ):
        raise ValueError(
            "Judge must return every saved assertion in order, with bool met"
        )
    info = result["reward_info"]
    info["nl_assertions"] = checks
    info["reward_breakdown"]["NL_ASSERTION"] = float(all(c["met"] for c in checks))
    reward = math.prod(info["reward_breakdown"].values())
    flags = info.get("info") or {}
    if (
        result.get("task_budget_failure")
        or flags.get("missing_transfer")
        or flags.get("unexpected_transfer")
    ):
        reward = 0.0
    # The fixed benchmark keeps the pre-transfer score as diagnostic evidence.
    if "official_reward" in flags:
        flags["official_reward"] = math.prod(info["reward_breakdown"].values())
    info["reward"] = result["official_score"] = reward
    return result


def rejudge_attempt(
    attempt: Path, label: str, backend: str, judge: Judge, tasks: Tasks | None = None
) -> Path:
    """Write isolated results and per-record agreement, refusing any overwrite."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError(
            "label must contain only letters, digits, dot, dash, underscore"
        )
    records = load_records(attempt)
    destination = attempt / f"rejudge-{label}"
    destination.mkdir(exist_ok=False)
    status: dict[str, Any] = {
        "status": "running",
        "judge_backend": backend,
        "requested_model": ROUTES[backend][0],
        "temperature": 0.0,
        "num_retries": 0,
        "input_plan": call_plan(attempt, tasks),
        "completed_records": 0,
        "scored_episodes": 0,
        "old_score_sum": 0.0,
        "new_score_sum": 0.0,
        "score_metric": "mean over completed saved records with an official score; infra excluded",
        "canonical_tasks_sha256": hashlib.sha256(
            json.dumps(
                sorted(
                    (domain, task_id, task) for (domain, task_id), task in tasks.items()
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if tasks is not None
        else None,
    }
    status_path = destination / "status.json"

    def write_status() -> None:
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    write_status()
    try:
        with (destination / "agreement.csv").open("x", newline="") as stream:
            fields = [
                "source_id",
                "trajectory",
                "source_sha256",
                "old_judge",
                "new_judge",
                "old_nl",
                "new_nl",
                "nl_agrees",
                "old_score",
                "new_score",
                "score_agrees",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for path, record in records:
                updated = rejudge_record(record, judge, tasks)
                relative = path.relative_to(attempt)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                selected = assertions_for(record, tasks) is not None
                updated["rejudge"] = {
                    "backend": backend,
                    "temperature": 0.0,
                    "source": str(path.resolve()),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "nl_recomputed": selected,
                }
                if selected:
                    updated["rejudge"]["old_nl_judge"] = record.get("nl_judge")
                    updated["nl_judge"] = {
                        "model": f"openai/{ROUTES[backend][0]}",
                        "backend": backend,
                        "source": "offline_rejudge",
                        "returned_model": getattr(judge, "returned_model", None),
                    }
                    updated["nl_judge_llm_calls"] = int(
                        bool(assertions_for(record, tasks))
                    )
                    updated["nl_judge_calls"] = []
                    if updated["nl_judge_llm_calls"]:
                        updated["nl_judge_calls"].append(
                            {
                                **updated["nl_judge"],
                                "temperature": 0.0,
                                "num_retries": 0,
                                "status": "completed",
                            }
                        )
                target.write_text(
                    json.dumps(updated, indent=2, ensure_ascii=False) + "\n"
                )
                old_nl = (record.get("reward_info") or {}).get("reward_breakdown") or {}
                new_nl = (updated.get("reward_info") or {}).get(
                    "reward_breakdown"
                ) or {}
                old_score, new_score = (
                    record.get("official_score"),
                    updated.get("official_score"),
                )
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            [
                                record["source_id"],
                                str(relative),
                                updated["rejudge"]["source_sha256"],
                                json.dumps(record.get("nl_judge")),
                                backend,
                                old_nl.get("NL_ASSERTION"),
                                new_nl.get("NL_ASSERTION"),
                                old_nl.get("NL_ASSERTION") == new_nl.get("NL_ASSERTION")
                                if selected
                                else "",
                                old_score,
                                new_score,
                                old_score == new_score,
                            ],
                            strict=True,
                        )
                    )
                )
                status["completed_records"] += 1
                if record.get("status") == "completed" and old_score is not None:
                    status["scored_episodes"] += 1
                    status["old_score_sum"] += old_score
                    status["new_score_sum"] += new_score
        if status["scored_episodes"]:
            status["old_mean_score"] = (
                status["old_score_sum"] / status["scored_episodes"]
            )
            status["new_mean_score"] = (
                status["new_score_sum"] / status["scored_episodes"]
            )
        status["status"] = "completed"
    except BaseException as exc:
        # Exception messages can contain provider credentials; persist only the type.
        status.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        write_status()
    return destination


class OfficialJudge:
    """Use tau2's official prompt/parser, with a fixed route and no retries."""

    def __init__(self, backend: str, runtime: dict[str, str]):
        self.model, self.url, variable, key_field = ROUTES[backend]
        # Shell env files may contain exports; never print their values.
        self.key = subprocess.check_output(
            [
                "bash",
                "-c",
                'set -a; source "$1"; printf "%s" "${!2}"',
                "tau2-rejudge",
                runtime[key_field],
                variable,
            ],
            text=True,
        )
        if not self.key:
            raise ValueError(f"Missing {variable}")
        self.returned_model: str | None = None

    def __call__(
        self, record: dict[str, Any], assertions: list[str]
    ) -> list[dict[str, Any]]:
        from tau2.evaluator import evaluator_nl_assertions as evaluator

        saved = (
            evaluator.DEFAULT_LLM_NL_ASSERTIONS,
            evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
            evaluator.generate,
        )
        returned_models: list[str] = []

        def generate(**kwargs: Any) -> Any:
            response = saved[2](**kwargs)
            returned = (response.raw_data or {}).get("model")
            if not returned:
                raise ValueError("Judge response has no raw_data.model")
            if self.returned_model is not None and returned != self.returned_model:
                raise ValueError("Judge returned model changed; stopping rejudge")
            self.returned_model = returned
            returned_models.append(returned)
            return response

        try:
            evaluator.DEFAULT_LLM_NL_ASSERTIONS = f"openai/{self.model}"
            evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
                "api_base": self.url,
                "api_key": self.key,
                "temperature": 0.0,
                "top_p": 1.0,
                "timeout": 120.0,
                "num_retries": 0,
            }
            evaluator.generate = generate
            messages = [
                SimpleNamespace(role=m["role"], content=m.get("content"))
                for m in record["messages"]
            ]
            checks = evaluator.NLAssertionsEvaluator.evaluate_nl_assertions(
                messages, assertions
            )
            return [
                dict(check.model_dump(), returned_model=returned_models[-1])
                for check in checks
            ]
        finally:
            (
                evaluator.DEFAULT_LLM_NL_ASSERTIONS,
                evaluator.DEFAULT_LLM_NL_ASSERTIONS_ARGS,
                evaluator.generate,
            ) = saved


def load_canonical_tasks(data_dir: Path, attempts: list[Path]) -> Tasks:
    """Validate original per-domain hashes before using runtime task criteria."""
    tasks: Tasks = {}
    digests = {}
    for path in (data_dir / "tau2" / "domains").glob("*/tasks.json"):
        payload = path.read_bytes()
        digests[path.parent.name] = hashlib.sha256(payload).hexdigest()
        for task in json.loads(payload):
            tasks[(path.parent.name, str(task["id"]))] = task
    if not tasks:
        raise ValueError("runtime tau2_data_dir contains no tasks.json")
    for attempt in attempts:
        manifest = attempt / "manifest.json"
        nl_domains = {
            record["domain"]
            for _, record in load_records(attempt)
            if assertions_for(record) is not None
        }
        if not manifest.is_file():
            if nl_domains:
                raise ValueError(
                    f"{attempt}: original manifest required for NL task hashes"
                )
            continue
        hashes = json.loads(manifest.read_text()).get("task_file_sha256") or {}
        missing = nl_domains - hashes.keys()
        if missing:
            raise ValueError(
                f"{attempt}: missing original NL task hashes for {sorted(missing)}"
            )
        for domain, expected in hashes.items():
            if digests.get(domain) != expected:
                raise ValueError(
                    f"{attempt}: canonical {domain} task hash differs from original"
                )
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempts", type=Path, nargs="+")
    parser.add_argument("--judge-backend", choices=ROUTES, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    runtime = json.loads(args.runtime.read_text())
    tasks = load_canonical_tasks(Path(runtime["tau2_data_dir"]), args.attempts)
    plans = [call_plan(attempt, tasks) for attempt in args.attempts]
    print(
        json.dumps(
            {"attempts": plans, "judge_calls": sum(p["judge_calls"] for p in plans)},
            indent=2,
        )
    )
    if not args.execute:
        return 0
    if not args.runtime or not args.label:
        parser.error("--execute requires --runtime and --label")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TAU2_DATA_DIR"] = runtime["tau2_data_dir"]
    judge = OfficialJudge(args.judge_backend, runtime)
    for attempt in args.attempts:
        rejudge_attempt(attempt, args.label, args.judge_backend, judge, tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
