# SPDX-License-Identifier: Apache-2.0

"""Validate and seal fixed-policy τ² episodes for offline critic training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from examples.tau2.contracts import (
    OFFICIAL_TAU2_REVISION,
    SUPPORTED_DOMAINS,
    validate_installed_tau2_revision,
    validate_official_splits,
)

TOKEN_FIELDS = (
    "input_ids",
    "attention_mask",
    "loss_mask",
    "action_origin_mask",
    "behavior_logprobs",
    "versions",
    "turn_ids",
    "token_roles",
)
PRIVATE_FIELDS = {
    "gold",
    "gold_actions",
    "private_info",
    "expected_state",
    "evaluation_criteria",
}
PROVENANCE_FIELDS = (
    "policy_id",
    "policy_revision",
    "simulator_id",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_official_train_ids() -> set[tuple[str, str]]:
    from tau2.registry import registry

    validate_installed_tau2_revision()
    splits_by_domain: dict[str, dict[str, list[str]]] = {}
    for domain in SUPPORTED_DOMAINS:
        splits_loader_fn = registry.get_task_splits_loader(domain)
        if splits_loader_fn is None:
            raise ValueError(f"No task splits loader found for domain {domain}")
        splits_by_domain[domain] = splits_loader_fn()
    validate_official_splits(splits_by_domain, domains=SUPPORTED_DOMAINS)
    identities = {
        (domain, str(task_id))
        for domain in SUPPORTED_DOMAINS
        for task_id in splits_by_domain[domain]["train"]
    }
    if len(identities) != 178:
        raise ValueError(
            f"Official τ² loader exposed {len(identities)} train tasks, expected 178"
        )
    return identities


def _iter_jsonl_paths(data_paths: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in data_paths:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            paths.extend(
                sorted(child for child in path.rglob("*.jsonl") if child.is_file())
            )
        elif path.is_file():
            paths.append(path)
        else:
            raise ValueError(f"Critic data path does not exist: {path}")
    if not paths:
        raise ValueError("No critic JSONL files were found")
    return paths


def _validate_row(
    row: dict[str, Any],
    *,
    line_no: int,
    official_train_ids: set[tuple[str, str]],
    expected_provenance: dict[str, str],
) -> tuple[str, str, str, float, str]:
    missing = {
        "domain",
        "task_id",
        "split",
        "critic_split",
        "episode_id",
        "attempt_id",
        "reward",
        "official_score",
        "terminated",
        "truncated",
        *TOKEN_FIELDS,
        *PROVENANCE_FIELDS,
    } - set(row)
    if missing:
        raise ValueError(f"line {line_no} missing fields: {sorted(missing)}")
    leaked = sorted(PRIVATE_FIELDS & set(row))
    if leaked:
        raise ValueError(f"line {line_no} contains evaluator-private fields: {leaked}")
    for key, expected in expected_provenance.items():
        if str(row.get(key, "")) != expected:
            raise ValueError(f"line {line_no} collection provenance mismatch for {key}")

    domain = str(row["domain"])
    task_id = str(row["task_id"])
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"line {line_no} has unsupported domain {domain}")
    if row["split"] != "train" or (domain, task_id) not in official_train_ids:
        raise ValueError(f"line {line_no} is not an official train task")
    critic_split = str(row["critic_split"])
    if critic_split not in ("train", "dev"):
        raise ValueError(f"line {line_no} has invalid critic_split {critic_split}")

    lengths = {field: len(row[field]) for field in TOKEN_FIELDS}
    width = lengths["input_ids"]
    if width < 2 or any(length != width for length in lengths.values()):
        raise ValueError(f"line {line_no} token fields are not aligned: {lengths}")
    attention = [bool(value) for value in row["attention_mask"]]
    loss = [bool(value) for value in row["loss_mask"]]
    action_origin = [bool(value) for value in row["action_origin_mask"]]
    roles = [str(value) for value in row["token_roles"]]
    if not any(loss):
        raise ValueError(f"line {line_no} has no action tokens")
    seen_padding = False
    for index, (attends, active, originated_as_action, role) in enumerate(
        zip(attention, loss, action_origin, roles, strict=True)
    ):
        if not attends:
            seen_padding = True
        elif seen_padding:
            raise ValueError(f"line {line_no} is not right padded")
        if originated_as_action != (attends and role == "assistant"):
            raise ValueError(
                f"line {line_no} action_origin_mask/token_roles mismatch at token {index}"
            )
        if active != originated_as_action:
            raise ValueError(
                f"line {line_no} loss_mask/action_origin_mask mismatch at token {index}"
            )
        if role in ("prompt", "observation", "user", "tool", "padding") and active:
            raise ValueError(f"line {line_no} trains on {role} token {index}")

    for index, (logp, version, turn_id, active) in enumerate(
        zip(
            row["behavior_logprobs"],
            row["versions"],
            row["turn_ids"],
            loss,
            strict=True,
        )
    ):
        if not math.isfinite(float(logp)):
            raise ValueError(f"line {line_no} non-finite logprob at token {index}")
        if active and (int(version) < 0 or int(turn_id) < 0):
            raise ValueError(
                f"line {line_no} action token {index} lacks version/turn identity"
            )

    terminated = bool(row["terminated"])
    truncated = bool(row["truncated"])
    if terminated == truncated:
        raise ValueError(f"line {line_no} requires terminated XOR truncated")
    if bool(row.get("bootstrap_mask", False)):
        raise ValueError("First τ² critic protocol fixes bootstrap_mask=false")
    reward = float(row["reward"])
    official = float(row["official_score"])
    if reward not in (0.0, 1.0) or official not in (0.0, 1.0):
        raise ValueError(f"line {line_no} requires binary reward and official score")
    if reward != official:
        raise ValueError(
            f"line {line_no} does not match the official-binary-v1 reward protocol"
        )
    episode_id = str(row["episode_id"])
    try:
        canonical_episode_id = int(episode_id)
        episode_tensor_id = int(row["episode_tensor_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_no} requires matching numeric episode_id/episode_tensor_id"
        ) from exc
    if canonical_episode_id < 0 or episode_tensor_id != canonical_episode_id:
        raise ValueError(
            f"line {line_no} episode_id and episode_tensor_id are not identical"
        )
    return domain, task_id, critic_split, reward, episode_id


def iter_critic_rows(
    *data_paths: Path,
    expected_provenance: dict[str, str] | None = None,
    official_train_ids: set[tuple[str, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Read checked rows; exhaust the iterator before consuming a training split."""

    if official_train_ids is None:
        official_train_ids = _load_official_train_ids()
    episode_ids: set[str] = set()
    task_splits: dict[tuple[str, str], str] = {}
    splits: set[str] = set()
    for path in _iter_jsonl_paths(data_paths):
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if expected_provenance is None:
                    expected_provenance = {
                        key: str(row.get(key, "")) for key in PROVENANCE_FIELDS
                    }
                domain, task_id, split, _, episode_id = _validate_row(
                    row,
                    line_no=line_no,
                    official_train_ids=official_train_ids,
                    expected_provenance=expected_provenance,
                )
                if episode_id in episode_ids:
                    raise ValueError(
                        f"{path}:{line_no} duplicates episode_id {episode_id}"
                    )
                episode_ids.add(episode_id)
                identity = (domain, task_id)
                if identity in task_splits and task_splits[identity] != split:
                    raise ValueError(f"Tasks cross critic train/dev split: {identity}")
                task_splits[identity] = split
                splits.add(split)
                yield row
    if not episode_ids:
        raise ValueError("Critic data is empty")
    if splits != {"train", "dev"}:
        raise ValueError("Critic data requires non-empty task-disjoint train and dev")


def inspect_critic_data(
    *data_paths: Path,
    policy_id: str,
    policy_revision: str,
    simulator_id: str,
    official_train_ids: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Validate fixed-policy critic rows and return a small data summary."""

    if not data_paths:
        raise ValueError("At least one critic data path is required")
    expected_provenance = {
        "policy_id": policy_id,
        "policy_revision": policy_revision,
        "simulator_id": simulator_id,
    }
    counts: Counter[str] = Counter()
    per_domain: Counter[str] = Counter()
    rewards: Counter[str] = Counter()
    task_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows = 0
    paths = _iter_jsonl_paths(data_paths)
    for row in iter_critic_rows(
        *paths,
        expected_provenance=expected_provenance,
        official_train_ids=official_train_ids,
    ):
        domain, task_id = str(row["domain"]), str(row["task_id"])
        critic_split, reward = str(row["critic_split"]), float(row["reward"])
        task_splits[(domain, task_id)].add(critic_split)
        counts[critic_split] += 1
        per_domain[f"{critic_split}:{domain}"] += 1
        rewards[f"{critic_split}:{int(reward)}"] += 1
        rows += 1
    return {
        "schema_version": 1,
        "status": "checked",
        "tau2_revision": OFFICIAL_TAU2_REVISION,
        "data_paths": [str(path) for path in paths],
        "data_sha256": {str(path): _sha256_file(path) for path in paths},
        "policy_id": policy_id,
        "policy_revision": policy_revision,
        "simulator_id": simulator_id,
        "reward_protocol": "official-binary-v1",
        "bootstrap_mask": False,
        "rows": rows,
        "episodes_by_split": dict(sorted(counts.items())),
        "episodes_by_split_domain": dict(sorted(per_domain.items())),
        "episodes_by_split_reward": dict(sorted(rewards.items())),
        "unique_tasks": len(task_splits),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def combine_critic_data(data_paths: tuple[Path, ...], output_path: Path) -> None:
    paths = _iter_jsonl_paths(data_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        output.write(line.rstrip("\n") + "\n")
    temporary.replace(output_path)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--simulator-id", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--summary-output")
    action.add_argument("--combine-output")
    args = parser.parse_args(argv)
    data_paths = tuple(Path(raw) for raw in args.data)
    report = inspect_critic_data(
        *data_paths,
        policy_id=args.policy_id,
        policy_revision=args.policy_revision,
        simulator_id=args.simulator_id,
    )
    if args.check:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.summary_output:
        _atomic_write(Path(args.summary_output), report)
    else:
        combine_critic_data(data_paths, Path(args.combine_output))


if __name__ == "__main__":
    main(sys.argv[1:])
