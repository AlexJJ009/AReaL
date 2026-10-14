# SPDX-License-Identifier: Apache-2.0
"""Strict read-only binding to the audited offline critic pool."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.tau2.critic_features import _reward, action_positions, validate_splits

SEED = 20261014
RETRY_OFFSET = 1 << 40


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: Any) -> str:
    import torch

    value = torch.as_tensor(value).reshape(-1).tolist()
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def load_pool(
    path: str | Path, synthetic: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate exact audited sources and return provenance-rich canonical rows.

    Synthetic mode only relaxes registered production counts/replay completeness;
    all byte, identity, mask and reward integrity checks still apply.
    """
    import torch

    torch.set_num_threads(1)
    path = Path(path)
    digest = sha256(path)
    if path.with_name(path.name + ".sha256").read_text().split()[0] != digest:
        raise ValueError("Pool manifest SHA256 mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "offline-critic-pool/1":
        raise ValueError("Unsupported pool schema")
    manifest["pool_manifest_sha256"] = digest
    sources, rows, seen = {}, [], set()
    for section, split in [("training", "train"), ("validation", "validation")]:
        entries = manifest[section]
        if (
            not synthetic
            and len(entries) != {"training": 2774, "validation": 120}[section]
        ):
            raise ValueError("Registered pool count mismatch")
        if len(entries) != manifest["counts"][section]["rows"]:
            raise ValueError("Declared pool count mismatch")
        for entry in entries:
            if entry["run"] not in ("r499", "r510"):
                raise ValueError("Unregistered source run")
            source = entry["rollout_file"] if split == "train" else entry["pt_file"]
            expected_sha = (
                entry["file_sha256"] if split == "train" else entry["pt_sha256"]
            )
            if source not in sources:
                if sha256(source) != expected_sha:
                    raise ValueError(f"Source SHA256 mismatch: {source}")
                sources[source] = (
                    expected_sha,
                    Path(source).read_bytes().splitlines()
                    if split == "train"
                    else torch.load(source, map_location="cpu", weights_only=True),
                )
            if sources[source][0] != expected_sha:
                raise ValueError("Conflicting source hashes")
            index = entry["line"] - 1 if split == "train" else entry["row_index"]
            locator = (source, index)
            if locator in seen:
                raise ValueError("Duplicate source row")
            seen.add(locator)
            raw = sources[source][1][index]
            if split == "train":
                if hashlib.sha256(raw).hexdigest() != entry["line_sha256"]:
                    raise ValueError("Source line SHA256 mismatch")
                row = json.loads(raw)
            else:
                row = dict(raw)
            for key in ("domain", "task_id", "source_id"):
                if str(row[key]) != str(entry[key]):
                    raise ValueError(f"Source identity mismatch: {key}")
            if _reward(row) != float(entry["reward"]):
                raise ValueError("Source reward mismatch")
            for field, plural in (
                ("original_reward", "original_rewards"),
                ("official_score", "official_scores"),
                ("task_budget_failure", "task_budget_failure"),
                ("terminated", "terminated"),
                ("truncated", "truncated"),
                ("bootstrap_mask", "bootstrap_mask"),
            ):
                if field in entry:
                    stored = row.get(field, row.get(plural))
                    if stored is None or float(
                        torch.as_tensor(stored).reshape(-1)[-1]
                    ) != float(entry[field]):
                        raise ValueError(f"Source terminal metadata mismatch: {field}")
            if split == "validation" and "rollout_file" in entry:
                if sha256(entry["rollout_file"]) != entry["file_sha256"]:
                    raise ValueError("Validation corroborating JSONL SHA mismatch")
            for key in ("input_ids", "loss_mask"):
                if array_hash(row[key]) != entry[key + "_sha256"]:
                    raise ValueError(f"Source tensor mismatch: {key}")
            ids = torch.as_tensor(row["input_ids"]).reshape(-1)
            if (
                len(ids) != entry["seqlen"]
                or len(action_positions(row)) != entry["loss_tokens"]
            ):
                raise ValueError("Source token count mismatch")
            if split == "train" and row.get("split") != "train":
                raise ValueError("Held-out source in training")
            physical = int(row.get("rollout_task_id", index))
            if physical < 0 or physical >= 2 * RETRY_OFFSET:
                raise ValueError("Invalid retry ID")
            order = physical % RETRY_OFFSET
            row.update(
                _source_path=str(Path(source).resolve()),
                _source_row=index,
                _split=split,
                _run=entry["run"],
                _episode_order=order,
                _trainer_step=order // 16,
                rollout_task_id=physical,
                _trajectory_id=f"{entry['run']}:{split}:{source}:{index}",
            )
            rows.append(row)
        observed = Counter(row["domain"] for row in rows if row["_split"] == split)
        for domain, count in manifest["counts"][section].get("by_domain", {}).items():
            if observed[domain] != count["rows"]:
                raise ValueError("Domain pool count mismatch")
    validate_splits(
        [r for r in rows if r["_split"] == "train"],
        [r for r in rows if r["_split"] == "validation"],
    )
    rows.sort(
        key=lambda r: (
            r["_split"] != "train",
            r["_run"],
            r["_episode_order"],
            r["_source_row"],
        )
    )
    replay = [
        r["_episode_order"]
        for r in rows
        if r["_split"] == "train" and r["_run"] == "r510"
    ]
    if not synthetic and replay != list(range(1280)):
        raise ValueError("r510 numeric replay is incomplete or duplicated")
    for index, row in enumerate(rows):
        row["_pool_index"] = index
    return manifest, rows


PLANTED_RULES = (
    "error",
    "sorry",
    "cancel",
    "refund",
    "reservation",
    "order",
    "flight",
    "thank",
)


def visible_prefix_texts(
    row: dict[str, Any], positions: Any, tokenizer: Any
) -> list[str]:
    """Decode exact ByteLevel causal byte prefixes, with a roundtrip fallback.

    ByteLevel tokens reversibly encode bytes. Slicing the concatenated byte stream
    at each token endpoint before UTF-8 decoding retains replacement characters at
    incomplete Unicode boundaries exactly as tokenizer.decode does. Other decoders
    and vocabularies that do not roundtrip use the ordinary prefix decoder.
    """
    import torch

    ids = torch.as_tensor(row["input_ids"]).reshape(-1).tolist()
    endpoints = [int(position) + 1 for position in positions]
    if not endpoints:
        return []
    ids = ids[: max(endpoints)]
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    state = json.loads(decoder.__getstate__()) if decoder is not None else {}
    if state.get("type") == "ByteLevel":
        ordinary = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        missing = [value for value in range(256) if value not in ordinary]
        inverse = {chr(value): value for value in ordinary}
        inverse.update({chr(256 + i): value for i, value in enumerate(missing)})
        chunks = []
        offsets = [0]
        for token in tokenizer.convert_ids_to_tokens(ids):
            try:
                chunk = bytes(inverse[char] for char in token)
            except (KeyError, TypeError):
                chunks = []
                break
            chunks.append(chunk)
            offsets.append(offsets[-1] + len(chunk))
        if chunks:
            content = b"".join(chunks)
            reference = tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if content.decode("utf-8", errors="replace") == reference:
                return [
                    content[: offsets[end]].decode("utf-8", errors="replace").lower()
                    for end in endpoints
                ]
    return [
        tokenizer.decode(
            ids[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).lower()
        for end in endpoints
    ]


def prefix_labels(row: dict[str, Any], positions: Any, tokenizer: Any, rule: str):
    """A label depends only on visible tokens <= the causal prefix position."""
    import torch

    return torch.tensor(
        [
            float(rule in text)
            for text in visible_prefix_texts(row, positions, tokenizer)
        ]
    )


def select_planted_rule(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """Choose a visible-text rule using training positions only, without rewards."""
    from scripts.tau2.critic_features import selected_positions

    totals = Counter()
    weighted_totals = Counter()
    n = 0
    token_mass = 0
    for row in rows:
        if row["_split"] != "train":
            continue
        positions, count = selected_positions(row, train_k=128, seed=SEED)
        for text in visible_prefix_texts(row, positions, tokenizer):
            for rule in PLANTED_RULES:
                label = int(rule in text)
                totals[rule] += label
                weighted_totals[rule] += label * count / len(positions)
        n += len(positions)
        token_mass += count
    rates = {rule: totals[rule] / n for rule in PLANTED_RULES}
    weighted_rates = {
        rule: weighted_totals[rule] / token_mass for rule in PLANTED_RULES
    }
    for rule in PLANTED_RULES:
        if 0.2 <= rates[rule] <= 0.6 and 0.2 <= weighted_rates[rule] <= 0.6:
            return {
                "rule": rule,
                "training_token_mass_base_rate": weighted_rates[rule],
                "candidate_token_mass_rates": weighted_rates,
                "training_token_base_rate": rates[rule],
                "candidate_rates": rates,
                "selection_split": "train",
                "label_semantics": "lowercase_substring_in_tokens_through_prefix_position",
            }
    raise ValueError(f"No planted rule has training base rate 20–60%: {rates}")
