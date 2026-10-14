# SPDX-License-Identifier: Apache-2.0
"""Strict read-only binding to the audited offline critic pool."""

from __future__ import annotations

import hashlib
import json
import re
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


PLANTED_RULES = ("earlier completed WRITE tool call",)
LABEL_SEMANTICS = "completed_assistant_xml_write_tool_call_at_or_before_prefix"
WRITE_TOOL_INVENTORY = json.loads(
    Path(__file__).with_name("critic_preexp_write_tools.json").read_text()
)
WRITE_TOOLS = {
    domain: frozenset(entry["write_tools"])
    for domain, entry in WRITE_TOOL_INVENTORY["domains"].items()
}


class PlantedRuleSelectionError(ValueError):
    """No admissible rule; retain every candidate's diagnostic rates."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("No planted rule passes pooled 20–60% and per-domain 5–95%")


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


def completed_write_calls(
    row: dict[str, Any], tokenizer: Any, last_position: int
) -> list[tuple[int, str]]:
    """Find complete Qwen XML WRITE calls wholly inside assistant-origin spans.

    The first assistant header can belong to the prompt; provenance comes from
    token_roles (or the exporter's action_origin_mask recovery), not text headers.
    A completion endpoint is the inclusive token containing the entire closing
    tool_call tag, independently of im_end or any subsequent tool response.
    """
    import torch

    ids = torch.as_tensor(row["input_ids"]).reshape(-1).tolist()
    roles = row.get("token_roles")
    if roles is None and "action_origin_mask" in row:
        origin = torch.as_tensor(row["action_origin_mask"]).reshape(-1).tolist()
        roles = []
        seen_action = False
        for active in origin:
            if active:
                roles.append("assistant")
                seen_action = True
            else:
                roles.append("observation" if seen_action else "prompt")
    if roles is None or len(roles) != len(ids):
        raise ValueError("Planted labels require token-aligned token_roles")
    if set(roles) - {"prompt", "assistant", "observation"}:
        raise ValueError("Unsupported token_roles for planted labels")
    prompt_end = next((i for i, role in enumerate(roles) if role != "prompt"), len(ids))
    if "prompt_len" in row and int(row["prompt_len"]) != prompt_end:
        raise ValueError("prompt_len disagrees with token_roles")
    domain = row.get("domain")
    if domain not in WRITE_TOOLS:
        raise ValueError(f"Unsupported WRITE tool domain: {domain}")
    calls = []
    limit = min(last_position + 1, len(ids))
    start = prompt_end
    while start < limit:
        if roles[start] != "assistant":
            start += 1
            continue
        stop = start + 1
        while stop < limit and roles[stop] == "assistant":
            stop += 1
        segment = ids[start:stop]

        def decode(end):
            return tokenizer.decode(
                segment[:end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        text = decode(len(segment))
        for match in re.finditer(
            r"<tool_call>((?:(?!<tool_call>).)*?)</tool_call>", text, re.S
        ):
            function = re.fullmatch(
                r"\s*<function=([A-Za-z_][A-Za-z_0-9]*)>"
                r"((?:\s*<parameter=[A-Za-z_][A-Za-z_0-9]*>.*?</parameter>)*)"
                r"\s*</function>\s*",
                match[1],
                re.S,
            )
            if function is None or function[1] not in WRITE_TOOLS[domain]:
                continue
            # Decode only logarithmically many prefixes to locate the exact token
            # endpoint, including tags split across tokens and merged suffixes.
            required = text[: match.end()]
            low, high = 1, len(segment)
            while low < high:
                middle = (low + high) // 2
                if decode(middle).startswith(required):
                    high = middle
                else:
                    low = middle + 1
            calls.append((start + low - 1, function[1]))
        start = stop
    return calls


def _first_tool_match(calls: list[tuple[int, str]], rule: str) -> int | None:
    if rule not in PLANTED_RULES:
        raise ValueError(f"Unsupported planted rule: {rule}")
    return calls[0][0] if calls else None


def prefix_labels(row: dict[str, Any], positions: Any, tokenizer: Any, rule: str):
    """Position p=t-1 is positive only after an assistant WRITE tool_call ends <=p."""
    import torch

    positions = torch.as_tensor(positions, dtype=torch.long)
    messages = (
        completed_write_calls(row, tokenizer, int(positions.max()))
        if positions.numel()
        else []
    )
    first = _first_tool_match(messages, rule)
    return (
        (positions >= first).float()
        if first is not None
        else torch.zeros_like(positions, dtype=torch.float32)
    )


def select_planted_rule(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    """Select in registered order, checking both rates pooled and within domain.

    Token mass is the extraction estimator: each sampled prefix carries N/K
    action tokens. Also report exact full-action prevalence and actual 0->1 flips.
    """
    from scripts.tau2.critic_features import selected_positions

    domains = sorted({row["domain"] for row in rows if row["_split"] == "train"})
    if not domains:
        raise ValueError("No training trajectories for planted rule selection")
    totals = {
        rule: {domain: Counter() for domain in [*domains, "pooled"]}
        for rule in PLANTED_RULES
    }
    for row in rows:
        if row["_split"] != "train":
            continue
        positions, count = selected_positions(row, train_k=128, seed=SEED)
        actions = action_positions(row)
        messages = completed_write_calls(row, tokenizer, int(actions[-1]))
        for rule in PLANTED_RULES:
            first = _first_tool_match(messages, rule)
            positives = int((positions >= first).sum()) if first is not None else 0
            exact = int((actions >= first).sum()) if first is not None else 0
            for domain in (row["domain"], "pooled"):
                stats = totals[rule][domain]
                stats.update(
                    trajectories=1,
                    sampled_prefixes=len(positions),
                    action_tokens=count,
                    positive_prefixes=positives,
                    positive_token_mass=positives * count / len(positions),
                    exact_positive_action_tokens=exact,
                    flipped_trajectories=int(0 < exact < count),
                )
    candidates = {}
    for rule, by_domain in totals.items():
        candidates[rule] = {}
        for domain, counts in by_domain.items():
            candidates[rule][domain] = {
                **counts,
                "token_mass_rate": counts["positive_token_mass"]
                / counts["action_tokens"],
                "sampled_prefix_rate": counts["positive_prefixes"]
                / counts["sampled_prefixes"],
                "exact_action_token_rate": counts["exact_positive_action_tokens"]
                / counts["action_tokens"],
                "trajectory_flip_rate": counts["flipped_trajectories"]
                / counts["trajectories"],
            }
    passing = []
    for rule, stats in candidates.items():
        if all(
            (0.2 <= rates[key] <= 0.6)
            if domain == "pooled"
            else (0.05 <= rates[key] <= 0.95)
            for domain, rates in stats.items()
            for key in ("token_mass_rate", "sampled_prefix_rate")
        ):
            passing.append(rule)
    chosen = passing[0] if passing else None
    report = {
        "rule": chosen,
        "status": "selected" if chosen else "no_candidate_passes",
        "candidate_order": list(PLANTED_RULES),
        "candidate_domain_rates": candidates,
        "candidate_rates": {
            rule: stats["pooled"]["sampled_prefix_rate"]
            for rule, stats in candidates.items()
        },
        "candidate_token_mass_rates": {
            rule: stats["pooled"]["token_mass_rate"]
            for rule, stats in candidates.items()
        },
        "passing_candidates": passing,
        "selection_split": "train",
        "label_semantics": LABEL_SEMANTICS,
        "write_tool_inventory": WRITE_TOOL_INVENTORY,
        "protocol_deviation": "Reviewer-authorized replacement of tool-return keyword candidates with the single earlier completed assistant WRITE call rule; acceptance thresholds unchanged",
        "token_mass_semantics": "sampled prefixes weighted by trajectory action_tokens / sampled_prefixes",
        "flip_semantics": "both 0 and 1 occur among all action positions of a trajectory",
        "acceptance": {
            "pooled": [0.2, 0.6],
            "each_domain": [0.05, 0.95],
            "rates": ["token_mass_rate", "sampled_prefix_rate"],
        },
    }
    if chosen is None:
        raise PlantedRuleSelectionError(report)
    report["training_token_mass_base_rate"] = candidates[chosen]["pooled"][
        "token_mass_rate"
    ]
    report["training_token_base_rate"] = candidates[chosen]["pooled"][
        "sampled_prefix_rate"
    ]
    return report
