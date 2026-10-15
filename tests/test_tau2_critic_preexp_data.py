# SPDX-License-Identifier: Apache-2.0
"""CPU integrity, causal prefix, and extraction/merge equivalence checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.tau2 import critic_preexp_data as data
from scripts.tau2 import critic_preexp_extract as extraction
from scripts.tau2 import critic_preexp_tools as inventory


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return "".join(
            {1: "<|im_start|>", 2: "tool\n", 4: "<|im_end|>", 9: "Error"}.get(t, "a")
            for t in tokens
        )

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 4}[token]


class Layer(nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, hidden):
        return hidden + self.amount


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 3)
        self.layers = nn.ModuleList([Layer(i + 1) for i in range(4)])
        self.norm = nn.LayerNorm(3)

    def forward(self, input_ids, **kwargs):
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = Backbone()
        self.score = nn.Linear(3, 1)

    def forward(self, **kwargs):
        hidden = self.model.language_model(**kwargs).last_hidden_state
        return self.score(hidden)


@pytest.fixture
def pool(tmp_path):
    """A byte-pinned one-train/one-validation pool with distinct task identities."""
    rows, manifest = (
        [],
        {
            "schema_version": "offline-critic-pool/1",
            "training": [],
            "validation": [],
            "counts": {},
        },
    )
    for index, (section, split) in enumerate(
        [("training", "train"), ("validation", "validation")]
    ):
        row = {
            "domain": "airline",
            "task_id": str(index),
            "source_id": f"test:{index}",
            "reward": float(index),
            "input_ids": [1, 2, 9, 4, 5, 5],
            "attention_mask": [1] * 6,
            "loss_mask": [0, 0, 0, 0, 1, 1],
            "token_roles": ["observation"] * 4 + ["assistant"] * 2,
            "split": split,
            "rollout_task_id": index,
        }
        path = tmp_path / f"{index}.jsonl"
        raw = json.dumps(row).encode()
        path.write_bytes(raw + b"\n")
        entry = {
            "run": "r499",
            "domain": row["domain"],
            "task_id": row["task_id"],
            "source_id": row["source_id"],
            "reward": row["reward"],
            "seqlen": 6,
            "loss_tokens": 2,
            "input_ids_sha256": data.array_hash(row["input_ids"]),
            "loss_mask_sha256": data.array_hash(row["loss_mask"]),
        }
        if split == "train":
            entry.update(
                rollout_file=str(path),
                line=1,
                file_sha256=data.sha256(path),
                line_sha256=hashlib.sha256(raw).hexdigest(),
            )
        else:
            path = tmp_path / "validation.pt"
            torch.save([row], path)
            entry.update(pt_file=str(path), row_index=0, pt_sha256=data.sha256(path))
        manifest[section].append(entry)
        manifest["counts"][section] = {"rows": 1}
        rows.append(row)
    path = tmp_path / "pool-manifest.json"
    path.write_text(json.dumps(manifest))
    path.with_name(path.name + ".sha256").write_text(data.sha256(path))
    return path


def test_manifest_modified_bytes_rejected(pool, monkeypatch):
    """A deliberately wrong expected hash must fail before source loading."""
    pool.write_text(pool.read_text() + " ")
    if os.environ.get("PREEXP_RED") == "manifest":
        original_sha = data.sha256
        monkeypatch.setattr(
            data,
            "sha256",
            lambda path: Path(str(path) + ".sha256").read_text()
            if str(path).endswith("pool-manifest.json")
            else original_sha(path),
        )
    with pytest.raises(ValueError, match="Pool manifest SHA256"):
        data.load_pool(pool, synthetic=True)


def test_load_pool_exact_source_identity_and_reward(pool):
    """Readback uses audited identities without synthesizing validation labels."""
    manifest, rows = data.load_pool(pool, synthetic=True)
    assert [(r["_split"], r["source_id"], r["reward"]) for r in rows] == [
        ("train", "test:0", 0.0),
        ("validation", "test:1", 1.0),
    ]
    assert manifest["pool_manifest_sha256"] == data.sha256(pool)


@pytest.mark.parametrize("requested_threads", [2, 4])
@pytest.mark.parametrize("invalid_manifest", [False, True])
def test_load_pool_restores_caller_threads_on_success_and_error(
    pool, monkeypatch, requested_threads, invalid_manifest
):
    """Pool validation uses one thread without changing later experiment fits."""
    previous_threads = torch.get_num_threads()
    observed_threads = []
    original_sha = data.sha256

    def record_threads(path):
        observed_threads.append(torch.get_num_threads())
        return original_sha(path)

    monkeypatch.setattr(data, "sha256", record_threads)
    if invalid_manifest:
        pool.write_text(pool.read_text() + " ")
    try:
        torch.set_num_threads(requested_threads)
        if invalid_manifest:
            with pytest.raises(ValueError, match="Pool manifest SHA256"):
                data.load_pool(pool, synthetic=True)
        else:
            _, rows = data.load_pool(pool, synthetic=True)
            assert len(rows) == 2
        assert observed_threads and set(observed_threads) == {1}
        assert torch.get_num_threads() == requested_threads
    finally:
        torch.set_num_threads(previous_threads)


def test_prefix_future_text_cannot_change_label():
    """A WRITE closing wrapper activates only when its final token is visible."""
    call = _tool_call("cancel_pending_order")
    row = _chat_row([("prompt", "policy"), ("assistant", call + "answer")])
    end = len("policy") + len(call) - 1
    labels = data.prefix_labels(
        row, torch.tensor([end - 1, end, end + 1]), ChatTokenizer(), WRITE_RULE
    )
    torch.testing.assert_close(labels, torch.tensor([0.0, 1.0, 1.0]), rtol=0, atol=0)


def test_intermediate_layer_index_and_final_score_input_exact(monkeypatch):
    """Layer1 means output after the first block; final equals actual head input."""
    model = Model()
    row = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    positions = torch.tensor([0, 1])
    if os.environ.get("PREEXP_RED") == "layer":
        original = extraction.extract_layers
        monkeypatch.setattr(
            extraction,
            "extract_layers",
            lambda model, row, positions, layers: {
                "1": original(model, row, positions, ["2"])["2"],
                "final": original(model, row, positions, ["final"])["final"],
            },
        )
    captured = extraction.extract_layers(model, row, positions, ["final", "1"])
    embeddings = model.model.language_model.embed(torch.tensor([[1, 2, 3]]))[0]
    torch.testing.assert_close(
        captured["1"], (embeddings + 1)[positions], rtol=0, atol=0
    )
    expected_final = embeddings
    for amount in (1, 2, 3, 4):
        expected_final = expected_final + amount
    torch.testing.assert_close(
        captured["final"],
        model.model.language_model.norm(expected_final)[positions],
        rtol=0,
        atol=0,
    )
    assert extraction.layer_indices(32) == [8, 16, 24]


def test_disjoint_merge_matches_single_process_and_rejects_duplicate_positions(
    pool, tmp_path, monkeypatch
):
    """Merged features must equal one-process extraction for every identity/token."""
    manifest, rows = data.load_pool(pool, synthetic=True)
    model, rule = (
        Model(),
        {
            "rule": WRITE_RULE,
            "selection_split": "train",
            "label_semantics": data.LABEL_SEMANTICS,
            "write_tool_inventory": data.WRITE_TOOL_INVENTORY,
            "status": "selected",
        },
    )
    for i in range(2):
        extraction.write_part(
            model,
            rows,
            tmp_path / f"part-{i}",
            pool_manifest=manifest,
            tokenizer=Tokenizer(),
            rule=rule,
            layers=["final", "1"],
            checkpoint=tmp_path,
            shard_index=i,
            shard_count=2,
        )
    extraction.write_part(
        model,
        rows,
        tmp_path / "single",
        pool_manifest=manifest,
        tokenizer=Tokenizer(),
        rule=rule,
        layers=["final", "1"],
        checkpoint=tmp_path,
        shard_index=0,
        shard_count=1,
    )
    extraction.merge_parts(
        [tmp_path / f"part-{i}" / "final" for i in range(2)],
        tmp_path / "merged",
        rows,
        manifest,
    )
    for row in rows:
        filename = f"features-{row['_pool_index']:05d}.pt"
        merged = torch.load(tmp_path / "merged" / filename, weights_only=True)
        single = torch.load(tmp_path / "single" / "final" / filename, weights_only=True)
        if os.environ.get("PREEXP_RED") == "merge":
            merged["features"] += 1
        torch.testing.assert_close(
            merged["features"], single["features"], rtol=0, atol=0
        )
        assert {k: v for k, v in merged.items() if k != "features"} == {
            k: v for k, v in single.items() if k != "features"
        }
    cache = tmp_path / "merged" / "manifest.json"
    record = json.loads(cache.read_text())
    filename = record["shards"][0]["path"]
    part = torch.load(cache.parent / filename, weights_only=True)
    part["position"][1] = part["position"][0]
    torch.save(part, cache.parent / filename)
    record["shards"][0]["sha256"] = data.sha256(cache.parent / filename)
    cache.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="numeric metadata|sampled positions"):
        extraction.verify_cache(cache, rows, manifest, synthetic=True)


def test_real_tiny_qwen_final_and_intermediate_inputs_match_hooks():
    """The actual Qwen critic gives exactly the same final and block1 tensors."""
    from tests.test_tau2_critic_features import rollout, tiny_critic

    model, row = tiny_critic(), rollout()
    positions = torch.tensor([1, 2, 4])
    expected_final, expected_middle = [], []
    hooks = [
        model.score.register_forward_pre_hook(
            lambda module, args: expected_final.append(
                args[0][0, positions].detach().clone()
            )
        ),
        model.model.language_model.layers[0].register_forward_hook(
            lambda module, args, output: expected_middle.append(
                (output[0] if isinstance(output, tuple) else output)[0, positions]
                .detach()
                .clone()
            )
        ),
    ]
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([row["input_ids"]]),
            attention_mask=torch.ones(1, 6, dtype=torch.bool),
            use_cache=False,
        )
    for hook in hooks:
        hook.remove()
    actual = extraction.extract_layers(model, row, positions, ["final", "1"])
    torch.testing.assert_close(actual["final"], expected_final[0], rtol=0, atol=0)
    torch.testing.assert_close(actual["1"], expected_middle[0], rtol=0, atol=0)
    with torch.inference_mode():
        hf_output = model.model.language_model(
            input_ids=torch.tensor([row["input_ids"]]),
            attention_mask=torch.ones(1, 6, dtype=torch.bool),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    torch.testing.assert_close(
        actual["1"], hf_output.hidden_states[1][0, positions], rtol=0, atol=0
    )


def test_bytelevel_prefix_decoder_matches_unicode_and_special_token_boundaries():
    """Every byte-token prefix, including incomplete Unicode, matches HF decode."""
    from tokenizers import Tokenizer as RustTokenizer
    from tokenizers import decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    ordinary = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    missing = [value for value in range(256) if value not in ordinary]
    alphabet = [chr(value) for value in ordinary] + [
        chr(256 + i) for i in range(len(missing))
    ]
    vocab = {char: i for i, char in enumerate(alphabet)}
    vocab["<|im_start|>"] = len(vocab)
    backend = RustTokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, additional_special_tokens=["<|im_start|>"]
    )
    ids = tokenizer.encode(
        "Error谢谢✅\n<|im_start|>tool\nfuture error", add_special_tokens=False
    )
    positions = torch.arange(len(ids))
    actual = data.visible_prefix_texts({"input_ids": ids}, positions, tokenizer)
    expected = [
        tokenizer.decode(
            ids[: i + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ).lower()
        for i in range(len(ids))
    ]
    assert actual == expected


class ChatTokenizer:
    """Character tokens plus atomic Qwen message delimiters, without downloads."""

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1000, "<|im_end|>": 1001}[token]

    def encode(self, text):
        ids = []
        while text:
            for marker in ("<|im_start|>", "<|im_end|>"):
                if text.startswith(marker):
                    ids.append(self.convert_tokens_to_ids(marker))
                    text = text[len(marker) :]
                    break
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ids

    def decode(self, tokens, **kwargs):
        return "".join(
            {1000: "<|im_start|>", 1001: "<|im_end|>"}.get(int(t), chr(int(t)))
            for t in tokens
        )


def _chat_row(messages, *, domain="retail", identity="synthetic"):
    tokenizer = ChatTokenizer()
    ids, roles = [], []
    for role, text in messages:
        encoded = tokenizer.encode(text)
        ids.extend(encoded)
        roles.extend([role] * len(encoded))
    return {
        "input_ids": ids,
        "token_roles": roles,
        "attention_mask": [1] * len(ids),
        "loss_mask": [int(role == "assistant") for role in roles],
        "_split": "train",
        "domain": domain,
        "task_id": identity,
        "_trajectory_id": identity,
    }


def _message(speaker, body):
    return f"<|im_start|>{speaker}\n{body}<|im_end|>"


WRITE_RULE = "earlier completed WRITE tool call"
WRITE_NAMES = {"retail": "cancel_pending_order", "airline": "cancel_reservation"}
READ_NAMES = {"retail": "get_order_details", "airline": "get_reservation_details"}


def _tool_call(name):
    return (
        f"<tool_call>\n<function={name}>\n"
        "<parameter=id>\nexample\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def _labels(row):
    return data.prefix_labels(
        row, torch.arange(len(row["input_ids"])), ChatTokenizer(), WRITE_RULE
    )


def _assert_negative_then_positive(messages, *, domain="retail"):
    """A negative case must also prove the labeler recognizes a later real WRITE."""
    before = _chat_row(messages, domain=domain)
    row = _chat_row(
        messages + [("assistant", _tool_call(WRITE_NAMES[domain]) + "answer")],
        domain=domain,
    )
    labels = _labels(row)
    torch.testing.assert_close(
        labels[: len(before["input_ids"])],
        torch.zeros(len(before["input_ids"])),
        rtol=0,
        atol=0,
    )
    assert labels[-1] == 1


@pytest.mark.parametrize("domain", ["retail", "airline"])
def test_system_schema_write_mentions_do_not_activate_label(domain):
    """Policy/schema WRITE names and complete quoted calls are prompt provenance."""
    body = "Tool schema " + json.dumps({"name": WRITE_NAMES[domain], "type": "write"})
    _assert_negative_then_positive(
        [("prompt", _message("system", body + _tool_call(WRITE_NAMES[domain])))],
        domain=domain,
    )


@pytest.mark.parametrize("domain", ["retail", "airline"])
def test_user_write_mentions_and_complete_call_do_not_activate_label(domain):
    """A user cannot create the planted label by quoting a valid WRITE wrapper."""
    _assert_negative_then_positive(
        [
            ("prompt", "policy"),
            ("observation", _message("user", _tool_call(WRITE_NAMES[domain]))),
        ],
        domain=domain,
    )


@pytest.mark.parametrize("domain", ["retail", "airline"])
def test_assistant_read_tool_call_never_sets_label(domain):
    """A completed READ call remains zero until a completed WRITE is generated."""
    assert READ_NAMES[domain] not in data.WRITE_TOOLS[domain]
    assert WRITE_NAMES[domain] in data.WRITE_TOOLS[domain]
    _assert_negative_then_positive(
        [("prompt", "policy"), ("assistant", _tool_call(READ_NAMES[domain]))],
        domain=domain,
    )


@pytest.mark.parametrize("domain", ["retail", "airline"])
def test_partial_write_activates_at_closing_wrapper_without_response(domain):
    """A complete call needs neither message terminator nor a tool response."""
    call = _tool_call(WRITE_NAMES[domain])
    row = _chat_row(
        [("prompt", _message("assistant", "")), ("assistant", call + "next")],
        domain=domain,
    )
    end = len(row["input_ids"]) - len("next") - 1
    labels = _labels(row)
    torch.testing.assert_close(labels[:end], torch.zeros(end), rtol=0, atol=0)
    torch.testing.assert_close(
        labels[end:], torch.ones(len(labels) - end), rtol=0, atol=0
    )
    partial = dict(
        row, input_ids=row["input_ids"][:end], token_roles=row["token_roles"][:end]
    )
    assert _labels(partial).sum() == 0


@pytest.mark.parametrize(
    "body",
    [
        "<tool_call><function=cancel_pending_order><parameter=id>x</function></tool_call>",
        "<tool_call><function=cancel_pending_order><parameter=id>x</parameter></tool_call>",
        "<tool_call><function=cancel_pending_order><parameter=id>x</parameter></function>",
        '<tool_call>{"name":"cancel_pending_order","arguments":{}}</tool_call>',
        _tool_call("unknown_tool"),
        _tool_call("cancel_reservation"),
        "cancel_pending_order({})",
    ],
)
def test_malformed_unknown_and_wrong_domain_calls_do_not_activate_label(body):
    """Only complete Qwen XML calls to a registered domain WRITE count."""
    _assert_negative_then_positive([("prompt", "policy"), ("assistant", body)])


def test_call_wrapper_across_role_segments_does_not_activate_label():
    """A wrapper assembled from assistant and user tokens is not an action."""
    call = _tool_call(WRITE_NAMES["retail"])
    cut = call.index("parameter")
    _assert_negative_then_positive(
        [
            ("prompt", "policy"),
            ("assistant", call[:cut]),
            ("observation", call[cut:]),
        ]
    )


def test_prefix_labels_unknown_domain_rejected():
    """Unknown domains cannot borrow another domain's WRITE inventory."""
    row = _chat_row(
        [("assistant", _tool_call("cancel_pending_order"))], domain="unknown"
    )
    with pytest.raises(ValueError, match="domain"):
        _labels(row)


def test_prefix_labels_missing_role_provenance_rejected():
    """Text alone cannot distinguish prompt, user and generated action tokens."""
    with pytest.raises(ValueError, match="token_roles|action_origin_mask"):
        data.prefix_labels(
            {"input_ids": [1, 2], "domain": "retail"},
            torch.tensor([1]),
            ChatTokenizer(),
            WRITE_RULE,
        )


def test_prefix_labels_action_origin_mask_matches_exported_roles():
    """Validation PT origin masks recover the same completed-WRITE transition."""
    row = _chat_row(
        [
            ("prompt", _message("system", _tool_call("cancel_pending_order"))),
            ("assistant", "reason"),
            ("observation", _message("user", _tool_call("cancel_pending_order"))),
            ("assistant", _tool_call("cancel_pending_order") + "answer"),
        ]
    )
    fallback = {key: value for key, value in row.items() if key != "token_roles"}
    fallback["action_origin_mask"] = [
        int(role == "assistant") for role in row["token_roles"]
    ]
    positions = torch.arange(len(row["input_ids"]))
    expected = data.prefix_labels(row, positions, ChatTokenizer(), WRITE_RULE)
    actual = data.prefix_labels(fallback, positions, ChatTokenizer(), WRITE_RULE)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0] == 0
    assert actual[-1] == 1


@pytest.mark.parametrize("rule", ["error", "error or not found", "refund"])
def test_prefix_labels_unregistered_rule_rejected(rule):
    """Old keyword rules and ad hoc words cannot replace the registered WRITE rule."""
    row = _chat_row([("prompt", "policy"), ("assistant", "answer")])
    with pytest.raises(ValueError, match="rule|registered"):
        data.prefix_labels(row, torch.tensor([0]), ChatTokenizer(), rule)


def _selection_row(domain, index, positive):
    row = _chat_row(
        [
            ("prompt", _message("system", _tool_call(WRITE_NAMES[domain]))),
            ("observation", _message("user", _tool_call(WRITE_NAMES[domain]))),
            (
                "assistant",
                _tool_call(WRITE_NAMES[domain] if positive else READ_NAMES[domain]),
            ),
            ("assistant", "a" * 100),
        ],
        domain=domain,
        identity=f"{domain}-{index}",
    )
    # Origin is generated-action provenance; sampled loss tokens need not include
    # the earlier call. This gives every row exactly 100 equal-mass test prefixes.
    row["loss_mask"] = [0] * (len(row["input_ids"]) - 100) + [1] * 100
    return row


@pytest.mark.parametrize("positive_domain", ["retail", "airline"])
def test_select_planted_rule_domain_determined_signal_rejected(positive_domain):
    """Pooled 40% passes, but domains with 0% or 100% must reject selection."""
    negative_domain = "airline" if positive_domain == "retail" else "retail"
    rows = [
        _selection_row(domain, index, domain == positive_domain)
        for domain, count in ((positive_domain, 8), (negative_domain, 12))
        for index in range(count)
    ]
    assert _labels(rows[0])[-1] == 1
    with pytest.raises(ValueError, match="No planted rule") as failure:
        data.select_planted_rule(rows, ChatTokenizer())
    report = failure.value.report
    assert set(report["candidate_rates"]) == {WRITE_RULE}
    assert report["candidate_rates"][WRITE_RULE] == pytest.approx(0.4)
    assert report["candidate_token_mass_rates"][WRITE_RULE] == pytest.approx(0.4)
    rates = report["candidate_domain_rates"][WRITE_RULE]
    assert rates[positive_domain]["token_mass_rate"] == 1
    assert rates[negative_domain]["token_mass_rate"] == 0


def test_select_planted_rule_training_only_balanced_writes_selected():
    """Balanced training WRITEs select 40%; held-out 100% cannot affect selection."""
    rows = [
        _selection_row(domain, index, index < 4)
        for domain in ("retail", "airline")
        for index in range(10)
    ]
    held_out = _selection_row("retail", 100, True)
    held_out["_split"] = "validation"
    held_out["domain"] = "held-out-unknown-domain"
    report = data.select_planted_rule(rows + [held_out] * 100, ChatTokenizer())
    assert report["rule"] == WRITE_RULE
    assert report["selection_split"] == "train"
    assert report["candidate_order"] == [WRITE_RULE]
    assert report["training_token_base_rate"] == pytest.approx(0.4)
    assert report["training_token_mass_base_rate"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "overrides",
    [
        {"label_semantics": "lowercase_substring_in_tokens_through_prefix_position"},
        {
            "label_semantics": "completed_post_prompt_tool_return_case_insensitive_substring"
        },
        {"rule": "error"},
        {"status": "rejected"},
        {"rule": "refund"},
    ],
)
def test_verify_cache_stale_planted_rule_rejected_without_tokenizer(
    tmp_path, overrides
):
    """Even metadata-only consumers must reject labels from the superseded rule."""
    rule = {
        "rule": WRITE_RULE,
        "label_semantics": data.LABEL_SEMANTICS,
        "write_tool_inventory": data.WRITE_TOOL_INVENTORY,
        "status": "selected",
    }
    rule.update(overrides)
    manifest = {
        "schema": extraction.SCHEMA,
        "position_semantics": "prefix_position=t-1_for_action_token_t",
        "pool_manifest_sha256": "pinned-pool",
        "layer": "final",
        "planted_rule": rule,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="planted rule semantics/status"):
        extraction.verify_cache(
            path, [], {"pool_manifest_sha256": "pinned-pool"}, synthetic=True
        )


def test_write_inventory_derived_from_annotations_with_source_provenance(
    tmp_path, monkeypatch
):
    """AST annotations classify tools; comments and name heuristics cannot add WRITEs."""
    content = (
        "# @is_tool(ToolType.WRITE)\n"
        "# def commented_write(): pass\n"
        "@is_tool(ToolType.WRITE)\n"
        "def surprising_name(): pass\n"
        "@is_tool(ToolType.READ)\n"
        "def cancel_read_only(): pass\n"
        "@is_tool(ToolType.GENERIC)\n"
        "async def generic_operation(): pass\n"
        "def unannotated_write(): pass\n"
    )
    for domain in ("airline", "retail"):
        path = tmp_path / "src" / "tau2" / "domains" / domain / "tools.py"
        path.parent.mkdir(parents=True)
        path.write_text(content)
    calls = []

    def pinned_revision(command, *, text):
        calls.append((command, text))
        return "pinned-revision\n"

    monkeypatch.setattr(inventory.subprocess, "check_output", pinned_revision)
    result = inventory.derive_write_inventory(tmp_path)

    assert calls == [(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], True)]
    assert result["source_revision"] == "pinned-revision"
    assert result["derivation"] == "AST is_tool(ToolType.WRITE)"
    for domain in ("airline", "retail"):
        entry = result["domains"][domain]
        assert entry["write_tools"] == ["surprising_name"]
        assert entry["source_file"] == f"src/tau2/domains/{domain}/tools.py"
        assert entry["source_sha256"] == hashlib.sha256(content.encode()).hexdigest()
        assert entry["tools"] == [
            {
                "name": "cancel_read_only",
                "type": "READ",
                "annotation_line": 5,
                "definition_line": 6,
            },
            {
                "name": "generic_operation",
                "type": "GENERIC",
                "annotation_line": 7,
                "definition_line": 8,
            },
            {
                "name": "surprising_name",
                "type": "WRITE",
                "annotation_line": 3,
                "definition_line": 4,
            },
        ]
