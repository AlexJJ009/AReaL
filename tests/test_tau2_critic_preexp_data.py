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


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return "".join("error" if t == 9 else "a" for t in tokens)


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
            "input_ids": [1, 2, 3, 4, 9],
            "attention_mask": [1] * 5,
            "loss_mask": [0, 1, 1, 0, 1],
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
            "seqlen": 5,
            "loss_tokens": 3,
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


def test_prefix_future_text_cannot_change_label(monkeypatch):
    """Future Error token is unavailable at the earlier causal prefix."""
    row = {"input_ids": [1, 1, 9]}
    if os.environ.get("PREEXP_RED") == "prefix":
        monkeypatch.setattr(
            data,
            "prefix_labels",
            lambda row, positions, tokenizer, rule: torch.ones(len(positions)),
        )
    labels = data.prefix_labels(row, torch.tensor([0, 1, 2]), Tokenizer(), "error")
    torch.testing.assert_close(labels, torch.tensor([0.0, 0.0, 1.0]), rtol=0, atol=0)


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
    model, rule = Model(), {"rule": "error", "selection_split": "train"}
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
