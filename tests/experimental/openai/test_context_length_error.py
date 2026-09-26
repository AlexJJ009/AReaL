# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from areal.api import ModelResponse
from areal.experimental.openai import ArealOpenAI
from areal.experimental.openai.types import (
    CONTEXT_LENGTH_EXCEEDED_MARKER,
    ContextLengthExceededError,
)


class _OneTokenEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def agenerate(self, req):
        self.calls += 1
        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=[42, req.tokenizer.eos_token_id],
            output_logprobs=[-0.2, -0.1],
            output_versions=[0, 0],
            tokenizer=req.tokenizer,
        )


class _PrefixAwareTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        assert tokenize is True
        tokens: list[int] = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "assistant":
                tokens.extend([42, self.eos_token_id])
            elif "overflow" in content:
                tokens.extend(range(100, 130))
            else:
                tokens.extend([11, self.eos_token_id])
        if kwargs.get("add_generation_prompt"):
            tokens.append(99)
        return {"input_ids": tokens}

    def decode(self, tokens) -> str:
        return "ok"


class _NoCallEngine:
    def __init__(self) -> None:
        self.calls = 0

    async def agenerate(self, req):  # pragma: no cover - test asserts no call
        self.calls += 1
        raise AssertionError("engine must not be called for context preflight failure")


class _FixedPromptTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, prompt_len: int) -> None:
        self.prompt_len = prompt_len

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        assert tokenize is True
        return {"input_ids": list(range(self.prompt_len))}

    def decode(self, tokens) -> str:  # pragma: no cover - generation is not reached
        return "decoded"


@pytest.mark.asyncio
async def test_chat_context_limit_preflight_skips_engine_and_removes_cache():
    engine = _NoCallEngine()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=_FixedPromptTokenizer(prompt_len=8),
        engine_max_tokens=8,
    )

    with pytest.raises(ContextLengthExceededError) as exc_info:
        await client.chat.completions.create(
            messages=[{"role": "user", "content": "too long"}],
            max_completion_tokens=4,
        )

    assert CONTEXT_LENGTH_EXCEEDED_MARKER in str(exc_info.value)
    assert exc_info.value.code == "context_length_exceeded"
    assert engine.calls == 0
    assert client.export_interactions(style="individual") == {}


@pytest.mark.asyncio
async def test_responses_context_limit_preflight_skips_engine_and_removes_cache():
    engine = _NoCallEngine()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=_FixedPromptTokenizer(prompt_len=4),
        engine_max_tokens=4,
    )

    with pytest.raises(ContextLengthExceededError) as exc_info:
        await client.responses.create(input="too long", max_output_tokens=4)

    assert CONTEXT_LENGTH_EXCEEDED_MARKER in str(exc_info.value)
    assert exc_info.value.limit_name == "engine_max_tokens"
    assert engine.calls == 0
    assert client.export_interactions(style="individual") == {}


@pytest.mark.asyncio
async def test_concat_context_limit_preserves_completed_parent_and_no_failed_leaf():
    engine = _OneTokenEngine()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=_PrefixAwareTokenizer(),
        chat_template_type="concat",
        engine_max_tokens=16,
    )
    client.chat.completions.chat_template_type = "concat"

    base_messages = [{"role": "user", "content": "start"}]
    root = await client.chat.completions.create(
        messages=base_messages, max_completion_tokens=4
    )
    client.set_reward(root.id, 0.0)
    root_messages = [
        choice.message.model_dump(exclude_none=True) for choice in root.choices
    ]

    with pytest.raises(ContextLengthExceededError):
        await client.chat.completions.create(
            messages=base_messages
            + root_messages
            + [{"role": "user", "content": "overflow"}],
            max_completion_tokens=4,
        )

    individual = client.export_interactions(style="individual")
    leaves = client.export_interactions(style="concat")
    assert engine.calls == 1
    assert set(individual) == {root.id}
    assert set(leaves) == {root.id}
    assert leaves[root.id].reward == 0.0
