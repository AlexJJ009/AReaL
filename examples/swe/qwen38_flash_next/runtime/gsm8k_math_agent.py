# SPDX-License-Identifier: Apache-2.0
"""Standard single-turn MathAgent, without unused tool-agent SDK imports."""

import os

from math_verify import parse, verify
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from qwen_template_defaults import with_template_defaults

from areal.api import AsyncRewardWrapper


def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


class MathAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs["extra_body"] = with_template_defaults(
            self.kwargs.get("extra_body")
        )
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        self._reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        comp: ChatCompletion = await client.chat.completions.create(
            messages=data["messages"], model="default", **self.kwargs
        )

        return await self._reward_fn(
            completions=comp.choices[0].message.content, answer=data["answer"]
        )
