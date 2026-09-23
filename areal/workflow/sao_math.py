# SPDX-License-Identifier: Apache-2.0
"""Auditable single-turn math rollout using the native RLVR tensor contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import aiofiles
import torch

from areal import workflow_context
from areal.api import AsyncRewardWrapper, InferenceEngine, ModelRequest, ModelResponse
from areal.reward.math_prd import extract_last_complete_boxed_answer
from areal.utils import stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import atrace_session_phase, session_context
from areal.workflow.rlvr import RLVRWorkflow


class AuditedMathWorkflow(RLVRWorkflow):
    """Keep native generation and masks, with strict scoring and per-sample evidence."""

    def __init__(
        self,
        *,
        audit_dir: str,
        reward_timeout_seconds: float = 120,
        reward_max_workers: int = 8,
        **kwargs: Any,
    ):
        # Resolve now so arun_episode does not replace the strict reward wrapper.
        reward_fn = kwargs["reward_fn"]
        if isinstance(reward_fn, str):
            kwargs["reward_fn"] = import_from_string(reward_fn)
        kwargs["include_termination"] = True
        super().__init__(**kwargs)
        self.async_reward_fn = AsyncRewardWrapper(
            self.reward_fn,
            timeout_seconds=reward_timeout_seconds,
            max_workers=reward_max_workers,
            max_retries=0,
            raise_on_timeout=True,
        )
        self.audit_dir = Path(audit_dir)
        self._audit_lock = asyncio.Lock()

    async def arun_episode(self, engine, data):
        result = await super().arun_episode(engine, data)
        # This math task ends and is scored at the response budget. Preserve the
        # observed stop reason, but do not credit an unobserved continuation.
        result["bootstrap_mask"] = torch.zeros_like(result["truncated"])
        context = workflow_context.get()
        # Small metadata tensors survive native group concatenation and remote
        # storage. The controller fetches these alone to prove actual consumption.
        source_key = int.from_bytes(
            hashlib.sha256(str(data["source_id"]).encode()).digest()[:8], "big"
        ) & ((1 << 63) - 1)
        result["audit_source_key"] = torch.tensor([source_key], dtype=torch.int64)
        result["audit_task_id"] = torch.tensor([context.task_id], dtype=torch.int64)
        result["audit_sample_idx"] = torch.tensor(
            [0 if context.sample_idx is None else context.sample_idx], dtype=torch.int64
        )
        return result

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        context = workflow_context.get()
        started_ns = time.time_ns()
        resp = None
        generated_ns = None
        scoring_error = None
        try:
            async with atrace_session_phase("generate"):
                resp = await engine.agenerate(req)
            generated_ns = time.time_ns()
            try:
                reward = await self._compute_rewards(resp, prompt_str, task_data)
            except Exception as exc:
                # Semantic timeout retries are bounded inside the scorer. Keep
                # this trajectory, but distinguish fallback zero from a wrong answer.
                scoring_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                reward = 0.0
        except Exception as exc:
            await self._write_record(
                {
                    "request_id": req.rid,
                    "source_id": task_data["source_id"],
                    "task_id": context.task_id,
                    "sample_idx": context.sample_idx,
                    "benchmark": task_data.get(
                        "benchmark", task_data.get("data_source")
                    ),
                    "answer": task_data.get("answer"),
                    "completion": (
                        self.tokenizer.decode(resp.output_tokens)
                        if resp is not None
                        else None
                    ),
                    "generation_completed_ns": generated_ns,
                    "is_eval": context.is_eval,
                    "started_ns": started_ns,
                    "failed_ns": time.time_ns(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "reward": None,
                },
                context.is_eval,
            )
            raise
        completed_ns = time.time_ns()
        if not math.isfinite(reward) or reward not in (0.0, 1.0):
            raise ValueError(f"Expected a verified binary math reward, got {reward}")
        if not resp.output_tokens:
            raise ValueError("Empty rollout cannot become a training trajectory")
        if resp.stop_reason == "abort":
            raise ValueError("Aborted generation cannot become a training trajectory")
        if len(resp.output_tokens) > req.gconfig.max_new_tokens:
            raise ValueError("Generation exceeded the response length contract")
        if resp.input_tokens != req.input_ids:
            raise ValueError("Returned prompt tokens differ from the submitted prompt")
        if len(resp.output_tokens) != len(resp.output_logprobs):
            raise ValueError("Missing behavior logprobs in rollout response")
        if len(resp.output_tokens) != len(resp.output_versions):
            raise ValueError("Missing behavior versions in rollout response")
        if not all(math.isfinite(x) for x in resp.output_logprobs):
            raise ValueError("Non-finite behavior logprobs in rollout response")
        completion = self.tokenizer.decode(resp.output_tokens)
        extracted = extract_last_complete_boxed_answer(completion)
        record = {
            "request_id": req.rid,
            "source_id": task_data["source_id"],
            "benchmark": task_data.get("benchmark", task_data.get("data_source")),
            "task_id": context.task_id,
            "sample_idx": context.sample_idx,
            "is_eval": context.is_eval,
            "started_ns": started_ns,
            "generation_completed_ns": generated_ns,
            "completed_ns": completed_ns,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "behavior_logprobs": resp.output_logprobs,
            "behavior_versions": resp.output_versions,
            "answer": task_data["answer"],
            "completion": completion,
            "parsed_answer": extracted.answer,
            "parse_status": extracted.status.value,
            "truncated": resp.stop_reason == "length",
            "stop_reason": resp.stop_reason,
            "reward": reward,
            "scoring_error": scoring_error,
            "reward_fallback_zero": scoring_error is not None,
        }
        await self._write_record(record, context.is_eval)
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward, scoring_failure=float(scoring_error is not None)
        )
        return resp, reward

    async def _write_record(self, record: dict[str, Any], is_eval: bool) -> None:
        # Each worker has its own file; the lock serializes concurrent samples
        # within this workflow without blocking the generation event loop.
        async with self._audit_lock:
            await asyncio.to_thread(self.audit_dir.mkdir, parents=True, exist_ok=True)
            async with aiofiles.open(
                self.audit_dir
                / f"{'eval' if is_eval else 'train'}-{os.getpid()}.jsonl",
                "a",
                encoding="utf-8",
            ) as stream:
                await stream.write(json.dumps(record, ensure_ascii=False) + "\n")
