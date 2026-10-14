# Qwen3 沿用现有 FSDP 配置，不再加其他优化

- 状态：已采纳（2026-10-11）

## Context

FSDP + FA2 + gradient checkpointing + packing 能直接跑通，在测过的组合里最快；Ulysses、tree
training、archon 也能启动，但都不更快（[infra-issues.md §2](../infra-issues.md)）。

## Decision

Qwen3 沿用现有 FSDP 配置，不再加其他优化。

## Consequences

- 训练 rollout 时每个实例 64 条并发、上下文更长，要关注 `num_retracted_reqs`，必要时调高 `mem_fraction_static`。
