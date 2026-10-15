# 用户模拟器和 NL judge 用各模型的默认模式

- 状态：已采纳（2026-10-12）

## Context

DeepSeek 关 thinking 后三次输出逐字相同，经 LiteLLM 传也有效；Doubao 关掉后仍有细微差别。关掉 thinking 后 DeepSeek
用户反而更好（test 60 合计 0.738 / 0.567，对比开 thinking 时 Retail 0.819 / 0.525），原因待 case study
确认。闭源模型多数不支持关 thinking 或设 temperature（[user-simulator.md §2](../user-simulator.md)）。

## Decision

用户模拟器和 NL judge 用各模型的默认模式（DeepSeek、Doubao 开 thinking）。被训练的 actor 关 thinking，评测时 policy
temperature 0。

## Consequences

- 统一用默认模式最容易与其他模型对齐。
- 关 thinking 的结果只作对照，不作为 setting。
