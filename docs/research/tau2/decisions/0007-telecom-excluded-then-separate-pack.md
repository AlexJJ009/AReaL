# Qwen3.5 训练不用 Telecom；Telecom 数据单独建 pack

- 状态：已采纳（2026-10-10，2026-10-11 补充）

## Context

Telecom 对 Qwen3.5-4B 太简单：训练前 320/320 全对，166 个 GRPO 组里 160 个全对，没有组内对比，却占约 42% 的
rollout（[ADR 0003](../../../decisions/0003-tau2-data-and-evaluation.md)）。换更弱的
agent（Qwen3-4B-2507）后，Telecom 有训练价值。

## Decision

Qwen3.5 的训练和评测只用 Airline 和 Retail（[training-setup.md §1](../training-setup.md)）。AReaL 的
Telecom 数据（271 题）单独筛选，建成与 Airline / Retail 分开的
pack（[training-data.md §6](../training-data.md)）。

## Consequences

- Telecom 对 Qwen3.5-4B 没有训练信号；是否对 Qwen3 有用取决于它在这份数据上的分数。
- Airline 的筛选 rubric 不能照搬到 Telecom（[training-data.md §5](../training-data.md)）。
