# 评测把 policy 超出上下文的 episode 当作基础设施错误排除（2026-10-15）

**状态**：未修，待决定。

**现象**：SAO 534 的第 30、40 步评测分别有 17、25 条 episode（共 240 条）报
`context_length_exceeded`（`areal_context_limit: prompt_tokens=32836 exceeds max_total_tokens=32768`），全部被记为
`Tau2InfrastructureError`，从 pass^k 中去掉。第 0–20 步几乎没有。不计这些 episode 时总体 pass^1 为
0.785、0.775；按失败计为 0.738、0.704。

**代码位置**：`examples/tau2/official_runner.py` 的 `official_metrics` 只用
`status == COMPLETED` 的记录计算 pass^k，`INFRASTRUCTURE_ERROR` 只计数。超出上下文的 400 错误在
`examples/tau2/agent.py` 里被包成 `Tau2InfrastructureError`。

**为什么是问题**：对话超出 32K 是 policy 的行为（循环、重复调用等），不是 API 或服务故障。排除它们会让训练后变长的 policy
显得更好。训练侧已经按失败处理：上下文超限和墙钟超时记 0
分（[training-setup.md §3](../training-setup.md)，[ADR 0011](../../../decisions/0011-tau2-lost-episodes-in-training.md)）。

**待查**：官方 tau2 在 agent 侧报上下文超限时是否记 0 分；改为记失败后，已有评测是否需要重算（可以离线重算，见
[sao-runs.md](../sao-runs.md) 的结果一节）。
