# SAO 使用 critic 510 第 55 步的 checkpoint（看 test 挑的）

- 状态：已采纳（用户，2026-10-14）；SAO 运行中

## Context

510 最后一步的验证 EV 回到约 0（Airline $-0.005$、Retail 0.001），第 55 步为 Airline 0.179、Retail
0.016。critic 的验证集就是 SAO 报分的 fixed-v3 test 60 题（[sao-runs.md](../sao-runs.md)）。

## Decision

SAO 用 510（`critic-head-qwen35-mix-r2`）第 55 步的 checkpoint，lr 5e-6，训练题为 mix pack 221 题加官方
fixed train 99 题；不另跑 v2 的 SAO，也不跑 GRPO 对照。

## Consequences

- **这是用 test 选模型**：SAO 的 test 分数会偏乐观，报告时必须注明；第 55 步的 0.179 本身误差约
  $\pm 0.1$（[critic/ev-case-study.md](../critic/ev-case-study.md)）。
- 选 mix 而不是 v2：v2 里约七成题对 Qwen3.5 几乎全对，advantage 接近 0，rollout 预算大多浪费。
