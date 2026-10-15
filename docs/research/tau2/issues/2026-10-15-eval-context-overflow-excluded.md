# 评测把 policy 超出上下文的 episode 当作基础设施错误排除（2026-10-15）

**状态**：已修（代码提交见下）。评测与训练一致：超出 32K 上下文的 episode 记 0
分并计入，不重试，单独报告条数（[ADR 0009](../../../decisions/0009-official-tau2-runner-semantics.md)）。

**现象**：SAO 534 的第 30、40 步评测分别有 17、25 条 episode（共 240 条）报
`context_length_exceeded`（`areal_context_limit: prompt_tokens=32836 exceeds max_total_tokens=32768`），被记为
`Tau2InfrastructureError` 并从 pass^k 中去掉，总体 pass^1 因此从 0.738、0.704 变成 0.785、0.775。

**原因**：评测照搬官方 `tau2 run` 的规则，provider 报错重试后排除。官方不设上下文窗口，不会遇到这种结局；我们为了成本把上下文限制在 32K，超出它是
policy 自己的失败。

**受影响的已有评测**（被排除的超长 episode 条数）：

| 评测                                                            | 条数        |
| --------------------------------------------------------------- | ----------- |
| `tau2-eval-20261010-fixed-v2/step0-base-t0-deepseek-fixed-v2`   | 2           |
| `tau2-eval-20261010/step0-base-t0-deepseek-fixed`               | 1           |
| `tau2-eval-20261011-areal-pack/step0-areal-pack-v1-t0-deepseek` | 1           |
| SAO 534 第 20 / 30 / 40 步                                      | 1 / 17 / 25 |

SAO 534 已按记 0 分重算（[sao-runs.md](../sao-runs.md)）。前三项每次最多 2 条，对结论没有影响，不重算。critic-only
训练本来就把这类 episode 记 0 分保留，验证集没有出现，不受影响。
