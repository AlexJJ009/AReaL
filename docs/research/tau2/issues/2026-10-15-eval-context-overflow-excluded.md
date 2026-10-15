# 评测把 policy 超出上下文的 episode 重试并按首个完成的 attempt 计分（2026-10-15）

**状态**：已修（`5df3356f`），已有评测已用 `official_metrics` 离线重算。评测与训练一致：第一次超出 32K 上下文的 episode 记 0
分并计入，不重试，单独报告条数（[ADR 0009](../../../decisions/0009-official-tau2-runner-semantics.md)）。

**现象**：SAO 534 的第 30、40 步评测分别有 17、25 条 episode 报
`context_length_exceeded`（`areal_context_limit: prompt_tokens=32836 exceeds max_total_tokens=32768`），被记为
`Tau2InfrastructureError` 并从 pass^k 中去掉，总体 pass^1 因此偏高。进一步核查发现问题比这更广：修复前，超出上下文的 episode
会重新跑（最多 4 次尝试），按第一个跑完的 attempt 计分，只有 4 次都超出才排除。重试会偏向短轨迹，所以所有修复前的评测都偏高。

**原因**：评测照搬官方 `tau2 run` 的规则，provider 报错重试后排除。官方不设上下文窗口，不会遇到这种结局；我们为了成本把上下文限制在 32K，超出它是
policy 自己的失败。

**重算**：`scripts/rescore/rescore.py <轨迹目录>`（用仓库自己的 `official_metrics`）按"取第一个完成的 attempt 的
`official_score`；在此之前先遇到 `context_length_exceeded` 的 attempt 则记 0"重算；配对 bootstrap 用
`scripts/rescore/boot.py`。Qwen3-2507 的运行几乎没有超出上下文，不受影响。

| 评测（Qwen3.5-4B step 0，除注明）   | 超出上下文的 episode  | 总体 mean@4 修复前 → 严格                     |
| ----------------------------------- | --------------------- | --------------------------------------------- |
| 五模拟器 v3：GLM 5.3                | 1                     | 0.758 → 0.758                                 |
| 五模拟器 v3：DeepSeek-Flash         | 6                     | 0.721 → 0.713                                 |
| 五模拟器 v3：Doubao 2.1 Lite        | 3                     | 0.700 → 0.696                                 |
| 五模拟器 v3：Doubao 2.1 Turbo       | 4                     | 0.633 → 0.625                                 |
| 五模拟器 v3：Qwen3.5-4B 自部署用户  | 3                     | 0.583 → 0.579                                 |
| official（10-09，DeepSeek 用户）    | 17                    | 0.838 → 0.817                                 |
| official（10-10）                   | 15                    | 0.808 → 0.793                                 |
| fixed v1                            | 26                    | 0.690 → 0.662                                 |
| fixed v2                            | 16                    | 0.708 → 0.694                                 |
| AReaL pack v1 / v2（813 / 648 题）  | 44 / 20               | 0.912 → 0.903 / 0.946 → 0.942                 |
| SAO 534 第 0 / 10 / 20 / 30 / 40 步 | 4 / 17 / 32 / 59 / 76 | 第 0 步 0.708 → 0.696，第 40 步 0.704 → 0.583 |

SAO 534 的结论因此改变：训练让对话变长，超出 32K 的 episode 增多，Airline
的下降由它驱动（[sao-runs.md](../sao-runs.md)）。其余评测的变化在 1–3pp，主要影响 Airline（Airline
对话更长）。critic-only 训练本来就把这类 episode 记 0 分保留，验证集没有出现，不受影响。
