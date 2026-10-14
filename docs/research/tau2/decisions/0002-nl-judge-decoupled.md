# NL judge 与用户模拟器分开配置，固定为 DeepSeek-Flash

- 状态：已采纳（`9e7082dc`，2026-10-13）

## Context

评测中换用户模拟器时，NL judge 如果跟着换，分数差里就混入了 judge 的差异。官方评测用 gpt-4.1（temperature 0）做 NL judge；我们用
DeepSeek-Flash，只影响含 NL_ASSERTION
的任务（[benchmark-variants.md §3.2](../benchmark-variants.md)）。

## Decision

NL judge 单独配置，固定为 DeepSeek-Flash，不随用户模拟器变化。critic 的生产配置 `config_critic_production.yaml`
显式写了两者（[critic/critic0-expanded-data.md](../critic/critic0-expanded-data.md)）。

## Consequences

- 比较不同用户模拟器时，judge 一致，差异只来自用户一侧。
- `9e7082dc` 之前两者没有分开配置，结果不可与之后的混比。
