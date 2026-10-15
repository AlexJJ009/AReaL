# 转人工规则：Retail 中 gold 有写操作的题设例外，Airline 不设

- 状态：已采纳（fixed v1 起，2026-10-10）

## Context

官方评分对"应该拒绝"的任务，agent 转人工也能拿满分，official 评分下约 31.5% 的 Airline episode
是靠这一条拿满分（[failure-taxonomy.md](../failure-taxonomy.md)）；official 到 fixed 的 Airline 总差距约
39pp（[benchmark-variants.md §1.3](../benchmark-variants.md)）。加两条规则后，Retail 有 17 条原本成功的
episode 被清零，其中 13 条在有 gold 写操作的题上；抽读到的都是写操作完成后用户提出 agent 做不了的要求，agent 才转人工，符合 Retail
policy 的"超出能力范围时转人工"。Airline 被清零 59
条，抽读到的是在回避拒绝（[benchmark-variants.md §1.3](../benchmark-variants.md)）。

## Decision

两条规则都在官方 evaluator 之后执行：agent 转了人工而 gold 没有转，记 0 分，例外是 Retail 中 gold 有写操作的题；gold 有转人工而
agent 没有转，记 0 分。Airline 不设领域级例外。

## Consequences

- 两个领域的规则不一致；例外的依据是抽读，不是 evaluator 能检查的条件。
- v3 核查纠正了 airline:37（NQNU5R 已飞，policy 要求转人工，补 gold）。
- 变体定义本身见
  [ADR 0010](../../../decisions/0010-tau2-official-and-fixed-benchmark-variants.md)。
