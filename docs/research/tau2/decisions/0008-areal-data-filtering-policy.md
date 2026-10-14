# AReaL-tau2-data 的筛选策略：只留 reward 正确的题，只允许指令侧小修

- 状态：已采纳（用户，2026-10-10）

## Context

AReaL-tau2-data 的 Airline / Retail 共 1711 题；机械检查没发现问题的 1243 题里，只有 424 题原样 keep，其余 819 题被
fix 或 drop，机械检查漏掉的问题远多于查出的（[training-data.md §2、§3](../training-data.md)）。

## Decision

从 1711 题里筛出 reward 正确的子集用于训练。每题的结论只能是 keep、fix、drop，并附证据。fix
只允许两种改法：给用户指令补一句用户本来就知道或能决定的信息，或在指令明确时替换 gold；超过约两处小改动就 drop，没有把握时也 drop。不允许把题修成 gold 不写
DB 的拒绝题。每个 keep 和 fix 都用 `score.py` 重放（gold 得 1 分、空轨迹得 0 分、gold 写操作不报错、去掉任一 gold 写操作得
0），并用评测证明可用。

## Consequences

- "评测通过不等于题对"：v1 里模板套用的 19 题有 13 题 4/4，说明 4/4 只证明 gold
  可达（[training-data.md §4](../training-data.md)）。
- 现行流程见 [training-data.md §5](../training-data.md)。
