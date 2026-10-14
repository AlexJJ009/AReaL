# 暂停方法草案（原 16.11），先找准现象

- 状态：已采纳（2026-10-12）

## Context

方法草案给 advantage 乘按用户估计的逆方差权重。Codex 的 ICML
标准审稿指出核心估计量有设计错误（S1、S2）：代码改动简单而数学过重，理论没有指导设计（[research-direction/appendix-math-review-codex.md](../research-direction/appendix-math-review-codex.md)）。现象证据目前只有两个模拟器、一个
benchmark。

## Decision

暂停
[appendix-method-draft-16.11.md](../research-direction/appendix-method-draft-16.11.md)
的方法推导。先换更多用户模拟器和 benchmark、做轨迹 case study，找准现象再定方法；轻量候选（用户回复前后各读一次
critic）与现象研究并行（[research-direction/problem-and-plan.md §3](../research-direction/problem-and-plan.md)）。

## Consequences

- 草案作为附录保留，不再往下推导。
- LLM RL 论文靠"现象 + 简单修正 + 消融"说服人，因此下一步按
  [problem-and-plan.md §1](../research-direction/problem-and-plan.md) 的 X1–X5
  补"伤害训练"这一环。
