# 评测协议：Airline + Retail，temperature 0，每题 4 次，报 mean@4 与 pass^k

- 状态：已采纳（2026-10-10）

## Context

旧口径是 temperature 1、每题 8 次、含 Telecom；训练前 Telecom 就是 320/320 全对，占了约 42% 的
rollout（[initial-audit.md §4](../initial-audit.md)）。

## Decision

评测 temperature 0（训练保持 1），每题 4 次，并发改为 8 组 x 4 次，保持 32 条在途；只评 Airline 和 Retail，题集用官方 test
60 题（Airline 20、Retail 40）。mean@4 等于 pass^1，另报 pass^2、pass^4。

## Consequences

- 历史结果保持原口径（temperature 1、N=8、含 Telecom），不与新口径混比。
- 训练只在 train 上做，test 只记录，不用来选 checkpoint；base 集包含 train，训练后的 base 分数不是 held-out
  结果（[training-setup.md §1](../training-setup.md)）。
