# 评测协议：Airline + Retail，temperature 0，每题 4 次，报 mean@4 与 pass^k

- 状态：早期 60 题协议已采纳（2026-10-10）；后续 PPO 100 题协议已扩展范围，见下文。

## Context

旧口径是 temperature 1、每题 8 次、含 Telecom；训练前 Telecom 就是 320/320 全对，占了约 42% 的
rollout（[initial-audit.md §4](../initial-audit.md)）。

## Decision

评测 temperature 0（训练保持 1），每题 4 次，并发改为 8 组 x 4 次，保持 32 条在途；只评 Airline 和 Retail，题集用官方 test
60 题（Airline 20、Retail 40）。mean@4 等于 pass^1，另报 pass^2、pass^4。

## 后续适用范围

60 题是早期 Airline/Retail
协议。[10-24 实验记录 §11–12、§17–18](../experiments/2026-10-24-qwenuser-no-telecom.md) 区分了
DeepSeek 用户下含 Telecom 的 100 题评测和 Qwen 用户下的 60 题评测，并记录完成结果。 本地
`examples/tau2/ppo_data.py:build_ppo_eval_dataset` 固定使用 20/40/40 共 100 题。 每题 4 次、mean@4
等指标定义保留；比较分数时仍须匹配题集、用户、judge 和 infra 剔除口径。

## Consequences

- 历史结果保持原口径（temperature 1、N=8、含 Telecom），不与新口径混比。
- 训练只在 train 上做，test 只记录，不用来选 checkpoint；base 集包含 train，训练后的 base 分数不是 held-out
  结果（[training-setup.md §1](../training-setup.md)）。
