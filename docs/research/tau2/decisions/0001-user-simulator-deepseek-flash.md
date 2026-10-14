# 训练和评测的用户模拟器用 DeepSeek-Flash

- 状态：已采纳

## Context

换用户模拟器会直接改变分数：同一份代码、同一个 actor，DeepSeek 用户下 Retail 0.851，Doubao 用户下 0.651，差距主要来自 Doubao
在确认的同一句话里结束对话（[recuris-and-leaderboard.md](../recuris-and-leaderboard.md)）。2026-10-13 在
fixed-v3 test 上换五个用户模拟器，总分 mean@4 从 0.583（Qwen3.5-4B 自部署）到 0.758（GLM 5.3），385 个失败里 168
个由用户造成（[user-simulator.md §2](../user-simulator.md)）。

## Decision

训练和评测都用 DeepSeek-Flash 做用户模拟器，key 从 `TAU2_DEEPSEEK_ENV_FILE`
读取（[training-setup.md §2](../training-setup.md)）。其他用户模拟器只用于 case study 和研究方向的对照实验。

## Consequences

- 不同用户模拟器下的分数不能直接比，报分必须注明用户模拟器。
- 与 Recuris
  论文和官方榜单的绝对分数有口径差（[recuris-and-leaderboard.md](../recuris-and-leaderboard.md)）。
- 用 gpt-5.2（reasoning low）做用户与榜单定标一次的计划还没有执行，见 [README](../README.md) 的"仍未完成"。
