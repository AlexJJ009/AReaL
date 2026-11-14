# 训练和评测的用户模拟器用 DeepSeek-Flash

- 状态：早期默认已采纳（2026-10-10 起）；后续对照实验范围见下文，当前 API 约定以仓库 `AGENTS.md` 为准。

## Context

换用户模拟器会直接改变分数：2026-10-13 在 fixed-v3 test 上换五个用户模拟器，同一个 actor，总分 mean@4 从
0.579（Qwen3.5-4B 自部署）到 0.758（GLM 5.3）；Retail 上 DeepSeek 用户 0.806，Doubao 2.1 Turbo 用户
0.650，差距主要来自 Doubao 在确认的同一句话里结束对话。385 个失败里 168
个由用户造成（[user-simulator.md](../user-simulator.md)）。

## Decision

训练和评测都用 DeepSeek-Flash 做用户模拟器，key 从 `TAU2_DEEPSEEK_ENV_FILE`
读取（[training-setup.md §2](../training-setup.md)）。其他用户模拟器只用于 case study 和研究方向的对照实验。

## 适用范围与后续记录

原决定对应早期 fixed-v3 训练和评测，不是所有历史 run 的模型说明。
[10-24 实验记录](../experiments/2026-10-24-qwenuser-no-telecom.md)保存了自部署 Qwen3.5-4B 用户的 run
3′/2′/4/5 与 seed-43（10-30 结果），同时注明各 run 的 DeepSeek judge 后端差异。 这些历史对照不构成当前换模型的授权；当前只批准仓库
`AGENTS.md` 列出的 DeepSeek 模型和 endpoint。

## Consequences

- 不同用户模拟器下的分数不能直接比，报分必须注明用户模拟器。
- 与 Recuris
  论文和官方榜单的绝对分数有口径差（[recuris-and-leaderboard.md](../recuris-and-leaderboard.md)）。
