# critic 训练数据配比：简单题降采样，保留全部有信号题

- 状态：已采纳（2026-10-13）

## Context

479 的 1494 条 episode 里约七成花在简单题上（6 次全对），每条约 11K DeepSeek prompt token；简单题所有 token 的目标都是
1，只能教 critic 整体水平，而整体水平已交给 head 的 bias（[critic/data-mix.md](../critic/data-mix.md)）。

## Decision

降采样，不加权，不整题删：有信号的 122 道全部保留，简单题按（领域，keep/fix）分层随机抽 99 道（种子 20261013），再加官方 fixed train 99
道，共 320 行，batch 16 下每个 epoch 正好 20 步。pack 题和官方题混在一起，每个 epoch
重新打乱（`train_dataset.shuffle=true`）。Instruct 的 critic 不调。

## Consequences

- 加权仍要把 rollout 跑完，费用不变，信息量也不变，所以不加权。
- SAO 训练时 actor 仍会遇到简单题，critic 要对它们预测接近 1，所以保留一小部分。
- 479 是 pack 在前、官方在后、每个 epoch 同一顺序，最后 7 个 batch 全是官方题；打乱后消除了这一点。
