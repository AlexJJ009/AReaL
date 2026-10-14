# critic head：w 零初始化、b 初始化为训练集均值，同一学习率

- 状态：已采纳（2026-10-13，`7da3a4f3`、`275548de`）

## Context

Qwen3.5 的 critic head 是无 bias 的线性层，权重按 std 0.02 初始化，第 0 步输出约 $\pm 3$，MSE 约 10。reward 只有
0/1，目标必然在 $[0,1]$ 内，无界线性 head 是沿用的默认（[critic/value-head.md](../critic/value-head.md)）。

## Decision

同时改两处：$w$ 零初始化，$b$ 初始化为训练集均值。head 和骨干用同一个学习率 5e-6，不单独设（`critic_head_lr`
字段保留，正式实验不设）。另起入口 `scripts/tau2/run.sh data critic-head`，`areal/api/cli_args.py`
和原训练路径不变。

## Consequences

- 只加 bias 不够：$b$ 会被 $\pm 3$ 的噪声盖住。
- Adam 每个参数每步大约移动 lr，$b$ 只有 1 个参数，靠大学习率修不了整体水平，所以起点靠初始化修好。
- 若仍漂，备选是 sigmoid + BCE。
