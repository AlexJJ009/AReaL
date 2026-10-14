# EV 上限实验

> 来源：原 README 第 16.14.4 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

**问题**：B 是 critic 训练没做好，还是 user simulator 带来的随机性本来就预测不了？

**做法**：fixed test 60 题，每题跑 $k=4$ 次。对题 $i$ 的成功率 $\hat p_i$：

$$
T = \bar p (1 - \bar p), \qquad
W = \frac{1}{n} \sum_i \frac{k}{k-1} \hat p_i (1 - \hat p_i), \qquad
\text{上限} = \frac{T - W}{T}.
$$

$T$ 是单条 episode reward 的总方差，$W$ 是同一道题内部的方差（无偏估计）。critic 在第一个 agent token 只知道是哪道题，最好也只能预测
$p_i$，所以第一个 token 的 EV 不超过"题间方差占比"。后面的前缀会透露更多信息，所以这是第一个 token 的上限，不是按 token 平均的 EV
的上限：上限高，说明 B 至少有题目层面的信号可学；上限低，不能单凭它说明 B 学不会。95% 区间按题 bootstrap。

**与 critic
对照（Qwen3.5）**：475（`user-sim-20261012/api-user/deepseek-test-u0p1`，另一个会话的用户模拟器实验）的采样与
critic 验证逐项一致：policy temperature 1、DeepSeek-Flash user temperature 0、NL judge
temperature 0、thinking 关闭、同样的 parser（解析后的配置只差训练用的 `n_samples`），两次提交之间 rollout 代码没有改动，240
条全部完成：

|         | 题数 | $\bar p$ | 总方差 | 题内方差 | 上限 \[95%\]        | critic 第 94 步 EV | critic 预测与 $\hat p_i$ 的相关 |
| ------- | ---- | -------- | ------ | -------- | ------------------- | ------------------ | ------------------------------- |
| 总体    | 60   | 0.713    | 0.205  | 0.104    | 0.49 \[0.31, 0.64\] | +0.15              | 0.58                            |
| Airline | 20   | 0.463    | 0.249  | 0.096    | 0.61 \[0.34, 0.85\] | +0.04              | 0.70                            |
| Retail  | 40   | 0.838    | 0.136  | 0.108    | 0.20 \[0.03, 0.35\] | −0.08              | 0.10                            |

- Airline 的上限远高于 critic 的实际 EV，而 critic 预测与题目成功率已经相关 0.70：Airline 有题目层面的信号，critic
  只学到了一部分。缺的不是 A（EV 已扣掉整体偏移），而是预测幅度太小（[ev-case-study.md](ev-case-study.md)）。
- Retail 一半以上的方差在题内（0.108 / 0.136），上限只有 0.20，critic 与题目难度几乎不相关。

**不同用户模拟器、不同温度**（Qwen3.5 actor，fixed test 60 题 × 4；后三行来自另一个会话的 473、476 和 474）：

| 用户模拟器                 | policy T | user T | 题内方差 | 总体上限 \[95%\]    | Airline | Retail |
| -------------------------- | -------- | ------ | -------- | ------------------- | ------- | ------ |
| DeepSeek-Flash（475）      | 1        | 0      | 0.104    | 0.49 \[0.31, 0.64\] | 0.61    | 0.20   |
| DeepSeek-Flash（474）      | 0        | 1      | 0.118    | 0.41 \[0.22, 0.57\] | 0.57    | 0.17   |
| DeepSeek-Flash（473）      | 0        | 0      | 0.124    | 0.39 \[0.19, 0.54\] | 0.55    | 0.11   |
| 豆包 Seed 2.1 Turbo（476） | 0        | 0      | 0.175    | 0.29 \[0.14, 0.44\] | 0.48    | 0.17   |

policy 用 greedy 解码、user temperature 0 时，题内方差没有变小（0.124 对
0.104）；换成豆包当用户，题内方差更大、上限更低。题内的随机性主要不来自 actor
采样，而来自用户一侧，且用户模拟器不同、噪声大小不同。这是论文"现象"部分的候选证据。保留：SGLang 的 greedy 解码在不同 batch
组合下不逐位确定，policy 一侧可能贡献一部分；要分开两者，需要一个输出可固定的自部署用户做对照。

**待补（不花 GPU 训练，每组 240 条、2–3M token）**：

- Instruct：pueue
  483（`tau2-training-20261012-critic-data/ceiling/launch_ceiling.sh ceiling-instruct-test-x4 instruct`，副本在
  `../scripts/critic-data`），按 475 的协议，parser 为 `qwen25`，固定在 `src-04be68ab`；check-config
  与 Instruct critic 验证逐项对比，只差 `n_samples`（evaluate 强制为 1）。
- DeepSeek 关闭 thinking：另一个会话的 481（T0/T0）和 482（T1/T0），跑完后直接计算。
- 豆包 T1/T0、自部署用户对照：尚未排队。

分析脚本：`../scripts/critic-data/ev_ceiling.py EPISODES [VALIDATION_REPORT]`。
