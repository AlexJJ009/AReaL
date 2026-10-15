# critic 不准时 PPO 能不能训：调研与判断

| 项       | 内容                                                                                                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 日期     | 2026-10-15                                                                                                                                                                                                   |
| 作者     | Claude（文献由子 agent 读原文核对，见第 5 节"核对程度"一列）                                                                                                                                                 |
| 触发     | Qwen3-4B-Instruct 的 critic 离线预实验（[protocol](../experiments/2026-10-15-critic-offline-preexp-qwen3.md)）显示冻结特征上没有线性信号；担心 critic 训不好，迟迟不敢启动 PPO                               |
| 相关文档 | [ev-ceiling.md](ev-ceiling.md)（按题目身份的 EV 上限）、[ev-ceiling-explainer.md](ev-ceiling-explainer.md)、[value-head.md](value-head.md)、[ADR 0012](../decisions/0012-critic-head-init.md)（head 初始化） |

## 1. 要回答的问题

1. critic 预测不准（EV 接近 0 甚至为负）时，PPO 会怎样？会不会训坏？
1. 同行在 LLM 上怎么训练 critic？有哪些已知的坑？
1. 在多轮工具调用（tau2）里，critic 实际能学到什么？
1. 我们观察到的现象是不是前人已经报告过？还有没有研究缺口？

## 2. 结论

- **critic 不准不会让梯度方向出错，但可能比不用 critic 更慢。** 在 $\gamma=\lambda=1$ 时，critic 只充当
  baseline，策略梯度仍然无偏。EV 正好衡量 critic 相对"最优常数 baseline"减少了多少优势方差。EV ≤ 0
  意味着它不但没减方差，反而增加了方差（第 3 节）。
- **我们的离线预实验测的是跨题泛化，PPO 用到的是训练分布。** 验证集 60 题与训练的 747 题不重叠，EV ≈ 0 只说明"冻结特征上的线性 head
  推断不出新题的难度"。PPO 每个 epoch 都会重新遇到同样的 747 题，critic 记住每道题的难度就已经是一个有用的 baseline（第 4 节）。
- **"critic 不准但 PPO 仍能工作"大体是已知结论，我们属于重新发现。** 已有的相关结论包括：value 网络拟合真值很差却仍有用（Ilyas
  2020）；PPO 的 value 网络给推理步骤排序接近随机（VinePPO）；EV ≤ 0 的 critic 会增大方差（EVPO）；critic 需要预训练或
  warmup（VC-PPO、VAPO、T1）。见第 5 节。
- **多轮工具调用场景里可能有缺口，前提是 PPO 先跑通。** 子 agent 没找到任何用带 critic 的 PPO 训练 tau-bench / tau2-bench
  的工作。它也没找到在多轮工具调用上区分"新题 EV"和"重复出现的训练题 EV"的分析，或者分析 critic 学的是题目难度还是对话中途状态价值的工作（第 8 节）。
- **对当前 PPO 的做法：** 不再等 critic 预训练，先跑 PPO。训练中把 critic 更新前在训练 batch 上的 EV 当作首要诊断指标（第 9 节）。

## 3. 背景：EV、baseline 与我们的设置

### 3.1 EV 的定义和含义

记 $R$ 为一条轨迹的最终 0/1 奖励，$V$ 为 critic 对该轨迹某个前缀的预测。按 token 汇总：

$$
\mathrm{EV} = 1 - \frac{\operatorname{Var}(R - V)}{\operatorname{Var}(R)} .
$$

当 $V$ 是任意常数 $c$ 时，$\operatorname{Var}(R-c)=\operatorname{Var}(R)$，所以 EV = 0。所以：

- EV > 0：critic 比"所有前缀都预测同一个数"更好，优势的方差变小了；
- EV = 0：critic 等价于常数 baseline；
- EV \< 0：critic 在不同 token 之间的预测波动与成败无关，额外给优势加了方差。

EV 对整体偏移不敏感：预测整体偏高 0.2，只要偏移是常数，EV 不变。这一点与 verl 的 `vf_explained_var` 口径一致。

不同论文里"EV"的参照目标不同。有的用 GAE 自举出来的 return，有的用最终 outcome 奖励。前者在训练不稳定时也可能接近 1，所以只有后者才能说明
critic 预测成败的能力（Best Practice Critic Optimization，见第 5 节）。我们用的都是最终奖励口径。

### 3.2 $\gamma=\lambda=1$ 时 critic 的作用

我们的 PPO 用 $\gamma=\lambda=1$，此时 GAE 化简为

$$
A_t = R - V(s_t),
$$

critic 只充当 baseline，不参与自举。由于 baseline 只依赖状态、不依赖动作，策略梯度的期望不变（无偏）；critic 的好坏只影响方差。加上 batch
级的优势白化后：

- critic 退化为常数时，PPO 等价于"REINFORCE + batch 均值 baseline"。ReMax、RLOO 一类方法证明这种做法本身能训。
- critic 在同一条轨迹内随机波动时，EV \< 0，方差比常数 baseline 更大。这时 PPO 不会被带偏方向，但学得更慢；严重时可能比不用 critic
  还差（EVPO 的观察）。

这修正了我此前在对话里的一个说法："critic 不准不会把 PPO 训坏"。无偏的部分没错，但"不会变差"说得过满：EV ≤ 0 的 critic 确实会让训练比常数
baseline 更差。

还要说明一点：策略梯度的方差还乘上了 score function，不完全等于优势的方差。EV 只是一阶指标，但它是同行普遍采用的判据（EVPO 直接用它决定是否启用
critic）。

### 3.3 $\lambda<1$ 时的风险（我们没有用）

$\lambda<1$ 时优势依赖 $V$ 的自举，critic 的系统性偏差会沿轨迹累积成优势的偏差。VC-PPO 报告，用 reward model 初始化的 critic
存在正向偏差，在长 CoT 上导致输出长度崩溃。我们用 $\lambda=1$，没有这个问题；但如果以后改成 VAPO 式的解耦 λ（actor 用
0.95），这个风险会回来。

## 4. 我们自己的证据

### 4.1 离线预实验（冻结骨干网络，只看 head）

Qwen3-4B-Instruct（protocol `2026-10-15-critic-offline-preexp-qwen3`，结果目录
`results-0f8f6aa0`）。验证集为 60 条轨迹，题目与训练集不重叠：

| 实验 | 内容                                          | Airline EV     | Retail EV      | 判定                                 |
| ---- | --------------------------------------------- | -------------- | -------------- | ------------------------------------ |
| E2   | 最后一层特征，ridge 共享探针                  | −0.13          | −0.04          | Q1（存在线性信号）**不成立**。       |
| E3   | 只训 head 的模拟，lr 5e-6 到 1e-3             | −0.02 到 −0.43 | −0.00 到 −0.24 | lr 越大越差；Q2 不成立，Q3 不确定    |
| E7   | 第 9 / 18 / 27 层，480 训完的骨干，按位置分组 | 全部 ≤ 0.04    | 全部 ≤ 0.04    | Q4 按判据成立但 EV 全为负；Q5 不确定 |

AUC 为 0.58（Airline）和 0.65（Retail），说明能粗略区分成败，但还不足以解释方差。

Qwen3.5-4B 的同类预实验（E2）：Airline 的 ridge EV 约 0.09，区间包含 0；Retail 约 0。

### 4.2 EV 上限：按题目身份能解释多少

[ev-ceiling.md](ev-ceiling.md) 在 fixed test 上每题跑 4 次，估计"只知道是哪道题"时 EV 的上限（Qwen3.5）：总体
0.49，Airline 0.61，Retail 0.20。剩下的方差来自同一道题内部，主要是 user simulator 和策略采样的随机性。

这给出了一个参照：\*\*critic 只要记住每道题的成功率，就能拿到相当一部分 EV。\*\*但这要求 critic 知道每道题的难度。在训练题上可以靠多个 epoch
记住；在新题上只能从文本推断，而离线预实验显示冻结特征推断不出来。

### 4.3 离线验证集与 PPO 实际所需之间的差别

| 维度       | 离线预实验的验证集   | PPO 实际用到的 baseline                  |
| ---------- | -------------------- | ---------------------------------------- |
| 题目       | 60 道新题            | 747 道训练题，每个 epoch 都会重新出现    |
| 需要的能力 | 从文本推断新题的难度 | 记住见过的题的难度，并追踪当前策略的变化 |
| 训练方式   | 冻结骨干，只训 head  | 全参数，在线训练，样本随策略变化         |
| 样本量     | 60 条                | 每步 128 条，48 步共 5976 条             |

因此离线结论要收窄为：\*\*冻结的 Qwen3 特征 + 线性 head 不能泛化到新题。\*\*它推不出"PPO 里的 critic 没用"。VinePPO
的结果支持这一区分：critic 在训练分布上的 EV 有 0.7–0.9，但给新的中间状态排序接近随机水平（第 5 节）。

### 4.4 PPO smoke 的训练 batch EV

`ppo-smoke-qwen3-v2` 的前两步：critic 更新前在训练 batch 上的 EV 为 −3e-9 和 0.003。head 按 ADR 0012
初始化（$w=0$），起步时预测是常数，所以 EV ≈ 0 是预期结果，还说明不了 critic 的学习能力。

## 5. 文献

"核对程度"一列说明这一条是怎么核对的。子 agent 用 pdftotext 读了 arXiv PDF 全文；标"摘要"的只看了搜索结果，没有读原文。2026 年的几篇论文
Claude 没有独立读过，引用前应再核对一次原文。

| 工作                                                                                   | 场景                               | 与 critic 相关的发现                                                                                                                                                                                                     | 核对程度                          |
| -------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| Ilyas et al., "A Closer Look at Deep Policy Gradients", ICLR 2020                      | MuJoCo（Walker2d）                 | value 网络能拟合自己的监督目标，但与真实 value 相差约 50%；相比真值 baseline，它只略微降低梯度方差，但比不用 baseline 好得多（不用 baseline 时 reward 差近一个数量级）。这里的"held-out"指同一任务内的新状态，不是新任务 | 原文 Fig. 3–4                     |
| Kazemnejad et al., VinePPO, ICML 2024                                                  | DeepSeekMath-7B，MATH              | PPO 的 value 网络给 5 个候选下一步排序，大部分训练时间接近随机；越往推理链后段误差越大，作者推测前段与训练数据相似、被记住了。critic 在训练分布上的 EV 为 0.7–0.9。用蒙特卡洛估计 value 后效果更好                       | 原文 §6–7，Fig. 8、D.5、D.12      |
| Yuan et al., VC-PPO, arXiv 2503.01491, 2025                                            | Qwen-32B，长 CoT，AIME             | 用 reward model 初始化的 critic 带正向偏差，导致长度崩溃。冻结 policy、λ=1 预训练 value 约 100 步后，AIME pass@1 从 5.6 提到 41.9（GRPO 为 35.8）                                                                        | 原文 §3.2，Table 2–3              |
| VAPO, arXiv 2504.05118, 2025                                                           | AIME24                             | 消融：vanilla PPO 5；去掉 value 预训练 11；去掉解耦 GAE 33；去掉长度自适应 GAE 45；完整 VAPO 60。critic warmup 50 步，critic λ=1.0，actor λ=0.95                                                                         | 原文 Table 1                      |
| Zheng et al., Secrets of RLHF Part I, arXiv 2307.04964, 2023                           | RLHF                               | critic 从 RM 或从 SFT 初始化结果相近；先只训 value head 到 loss 接近 0 再开始 PPO，训练更稳。他们建议用 critic 预训练代替学习率 warmup                                                                                   | 原文 §5.3.3，Fig. 8               |
| Ahmadian et al., Back to Basics, arXiv 2402.14740, 2024                                | RLHF，单轮                         | 带 baseline 的 REINFORCE 和 RLOO 稳定优于 PPO，作者认为在 RLHF 中不需要给不完整的回复建模 value                                                                                                                          | 原文 §5                           |
| ReMax（2310.10505）、GRPO（2402.03300）                                                | 单轮                               | 去掉 value 模型                                                                                                                                                                                                          | 只确认了"去掉 critic"，没有读细节 |
| EVPO（Pan et al.），arXiv 2604.19485                                                   | WebShop、Sokoban、FrozenLake、MATH | 把 batch EV 当作卡尔曼式的判据：EV ≤ 0 时 critic 增大优势方差，此时退回 batch 均值 baseline。Sokoban 早期 EV 为负；往收敛的 critic 里加噪声会损害性能（Fig. 2）。没找到新任务上的划分                                    | 原文，Fig. 2                      |
| T1，arXiv 2609.11042                                                                   | 终端 agent，PPO                    | 冷启动 critic 的 EV 最低 −33.6，58 个记录点中 30 个为负（Fig. 7）；有 critic warmup 时 EV 在 0.71–0.86。一个 27B 的 run 改了 critic 学习率后，EV 从 −39 变成 +0.11                                                       | 原文 Fig. 7                       |
| SPPO，arXiv 2604.08865                                                                 | 200 个 held-out prompt             | value 预测与经验通过率的 Pearson 相关 0.642、Spearman 0.664；难题上 critic 预测约 0.5                                                                                                                                    | 原文                              |
| Best Practice Critic Optimization，arXiv 2608.23566                                    | —                                  | 以自举目标计算的 EV 在训练不稳定时也接近 1；只有以 outcome 计算的 EV 有意义                                                                                                                                              | 原文                              |
| Le Critique，arXiv 2608.16739                                                          | 多环境                             | 普通 value function 的 EV 约 0.03–0.6；作者把 critic 与组均值 baseline 的对比理解为"估计 prompt 难度"。图中文字乱码，具体数值没读出                                                                                      | 原文（图乱码）                    |
| GenAC，arXiv 2604.10701                                                                | AIME24                             | 在分布外 prompt 上，生成式 critic 的 MSE 比判别式 critic 低 52.5%                                                                                                                                                        | 原文                              |
| ArCHer（2402.19446）、SWEET-RL（2503.15478）                                           | 多轮 agent                         | 分别用 utterance 级 TD critic 和带特权信息的 turn 级 critic（Bradley-Terry 目标）；都没有报告 EV，也没有分析新任务                                                                                                       | 原文                              |
| SPA-RL（2505.20732）、RAGEN（2504.20073）、GiGPO（2505.10978）                         | 多轮 agent                         | SPA-RL 断言稀疏奖励下 PPO 的 value 学习会退化，但没有测量；RAGEN 发现 PPO 在 Bandit、Sokoban 上比 GRPO 稳，在 FrozenLake、WebShop 上不如 GRPO；GiGPO 不用 critic                                                         | 原文（GiGPO 只确认了无 critic）   |
| tau-bench RL 相关：Fission-GRPO、MUA-RL、verl-recipe tau2_airline、SENTINEL、ToolVerse | tau-bench / tau2                   | 全部是不用 critic 的方法，没找到带 critic 的 PPO                                                                                                                                                                         | 摘要                              |
| verl / OpenRLHF 的 `vf_explained_var`                                                  | —                                  | 没找到权威的公开数值。一篇社区文章（Qwen2.5-0.5B，GSM8K）报告约 0.04–0.40                                                                                                                                                | 摘要                              |

## 6. 同行怎么训练 critic，和我们的选择对照

| 做法                                 | 出处                                            | 我们当前的选择                                                       | 备注                                                                     |
| ------------------------------------ | ----------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 从 RM 或 SFT / policy 初始化 critic  | InstructGPT 一类；Secrets（两者相近）           | 从 policy 初始化，head 用 $w=0$、$b$ = 训练成功率 0.6205（ADR 0012） | 不存在"RM 偏差"问题                                                      |
| 冻结 policy，先预训练 / warmup value | VC-PPO（约 100 步）、VAPO（50 步）、Secrets、T1 | **不做 warmup**（用户决定，2026-10-15）                              | 文献普遍建议 warmup；这是当前设置里最大的风险点，由第 9 节的 EV 监控兜底 |
| critic λ=1，actor λ\<1（解耦 GAE）   | VC-PPO、VAPO                                    | actor 和 critic 都用 λ=1                                             | 不引入自举偏差                                                           |
| 调 critic 学习率                     | T1（27B，EV 从 −39 到 +0.11）                   | critic lr 1e-5，全参数，不冻结 attention                             | Qwen3 离线 E3 不支持调高 head lr                                         |
| EV ≤ 0 时退回 batch 均值 baseline    | EVPO                                            | 没有实现                                                             | 备选方案                                                                 |
| 用蒙特卡洛估计代替 value 网络        | VinePPO                                         | 没有实现                                                             | tau2 的 rollout 要调用 user simulator API，MC 代价很高                   |
| turn 级 critic、特权信息 critic      | ArCHer、SWEET-RL                                | 没有实现，用 token 级 critic                                         | 多轮场景的备选方向                                                       |
| 不用 critic（GRPO、RLOO、ReMax）     | 多篇                                            | 暂时不跑 GRPO（用户决定）                                            | 可以作为对照                                                             |

## 7. 多轮工具调用场景下，critic 可能在学什么（待验证的假设）

\*\*H-a：每道题的难度。\*\*critic 在多个 epoch 里记住每道训练题的成功率，作用接近 GRPO 的组均值 baseline。支持的证据：EV
上限表明题目身份能解释约一半方差（4.2 节）；VinePPO 的"记忆"假设；我们在新题上 EV ≈ 0。按这个假设，训练 batch 上的 EV 应在第 2 个 epoch
以后明显上升，而在评测题上保持接近 0。

\*\*H-b：对话中途的状态价值。\*\*例如一次错误的工具调用或违规操作之后，成功概率下降，critic 若能捕捉到，就能把 credit 分配到具体的轮次，这是 GRPO
做不到的。目前没有支持证据：E7 按位置分组的结果显示，新题上任何位置都没有线性信号。

\*\*不可约的噪声。\*\*即使完全知道题目和当前前缀，user simulator 和策略采样的随机性也会留下题内方差，EV 上限表里的 $W$ 就是这部分（4.2
节）。所以 critic 的 EV 本身就不可能接近 1。

\*\*如何区分 H-a 和 H-b：\*\*对比训练 batch 上的 EV 在第 1 个 epoch（题目对 critic 都是新的）和之后 epoch 的差别；对比第一个
action token 和对话后段的 EV；在评测 rollout 上算 critic 的 EV（新题）。

## 8. 研究缺口

以下是子 agent 没有找到的内容。"没找到"不等于"不存在"，写进论文前要再做一轮检索。

1. 用带 critic 的 PPO 训练 tau-bench / tau2-bench 这类多轮工具调用 agent 的结果。
1. 在多轮工具调用上分开测量 critic 在"新题"和"重复出现的训练题"上的 EV。
1. critic 主要学的是每道题的难度（跨 epoch 记忆），还是对话中途的状态价值。
1. $\gamma=\lambda=1$ 加 batch 白化的 PPO，实际是否退化为"学习出来的每题 baseline"；它和 RLOO 式的组 baseline
   相比如何。
1. EVPO 式的 EV 门控，或者每题组 baseline，能否在 critic 学不好时救回性能。

前提是 PPO 先在 tau2 上跑出有效的提升，否则这些问题没有讨论基础。

## 9. 对当前 PPO（`ppo-formal-qwen3-v3`）的诊断计划

以下是建议，没有全部实施。

| 指标 / 实验                                                               | 怎么看                                                     | 状态                      |
| ------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------- |
| critic 更新前在训练 batch 上的 EV，`ppo_critic/critic_explained_variance` | 每步记录；对比第 1 个 epoch 与第 2 个及之后的 epoch（H-a） | 已在记录                  |
| 评测分数                                                                  | 训练前和每 10 步评测一次                                   | 已在记录                  |
| 评测 rollout 上 critic 的 EV（新题）                                      | 衡量跨题泛化；与训练 batch EV 对比                         | 需要改代码，未实施        |
| EV 按轮次 / 位置分组                                                      | 检验 H-b                                                   | 需要改代码，未实施        |
| 常数 critic 对照（$V\equiv c$，只保留 batch 白化）                        | 直接量化 critic 的贡献；配置只改一处                       | 待 PPO 正式版有结果后决定 |

如果到第 2 个 epoch 以后训练 batch EV 仍长期 ≤ 0，说明 critic 在添乱。可选补救是 critic warmup（K2），或者 EVPO
式门控；到时由用户决定。

## 10. 参考链接

- Ilyas et al., A Closer Look at Deep Policy Gradients:
  <https://arxiv.org/abs/1811.02553>
- VinePPO: <https://arxiv.org/abs/2410.01679>
- VC-PPO: <https://arxiv.org/abs/2503.01491>
- VAPO: <https://arxiv.org/abs/2504.05118>
- Secrets of RLHF in Large Language Models Part I: PPO:
  <https://arxiv.org/abs/2307.04964>
- Back to Basics: <https://arxiv.org/abs/2402.14740>
- ReMax: <https://arxiv.org/abs/2310.10505>
- GRPO（DeepSeekMath）: <https://arxiv.org/abs/2402.03300>
- EVPO: <https://arxiv.org/abs/2604.19485>
- T1: <https://arxiv.org/abs/2609.11042>
- SPPO: <https://arxiv.org/abs/2604.08865>
- GenAC: <https://arxiv.org/abs/2604.10701>
- Le Critique: <https://arxiv.org/abs/2608.16739>
- Best Practice Critic Optimization: <https://arxiv.org/abs/2608.23566>
- ArCHer: <https://arxiv.org/abs/2402.19446>
- SWEET-RL: <https://arxiv.org/abs/2503.15478>
- SPA-RL: <https://arxiv.org/abs/2505.20732>
- RAGEN: <https://arxiv.org/abs/2504.20073>
- GiGPO: <https://arxiv.org/abs/2505.10978>
- verl-recipe tau2_airline（GRPO）: <https://github.com/verl-project/verl-recipe/pull/121>
