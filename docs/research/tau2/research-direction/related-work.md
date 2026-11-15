# 相关工作与撞车检查

> 来源：原 README 第 16.8、16.10 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 相关工作（2026-10-11 检索）

三路检索各一个 agent，结果在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-capability-gap-20261011/out/`：
`related_user_rl.md`（带用户模拟器的 agent RL、反馈可靠度加权）、`related_credit.md`（多轮 credit
assignment、critic、自适应 clip）、 `related_classical_rl.md`（经典 RL 中的外部噪声与反事实 baseline）。下表中我打开
arXiv 页面核对过的标"已核对"，其余以 agent 报告为准。

**最接近的工作：**

| 工作                                                                                         | 做了什么                                                                                                                                                                                                                         | 与本方向的关系                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AutoForge（2512.22857）](https://arxiv.org/abs/2512.22857)，已核对 HTML 正文                | LLM judge 判断用户是否给出幻觉或遗漏信息，判错的整条轨迹不进 advantage 和 loss，不重采样；std 在整个环境上算。Qwen3-Thinking-30B-A3B τ² Retail/Airline 58.8/58.0 → 74.8/62.0；去掉 mask 后训练曲线后期下降（只有曲线，没给数字） | 覆盖了"轨迹级、二值、把用户出错从 advantage 去掉"。未覆盖：按轮次定位（确认后结束这类失败，出错前 agent 已经做对）、区分 agent 能否影响、critic/GAE。它的 judge 只查幻觉和遗漏，不查确认后结束 |
| [FACA（2608.17499）](https://arxiv.org/abs/2608.17499)，已核对摘要                           | 把用户下一句回复当作对前一段 agent 行为的局部评价，算出的 advantage 加到终局 advantage 上，记在 agent 名下。8B/14B 在 τ 系列 9 个 domain 平均 +5.9/+10.2pp，主要来自 Telecom                                                     | 方向相反：它把用户反应当作 reward 来源，本方向把其中与 agent 无关的部分当作噪声。合起来可以表述为"用户反应 = 对 agent 的评价 + 用户自身噪声"                                                   |
| [Noisy RLVR（2510.00915）](https://arxiv.org/abs/2510.00915)，已核对摘要；UARM（2606.19818） | 把 verifier 错误建模为有固定假阳性/假阴性率的信道；2510.00915 在不对称 FP/FN 信道下给出 backward / forward 两种修正（GRPO 的无偏梯度见 [2510.18924](https://arxiv.org/abs/2510.18924)）；UARM 按单样本 RM 不确定度加权 advantage | 提供了修正的数学形式。"用户出错让正确的 agent 得 0"对应假阴性；但它们假设噪声与策略无关，而用户是否出错部分取决于 agent                                                                        |
| Off-policy DAE（ICLR 2024）                                                                  | 把回报分解为 agent 的 skill 和环境的 luck，luck 项 $V(s') - \mathbb{E}[V(s')]$ 用 CVAE 估计                                                                                                                                      | $\Delta_u$ 在理论上就是这里的 luck 项。LLM 对话中用户回合的位置已知，不需要 CVAE，这是可以主张的简化                                                                                           |
| [SWEET-RL（2503.15478）](https://arxiv.org/abs/2503.15478)                                   | 非对称 critic，训练时给 critic 参考答案，逐轮 credit                                                                                                                                                                             | 可借用结构：给 critic 用户的隐藏剧本和可靠度。它假设用户可靠                                                                                                                                   |

tau2 上目前报告的 4B 结果：IAPO（2608.24588）Qwen3-4B 38.4，GRPO 34.8（agent 报告，未核对）。

**2026-11-15 补充**（噪声注入相关的 benchmark 和训练方法）：

| 工作                                                                                                                                   | 做了什么                                                                                                                                                                                                 | 与本方向的关系                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AgentNoiseBench（2602.11348）](https://arxiv.org/abs/2602.11348)，[GitHub](https://github.com/keven-cyber/agentnoisebench)，ICML 2026 | 噪声鲁棒性 benchmark，覆盖 τ²、VitaBench、Search-R1。在全局用户 prompt 中注入幻觉、遗漏、误导等噪声类型，测试 agent 在噪声下的表现。定义 $\mathbb{I}_{solvable}$ 但未给出计算方法（DESIGN.md §5 已指出） | 提供外部评测基准（DESIGN.md §9 第三张表）。差异：改全局用户 prompt vs 我们改单题剧本；他们的 τ² 噪声版本可直接用作迁移评测                            |
| [NoisyAgent（2605.27209）](https://arxiv.org/abs/2605.27209)                                                                           | 混合 clean/noisy 用户 rollout 训练 agent，GSPO（Group Selective Policy Optimization）在组内选择性地从干净 rollout 学、丢弃有噪 rollout。在 τ² 等 benchmark 上报告了结果                                  | 覆盖"训练时按 episode 选择性丢弃有噪 rollout"，与 AutoForge 类似但不需要看剧本的 judge，靠组内对比识别噪声。未覆盖：轮级 credit、critic、单条 rollout |

tau2-bench
官方仓库（[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)）的变体定义和我们的修补见
[benchmark-variants.md](../benchmark-variants.md)。AReaL-tau2-data（[inclusionAI/AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)，[ModelScope 镜像](https://modelscope.cn/datasets/inclusionAI/AReaL-tau2-data)，文件一致）的训练数据用法见
[training-data.md](../training-data.md)。

**经典 RL 中的对应理论**（`related_classical_rl.md`，14 篇读了方法和定理）：

| 工作                                                                                                                                                               | 结论                                                                                                                                                                                                   | 对本方向的含义                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| [Mao et al., ICLR 2019](https://arxiv.org/abs/1807.02264)；Counterfactual Credit Assignment（Mesnard et al. 2021，[2011.09464](https://arxiv.org/abs/2011.09464)） | 基准可以以未来信息为条件而不引入偏差，前提是这份信息与 agent 的动作独立                                                                                                                                | 以用户的**采样噪声**为条件：无偏；以用户**说出的文字**为条件：有偏，因为文字依赖 agent 说了什么                         |
| COMA（反事实多智能体 baseline）                                                                                                                                    | 固定另一个智能体的动作、对自己的动作求平均                                                                                                                                                             | 假设双方同时行动。用户是在 agent 之后回复的，直接套用有偏                                                               |
| Gumbel-max 结构因果模型（Oberst & Sontag 2019、Chatzi et al. 2025）；按事件编号的随机数（2603.11084）                                                              | LLM 采样可以写成 $\arg\max(\log p + g)$，噪声 $g$ 是外部变量；随机数要按事件编号生成，不能按"第几次调用"生成                                                                                           | 自托管用户时，可以让用户噪声在构造上与 agent 动作独立，同时用户的行为分布不变                                           |
| [Luck Is Not Skill（2609.24144）](https://arxiv.org/abs/2609.24144)，已核对摘要                                                                                    | GRPO 组内共用同一份环境噪声（paired rollouts），用 leave-one-out 基准时无偏，除以组内 std 会破坏这个性质。2B 工具 agent：工具故障下最终成功率 +5.1pp，但没达到预注册的学习曲线判据；评分翻转下基本无效 | 最接近的方法论先例，噪声来自工具和评分，不是用户。它也说明：reward 对比的方差下降，不等于梯度方差下降，更不等于学得更快 |

**经典方法在我们场景里没有解决的问题：**

1. **我们拿不到用户的噪声**。用户是 DeepSeek API、temperature
   0，剩下的随机性来自服务端的非确定性（[capability-gaps.md §1.3](../capability-gaps.md)：同组首句用户消息 386/386
   组都不全相同），我们控制不了。
1. **"运气"的分解不唯一**。同一个用户分布可以用不同方式写成"确定函数 + 噪声"，每种都无偏，但给出不同的运气/能力划分。
1. **对话分叉后共用噪声失效**。agent 第一轮说法不同，用户后面每一轮的上下文就不同，同一份噪声对应的回复不再可比。
1. **只去噪声，不去偏差**。用户模拟器系统性的毛病（过度配合、确认后结束）仍然保留，这是问题 B，经典方法不处理。

所以问题 A 在自托管用户下有现成的无偏做法；我们能贡献的是问题 B，以及在拿不到用户噪声时（API 用户、真实用户）如何估计。

**检索中没找到的**（检索次数有限，"没找到"不等于"没有"）：

1. 把用户出错的轮次当作截断点：之后的结果不传回给 agent，用 $V(s_k)$ bootstrap 代替 0 reward，出错之前的 agent 行为得到
   critic 估计的 credit。 现有做法只有"整条丢弃"（AutoForge）和"整条计 0"（其余所有）。
1. 区分"agent 能影响"和"外生"的用户错误：固定 agent 前缀、重采样 K 次用户回合估计 $P(\text{用户出错} \mid \text{前缀})$，
   这同时也是一个"只重采样用户回合"的 baseline，也没有公开工作。它要求能重采样，只适用于模拟器场景。
1. 按用户或环境可靠度设置 clip。现有自适应 clip 用的是 token 概率、entropy、information gain。
1. "弱用户加修正能否追平直接换强用户"这个对照。多篇工作报告强用户训练更好（AutoForge 换 GPT-5-thinking 用户 Telecom
   +14，AReaL-SEA（2601.22607）在 Telecom 上用 RL 微调后的用户，pass^1 95.6 对基础用户 75.6，+20，只有 Telecom
   这一个领域），但没人回答损失有多少能靠修正 credit 挽回。

**对实验设计的修正：**

- E1 改称"AutoForge 式整条 mask"，作为 baseline（已改入
  [analysis.md §4](analysis.md)、[analysis.md §5](analysis.md)）。
- 不再引用 Coverage, Not Targeting（2609.02417）：τ² 结论只来自一次 policy 更新（LoRA），主结果
  +3.6pp，作者报告的评测噪声与效应同级（一个 seed 重评测变化 +0.159），4B 上无差异。
- 不与 GiGPO（[2505.10978](https://arxiv.org/abs/2505.10978)，2025-05，"Group-in-Group Policy
  Optimization"）比较。它不训练 critic， 在同一组 rollout 中找完全相同的中间状态（anchor state），在这些状态上比较后续回报，得到
  step 级优势；在 ALFWorld、WebShop 上有效。 τ² 里用户回复各不相同，中间状态几乎不会完全重合，它退化为 GRPO；在线场景每题只有一条
  rollout，更用不了。IAPO 论文里 Qwen3-4B 上 GiGPO 37.2、 GRPO 34.8（agent 报告），差距在噪声量级。
- 做 $\Delta_u$ 之前，先按用户回合边界分桶报告 SAO critic 的 EV；critic 在中间状态预测偏平时，$\Delta_u$ 会被压向 0。
- UserProxyBench 报告成功 episode 中 24.4% 含用户违规，与
  [capability-gaps.md §1.3](../capability-gaps.md) 的 29% 可以互相印证，可写入 motivation（agent
  报告，未核对）。

文献部分除标注"已核对"的以外，都以检索 agent 的报告为准，引用前要核实全文。

## 2 在线场景的撞车检查（2026-10-11 检索）

一个检索 agent，38 次检索，结果在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-capability-gap-20261011/out/related_online_user_rl.md`
（26 篇，按撞车程度分级，高、中度的读了方法部分）。P-GRPO、RGPO 的编号和标题我用 arXiv 页面核对过，其余以 agent 报告为准。

**结论：没有高度撞车的工作**。没有一篇同时做到：每个（用户、题目）只有一条 rollout 并用 critic；用户反复出现、带 ID；在线估计每个用户的可靠度，并用它缩放
advantage、clip 或 critic 目标。但每个部件都有人做过，评审可以把本方向拆成"AutoForge 的软化在线版 + P-GRPO 的 critic 版 +
RGPO 的 RL 版"，必须正面回答这个拆分。

| 工作                                                                                   | 已覆盖的部分                                                                                                                                                | 没覆盖的部分                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [P-GRPO（2603.10009，Apple，2026）](https://arxiv.org/abs/2603.10009)，已核对标题      | 按用户或用户组 ID 在线维护 reward 的均值和方差，advantage 取 $(R-\mu_u)/\sigma_u$。Qwen3-8B MovieLens 0.719 → 0.757；随机分组无增益                         | 目的是偏好不同，不是用户可靠；没有 critic。2026-10-13 更正：Eq.(7) 等价于组内优势乘按用户组的缩放 $\sigma_G/\sigma_p$ 再加平移，所以也是按用户缩放；历史统计的公式本身不要求同一 prompt 多条，只是实验用了组采样。与 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 的按用户逆方差加权很接近 |
| [RGPO（2607.20515，2026）](https://arxiv.org/abs/2607.20515)，已核对标题；Crowd-PrefRL | 估计每个标注者的可靠度，按它给样本加权                                                                                                                      | 离线；靠多人标注同一样本来估计；不是 RL 交互                                                                                                                                                                                                                                                                   |
| AutoForge MEU（2512.22857）                                                            | 在 τ/τ² 上去掉用户造成失败的轨迹                                                                                                                            | judge 看用户意图（特权信息）；按 episode 0/1；要组采样；不跨 episode 积累用户信息                                                                                                                                                                                                                              |
| FACA（2608.17499）                                                                     | 每轮用户反应进入 advantage，τ 系列                                                                                                                          | 读模拟器私有字段；没有可靠度；GRPO                                                                                                                                                                                                                                                                             |
| SAO（2607.07508）、CLaaS                                                               | 单条 rollout 加学得的 critic（value model 由微调后的初始化而来，$\lambda_{critic}=1$，10 步 warmup，K=2，冻结 attention），Skip-Observation GAE；一次性场景 | 没有用户维度                                                                                                                                                                                                                                                                                                   |
| RLHI（2509.25137）                                                                     | 在真实用户日志上按用户质量过滤很关键（+23.4，不过滤 +2.5）                                                                                                  | 离线、LLM 判断、硬过滤                                                                                                                                                                                                                                                                                         |

**仍然空着的：**

1. 单条 rollout 下，只用用户 ID、可观测对话和结局，跨 episode 在线估计用户可靠度。
1. 用可靠度连续缩放更新，并在某个用户样本还很少时向总体均值收缩（empirical Bayes shrinkage）。
1. 把用户池的构成（提前结束、隐瞒信息、偏离目标的比例）作为受控变量。
1. **区分"用户不可靠"和"用户难但有信息量"**。这一点决定方法是否成立，是这个方向真正的研究问题：agent 在难用户上做得差，恰恰最该学。

**对 [analysis.md §7](analysis.md) 方案的两处修正**（推导见
[appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)）：

- 最初设想的可靠度 $\bar r_u = \text{mean}(R - V(s_0))$ 不可用：critic 训练好之后，它对每个用户都趋于
  0（[appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 第 2 步）。
- 降低某些用户的权重，等于改变训练目标中各用户的比重，可能把训练推向容易的用户（[appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)
  第 1 步）。评测必须按用户类型分别报告，并报告最差用户组的成功率。

**必须比较的 baseline**（都满足"单条 rollout、不用特权信息"）：

| #   | baseline                                      | 回答的问题                         |
| --- | --------------------------------------------- | ---------------------------------- |
| 1   | 不做用户加权的单条 rollout SAO                | 主 baseline                        |
| 2   | REINFORCE++，batch 全局归一化，无 critic      | 没有 critic 的下界                 |
| 3   | 按用户的 running baseline（P-GRPO 单条版）    | 缩放是否优于平移                   |
| 4   | AutoForge 式 judge mask：看剧本版、不看剧本版 | 有特权信息时的参照；现实可行的版本 |
| 5   | 逆方差加权，不用用户 ID                       | 是否需要用户 ID                    |
| 6   | 以用户 ID 为条件的 critic，不缩放             | 把可靠度吸收进 $V$ 是否就够了      |
| 7   | 按用户可靠度的样本权重（RGPO 式，带收缩）     | 与已有可靠度加权的区别             |

baseline 数量多，第一版只需要 1、3、6 和我们的方法，分别回答"有没有用""缩放还是平移""critic 自己能不能吸收"。
