# 研究方向：用户不可靠时怎么训练多轮 agent（2026-10-19 重写）

本文是当前版本，只写现在有效的问题定义、方法和实验计划。2026-10-11 到 10-19 的旧内容（β 拆分、截断 +
bootstrap、检测器、逆方差加权草案的完整讨论和决策表）在 git 提交 `5f66f66` 的同名文件里；§7 只保留改变过方向的决策。其他文档里在 10-19
之前引用的 problem-and-plan §1–§5，指的是那个旧版本。

## 1 问题

**可验证但不可归因。** 用 RL 训练多轮客服 agent 时，结局 reward 由 agent 和用户共同决定。verifier
能判断任务是否完成，判断不了是谁导致了结局。用户在确认的同一句里离开（U1）、违反自己的任务或编造事实（U3）、没办完就走（U4b）时，agent 即使每一步都做对也得 0
分，负 advantage 落在 agent 此前的正确行为上。

**现象。** 同一个 Qwen3.5-4B 在 τ² Retail 上只换用户模拟器，成功率从 85.0% 降到 65.0%；agent 自身造成的失败始终是 10–15
条，用户造成的从 7 到 45 条（[user-simulator.md §2](../user-simulator.md)）。最强的模拟器下，用户仍造成 29% 的失败。

**用户不是一个。** 即使只用一个模拟器模型，τ² 每道题给用户的 instruction 和 persona
都不同（有的就是要扮演难缠或古怪的用户），训练里天然是多用户。换模拟器模型只是让用户分布变化更大。

**设定。** PPO（带 critic），每题一条 rollout。可用的信息：对话、结局 reward，以及**模拟器里才有的用户剧本**（这道题给用户的
instruction 和 persona）。评测用固定的可靠用户，并检查 policy 合规。部署时 actor 不需要剧本；剧本只给训练时的 critic
用，所以本方法针对"用模拟器训练"，不覆盖"从真人部署中学习"。

## 2 从 PPO 出发：用户错误怎么进入梯度

### 2.1 目标与 policy gradient

训练目标是期望成功率 $J(\theta)=\mathbb{E}_{x,\,u,\,\tau\sim\pi_\theta}[R]$，其中 $x$ 是任务，$u$
是用户，$\tau$ 是整段对话，$R\in\{0,1\}$ 是结局 reward。policy gradient 定理给出

$$
\nabla_\theta J=\mathbb{E}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(Q(s_t,a_t)-b(s_t)\big)\Big].
$$

$b(s_t)$ 是 baseline。只要它不依赖当前动作 $a_t$，就有
$\mathbb{E}_{a_t}[\nabla\log\pi(a_t\mid s_t)\,b(s_t)]=b(s_t)\,\nabla\sum_a\pi(a\mid s_t)=0$，所以
**减任何不依赖动作的 baseline 都不改变梯度的期望，只改变方差**。取 $b=V(s_t)$ 时，$Q-V$ 就是 advantage
$A_t$。后面所有关于"有偏/无偏"的判断都回到这一条。

### 2.2 PPO 的 loss：advantage 是每个 token 的权重

PPO 用重要性比 $\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t)$ 和裁剪：

$$
L^{\text{CLIP}}(\theta)=-\,\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big].
$$

AReaL 的实现是 `pg_loss = max(-A·ρ, -A·clip(ρ))`，再按 `loss_mask` 对 agent token
求平均（`areal/utils/functional/functional.py:658-678`）。 critic 用平方误差拟合 return $\hat G_t$（带
value clip）。

在 $\rho\approx1$ 附近，$L^{\text{CLIP}}$ 的梯度就是
$-\hat A_t\nabla\log\pi(a_t\mid s_t)$。**reward 不进 loss，进 loss 的只有逐 token 的
$\hat A_t$**。所以"改方法"就是"换一种 $\hat A_t$"：可以平移它（加减一个量），也可以缩放它（乘一个系数）。这两种操作性质不同，§4、§6 会用到。

### 2.3 GAE：稀疏的 reward 怎么变成每个 token 的 advantage

GAE 从后往前递推：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad \hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1},\qquad \hat G_t=\hat A_t+V(s_t).
$$

代码是 `gae.py:66`：`delta = rewards[t] + discount * nextvalues - values[t]`。我们的配置下有三处特殊：

1. **$r_t$ 几乎处处为 0。** `kl_ctl=0`，没有逐 token 的 KL 惩罚；τ² 只在结束时给分，$R$ 只放在最后一个 agent token
   上（`actor.py:272-295`）。
1. **$\gamma=\lambda=1$。** 递推化简为 $\hat A_t=R-V(s_t)$，$\hat G_t=R$。critic 拟合的是 Monte
   Carlo 回报。
1. **user 和 tool token 被跳过。** 它们 `loss_mask=0`，GAE 把 `nextvalues`
   原样带过去（`gae.py:86`），相当于只在 agent token 构成的子序列上做 GAE。

所以中间 token 没有 reward，但每个 token 都有自己的 $\hat A_t=R-V(s_t)$。$R$ 是共享的，各 token 不同的地方在于 critic
给的 $V(s_t)$。

如果你对 TD error、$\lambda$、为什么从后往前算还不熟，先读
[附录 B](#%E9%99%84%E5%BD%95-btd%CE%BB-%E4%B8%8E-%CE%B2-%E7%9A%84%E7%9B%B4%E8%A7%82%E6%8E%A8%E5%AF%BC)，它从最小的例子推到
β。

### 2.4 跨过一次用户回复：一个 TD 被恒等拆成两段

记 agent 第 $k$ 轮开口前的读数为 $V_k$，第 $k$ 轮说完、用户还没回复时的读数为 $U_k$（value\[$b$\]，见 附录
A），用户回复后、下一轮开口前为 $V_{k+1}$，最后一轮之后 $V_{K+1}=R$。对第 $t$ 轮里的 agent token $i$，$R-V_i$
可以逐项展开（中间读数正负抵消）：

$$
R-V_i=\underbrace{(U_t-V_i)+\sum_{k>t}(U_k-V_k)}_{\text{agent 段：agent 自己的话和工具调用带来的变化}}+\underbrace{\sum_{k\ge t}(V_{k+1}-U_k)}_{\text{用户段：用户回复带来的变化}}.
$$

这是恒等式，没有任何假设。它把 PPO 已经在用的 advantage 按"谁说话"切开。工具返回是确定的数据库结果，留在 agent 段。

**用户段的条件均值。** 给定 agent 第 $k$ 轮说完的历史 $h_k^+$，如果 critic 在用户回复前后自洽，

$$
U_k=\mathbb{E}\big[V_{k+1}\mid h_k^+\big],\tag{C}
$$

那么用户段条件均值为 0：它是"给定 agent 这句话，用户回复得好或坏"的纯噪声。推导和"critic 整体偏高也不破坏 (C)"的说明见旧版
§4.3（`5f66f66`）。

### 2.5 两种伤害

用 §2.4 的分解，用户带来的伤害可以精确地分成两种。

**伤害 1：方差。** 用户段是围绕 0 的随机波动。agent 每一步都做对，用户随机地在确认时离开，用户段就是一个大负数，落到 agent 此前所有 token
上。期望梯度不变，但方差大，同样的预算学得少。

**伤害 2：目标本身被带偏。** 如果用户的不可靠是**可预测**的，例如 agent 一问"确认吗？"，这个用户就有 40% 在同一句里 STOP，那么在 critic
准确时，$U_k$ 在问确认之后本身就低。这一项在 agent 段里，梯度会正确地告诉 agent："在这个用户下，问确认是坏动作"。对
$J$（在这个用户下的成功率）来说这是对的，对"agent 自己的能力和 policy 合规"来说是错的。

由此得到一个硬结论：**任何对 $J$ 无偏的 advantage 估计都改不了伤害 2**，因为伤害 2 就是 $J$
本身。要处理它，只能改目标：给不同的轨迹或用户不同的权重，或者把某些 token 移出 loss。降方差的方法只能处理伤害 1。

## 3 已经试过的方法，以及它们为什么不够

| 方法             | 做法                               | 结果                                                                                                                                                                                                                          | 证据                                                                                                                                          |
| ---------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 统一 β           | 所有用户段乘 $\beta=0.5$           | 中间用户段几乎为 0（$+0.006\pm0.061$），β 实际只作用在最后一次回复；那里 U1 只占 11–12%，其余是结算才暴露的 agent 错误。631 的 Retail 在 checkpoint 120 比 PPO 低 13.75 pp，后期多出的正是工具不报错、用户还道谢的 agent 写错 | [critic/user-boundary-forward.md](../critic/user-boundary-forward.md)、[ppo-beta05-eval-drop/REPORT.md §5](../ppo-beta05-eval-drop/REPORT.md) |
| 规则门控 β       | 只在 U1 规则命中的边界取 $\beta=0$ | 规则召回高，但反事实 precision 约 58%：很多"用户 yes+STOP"之前 agent 的方案本来就错                                                                                                                                           | [user-error-detectors/REPORT.md §0a–0b](../user-error-detectors/REPORT.md)                                                                    |
| judge 门控       | LLM 读全轨迹判责任方和边界         | 认得出 U3，但在训练分布上 precision 76%，把 12.8% 的 agent 失败判给用户；每次 33.5k token                                                                                                                                     | [user-error-detectors/REPORT.md §0c](../user-error-detectors/REPORT.md)                                                                       |
| 截断 + bootstrap | 用户出错处截断，用 $U_k$ bootstrap | 依赖检测器；$U_k$ 和 $V_t$ 差不多，效果接近 mask                                                                                                                                                                              | 旧版 §5                                                                                                                                       |

共同的原因有两层：

1. **critic 不知道真相。** 现在的 critic 只看对话，看不出用户说的话违反了剧本，也看不出 agent
   的写操作不符合用户的真实意图。所以两类错误都要到结算才暴露，挤在最后一次回复的用户段里（[critic/user-boundary-forward.md](../critic/user-boundary-forward.md)
   结论 1、3：$U$ 对"提议是否正确"的 AUC 只有 0.573）。§2 的分解在恒等式层面成立，但在这个 critic 下，"用户段"并不等于"用户的作用"。
1. **检测器做不准，而且依赖 domain。** 规则只认得 U1；judge 要再引入一个 LLM，还要按 domain
   维护错误分类。老师和我们都判断这条路做不通（2026-10-19）。

β、规则门控和 judge 保留实现，作为消融。

## 4 方法：给 critic 看用户剧本

### 4.1 做法

critic 是独立于 actor 的模型，现在读的是和 actor 完全相同的 token 序列：system 消息（domain policy、工具列表）加对话。我们只在
critic 读的那份序列里，在对话开始前插入这道题的用户剧本：

```text
<|im_start|>system
<policy>...</policy>  # Tools ...
<user_profile>                        ← 只有 critic 读到
reason_for_call / known_info / unknown_info / task_instructions / persona
</user_profile><|im_end|>
<|im_start|>user
Hi, I want to return ...              ← 从这里起两边完全一样
```

- 剧本取自 τ² 任务的
  `user_scenario`（用户模拟器拿到的就是它，`examples/tau2/agent.py:457`），真实样例见[附录 C](#%E9%99%84%E5%BD%95-c%E5%89%A7%E6%9C%AC%E6%A0%B7%E4%BE%8B)。**不放**
  `evaluation_criteria`：没有 gold 动作序列、没有 NL 断言、没有 reward 规则。
- actor 的输入、loss、GAE 都不变。critic 的训练目标仍是 $R$（$\lambda_{\text{critic}}=1$）。
- 插在对话之前，critic 序列里此后所有位置统一偏移 $L$ 个 token，读 value 时减掉偏移即可与 actor 对齐。

### 4.2 为什么无偏

剧本 $z$ 在对话开始前就定了，不依赖 agent 的任何动作。按 §2.1 的同一个论证，$V_\phi(h_t, z)$ 是合法的 baseline，期望梯度不变。这与
Le Critique（[2608.16739](https://arxiv.org/abs/2608.16739)）的约束相同：特权信息不能包含当前回答的未来
token、realized reward 或 verifier 反馈。

### 4.3 它改变了什么

critic 知道用户应该怎么做之后：

- **用户违反剧本时**（U3），价值在那次用户回复处下跌，落进用户段；
- **agent 做了不符合用户意图的写操作时**（例如该退单件却取消整单），价值在 agent 那一轮下跌，落进 agent 段；
- **用户随机地在确认时离开**（U1），仍落在最后一次回复的用户段。

于是 §2 的分解第一次有机会真正按"谁的错"切开。这有两个用处：

1. **更好的 baseline。** 单是 $V(h,z)$ 比 $V(h)$ 准，advantage 方差就更小。Le Critique 在单轮推理和多轮数独上都看到了
   EV 提升。
1. **让 β 只打折用户的作用。** $\lambda=1$ 时 $A_t=R-V_t$。如果用户在中途犯 U3，agent 此前正确的 token 拿到的负
   advantage 来自那一轮的用户段；只有把用户段缩掉（β），这些 token 才不被罚。旧的统一 β 失败，是因为用户段里混着 agent 的错误；剧本让 agent
   的错误移出用户段，β 就只作用在用户身上。

所以完整的方法是 **带剧本的 critic + 用户段乘 β**。β 的无偏条件仍是 §2.4 的 (C)。伤害 2（可预测的不可靠诱导 agent 少确认）在 agent
段里，这个方法不处理，留给 §6。

### 4.4 critic 需不需要先离线训好

不需要，直接在 PPO 里从现有初始化开始训。理由：

- **原理上**：$\lambda=1$ 时 critic 只是 baseline，不准只增加方差、不引入偏差（§2.1）。PPO 每一步都用 on-policy 的
  Monte Carlo 回报训练它，它跟着当前策略的分布走；离线拟合的是旧策略的分布。
- **经验上**：critic-only 离线训练（479、499、510）验证 EV 一直在 0 附近，最好
  0.1–0.2（[critic/README.md](../critic/README.md)）；放进 PPO 后 EV 在第 21–40 步到 0.19–0.23，最后
  20 步 0.34–0.36（[ppo-beta05-eval-drop/REPORT.md §5](../ppo-beta05-eval-drop/REPORT.md)
  引的 644 学习信号表）。
- **初始化**：沿用 ADR 0012（从 policy 初始化、head bias 取训练成功率），不做 warmup（2026-10-15 的决定）。critic
  的骨干没见过 `<user_profile>` 块，但它是普通文本，在线训练会学会用它；学没学会由 §5 的诊断回答。

## 5 实验设计

### 5.1 对照：只改 critic 的输入

同一个用户、同一个 judge（都是火山 plan 上的 `deepseek-v4-1-flash-261017`）、同一套配置，2×2：

|             | 普通 critic          | 带剧本的 critic                 |
| ----------- | -------------------- | ------------------------------- |
| $\beta=0.5$ | 新 run 0（对照）     | **新 run 1**（完整方法）        |
| $\beta=1$   | 新 run 3（PPO 基线） | 新 run 2（只有更好的 baseline） |

- **用户模型（用户 2026-10-20 决定）**：先用 DeepSeek 261017 跑通，它比豆包 Turbo 快、便宜（642 的豆包 rollout 中位数
  378 s，631 的 DeepSeek 89 s）。代价：642（豆包，停在 step 24）不再是单变量对照，对照 run 0 要用 DeepSeek
  重跑；DeepSeek 用户在训练分布里 U3 只占约 1%，训练指标上方法能修的用户错误少，D2 仍靠 §5.2 的多模拟器数据测。user 和 judge
  是同一个模型。
- **β（用户 2026-10-20 决定 β=1 和 β=0.5 都跑）**：(C) 成立时，β 取任何值都不改变期望梯度，用户段方差按 $\beta^2$ 缩小；(C)
  不成立时，偏差和 $1-\beta$ 成正比，而训练早期 critic 最不准。顺序待定，建议先跑 run 2（β=1）。
- **其余（用户 2026-10-20 确认）**：value head 的 bias 先用 0.6205；telecom 不进训练集，以便和已有 telecom
  结果比较；剧本块前面加一段对所有 task 都一样的说明（647 实现）。
- **先跑 run 1。** 它是完整方法，先确认能跑通、critic 学会用剧本（D1）。run 0 和它只差 critic 的输入，单变量。critic 训练不依赖
  β（target 都是 $R$），所以 run 1 和 run 0 的 critic 可以直接比较诊断指标。
- run 2、run 3 在 run 1 的诊断通过后再跑，补齐"剧本的作用"和"β 的作用"。
- 每个 run 都需要用户批准 GPU；gpu8 只有一台，run 串行。

### 5.2 诊断：剧本有没有让用户段等于用户的作用

critic 每 10 步保存一次。训练结束后（或中途），用 run 1 和 run 0 的同步数 critic checkpoint，在**同一批有标签的轨迹**上前向，复用
`tau2-critic-boundary-20261018/` 的边界读数脚本。带剧本的 critic 输入里加剧本，普通 critic 不加。

数据：v3/v4 多模拟器轨迹（qwen 用户 U3 多、Turbo 用户 U1 多，主因和出错消息已标注）、639/640 的 U1 标签、643 的 J 输出。训练分布的
DeepSeek 用户 U3 只占约 1%，测不出第 2 项，所以必须用多模拟器数据。

判据在跑之前固定（数值待用户确认）：

| #   | 检查                                           | 普通 critic 的现状          | 带剧本的 critic 通过的条件                                 |
| --- | ---------------------------------------------- | --------------------------- | ---------------------------------------------------------- |
| D1  | 训练中的 EV（以 outcome 计）                   | run 0 同步数的值            | 同步数窗口均值高 ≥ 0.05                                    |
| D2  | 标注为 U3 的那次用户回复，用户段 $V_{k+1}-U_k$ | 中间用户段约 0              | 均值 ≤ −0.2，且区分 U3 回复与正常回复的 AUC ≥ 0.7          |
| D3  | 主因是 agent 写错的失败，跌幅落在哪            | 主要在最后一次回复（$R-U$） | 落在最后一次回复的比例降到一半以下，跌幅前移到写错的那一轮 |
| D4  | $U$ 对"提议是否正确"的 AUC                     | 0.573                       | ≥ 0.65                                                     |

- D2、D3 都不通过：剧本没有让 critic 分开两类错误，β 部分没有依据，方法退化为"更好的 baseline"，看 run 2 是否还有收益。
- D1 不通过：critic 没学会用剧本，先查输入构造和对齐。

### 5.3 训练结果怎么读

- 主指标：固定可靠用户上的 eval 成功率（Airline/Retail/Telecom 分开），以及写前确认率等 policy 合规指标。
- run 1 对 run 0：带剧本是否让 β 不再伤 Retail（631 的主要问题）。
- 单 seed；按 task 聚类给区间，窗口在跑前固定。

### 5.4 实现（Codex 实现，Claude review，测试先红后绿）

1. critic 输入构造：从任务取 `user_scenario`，渲染 `<user_profile>` 块，只插进 critic 的序列；记录偏移 $L$，value
   读数减去 $L$ 后与 actor 位置对齐。测试：插入前后 actor 输入逐 token 不变；对齐后 critic 的 $V_t$、$U_k$
   位置与不插入时一一对应；`evaluation_criteria` 的任何字段都不出现在 critic 输入里。
1. 配置开关放在 tau2 的配置类（`examples/tau2/utils.py`），不改 `areal/api/cli_args.py`。
1. 序列长度：剧本一般几百 token，只加在 critic 侧；检查 critic 的 max length 和显存。

**状态（2026-10-20）**：pueue 645（Codex）实现，Claude review 后合入
`codex/sao-math`（`6bf88fbd`，merge `e12482ef`）。

- 开关 `critic_user_scenario`，与 `user_boundary_detector` 互斥，可与 `user_td_beta` 同开。
- 剧本作为独立的 system 消息放在 critic 序列**最前面**（不是 §4.1 示意的插进原 system 消息），每条样本只整体右移 $L_i$。critic
  前向后逐行丢掉前 $L_i$ 个读数，外部的 GAE、β、EV 都看原 shape。
- 剧本逐字段从 `task.user_scenario` 取（persona 与 domain / reason_for_call / known_info /
  unknown_info / task_instructions），不读 `Task` 的其他字段，多出字段直接报错。用真实 tokenizer
  抽查：`<|im_start|>`、`<|im_end|>` 各编码成一个特殊 token。
- 测试：先红后绿；故意错一位对齐时测试变红；CPU 主套件 372 passed。没有在 GPU 上跑过。
- 长度：剧本 Airline 中位 416、最长 1917 token，Retail 中位 784、最长 1532 token。按 642 保存的 2991 条序列算，2
  条加剧本后超过 critic 的 `max_tokens_per_mb=32768`，而单条超出时 micro-batch 分配会直接报错。所以 run 配置把
  critic 的 `max_tokens_per_mb` 提到 36864（vLLM `max_model_len=32768`，加最长剧本也不超过
  34685），actor 不变。
- run 1 的 launcher
  与代码快照：`_artifacts/AReaL/codex-sao-math/tau2-ppo-privileged-deepseek-20261020/`（`src-e12482ef`，用户和
  judge 都是 DeepSeek 261017，用新 key），check-config 已通过。

### 5.5 run 1 的设置（跑之前逐项确认）

来源：`tau2-ppo-privileged-deepseek-20261020/check-config.log`（2026-10-20，代码快照
`src-e12482ef`）。

| 类别         | 设置                                                                                                                         | 备注                                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 模型         | actor、critic 都是 Qwen3-4B-Instruct-2507@`cdbee75f`                                                                         | critic 从同一个 checkpoint 初始化，value head 权重 0、bias 0.6205                                        |
| 算法         | PPO，$\gamma=1$，actor/critic $\lambda=1$，advantage 白化，clip 0.2，value clip 0.5                                          | decoupled loss，重算 logprob，`max_head_offpolicyness=2`                                                 |
| 学习率       | actor 1e-6，critic 1e-5，常数，无 warmup，梯度裁剪 1.0                                                                       | critic 的 attention 不冻结                                                                               |
| 数据         | 每步 128 条 episode，每个 task 每个 epoch 采 1 条，每个 epoch 747 个 task（airline 410、retail 337），20 个 epoch，共 120 步 | 训练集不含 telecom，telecom 只出现在 eval 里                                                             |
| 生成         | temperature 1.0，每轮最多 4096 token，上下文 32768，最多 200 步                                                              |                                                                                                          |
| 用户 / judge | 火山 plan 上的 `deepseek-v4-1-flash-261017`，temperature 0，新 key                                                           | 646 合入后：偶发错误重试 8 次、每个请求超时 120 s、总预算 600 s；额度用完 5 分钟内 3 次就熔断，退出码 86 |
| 超时         | 每个 episode 1800 s，并发 rollout 24                                                                                         |                                                                                                          |
| 方法         | `critic_user_scenario=true`，`train_user_boundary_values=true`，$\beta$ 待定（§5.1）                                         | critic 的 `max_tokens_per_mb=36864`（§5.4）                                                              |
| eval         | 100 个 task（airline 20、retail 40、telecom 40）× 4 次，temperature 0；第 0 步和之后每 10 步一次                             | eval 的用户也是 261017，会消耗额度                                                                       |
| 保存         | recover 和 checkpoint 都是每 10 步                                                                                           | seed 42                                                                                                  |

需要确认的地方：

1. **value head 的 bias 初值 0.6205**，来自旧用户下 run 480 的训练成功率（927/1494）。DeepSeek 261017
   下的初始成功率还没测过。在线训练几步后会自己纠正，但前几步的 EV 会受影响。
1. **用户 temperature 为 0。** 给定同一段历史，用户的回复几乎是确定的（provider 不保证完全确定）。所以 $V_{k+1}-U_k$
   里"用户的运气"这部分会很小，β 降方差的空间也跟着变小。
1. **eval 数字不能和 631/642 直接比**：用户模型不同。
1. **每个 epoch 最后一步的 batch 特别大**：642 里最后一步是 428 条或 212 条，其他步是 128 条。647 查明这是有意为之的复制，不是
   bug。训练用 4 个 DP rank，每个 rank 要分到同样多的组，所以 `repeat_groups_for_dispatch` 把 $G$ 组整体复制
   $r=4/\gcd(G,4)$ 次：107 组复制 4 次得 428 条，106 组复制 2 次得 212 条。复制的是同一份
   rollout，不会再调用模拟器。token-mean loss 的分子和分母同比例放大，所以每个 task 的权重不变，optimizer
   也只走一步。唯一的差别是无偏白化的分母变成 $rN-1$，带来约 2–3 ppm 的尺度变化，可以忽略。代价是这一步的更新计算量是平时的 2–4 倍。前 4 个
   epoch 里，每个被训练到的 task 都只出现一次；第 3、4 个 epoch 各有 1 个 task 连续失败两次，按
   `max_skipped_tasks_per_epoch=2` 被跳过。642 第 18、24 步慢（rollout 阶段约 2 小时），主要是 controller
   超时后重放造成的，不是复制造成的。报告：
   `_artifacts/AReaL/codex-sao-math/tau2-critic-header-batch-20261020/REPORT.md`。
1. **β 的取值**：见 §5.1。

### 5.6 风险

- **critic 学不会用剧本**：D1、D2 会显示出来。
- **剧本泄露答案**：剧本里有订单号、用户真实意图，比用户身份多很多信息。它不是 gold；用户 2026-10-20 批准剧本进
  critic，`evaluation_criteria` 仍不进。
- **新颖性**：特权 critic 本身不新（Le Critique 给答案，SWEET-RL 给参考答案，asymmetric actor-critic
  给完整状态）。差别在于给的是用户剧本、目的是把用户的作用和 agent 的作用分到不同的段；能不能站住取决于 D2、D3。2026-10-19 的定向检索（约 5
  次检索，打开了下列页面）没有找到"critic 看用户隐藏剧本、并在用户回复边界上分开用户偏离和 agent 错误"的工作，这是"没找到"，不是"不存在"。最近的邻居：
  - Le Critique（2608.16739）：critic 看标准答案、同组其他回答、数独解盘；没有用户。
  - SWEET-RL（2503.15478）：advantage 以参考代码或网页为条件，Bradley-Terry 训练后做
    DPO；模拟用户也看参考答案，不区分用户错误。
  - SIAPO（2605.22240）：模拟用户有隐藏的顾虑和逐轮意愿；隐藏状态用于自蒸馏的 teacher，不进 value；GRPO，无 critic。
  - AutoForge（2512.22857）：judge 读用户意图，整条 mask；无 critic。
  - FACA（2608.17499）：用户下一句回复当作局部 advantage 加到结局上；不看剧本。
  - Asymmetric Actor-Critic for Multi-turn LLM Agents（2604.00304）：运行时 critic 监督固定的
    actor，不看用户目标，也不是 RL 的 value。

## 6 之后：按 persona 缩放（D2、D3 通过后再做）

老师的直觉是"置信度低的用户少学"。卡点是可辨识性：成功率低可能是用户不可靠，也可能是用户难缠但 agent 能应对（后者该多学）。带剧本的 critic
下两者落在不同的段：难缠但可应对的用户，结局由 agent 怎么应对决定，变化在 agent 段；不可靠的用户，agent 说完之后结局仍取决于他乱不乱来，用户段波动大。

所以可靠度可以定义为"agent 说完之后，这类用户的回复还会让结局波动多少"：按 persona 用过去的 episode 估计用户段方差
$\hat\sigma_p^2$，episode 乘

$$
w_p=\frac{1}{1+\hat\sigma_p^2/\tau},
$$

$\tau$ 取全体平均，样本少的 persona 向全局收缩。这是逆方差加权；用户只加噪声时任意正权重不改梯度方向。它是唯一能动伤害 2
的部分。旧的逆方差草案（16.11）被否掉的两个原因（测量位置事后选、残差混着 agent 错误）在这里分别由固定的用户边界和剧本解决。

## 7 决策轨迹（只保留改变方向的决策；完整表见 `5f66f66`）

| 日期  | 决策                                                                      | 原因                                                                                                                |
| ----- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 10-11 | 起点：老师提出按用户反馈的置信度调整 advantage，critic 也考虑用户         | 好用户下分数高、差用户下学不会                                                                                      |
| 10-12 | 逆方差加权草案暂停，先找准现象                                            | 数学过重，核心估计量有设计错误                                                                                      |
| 10-12 | 轻量方法：用户段乘 β                                                      | 粒度到每次回复，一个超参数                                                                                          |
| 10-13 | 研究对象定为不可靠用户；模拟器是可控的代理                                | 真人同样会确认后离开、改主意                                                                                        |
| 10-18 | 加截断 + bootstrap，检测器可替换                                          | U1 能用 prompt 修掉，U3 修不掉、β 管不到                                                                            |
| 10-19 | 放弃检测器路线（规则、judge），保留作消融                                 | 643：judge 在训练分布上把 12.8% 的 agent 失败判给用户；规则只认 U1；老师判断做不通                                  |
| 10-19 | **新方法：给 critic 看用户剧本 + 用户段乘 β；先跑 run 1 并用 D1–D4 诊断** | 旧方法失败的共同原因是 critic 不知道真相；剧本在开始前确定，作为 baseline 输入无偏；不需要 judge 和 domain 专属分类 |
| 10-20 | 剧本进 critic（不含 `evaluation_criteria`）；用户模型先用 DeepSeek 261017 | 用户批准剧本作 critic 输入；DeepSeek 比豆包快、便宜，先跑通；对照 run 0 用同一用户重跑                              |
| 10-19 | critic 不做离线预训练，直接在 PPO 中训练                                  | $\lambda=1$ 时 critic 只是 baseline；离线 critic-only 的 EV 一直在 0.1–0.2 以下，PPO 中可到 0.3 以上                |

## 附录 A：一段对话在 token 序列上的样子，以及 critic 在哪里读数

方法要在"agent 说完、用户还没回复"这一刻读一次 critic。这一刻对应训练序列里的哪个 token，取决于 chat template、loss mask 和
critic 的输出约定。本节把这三件事讲清楚，后面的公式都用这里定义的位置。论文正文不一定写这些，但没有它们，$V_t$、$U_t$ 落在哪里是看不懂的。

**chat template 是什么**。语言模型只读一条 token 序列，不认识"system / user / assistant /
tool"这些角色。多轮对话要先按一套固定规则拼成一条字符串， 用特殊 token 标出每条消息的开始、结束和角色，这套规则就是 chat template。它随
tokenizer 发布（`tokenizer_config.json` 里的 Jinja 模板），训练和推理必须用同一套，否则模型看到的格式和它学过的不一样。

我们用的是 Qwen3-4B-Instruct-2507（revision `cdbee75f`）自带的模板，规则是：

- 每条消息写成 `<|im_start|>{角色}\n{内容}<|im_end|>\n`。`<|im_start|>`（id 151644）和 `<|im_end|>`（id
  151645）是单个特殊 token。
- system 消息里放 tau2 的 policy，后面接 `# Tools` 段，列出全部工具的 JSON schema。
- agent 调工具时，在 assistant 内容里写
  `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`。
- **工具返回也用 `user` 角色**，内容包在 `<tool_response>…</tool_response>` 里。所以只看角色 token
  分不出"用户回复"和"工具返回"，要看 `<tool_response>` 或原始消息的 role。
- 轮到 agent 时，模板先追加 `<|im_start|>assistant\n`（generation prompt），模型从这之后开始生成，直到生成
  `<|im_end|>` 为止。

用这个模板渲染一段简化的 tau2 对话（policy 和工具列表省略），得到的字符串是：

```text
<|im_start|>system
<policy>...</policy>

# Tools
...（工具 schema 与调用格式说明）...<|im_end|>
<|im_start|>user
Hi, I want to cancel reservation EHGLP3.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_reservation_details", "arguments": {"reservation_id": "EHGLP3"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"reservation_id": "EHGLP3", "cabin": "basic_economy"}
</tool_response><|im_end|>
<|im_start|>assistant
Your reservation is basic economy without insurance, so it cannot be cancelled.<|im_end|>
<|im_start|>user
OK, thanks. ###STOP###<|im_end|>
```

**训练序列和 loss mask**。AReaL 的 tau2 配置用 `chat_template_type: concat`：一个 episode 是一条 token
序列，每次 agent 生成都在前文后面接上，前文的 token 原样保留，不重新渲染。每个 token 带一个 `loss_mask`（下文叫 raw mask）：agent
生成的 token（包括它生成的 `<|im_end|>`）为 1，system、用户、工具返回和模板插入的 token 为 0。最后那条用户结束消息（如
`###STOP###`）不在序列里，序列停在 agent 最后一次生成的 `<|im_end|>`。

**value\[i\] 的含义**。critic 和 policy 一样是 causal 模型，对序列的每个位置都输出一个数。位置 $i$ 的输出读过了第 $0..i$ 个
token，所以 value\[i\] 是"站在 token $i$ 之后这个状态上"对最终 reward 的预测；同一位置上 policy 预测的是下一个 token
$x_{i+1}$。

**rolled mask：哪些位置参与 loss**。训练在位置 $i$ 学的是"预测 $x_{i+1}$"，所以位置 $i$ 参与 loss 当且仅当 $x_{i+1}$
是 agent 生成的。代码把 raw mask 左移一位得到 rolled
mask：$\mathrm{roll}[i] = \mathrm{raw}[i+1]$（`actor.py:226-227`）。只有 mask 左移，value
本身不移。actor loss、token 级 GAE（`gae.py:27`，跳过 mask 为 0 的位置，等价于 SAO 的 Skip-Observation GAE）和
critic loss（`critic.py:178-195`）都只用 rolled mask 为 1 的位置。用户 token 和工具返回 token 上的 value
也会被算出来，但没有 target，不训练也不使用。

**位置定义**。设 agent 第 $t$ 轮的生成段占位置 $a..b$（$x_a$ 是第一个生成 token，$x_b$ 是生成的 `<|im_end|>`）：

| 记号      | 位置            | 这个 token 是什么                                                               | 读到了什么                              | 现在是否训练     |
| --------- | --------------- | ------------------------------------------------------------------------------- | --------------------------------------- | ---------------- |
| $V_t$     | value\[$a-1$\]  | generation prompt `<\|im_start\|>assistant\n` 的最后一个换行（模板插入，raw=0） | agent 开口之前的全部上下文              | 是（roll=1）     |
| —         | value\[$b-1$\]  | agent 生成内容的最后一个 token（如 `.` 或 `</tool_call>`）                      | 整句话，但还没看到 agent 决定在这里结束 | 是（roll=1）     |
| $U_t$     | value\[$b$\]    | agent 生成的 `<\|im_end\|>`（raw=1）                                            | 整句话及结束标记，还没看到用户回复      | **否**（roll=0） |
| $V_{t+1}$ | value\[$a'-1$\] | 下一轮 generation prompt 的换行（$a'$ 是下一段的起点）                          | 用户回复之后的上下文                    | 是               |

注意 agent 的第一个生成 token 不是换行：`<|im_start|>assistant\n` 整段是模板插入的，换行只是"agent 开口之前"上下文的最后一个
token，$V_t$ 在那里读。`<|im_end|>` 的 raw=1、roll=0 有两层意思："生成 `<|im_end|>`"这个动作在位置 $b-1$ 上训练；位置
$b$ 本身的输出不进任何 loss，因为它的下一个 token 是模板。

**一个真实片段**（PPO v3 rollout `rollout/0/202.jsonl`，episode `2817075289978133925`，0-based
下标；省略号处见 `_artifacts/.../tau2-token-positions-20261016/selected-token-dumps.md`）：

```text
位置   token            raw roll  说明
4944   <|im_start|>      0   0    模板
4945   assistant         0   0    模板
4946   \n                0   1    模板          ← V_t = value[4946]
4947   To                1   1    agent 生成的第一个 token
 ...
5010   .                 1   1    agent 生成     ← value[b-1]
5011   <|im_end|>        1   0    agent 生成     ← U_t = value[5011]
5012   \n                0   0    模板
5013   <|im_start|>      0   0    模板
5014   user              0   0    模板
5015   \n                0   0    模板
5016   Sure              0   0    用户回复开始
 ...
5045   <|im_end|>        0   0    用户回复结束
5046   \n                0   0    模板
5047   <|im_start|>      0   0    模板
5048   assistant         0   0    模板
5049   \n                0   1    模板          ← V_{t+1} = value[5049]
5050   <tool_call>       1   1    下一轮 agent 生成
```

**在真实数据上的核对**（2026-10-16，Codex 报告
`_artifacts/AReaL/codex-sao-math/tau2-token-positions-20261016/codex-report.md`，Claude
独立重跑）：PPO v3 `rollout/0` 的 251 个 episode、2,683 个生成段，全部以生成的 `<|im_end|>` 结尾，全部满足
$\mathrm{roll}[a-1]=1$、$\mathrm{roll}[b-1]=1$、$\mathrm{roll}[b]=0$，违例 0 个。生成段之间的间隔有 977
个是用户回复、1,455 个是工具返回，另有 251 个是 episode 的最后一次生成。所以每个 episode 平均有
$(977+251)/251 \approx 4.9$ 个用户边界需要读 $U_t$，工具边界不读。outcome reward 加在最后一段的 $b-1$
位置（`trajectory.py:29-57` 取 rolled mask 的最后一个位置）。

**$U_t$ 用 value\[$b$\]，不用 value\[$b-1$\]**。在位置 $b-1$，agent 还要决定是在这里结束（生成
`<|im_end|>`）还是继续写。记 $\mathrm{EOS}$ 为 `<|im_end|>`，按全概率公式，真实的 value 函数满足（$\gamma=1$，中间没有
reward）：

$$
V(s_{b-1}) = \pi(\mathrm{EOS} \mid s_{b-1})\, V(s_b) + \sum_{x \neq \mathrm{EOS}} \pi(x \mid s_{b-1})\, V(s_{b-1} \oplus x)
$$

其中 $s_{b-1} \oplus x$ 表示在 $s_{b-1}$ 后面接上 token
$x$。只有当结束几乎确定（$\pi(\mathrm{EOS}\mid s_{b-1}) \approx 1$）时才有
$V(s_{b-1}) \approx V(s_b)$。上面那批数据里，生成 `<|im_end|>` 的概率中位数是 0.99999917，但 2,683 段里有 180
段低于 0.9，而且**全部在用户边界上**（工具边界最低是 0.997）。也就是说，恰恰在方法要读 $U_t$ 的位置，"说到这里就停"本身是 agent
的一个有随机性的决定。用 value\[$b-1$\] 当 $U_t$，会把这个决定的一部分算进用户段。所以 $U_t$ 取
value\[$b$\]，代价是这个位置现在不训练，要在 critic loss 里补上。

## 附录 B：TD、λ 与 β 的直观推导

§2.3 和 §2.4 直接写了 $\delta_t$、$\hat A_t=\delta_t+\lambda\hat A_{t+1}$ 和 β，默认读者已经懂
GAE。本附录从最小的例子开始，一步一步推到 β。每一节只用前面讲过的东西。

约定：本附录里 $\gamma=1$，$r_t=0$，只有最后有 $R\in\{0,1\}$（和 τ² 一致）。$V_t$ 是 critic 在第 $t$
步的读数，即"站在这里，最后能拿多少分"的预测。对话的一步先当成一个抽象的"步"，B.8 再讲在 LLM 里一步是什么。

### B.1 $V_t$ 是一个实时预测，TD error 是"新消息"

看 NBA 转播：屏幕角落有一个实时胜率。开场 50%，某队连进两球变成 62%，对手三分命中又回到 55%。这个数字就是
$V_t$：站在当前比分和局势上，对"最后谁赢"的预测。每次比分变化，预测会动一下，动多少，就是这一步带来的新消息。

TD error 就是这个"动了多少"：

$$
\delta_t=V_{t+1}-V_t\qquad(\text{最后一步：}\ \delta_T=R-V_T).
$$

你熟悉的 $\delta_t=r_t+V_{t+1}-V_t$ 在 $r_t=0$ 时就是它。体育分析里的 Win Probability
Added（一次回合让胜率增加多少）是同一个量。

把所有 $\delta$ 加起来，中间的预测前后抵消，剩下"最终结果 − 开场预测"：

$$
\delta_1+\delta_2+\dots+\delta_T=(V_2-V_1)+(V_3-V_2)+\dots+(R-V_T)=R-V_1.
$$

这是 B.2 要用的恒等式。

### B.2 把 $\delta$ 加起来：Monte Carlo 与 $\lambda=1$、$\lambda=0$

一个 3 步的小例子（$R=0$，即这局输了）：

| 步  | 读数 $V$ | $\delta$ 的算法   | $\delta$ |
| --- | -------- | ----------------- | -------- |
| 1   | 0.5      | $V_2-V_1=0.7-0.5$ | $+0.2$   |
| 2   | 0.7      | $V_3-V_2=0.3-0.7$ | $-0.4$   |
| 3   | 0.3      | $R-V_3=0-0.3$     | $-0.3$   |

$\delta$ 之和 $=0.2-0.4-0.3=-0.5$，而 $R-V_1=0-0.5=-0.5$。中间的 $V_2$、$V_3$ 在加法里正负抵消了，所以不管
critic 在中间报了什么数，总和都是 $R-V_1$。

GAE 的递推 $\hat A_t=\delta_t+\lambda\hat A_{t+1}$ 在 $\lambda=1$ 时就是"把从 $t$ 起的所有 $\delta$
加起来"，于是

$$
\hat A_t=R-V_t.
$$

按上表：$\hat A_3=-0.3$，$\hat A_2=-0.4-0.3=-0.7$，$\hat A_1=0.2-0.7=-0.5$，分别等于
$R-V_3,R-V_2,R-V_1$。

**为什么叫 Monte Carlo。** Monte Carlo 的意思是：要估计一个期望，就真的跑随机样本，用样本的结果代替期望。真正的 advantage
里有一个期望（"从这里往后平均能拿多少分"），$R-V_t$ 用的是这一条 episode 实际拿到的 $R$，等于只抽了一个样本。平均发生在 batch
之间和训练过程中：很多 episode 各自给出一个样本，梯度把它们平均。

**$\lambda=1$ 时 critic 只是参考线。** $\hat A_t=R-V_t$ 里，方向由 $R$ 决定，$V_t$ 只是减掉的一条参考线。由于 $V_t$
是在第 $t$ 步动作之前读的，不依赖这个动作，§2.1 说过它只降方差、不改方向。所以 $\lambda=1$ 时 critic 不准，只是方差变大，不会把梯度带偏。

**$\lambda=0$ 时。** $\hat A_t=\delta_t=V_{t+1}-V_t$，只看一步的新消息。这条 episode 的 $R$ 只出现在最后一步的
$\delta_T=R-V_T$，中间步完全看不到它。$R$ 仍然间接起作用：critic 是用很多条过去 episode 的 $R$ 训出来的，$V_{t+1}$ 里装着这些
episode 的总结，只是不含这一条的 $R$。

### B.3 reward 只在最后一个 token：没有广播，也没有偏差

问题：reward 只加在最后一个 agent token 上（`actor.py:272-295`），前面的 token 是怎么用到它的？是不是把 $R$ 复制给所有
token？

不是复制。它沿着递推从后往前传。GAE 的循环（[gae.py:65](../../../../areal/trainer/ppo/gae.py:65)
起，`for t in reversed(...)`）每一步做

$$
\hat A_t=\delta_t+\lambda\hat A_{t+1}.
$$

最后一步的 $\delta_T$ 里有 $R$；它通过 $\lambda\hat A_{t+1}$ 传到第 $T-1$ 步，再传到第 $T-2$ 步，每往前一步乘一个
$\lambda$。B.2 的例子里，$R$ 只出现在 $\delta_3$ 里，但 $\hat A_1=\delta_1+\delta_2+\delta_3$ 也含有它。

**为什么奖励放在最后不是偏差。** reward 放在哪一步，取决于结果在哪一步才知道，不是我们选的。结果只有结束时才揭晓，所以信息只能从结局流回原因。偏差指的是用
critic 的预测顶替真实结果；$\lambda=1$ 时每个 token 拿到的都是真实的 $R$（B.2），没有这种顶替。

**为什么必须从后往前。** 这不是为了省计算。两条理由：

1. $V$ 本身是递归定义的（Bellman 方程）：$V(s_t)=\mathbb{E}[r_t+V(s_{t+1})]$。第 $t$ 步的价值由第 $t+1$
   步的价值定义，要先知道后面，才能估前面。
1. 一个动作只能影响未来，影响不到过去，所以第 $t$ 步动作的 advantage 只和 $t$ 之后发生的事有关。算 $\hat A_t$ 需要
   $\hat A_{t+1}$，不需要 $\hat A_{t-1}$。顺序自然是从后往前。

**GRPO 为什么看起来像广播。** GRPO 路径不用 critic，`values` 全是 0（`actor.py:300`
附近，`values = torch.zeros_like(rewards)`），于是 $\delta_t=0$（$t<T$），$\delta_T=R$，$\lambda=1$
时 $\hat A_t=R$ 对每个 token 都一样。这是 critic 取 0 时的特例，看起来像把 $R$ 复制到每个 token，实际仍是同一个递推，只是中间的
$V$ 是 0。（GRPO 里通常再减去组内均值，那是另一个 baseline，这里不展开。）

### B.4 TD 为什么自然出现：一步真实信息，剩下的交给预测

policy gradient 需要 $Q(s_t,a_t)-V(s_t)$（§2.1）。$Q(s_t,a_t)$ 是"做了 $a_t$
之后，往后平均能拿多少分"。我们手里只有一次采样。做完 $a_t$ 之后，**真正新出现的信息只有下一个状态 $s_{t+1}$**（以及这一步的
$r_t$）。后面的事还没发生，如果我们已经有了一个预测 $V_{t+1}$，就直接用它代替："走一步，剩下的交给预测"：

$$
Q(s_t,a_t)\approx r_t+V_{t+1}\quad\Rightarrow\quad Q-V_t\approx r_t+V_{t+1}-V_t=\delta_t.
$$

"一步真实，其余用预测"可以推广到走 $n$ 步，再交给预测：

| 估计量      | 真实走几步 | 之后用什么       | $r=0$ 时的形式                              |
| ----------- | ---------- | ---------------- | ------------------------------------------- |
| TD（1 步）  | 1          | $V_{t+1}$        | $V_{t+1}-V_t=\delta_t$                      |
| $n$ 步      | $n$        | $V_{t+n}$        | $V_{t+n}-V_t=\delta_t+\dots+\delta_{t+n-1}$ |
| Monte Carlo | 到结束     | 无（直接用 $R$） | $R-V_t=\sum_{l\ge0}\delta_{t+l}$            |

最后一列的等号是 B.1 的抵消恒等式。TD 到 MC 是一条连续的谱，差别只在"真实走多远"。这个框架也是 Sutton & Barto
教材里"开车回家"例子的思路（[教材](http://incompleteideas.net/book/the-book-2nd.html)）：路上每过一个路口就更新一次"几点到家"的预测，不必等到家再更新。

### B.5 为什么单个 $\delta$ 是 advantage 的无偏样本

这一点容易觉得奇怪：$\delta_t$ 只用了一个采样的下一个状态，凭什么是 advantage 的无偏样本？

**一个校准的预测不会系统性地漂移。** 如果 critic 是校准的，即
$V_t=\mathbb{E}[V_{t+1}\mid s_t]$（"现在的预测等于下一刻预测的平均"），那么

$$
\mathbb{E}[\delta_t\mid s_t]=\mathbb{E}[V_{t+1}\mid s_t]-V_t=0.
$$

道理是：如果你能预知下一刻预测平均会更高，现在的预测就应该已经更高了。新消息平均为零，才说明预测已经吸收了所有已知信息。

**按动作拆开。** 同一个状态 $s_t$ 下，不同动作带来的新消息均值不同。把条件再加上 $a_t$：

$$
\mathbb{E}[\delta_t\mid s_t,a_t]=\mathbb{E}[V_{t+1}\mid s_t,a_t]-V_t=Q(s_t,a_t)-V(s_t)=A(s_t,a_t).
$$

**篮球例子。** 一次进攻开始时胜率 60%。教练在两个战术里等概率选一个：

| 战术 | 结果           | 概率 | 之后胜率 | $\delta$ |
| ---- | -------------- | ---- | -------- | -------- |
| 三分 | 投中           | 50%  | 70%      | $+10$    |
| 三分 | 没中           | 50%  | 54%      | $-6$     |
| 突破 | （简化为确定） | 100% | 58%      | $-2$     |

- 三分的新消息均值：$0.5\times10+0.5\times(-6)=+2$。
- 突破：$-2$。
- 两种战术各一半：$0.5\times2+0.5\times(-2)=0$，和"校准的预测平均不漂移"一致。进一步核对 $V=60$：三分之后平均胜率
  $0.5\times70+0.5\times54=62$，突破 58，再平均得 $(62+58)/2=60$ ✓。

三分的 advantage 是 $+2$（平均而言选三分比平均水平好 2 个百分点），但单次的 $\delta$ 要么 $+10$ 要么
$-6$，噪声很大。"无偏"的意思是：一次 $\delta$ 像掷一次骰子，单次不等于均值，平均起来等于骰子的均值，即 $+2$。

前提是 critic 校准。critic 不准时，$\mathbb{E}[\delta_t\mid s_t,a_t]\ne A$，下面 B.6 的反例 2
就是这种情况。§2.4 的条件 (C) $U_k=\mathbb{E}[V_{k+1}\mid h_k^+]$，是这个条件用在"agent 说完、用户回复前"这一个点上。

### B.6 为什么要离开 $\lambda=1$，又为什么不去 $\lambda=0$

常见的说法是"离开 $\lambda=1$ 是因为 critic 不准"。方向正好相反：

- **离开 $\lambda=1$ 的理由是 MC 噪声大。** critic 越准，TD 越省噪声。
- **不直接用 $\lambda=0$ 的理由才是 critic 可能不准。**

用一个最小的游戏把两边都算出来。

**设定（上帝视角，用来检验估计量；算法本身只能看到每局的输赢）。** 起点 $s_0$，policy 等概率选动作 G 或 B。选 G 之后真实胜率是 0.6，选 B 之后是
0.4，之后纯靠运气，结局 $R\in\{0,1\}$。"60%"的意思是：很多局都选 G 的话，其中 60% 赢。

**准确的 critic** 指 $V$ 等于当前 policy 下的真实均值：$V(s_G)=0.6$，$V(s_B)=0.4$，
$V(s_0)=0.5\times0.6+0.5\times0.4=0.5$。G 的真实 advantage 是 $0.6-0.5=+0.1$。

#### 反例 1：MC 噪声大

G 的 MC advantage 是 $R-V(s_0)=R-0.5$：

| 结局 | 概率 | $R-0.5$ |
| ---- | ---- | ------- |
| 赢   | 0.6  | $+0.5$  |
| 输   | 0.4  | $-0.5$  |

- 均值：$0.6\times0.5+0.4\times(-0.5)=0.30-0.20=+0.1$，是对的，没有偏差。
- 方差：先算每个取值与均值 $0.1$ 的偏差，$0.5-0.1=0.4$，$-0.5-0.1=-0.6$；平方 $0.16$、$0.36$；按概率加权
  $0.6\times0.16+0.4\times0.36=0.096+0.144=0.24$。标准差 $\sqrt{0.24}\approx0.49$。
- 噪声（0.49）是信号（0.1）的约 5 倍。只看 1 条样本，$R-0.5$ 有 40% 的概率是负的，会得出"G 不好"。
- 平均 $n$ 个独立样本：$n$ 个独立量之和的方差是 $n\times0.24$，除以 $n$ 得到平均，方差再除以 $n^2$，结果是 $0.24/n$，标准差
  $0.49/\sqrt n$。要让这个标准差 $\le$ 信号大小 $0.1$（取 0.1 是因为要分辨的差别就是 0.1），需要
  $0.49/\sqrt n\le0.1$，$n\ge24$。要有更可靠的把握（标准差降到 $0.05$），$n\ge0.24/0.05^2=96$。

**用 TD 且 critic 准确**：G 的
$\delta=V(s_G)-V(s_0)=0.6-0.5=+0.1$，每次都是，零噪声。这个游戏里动作之后的状态是确定的，所以 TD
没有噪声。如果下一个状态本身是随机的（例如用户的回复），TD 也有噪声，那正是 B.10 的用户段 $V_{k+1}-U_k$。

#### 反例 2：TD 在 critic 学反时有偏

假设 critic 学反了：$V(s_G)=0.4$，$V(s_B)=0.6$（$V(s_0)=0.5$ 不变）。

"学反"在 LLM 里的例子：G 是真的调用工具改了订单，然后说"已为您完成"；B 是没调用工具，直接说"已为您完成"。早期 critic
可能只学到"出现这句话就常成功"，没学到"有没有真的调用工具"，于是给 B 的价值不低于 G，甚至更高。

|     | $V(s_0)$ | 动作之后的 $V$ | TD：$V(s_{\text{后}})-V(s_0)$ | MC 的均值：$\mathbb{E}[R]-V(s_0)$ |
| --- | -------- | -------------- | ----------------------------- | --------------------------------- |
| G   | 0.5      | 0.4            | $-0.1$                        | $0.6-0.5=+0.1$                    |
| B   | 0.5      | 0.6            | $+0.1$                        | $0.4-0.5=-0.1$                    |

TD 稳定地把 policy 推向 B，没有噪声，推错了。MC 不受影响，G 的均值仍是 $+0.1$。

**原因。** MC 用的是 $V(s_t)$，即动作之前的读数，G 和 B 共用同一个值，只会同时平移两个动作的 advantage，大小顺序不变。TD 用的是
$V(s_{t+1})$，即动作之后的读数，两个动作的值不同，critic 的误差就变成了"哪个动作更好"的判断。

**会自己纠正吗？** 最终会：critic 用真实的 $R$ 训练，会逐渐学对。但在这之前，policy 一直在往 B 漂，并产生更多 B 的数据。

### B.7 GAE 的权重从哪来

B.4 的每一种"走 $n$ 步再交给预测"都是合理的估计，记 $\hat A^{(n)}_t=\delta_t+\dots+\delta_{t+n-1}$。GAE
是它们的加权平均：

$$
\hat A_t=\sum_{n\ge1}(1-\lambda)\lambda^{n-1}\hat A^{(n)}_t.
$$

权重 $(1-\lambda)\lambda^{n-1}$ 加起来等于 1（几何级数），所以这是平均，不只是求和。

**一个故事帮助记住它。** 每往前看一步，以概率 $\lambda$ 继续看，以概率 $1-\lambda$ 停下。恰好看了 $n$ 步就停的概率是
$\lambda^{n-1}(1-\lambda)$，正是上面的权重。

**合并同类项。** $\delta_{t+l}$ 出现在所有 $n\ge l+1$ 的 $\hat A^{(n)}$ 里，即"走到第 $l+1$ 步还没停"，这个概率是
$\lambda^l$。所以

$$
\hat A_t=\sum_{l\ge0}\lambda^l\,\delta_{t+l},
$$

展开后与递推 $\hat A_t=\delta_t+\lambda\hat A_{t+1}$ 完全一致。

**平均看多远。** 平均步数是 $1/(1-\lambda)$。$\lambda=0.95$ 平均约 20 步（不是每次恰好 20 步）；$\lambda=1$
看到结束；$\lambda=0$ 只看 1 步。

**数值：$\lambda=0.5$，B.2 的例子。**

| 步  | $\delta$ | $\lambda=0.5$ 的 $\hat A$     | $\lambda=1$ 的 $\hat A$ |
| --- | -------- | ----------------------------- | ----------------------- |
| 3   | $-0.3$   | $-0.3$                        | $-0.3$                  |
| 2   | $-0.4$   | $-0.4+0.5\times(-0.3)=-0.55$  | $-0.4-0.3=-0.7$         |
| 1   | $+0.2$   | $0.2+0.5\times(-0.55)=-0.075$ | $0.2-0.7=-0.5$          |

第 1 步在 $\lambda=1$ 时被判 $-0.5$，其中大半是第 2 步的坏消息（$\delta_2=-0.4$）；$\lambda=0.5$ 时只剩
$-0.075$：第 1 步不再替第 2 步背锅。代价是第 1 步看到的 $R$ 更少，更依赖 critic 的 $V_2$ 是否准（B.6 的反例 2）。

几何权重只是一个选择：它只有一个参数，而且能一遍递推算完。别的加权方式也合理，没有更本质的理由。原文见
[GAE 论文](https://arxiv.org/abs/1506.02438)。

### B.8 在 LLM 里：为什么固定 $\lambda$ 会让 reward 衰减掉

LLM 里"一步"是一个 token，reward 在最后一个 token。距离结尾 $k$ 个 token 的位置，$R$ 的权重是 $\lambda^k$。

- $\lambda=0.95$，$k=1000$：$0.95^{1000}=e^{1000\ln0.95}\approx e^{-51.3}\approx5\times10^{-23}$。
  平均只往后看约 20 个 token，而 reward 在 1000 个 token 之外，这些 token 的 advantage 基本只靠 critic，等于回到
  B.6 的反例 2。
- 对话长度不一。$k=100$ 时 $0.95^{100}\approx0.6\%$。同一个固定 $\lambda$ 对长短不同的 episode 很不公平。

SAO
的做法（[lambda_fn.py:38](../../../../areal/trainer/ppo/lambda_fn.py:38)，`sao_length_adaptive_gae`）：

$$
\lambda=1-\frac{1}{\alpha L},\quad\alpha=1.5,
$$

$L$ 是这条 episode 的 action token 数。平均看 $1/(1-\lambda)=\alpha L=1.5L$ 步，比 episode
本身还长；$\lambda^L\approx e^{-1/\alpha}\approx0.51$，从第一个 token 到结尾，$R$ 的权重约一半，不会衰减到零。

原版 PPO（[Schulman 2017](https://arxiv.org/abs/1707.06347)）用固定 $\lambda=0.95$，它的实验是 MuJoCo
和 Atari，轨迹很短。针对长文本 LLM 做长度自适应 $\lambda$ 的有 [VAPO](https://arxiv.org/abs/2504.05118)（仓库里是
`vapo_length_adaptive_gae`）和 [SAO](https://arxiv.org/abs/2607.07508)。

**我们的设定**：actor 和 critic 都用
$\lambda=1$，$\gamma=1$（[config_ppo.yaml:45-46](../../../../examples/tau2/config_ppo.yaml:45)），所以没有衰减问题；代价是
B.6 反例 1 的 MC 噪声。

### B.9 Skip-Observation：跳过的是什么

容易产生一个误解："SAO 的 GAE 忽略 observation，PPO 的不忽略"。本仓库里两者用同一个 `gae.py`。`loss_mask=0` 的
token（用户回复、工具返回）不算时间步：循环里遇到它们，`nextvalues` 和 `lastgaelam`
原样带过去（[gae.py:83-87](../../../../areal/trainer/ppo/gae.py:83)），GAE 实际只在 agent token
组成的子序列上算。

"跳过"**不是**说 critic 不看用户的话。critic 在下一个 agent token 之前读到了用户回复，$W$ 已经包含了它；跳过的意思是用户 token 不是
agent 的决策，不需要给它们分配功劳。

$\lambda=1$ 时跳不跳对 agent token 的 advantage 没有影响（$\hat A_t=R-V_t$，与中间 token
无关）。$\lambda<1$ 时才有区别：不跳的话，$\lambda$ 会在几百个工具返回 token 上白白衰减；跳过之后，B.8 的 $L$ 只数 action
token。

### B.10 β：从一轮对话讲起

**三个读数。** 只看一轮 agent 发言和紧接着的一次用户回复。critic 在三个时刻各读一个数（记号与 §2.4 相同）：

| 记号      | 时刻                             | 例子         |
| --------- | -------------------------------- | ------------ |
| $V_k$     | agent 第 $k$ 轮开口前            | 0.50         |
| $U_k$     | agent 说完、用户还没回复         | 0.55         |
| $V_{k+1}$ | 用户回复之后、agent 下一轮开口前 | 看用户怎么回 |

代码注释里把 $V_{k+1}$ 叫作
$W$（[gae.py:77](../../../../areal/trainer/ppo/gae.py:77)），二者是同一个量。最后一轮之后 $V_{K+1}=R$。

**例子。** 用户要取消预订 EHGLP3。agent 这一轮问"请提供预订号"，问得规范，读数从 0.50 升到 0.55。用户的回复有两种，各一半：

| 用户回复      | $V_{k+1}$ |
| ------------- | --------- |
| 报对了 EHGLP3 | 0.75      |
| 报错成 EHGLP8 | 0.35      |

**这一轮 agent token 拿到什么。** $\lambda=1$ 时 advantage 是 $R-V_k$，按 B.1 拆成每一步的变化，开头两段是

$$
R-V_k=\underbrace{(U_k-V_k)}_{\text{agent 这句话：}+0.05}+\underbrace{(V_{k+1}-U_k)}_{\text{用户这次回复：}+0.20\text{ 或 }-0.20}+(\text{之后各轮的变化}).
$$

同一句"请提供预订号"，用户报对时 agent 被奖励 $+0.25$，报错时被罚 $-0.15$。第二段是用户的运气，幅度是 agent 自己贡献的 4 倍。这就是 §2.5
的伤害 1：噪声。

**β 做什么。** 只把第二段（用户段）乘 β，agent 段和之后各轮不动：

$$
(U_k-V_k)+\beta\,(V_{k+1}-U_k)+(\text{之后各轮的变化}).
$$

代码在 agent 这一轮最后一个 token
上做这件事（[gae.py:75-80](../../../../areal/trainer/ppo/gae.py:75)），再由递推传给这一轮和之前所有 agent
token。

| β   | 用户报对          | 用户报错          | 平均    |
| --- | ----------------- | ----------------- | ------- |
| 1   | $0.05+0.20=+0.25$ | $0.05-0.20=-0.15$ | $+0.05$ |
| 0.5 | $0.05+0.10=+0.15$ | $0.05-0.10=-0.05$ | $+0.05$ |
| 0   | $+0.05$           | $+0.05$           | $+0.05$ |

**为什么平均不变。** 用户段的平均是 $0.5\times0.20+0.5\times(-0.20)=0$。它为 0，是因为 critic 说完时报的 $U_k=0.55$
恰好等于用户回复后两种读数的平均 $0.5\times0.75+0.5\times0.35=0.55$。这就是条件
(C)：$U_k=\mathbb{E}[V_{k+1}\mid h_k^+]$。0 乘任何 β 还是 0，所以 β 不改变这句话的平均得分，只缩小用户运气带来的摆动。β
处理的是伤害 1（方差）。

**可预测的不可靠不受影响。** 如果这个用户一被问就有很大概率报错，准确的 critic 在 agent 说完时就会把 $U_k$ 报低。这部分落在 agent 段
$U_k-V_k$，β 管不到，agent 仍会学"对这个用户换一种问法"。这是伤害 2，β 本来就不处理（§2.5）。

**(C) 不成立时出问题。** 换一个例子：agent 这一轮其实调错了工具，取消了另一个预订，但普通 critic 当时没看出来，仍报
$U_k=0.55$。真实情况是之后基本必败，错误要到最后 $R$ 才暴露，落在用户段（最后一轮的用户段是 $R-U_K$）。这一段的平均不再是 0，而是负的，而且是
agent 的错。β 把它缩小，就等于替 agent 打了折，这是偏差。旧的统一 β 就是这样失败的。带剧本的 critic 知道用户要取消的是 EHGLP3，agent
取消错时 $U_k$ 就该掉下去，错误回到 agent 段，β 不再碰它。§5.2 的 D3 测的就是这一点。

**β 是不是只在用户边界上的 λ。** 作用相似：都是"这一段少看真实结果，多信 critic 的平均"。区别是 λ 对每一步都打折，并把衰减传给后面所有步；β
只缩这一次用户回复，之前和之后的变化照常全额累加（[gae.py:81](../../../../areal/trainer/ppo/gae.py:81)）。agent 自己的
token 仍是 $\lambda=1$，critic 的误差只在用户边界进入 advantage。

**分工与实验。** 带剧本的 critic 让 (C) 尽量成立，β 才只缩用户的运气；β 把这份运气从 agent 的分数里拿掉。2×2 实验（§5.1）里 run
1（带剧本 + β=0.5）对 run 2（带剧本 + β=1），测的就是：在带剧本的 critic 下，β 降方差的收益是否大于 (C) 不完全成立带来的偏差。

## 附录 C：剧本样例

critic 前缀就是下面这样的文本。它和用户模拟器拿到的 `str(task.user_scenario)` 完全一样，外面套上 `<user_profile>`
块。Token 数用 Qwen3 tokenizer 计算，包括首尾的特殊 token。训练 pack 里的剧本长度：airline 中位数 416、最长
1917；retail 中位数 784、最长 1532。

**官方 airline_0（150 token）**：

```text
<|im_start|>system
<user_profile>
Instructions:
	Domain: airline
	Reason for call:
		You want to cancel reservation EHGLP3.

		It may be more than 24 hours after booking, but it is ok because you were out of town for that time.
	Known info:
		You are Emma Kim.
		Your user id is emma_kim_9957.
	Task instructions:
		If Agent tells you that cancellation is not possible,
		mention that you were told that you didn't need to get insurance because your previous trip was booked with the same agency with insurance.

		You don't want to cancel if you don't get a refund.
</user_profile><|im_end|>
```

**pack airline_1147（416 token，airline 的中位长度）**：

```text
<|im_start|>system
<user_profile>
Instructions:
	Domain: airline
	Reason for call:
		Upgrade cabin class from basic economy to economy for existing reservation
	Known info:
		user_id: anya_anderson_8585, reservation_id: 2KZLFK, current_cabin: basic_economy, desired_cabin: economy, payment_method: credit_card_4619444
	Task instructions:
		You are Anya Anderson, a gold member with user_id 'anya_anderson_8585'. You have a basic economy reservation (reservation ID: 2KZLFK) for a one-way trip from Boston (BOS) to Phoenix (PHX) on May 16, 2024, with a connection in Orlando (MCO). You want to upgrade to economy class for more comfort and flexibility.

		Your task is to:
		1. Contact the airline agent and provide your user_id when asked: 'anya_anderson_8585'
		2. Tell the agent you want to upgrade your reservation from basic economy to economy class
		3. Provide your reservation ID when asked: '2KZLFK'
		4. When the agent explains the price difference, confirm that you understand and want to proceed
		5. When asked for a payment method, provide one of your credit cards: 'credit_card_4619444'
		6. Confirm the modification when the agent requests final confirmation
		7. After the modification is completed, thank the agent

		Important guidelines:
		- Be clear that you want to upgrade the cabin class to economy, not change the flights
		- You are willing to pay the price difference
		- If asked about baggage, mention that you don't need to add any checked bags at this time
		- Answer any verification questions truthfully based on your user information
</user_profile><|im_end|>
```

**pack retail_492（784 token，retail 的中位长度）**：

```text
<|im_start|>system
<user_profile>
Instructions:
	Domain: retail
	Reason for call:
		Customer wants to cancel all pending orders and update addresses across multiple orders
	Known info:
		{"name": {"first_name": "Iris", "last_name": "Cox"}, "user_id": "iris_cox_9643", "email": "iris.cox7821@example.com", "order_ids": ["#W3310761", "#W6417791", "#W7904368"], "order_items": [...5 个商品及其选项...]}
	Task instructions:
		**User Goals:**
		1) Cancel ALL pending orders - customer has multiple orders with different statuses and wants to cancel every order that is still pending
		2) Update shipping addresses on all three orders - customer recently moved and needs to change the delivery address on all their orders to the new address

		**Details:**
		- Customer has 3 orders total: 2 pending orders and 1 delivered order
		- Customer wants to cancel both pending orders with reason 'no longer needed'
		- Customer needs to update addresses on all 3 orders to new address: 742 Elm Street, Apt 5B, Seattle, WA 98101
		- Customer will authenticate via email: iris.cox7821@example.com

		**Constraints & Behavior:**
		[Pattern A] User behavior: User mentions ALL goals at once in the first message - ...
		- User will confirm each cancellation when prompted
		- User will provide new address details when asked
		- For address updates, user wants the same new address applied to all orders
		Update the shipping addresses of my eligible pending orders before cancelling them. Do not change my default profile address.
		Whenever you give an address, the country is USA (say exactly "USA").
</user_profile><|im_end|>
```

（retail_492 的 `order_items` 和 `[Pattern A]` 那一行在这里有删节，critic 实际读到的是全文。）

**要注意的一点**：pack 里的剧本比官方任务详细得多。它常常写出逐步的流程（"1. 提供 user_id……6.
确认修改"）、要用的参数（取消原因、新地址），有时还写了操作顺序（"先改地址再取消"）。这虽然不是
`evaluation_criteria`，但已经接近正确的动作序列，critic 几乎能直接判断 agent 每一步对不对。对 baseline 的无偏性没有影响（剧本在
episode 开始前就固定了），但论文里不能把它说成"只给了用户身份"，应该写成"给了用户模拟器拿到的全部剧本，其中可能含有接近解答的步骤"。
