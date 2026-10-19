# 研究方向：用户不可靠时怎么训练多轮 agent（2026-10-19 重写）

本文是当前版本，只写现在有效的问题定义、方法和实验计划。2026-10-11 到 10-19 的旧内容（β 拆分、截断 + bootstrap、检测器、逆方差加权草案的完整讨论和决策表）在 git 提交 `5f66f66` 的同名文件里；§7 只保留改变过方向的决策。其他文档里在 10-19 之前引用的 problem-and-plan §1–§5，指的是那个旧版本。

## 1 问题

**可验证但不可归因。** 用 RL 训练多轮客服 agent 时，结局 reward 由 agent 和用户共同决定。verifier 能判断任务是否完成，判断不了是谁导致了结局。用户在确认的同一句里离开（U1）、违反自己的任务或编造事实（U3）、没办完就走（U4b）时，agent 即使每一步都做对也得 0 分，负 advantage 落在 agent 此前的正确行为上。

**现象。** 同一个 Qwen3.5-4B 在 τ² Retail 上只换用户模拟器，成功率从 85.0% 降到 65.0%；agent 自身造成的失败始终是 10–15 条，用户造成的从 7 到 45 条（[user-simulator.md §2](../user-simulator.md)）。最强的模拟器下，用户仍造成 29% 的失败。

**用户不是一个。** 即使只用一个模拟器模型，τ² 每道题给用户的 instruction 和 persona 都不同（有的就是要扮演难缠或古怪的用户），训练里天然是多用户。换模拟器模型只是让用户分布变化更大。

**设定。** PPO（带 critic），每题一条 rollout。可用的信息：对话、结局 reward，以及**模拟器里才有的用户剧本**（这道题给用户的 instruction 和 persona）。评测用固定的可靠用户，并检查 policy 合规。部署时 actor 不需要剧本；剧本只给训练时的 critic 用，所以本方法针对"用模拟器训练"，不覆盖"从真人部署中学习"。

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

### 2.4 跨过一次用户回复：一个 TD 被恒等拆成两段

记 agent 第 $k$ 轮开口前的读数为 $V_k$，第 $k$ 轮说完、用户还没回复时的读数为 $U_k$（value\[$b$\]，见
附录 A），用户回复后、下一轮开口前为 $V_{k+1}$，最后一轮之后 $V_{K+1}=R$。对第 $t$ 轮里的 agent token $i$，$R-V_i$
可以逐项展开（中间读数正负抵消）：

$$
R-V_i=\underbrace{(U_t-V_i)+\sum_{k>t}(U_k-V_k)}_{\text{agent 段：agent 自己的话和工具调用带来的变化}}+\underbrace{\sum_{k\ge t}(V_{k+1}-U_k)}_{\text{用户段：用户回复带来的变化}}.
$$

这是恒等式，没有任何假设。它把 PPO 已经在用的 advantage 按"谁说话"切开。工具返回是确定的数据库结果，留在 agent 段。

**用户段的条件均值。** 给定 agent 第 $k$ 轮说完的历史 $h_k^+$，如果 critic 在用户回复前后自洽，

$$
U_k=\mathbb{E}\big[V_{k+1}\mid h_k^+\big],\tag{C}
$$

那么用户段条件均值为 0：它是"给定 agent 这句话，用户回复得好或坏"的纯噪声。推导和"critic 整体偏高也不破坏 (C)"的说明见旧版 §4.3（`5f66f66`）。

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

| 方法 | 做法 | 结果 | 证据 |
| --- | --- | --- | --- |
| 统一 β | 所有用户段乘 $\beta=0.5$ | 中间用户段几乎为 0（$+0.006\pm0.061$），β 实际只作用在最后一次回复；那里 U1 只占 11–12%，其余是结算才暴露的 agent 错误。631 的 Retail 在 checkpoint 120 比 PPO 低 13.75 pp，后期多出的正是工具不报错、用户还道谢的 agent 写错 | [critic/user-boundary-forward.md](../critic/user-boundary-forward.md)、[ppo-beta05-eval-drop/REPORT.md §5](../ppo-beta05-eval-drop/REPORT.md) |
| 规则门控 β | 只在 U1 规则命中的边界取 $\beta=0$ | 规则召回高，但反事实 precision 约 58%：很多"用户 yes+STOP"之前 agent 的方案本来就错 | [user-error-detectors/REPORT.md §0a–0b](../user-error-detectors/REPORT.md) |
| judge 门控 | LLM 读全轨迹判责任方和边界 | 认得出 U3，但在训练分布上 precision 76%，把 12.8% 的 agent 失败判给用户；每次 33.5k token | [user-error-detectors/REPORT.md §0c](../user-error-detectors/REPORT.md) |
| 截断 + bootstrap | 用户出错处截断，用 $U_k$ bootstrap | 依赖检测器；$U_k$ 和 $V_t$ 差不多，效果接近 mask | 旧版 §5 |

共同的原因有两层：

1. **critic 不知道真相。** 现在的 critic 只看对话，看不出用户说的话违反了剧本，也看不出 agent 的写操作不符合用户的真实意图。所以两类错误都要到结算才暴露，挤在最后一次回复的用户段里（[critic/user-boundary-forward.md](../critic/user-boundary-forward.md) 结论 1、3：$U$ 对"提议是否正确"的 AUC 只有 0.573）。§2 的分解在恒等式层面成立，但在这个 critic 下，"用户段"并不等于"用户的作用"。
1. **检测器做不准，而且依赖 domain。** 规则只认得 U1；judge 要再引入一个 LLM，还要按 domain 维护错误分类。老师和我们都判断这条路做不通（2026-10-19）。

β、规则门控和 judge 保留实现，作为消融。

## 4 方法：给 critic 看用户剧本

### 4.1 做法

critic 是独立于 actor 的模型，现在读的是和 actor 完全相同的 token 序列：system 消息（domain policy、工具列表）加对话。我们只在 critic 读的那份序列里，在对话开始前插入这道题的用户剧本：

```text
<|im_start|>system
<policy>...</policy>  # Tools ...
<user_profile>                        ← 只有 critic 读到
reason_for_call / known_info / unknown_info / task_instructions / persona
</user_profile><|im_end|>
<|im_start|>user
Hi, I want to return ...              ← 从这里起两边完全一样
```

- 剧本取自 τ² 任务的 `user_scenario`（用户模拟器拿到的就是它，`examples/tau2/agent.py:457`）。**不放** `evaluation_criteria`：没有 gold 动作序列、没有 NL 断言、没有 reward 规则。
- actor 的输入、loss、GAE 都不变。critic 的训练目标仍是 $R$（$\lambda_{\text{critic}}=1$）。
- 插在对话之前，critic 序列里此后所有位置统一偏移 $L$ 个 token，读 value 时减掉偏移即可与 actor 对齐。

### 4.2 为什么无偏

剧本 $z$ 在对话开始前就定了，不依赖 agent 的任何动作。按 §2.1 的同一个论证，$V_\phi(h_t, z)$ 是合法的 baseline，期望梯度不变。这与 Le Critique（[2608.16739](https://arxiv.org/abs/2608.16739)）的约束相同：特权信息不能包含当前回答的未来 token、realized reward 或 verifier 反馈。

### 4.3 它改变了什么

critic 知道用户应该怎么做之后：

- **用户违反剧本时**（U3），价值在那次用户回复处下跌，落进用户段；
- **agent 做了不符合用户意图的写操作时**（例如该退单件却取消整单），价值在 agent 那一轮下跌，落进 agent 段；
- **用户随机地在确认时离开**（U1），仍落在最后一次回复的用户段。

于是 §2 的分解第一次有机会真正按"谁的错"切开。这有两个用处：

1. **更好的 baseline。** 单是 $V(h,z)$ 比 $V(h)$ 准，advantage 方差就更小。Le Critique 在单轮推理和多轮数独上都看到了 EV 提升。
1. **让 β 只打折用户的作用。** $\lambda=1$ 时 $A_t=R-V_t$。如果用户在中途犯 U3，agent 此前正确的 token 拿到的负 advantage 来自那一轮的用户段；只有把用户段缩掉（β），这些 token 才不被罚。旧的统一 β 失败，是因为用户段里混着 agent 的错误；剧本让 agent 的错误移出用户段，β 就只作用在用户身上。

所以完整的方法是 **带剧本的 critic + 用户段乘 β**。β 的无偏条件仍是 §2.4 的 (C)。伤害 2（可预测的不可靠诱导 agent 少确认）在 agent 段里，这个方法不处理，留给 §6。

### 4.4 critic 需不需要先离线训好

不需要，直接在 PPO 里从现有初始化开始训。理由：

- **原理上**：$\lambda=1$ 时 critic 只是 baseline，不准只增加方差、不引入偏差（§2.1）。PPO 每一步都用 on-policy 的 Monte Carlo 回报训练它，它跟着当前策略的分布走；离线拟合的是旧策略的分布。
- **经验上**：critic-only 离线训练（479、499、510）验证 EV 一直在 0 附近，最好 0.1–0.2（[critic/README.md](../critic/README.md)）；放进 PPO 后 EV 在第 21–40 步到 0.19–0.23，最后 20 步 0.34–0.36（[ppo-beta05-eval-drop/REPORT.md §5](../ppo-beta05-eval-drop/REPORT.md) 引的 644 学习信号表）。
- **初始化**：沿用 ADR 0012（从 policy 初始化、head bias 取训练成功率），不做 warmup（2026-10-15 的决定）。critic 的骨干没见过 `<user_profile>` 块，但它是普通文本，在线训练会学会用它；学没学会由 §5 的诊断回答。

## 5 实验设计

### 5.1 对照：只改 critic 的输入

同一个用户（豆包 Turbo，`doubao-seed-2-1-turbo-260628`）、同一个 judge（`deepseek-v4-1-flash-261017`）、同一套配置，2×2：

| | 普通 critic | 带剧本的 critic |
| --- | --- | --- |
| $\beta=0.5$ | **642**（在跑） | **新 run 1**（完整方法） |
| $\beta=1$ | 新 run 3（PPO 基线） | 新 run 2（只有更好的 baseline） |

- **先跑 run 1。** 它和 642 只差 critic 的输入，单变量；它也是完整方法。critic 训练不依赖 β（target 都是 $R$），所以 run 1 和 642 的 critic 可以直接比较诊断指标。
- run 2、run 3 在 run 1 的诊断通过后再跑，补齐"剧本的作用"和"β 的作用"。
- 每个 run 都需要用户批准 GPU；gpu8 现在被 642 占用。

### 5.2 诊断：剧本有没有让用户段等于用户的作用

critic 每 10 步保存一次。训练结束后（或中途），用 run 1 和 642 的同步数 critic checkpoint，在**同一批有标签的轨迹**上前向，复用 `tau2-critic-boundary-20261018/` 的边界读数脚本。带剧本的 critic 输入里加剧本，普通 critic 不加。

数据：v3/v4 多模拟器轨迹（qwen 用户 U3 多、Turbo 用户 U1 多，主因和出错消息已标注）、639/640 的 U1 标签、643 的 J 输出。训练分布的 DeepSeek 用户 U3 只占约 1%，测不出第 2 项，所以必须用多模拟器数据。

判据在跑之前固定（数值待用户确认）：

| # | 检查 | 普通 critic 的现状 | 带剧本的 critic 通过的条件 |
| --- | --- | --- | --- |
| D1 | 训练中的 EV（以 outcome 计） | 642 同步数的值 | 同步数窗口均值高 ≥ 0.05 |
| D2 | 标注为 U3 的那次用户回复，用户段 $V_{k+1}-U_k$ | 中间用户段约 0 | 均值 ≤ −0.2，且区分 U3 回复与正常回复的 AUC ≥ 0.7 |
| D3 | 主因是 agent 写错的失败，跌幅落在哪 | 主要在最后一次回复（$R-U$） | 落在最后一次回复的比例降到一半以下，跌幅前移到写错的那一轮 |
| D4 | $U$ 对"提议是否正确"的 AUC | 0.573 | ≥ 0.65 |

- D2、D3 都不通过：剧本没有让 critic 分开两类错误，β 部分没有依据，方法退化为"更好的 baseline"，看 run 2 是否还有收益。
- D1 不通过：critic 没学会用剧本，先查输入构造和对齐。

### 5.3 训练结果怎么读

- 主指标：固定可靠用户上的 eval 成功率（Airline/Retail/Telecom 分开），以及写前确认率等 policy 合规指标。
- run 1 对 642：带剧本是否让 β 不再伤 Retail（631/642 的主要问题）。
- 单 seed；按 task 聚类给区间，窗口在跑前固定。

### 5.4 实现（Codex 实现，Claude review，测试先红后绿）

1. critic 输入构造：从任务取 `user_scenario`，渲染 `<user_profile>` 块，只插进 critic 的序列；记录偏移 $L$，value 读数减去 $L$ 后与 actor 位置对齐。测试：插入前后 actor 输入逐 token 不变；对齐后 critic 的 $V_t$、$U_k$ 位置与不插入时一一对应；`evaluation_criteria` 的任何字段都不出现在 critic 输入里。
1. 配置开关放在 tau2 的配置类（`examples/tau2/utils.py`），不改 `areal/api/cli_args.py`。
1. 序列长度：剧本一般几百 token，只加在 critic 侧；检查 critic 的 max length 和显存。

### 5.5 风险

- **critic 学不会用剧本**：D1、D2 会显示出来。
- **剧本泄露答案**：剧本里有订单号、用户真实意图，比用户身份多很多信息。它不是 gold，但"gold 不进 critic"的旧决定是否允许剧本进 critic，需要用户确认。
- **新颖性**：特权 critic 本身不新（Le Critique 给答案，SWEET-RL 给参考答案，asymmetric actor-critic 给完整状态）。差别在于给的是用户剧本、目的是把用户的作用和 agent 的作用分到不同的段；能不能站住取决于 D2、D3。动手前还要做一次定向检索：LLM 多轮 agent RL 里有没有人把用户的隐藏目标给 critic。

## 6 之后：按 persona 缩放（D2、D3 通过后再做）

老师的直觉是"置信度低的用户少学"。卡点是可辨识性：成功率低可能是用户不可靠，也可能是用户难缠但 agent 能应对（后者该多学）。带剧本的 critic 下两者落在不同的段：难缠但可应对的用户，结局由 agent 怎么应对决定，变化在 agent 段；不可靠的用户，agent 说完之后结局仍取决于他乱不乱来，用户段波动大。

所以可靠度可以定义为"agent 说完之后，这类用户的回复还会让结局波动多少"：按 persona 用过去的 episode 估计用户段方差 $\hat\sigma_p^2$，episode 乘

$$
w_p=\frac{1}{1+\hat\sigma_p^2/\tau},
$$

$\tau$ 取全体平均，样本少的 persona 向全局收缩。这是逆方差加权；用户只加噪声时任意正权重不改梯度方向。它是唯一能动伤害 2 的部分。旧的逆方差草案（16.11）被否掉的两个原因（测量位置事后选、残差混着 agent 错误）在这里分别由固定的用户边界和剧本解决。

## 7 决策轨迹（只保留改变方向的决策；完整表见 `5f66f66`）

| 日期 | 决策 | 原因 |
| --- | --- | --- |
| 10-11 | 起点：老师提出按用户反馈的置信度调整 advantage，critic 也考虑用户 | 好用户下分数高、差用户下学不会 |
| 10-12 | 逆方差加权草案暂停，先找准现象 | 数学过重，核心估计量有设计错误 |
| 10-12 | 轻量方法：用户段乘 β | 粒度到每次回复，一个超参数 |
| 10-13 | 研究对象定为不可靠用户；模拟器是可控的代理 | 真人同样会确认后离开、改主意 |
| 10-18 | 加截断 + bootstrap，检测器可替换 | U1 能用 prompt 修掉，U3 修不掉、β 管不到 |
| 10-19 | 放弃检测器路线（规则、judge），保留作消融 | 643：judge 在训练分布上把 12.8% 的 agent 失败判给用户；规则只认 U1；老师判断做不通 |
| 10-19 | **新方法：给 critic 看用户剧本 + 用户段乘 β；先跑 run 1 并用 D1–D4 诊断** | 旧方法失败的共同原因是 critic 不知道真相；剧本在开始前确定，作为 baseline 输入无偏；不需要 judge 和 domain 专属分类 |
| 10-19 | critic 不做离线预训练，直接在 PPO 中训练 | $\lambda=1$ 时 critic 只是 baseline；离线 critic-only 的 EV 一直在 0.1–0.2 以下，PPO 中可到 0.3 以上 |


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

