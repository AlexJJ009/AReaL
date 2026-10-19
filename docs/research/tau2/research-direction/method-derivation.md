# 从 PPO 推到我们的方法，以及转向"用户池 + 用户能力"（2026-10-19）

本文写给第一次读方法的人。从 PPO 的 loss 和 GAE 出发，一步步推到 β 方法，说明它为什么只管得到 U1 这类错误、管不到 U3；然后按老师 2026-10-19
的意见，推导"用户池 + 用户能力系数"的改法，并检查其中需要修正和要先验证的地方。

token 位置、chat template 和 critic 读数点的细节见
[problem-and-plan.md §4.1](problem-and-plan.md)，这里不重复。代码行号对应本仓库当前分支。

## 1 起点：PPO 在我们这里到底算了什么

### 1.1 目标与 policy gradient

训练目标是期望成功率 $J(\theta)=\mathbb{E}_{x,\,u,\,\tau\sim\pi_\theta}[R]$，其中 $x$ 是任务，$u$
是用户，$\tau$ 是整段对话，$R\in\{0,1\}$ 是结局 reward。policy gradient 定理给出

$$
\nabla_\theta J=\mathbb{E}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(Q(s_t,a_t)-b(s_t)\big)\Big].
$$

$b(s_t)$ 是 baseline。只要它不依赖当前动作 $a_t$，就有
$\mathbb{E}_{a_t}[\nabla\log\pi(a_t\mid s_t)\,b(s_t)]=b(s_t)\,\nabla\sum_a\pi(a\mid s_t)=0$，所以
**减任何不依赖动作的 baseline 都不改变梯度的期望，只改变方差**。取 $b=V(s_t)$ 时，$Q-V$ 就是 advantage
$A_t$。后面所有关于"有偏/无偏"的判断都回到这一条。

### 1.2 PPO 的 loss：advantage 是每个 token 的权重

PPO 用重要性比 $\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\text{old}}(a_t\mid s_t)$ 和裁剪：

$$
L^{\text{CLIP}}(\theta)=-\,\mathbb{E}_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,\hat A_t\big)\Big].
$$

AReaL 的实现是 `pg_loss = max(-A·ρ, -A·clip(ρ))`，再按 `loss_mask` 对 agent token
求平均（`areal/utils/functional/functional.py:658-678`）。 critic 用平方误差拟合 return $\hat G_t$（带
value clip）。

在 $\rho\approx1$ 附近，$L^{\text{CLIP}}$ 的梯度就是
$-\hat A_t\nabla\log\pi(a_t\mid s_t)$。**reward 不进 loss，进 loss 的只有逐 token 的
$\hat A_t$**。所以"改方法"就是"换一种 $\hat A_t$"：可以平移它（加减一个量），也可以缩放它（乘一个系数）。这两种操作性质不同，§4 会反复用到。

### 1.3 GAE：稀疏的 reward 怎么变成每个 token 的 advantage

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

### 1.4 跨过一次用户回复：一个 TD 被恒等拆成两段

记 agent 第 $k$ 轮开口前的读数为 $V_k$，第 $k$ 轮说完、用户还没回复时的读数为 $U_k$（value\[$b$\]，见
[§4.1](problem-and-plan.md)），用户回复后、下一轮开口前为 $V_{k+1}$，最后一轮之后 $V_{K+1}=R$。对第 $t$ 轮里的 agent
token $i$，$R-V_i$ 可以逐项展开（中间读数正负抵消）：

$$
R-V_i=\underbrace{(U_t-V_i)+\sum_{k>t}(U_k-V_k)}_{\text{agent 段：agent 自己的话和工具调用带来的变化}}+\underbrace{\sum_{k\ge t}(V_{k+1}-U_k)}_{\text{用户段：用户回复带来的变化}}.
$$

这是恒等式，没有任何假设。它把 PPO 已经在用的 advantage 按"谁说话"切开。工具返回是确定的数据库结果，留在 agent 段。

**用户段的条件均值。** 给定 agent 第 $k$ 轮说完的历史 $h_k^+$，如果 critic 在用户回复前后自洽，

$$
U_k=\mathbb{E}\big[V_{k+1}\mid h_k^+\big],\tag{C}
$$

那么用户段条件均值为 0：它是"给定 agent 这句话，用户回复得好或坏"的纯噪声。推导和"critic 整体偏高也不破坏 (C)"的说明见
[§4.3](problem-and-plan.md)。

## 2 问题：用户错误怎么进入梯度

用 §1.4 的分解，用户带来的伤害可以精确地分成两种。

**伤害 1：方差。** 用户段是围绕 0 的随机波动。agent 每一步都做对，用户随机地在确认时离开，用户段就是一个大负数，落到 agent 此前所有 token
上。期望梯度不变，但方差大，同样的预算学得少。

**伤害 2：目标本身被带偏。** 如果用户的不可靠是**可预测**的，例如 agent 一问"确认吗？"，这个用户就有 40% 在同一句里 STOP，那么在 critic
准确时，$U_k$ 在问确认之后本身就低。这一项在 agent 段里，梯度会正确地告诉 agent："在这个用户下，问确认是坏动作"。对
$J$（在这个用户下的成功率）来说这是对的，对"agent 自己的能力和 policy 合规"来说是错的。

由此得到一个硬结论：**任何对 $J$ 无偏的 advantage 估计都改不了伤害 2**，因为伤害 2 就是 $J$
本身。要处理它，只能改目标：给不同的轨迹或用户不同的权重，或者把某些 token 移出 loss。降方差的方法只能处理伤害 1。

## 3 现有方法：β 只缩用户段

### 3.1 推导

把用户段乘 $\beta\in[0,1]$：

$$
\tilde A_i=(R-V_i)-(1-\beta)\sum_{k\ge t}(V_{k+1}-U_k).
$$

token 级实现是只改跨过用户回复的那一个 TD：$\tilde\delta_{b-1}=(U-V_{b-1})+\beta\,(W-U)$（`gae.py:79`），其余
token 的 TD 不变。

- **无偏条件**：(C) 成立时，第二项条件均值为 0，$\tilde A$ 只比 $A$ 少了噪声。(C) 不成立时，偏差
  $\propto(1-\beta)\,e_k$，$e_k=\mathbb{E}[V_{k+1}\mid h_k^+]-U_k$，与 GAE 的 $\lambda<1$
  是同一类交换。
- **它处理的是伤害 1**。伤害 2 在 agent 段里，β 不碰。

### 3.2 为什么它只管得到 U1，管不到 U3

关键是：用户段要有东西可缩，critic 必须在**用户回复的那一刻**看出局面变了。

- **U1（确认的同一句里 STOP）。** 这是终止边界上的随机事件：agent 已经请求确认，用户回复 STOP，写操作没执行，$R=0$。终止边界上
  $V_{K+1}=R$，用户段就是 $R-U_{\text{last}}$，数值大，β 能缩掉它。
- **U3（违反任务、编造事实）。** 用户在中间某一轮给了错误信息，agent 照着做，最后数据库对不上。critic
  没有剧本、没有正确答案，看不出这条信息是错的，所以在那一轮 $V_{k+1}\approx U_k$，用户段几乎为 0。失败要到结局才暴露，落在终止边界的
  $R-U_{\text{last}}$ 里。

第二点有一个副作用：**终止边界的 $R-U_{\text{last}}$ 混着所有"只有结局才暴露"的错误，包括 agent 自己的错误**。统一的 β
会把它们一起打折。pueue 644 在 631 上看到的正是这种情况：Retail 后期多出的错误是写入没有来源的地址、整单取消代替单件取消这类工具不报错、用户还道谢的
agent 错误 （[ppo-beta05-eval-drop/REPORT.md §5](../ppo-beta05-eval-drop/REPORT.md)）。

### 3.3 门控 β 和 judge：为什么暂停，留作消融

要让 β 只作用在用户出错的边界上，必须先定位"哪次回复是用户出错"。规则只认得出 U1；U3 要靠能读剧本、读数据库事实的 judge。实测结果：

- pueue 643 的全轨迹归因 judge 在旧 Doubao 标注集上 any-user precision 92%；但在训练分布的标签上 precision
  76%，把 agent 失败误归给用户的比例为
  12.8%（20/156）（[user-error-detectors/REPORT.md §0c](../user-error-detectors/REPORT.md)）。
- 这类方法依赖 domain 专属的错误分类和 judge prompt，每换一个环境都要重做。

老师判断这条路做不通；我们保留实现，作为 §4 新方向的消融（"按回复内容条件化"的代表）。

## 4 新方向：用户池与用户能力（老师 2026-10-19 的意见）

### 4.1 设定

训练时有一个用户池 $\mathcal U=\{u_1,\dots,u_m\}$：不同模型、不同 persona 的模拟器，可靠度不同。每个 episode 在开始前抽一个用户
$u\sim p(u)$，抽法与 agent 的策略无关；训练器知道这条轨迹的用户是谁（真实部署里对应用户 ID）。

**注意：只有一个用户时，下面的修正恒为 0。** 这个方向要求用混合用户池训练。当前的 642（只用 Doubao Turbo）测不了它，原计划的 Doubao β=1
对照也只是在测旧方法。

### 4.2 老师的公式

用户能力的总体平均为 $\eta$，用户 $u$ 的累计估计为 $\hat\eta_u$（例如这个用户最近的平均成功率），能力优势
$A_2=\hat\eta_u-\eta$，代入 PPO：

$$
\min\big(\rho_t(A_1+A_2),\ \mathrm{clip}(\rho_t)\,(A_1+A_2)\big),\qquad A_1=\text{GAE advantage}.
$$

critic 同时考虑 agent 和 user。下面逐项检查。

### 4.3 检查一：$A_2$ 的符号

$A_2$ 只依赖用户身份 $u$，而 $u$ 在 episode 开始前抽取、与动作无关。按 §1.1 的同一个论证：

$$
\mathbb{E}\big[\nabla\log\pi(a_t\mid s_t)\,c(u)\big]=\mathbb{E}_u\Big[c(u)\,\mathbb{E}\big[\nabla\log\pi(a_t\mid s_t)\mid u\big]\Big]=0 .
$$

所以**加 $+A_2$ 还是 $-A_2$ 都无偏**，二者的区别只在方差。方差最小的平移量是 $c(u)\approx-\mathbb{E}[A_1\mid u]$。当
$A_1$ 的 baseline 不认识用户时（例如 GRPO 组里混着不同用户，或 critic
认不出用户），$\mathbb{E}[A_1\mid u]\approx\eta_u-\eta$，于是

$$
\hat A=A_1-(\hat\eta_u-\eta),
$$

**符号是减号**。用 $+A_2$ 会把用户造成的差异放大一倍。

**例子。** 同一个 agent，用户 G 的成功率 0.8，用户 B 是 0.4，总体 $\eta=0.6$。不认识用户的 baseline 都取 0.6：

| 情形         | $A_1=R-0.6$ | $A_1-(\hat\eta_u-\eta)$ | $A_1+(\hat\eta_u-\eta)$ |
| ------------ | ----------- | ----------------------- | ----------------------- |
| G 用户下成功 | $+0.4$      | $+0.2$                  | $+0.6$                  |
| G 用户下失败 | $-0.6$      | $-0.8$                  | $-0.4$                  |
| B 用户下成功 | $+0.4$      | $+0.6$                  | $+0.2$                  |
| B 用户下失败 | $-0.6$      | $-0.4$                  | $-0.8$                  |

减号那一列就是老师说的"置信度"：在不可靠的用户 B 下失败，信息量小，少罚；在可靠的用户 G 下失败，多半是 agent 的错，多罚。加号那一列正好反过来。

这个平移等价于以用户为条件的 baseline $b(u)$：P-GRPO（2603.10009）按用户 ID 维护 reward
均值和方差，做的就是这件事（[related-work.md §2](related-work.md)）。

### 4.4 检查二：PPO 里 critic 可能已经认出了用户

PPO 的 $A_1=R-V(s_t)$。critic 在 agent 第一次开口之前就读到了用户的第一句话，如果不同用户的说话风格可以分辨，$V(s_t)$ 已经近似以
$u$ 为条件，$\mathbb{E}[A_1\mid u]\approx0$，再减 $A_2$ 就是重复扣。这时干净的写法是把用户身份直接给 critic：

$$
\hat A_t=R-V_\phi(s_t,u).
$$

baseline 可以用 actor 看不到的信息，仍然无偏（非对称 critic）。这就是"critic 同时考虑 agent 和 user"的一种落地方式，也是
[related-work.md §2](related-work.md) baseline 表里的第 6 项。critic 是否已经认出用户，只能用 critic
forward 测（训练导出里没有 value，见 644）。

**batch 级 advantage 归一化的影响**：减均值只去掉全局均值，按用户的平移在 batch 内保留；除以 std 是统一缩放，不改变用户之间的相对大小。

### 4.5 检查三：能力系数是缩放，不是平移

把 HACPO 的能力系数搬到用户上，是乘法：

$$
\hat A=\omega_u\,A_1,\qquad \omega_u=\hat\eta_u/\eta .
$$

乘法不是 baseline，它改的是目标：$\nabla J_\omega=\mathbb{E}_u[\omega_u\nabla J_u]$，可靠的用户权重大。这正是 §2
说的"只有改目标才能动伤害 2"：钻不可靠用户空子的捷径（例如少确认）只在不可靠用户下有回报，而这些用户的权重被降低了。所以**能力系数是这个方向里唯一能动伤害 2
的部分**，平移只降方差。

代价是可辨识性问题（决策 11）：用户成功率低，可能是因为他不可靠（该少学），也可能是因为 agent 还不会应对这类用户（例如用户不主动给订单号，agent
该追问），后者恰恰该多学。降权会把训练推向容易的用户。

和 HACPO 的类比在这里变弱了：HACPO 的 $\omega$ 缩放的是**别的 agent 生成的 off-policy
样本**，衡量的是"这份数据有多可信"；我们的样本全是自己的 on-policy 轨迹，用户只改变环境。平移那一半可以直接搬过来，系数这一半需要单独论证。

### 4.6 怎么估计 $\hat\eta_u$

- **在同一时间窗口里取相对值。** agent 在进步，所有用户的成功率一起上涨。HACPO 用同一个 batch 内的比值 $P^{(k)}/P^{(j)}$，agent
  的进步在比值里约掉。我们也应该用同一窗口内的相对量，而不是累计的绝对成功率。
- **按任务配对。** 用户 $u$ 碰到的题目可能偏难。最直接的做法是同一道题、不同用户之间比较；也可以拟合
  $\Pr(R=1)=\sigma(a+d_x+q_u)$（agent 能力 + 题目难度 + 用户可靠度），取 $q_u$。
- **成功率能不能当可靠度用。** 我们已有一条证据：同一个 Qwen3.5-4B 在 Retail 上换五个模拟器，agent 自身造成的失败始终是 10–15
  条，用户造成的从 7 到 45 条（[user-simulator.md §2](../user-simulator.md)）。对固定的
  agent，用户之间成功率的差异几乎全部来自用户。这支持在**用户之间**用成功率比较可靠度；它不说明单条轨迹里谁犯了错。

### 4.7 统一的框架：按"用户是谁"条件化，还是按"用户说了什么"条件化

把 §3 和 §4 放在一起，可以得到一个比单个方法更干净的主张：

| 条件化的对象                            | 例子                             | 是否无偏                                           | 管什么                          |
| --------------------------------------- | -------------------------------- | -------------------------------------------------- | ------------------------------- |
| 用户是谁（$u$，开始前抽取，与动作无关） | 以用户为条件的 baseline / critic | 恒无偏                                             | 伤害 1 中用户之间的那部分       |
| 用户是谁，乘法                          | 能力系数 $\omega_u$              | 改目标（有意为之）                                 | 伤害 2 的激励                   |
| 用户说了什么（回复内容，依赖 agent）    | β 拆分、门控 β、judge mask       | 只在 (C) 成立时无偏；judge 误判时把 agent 的错免掉 | 伤害 1 中单条回复的那部分（U1） |

经典 RL 里，Mao et al.（ICLR 2019）的结论是：baseline 可以以外部输入为条件，前提是这份输入与 agent
的动作独立。用户身份满足这个前提，用户的回复内容不满足。631/644 的 Retail 结果是"按回复内容条件化"出问题的一个实例。这个对比可以作为论文的主线；β 和
judge 成为它的消融。

### 4.8 新颖性：要正面说清楚

- 按用户的 baseline 平移：P-GRPO 已做（推荐和偏好场景，GRPO，无 critic）。
- 按成功率加权 advantage：ProxMO（2602.19225）已做（决策 4）。
- 按标注者可靠度加权样本：RGPO（离线标注）。
- 我们 2026-10-11 至 10-12 的决策 4、11、12 已经分析过这些部件，当时因为可辨识性和 critic 吸收残差的问题暂停。

所以单独的"$A_1-A_2$"或"$\omega_u A_1$"新颖性都弱。能站住的是：(1) 多轮交互 agent、多用户池训练这个设定；(2) §4.7 的"按用户是谁
vs 按用户说了什么"的区分及其无偏性，配上 β/judge 失败的实证；(3) "便宜、不可靠的模拟器池 + 用户感知的
advantage，接近用强模拟器训练"这个实用结论。

论文叙事也要跟着改：intro
现在把"让模拟器多样化"列为"把用户做可靠"的昂贵补救之一，而用户池训练正站在那一边。转向后要改成"多样但不可靠的用户池是常态，问题是怎么从中学"。

## 5 动手前的检查（离线、无 GPU、无 API）

用 v3/v4 多模拟器数据（同一 actor、同一批题、多个模拟器、已有失败主因标签）：

1. **方差分解。** 结局 $R$ 的方差里，"题目 × 用户"中用户身份解释多少。很小的话，这个方向能拿到的就很少；这个结果本身也有效。
1. **成功率是否可以当可靠度。** 每个用户的成功率与"用户造成的失败占比"的相关性。
1. **梯度方差的数值模拟。** 用实测的各用户成功率，比较 $A_1$、$A_1+A_2$、$A_1-A_2$、$\omega_uA_1$ 的梯度估计方差。

需要 GPU 的一项：用已保存的 critic 在多用户轨迹上 forward，看 $V$ 在第一轮是否已经按用户分开（决定 PPO 里要不要显式给 critic 用户身份）。

之后的最小训练实验：用户池（例如 GLM、DeepSeek、Doubao Turbo、Qwen 自部署）上训练 PPO，对比 (a) 不区分用户；(b) 以用户为条件的
critic 或 $A_1-A_2$；(c) 再加能力系数；(d) β / judge 消融。评测用训练没见过的固定可靠用户，并报 policy 合规。都需要用户批准。
