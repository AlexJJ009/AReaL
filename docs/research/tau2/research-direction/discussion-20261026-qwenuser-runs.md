# 10月26日讨论记录：Qwen 用户训练的几个 run，以及要补的 case study

2026-10-26。数据读于 01:57–03:30 PDT；run 4 当时训练到第 60 步左右，还没有评测。图由
[`scripts/figures/plot_qwenuser.py`](../scripts/figures/plot_qwenuser.py)
在开发机上生成，评测和训练日志新增后重跑即可更新。

## 0 先看手里已有的结果

1. **用 Qwen3.5-4B 当用户训练，换 DeepSeek 当用户评测，带剧本 critic 的 run 2′ 比标准 critic 的 run 3′ 好。**
   Retail step 120：65.6 对 54.4，差 +11.3pp \[+1.3, +21.2\]；Telecom（训练里没有）+12.5pp \[+3.7,
   +21.9\]；Airline 无差别。run 2′ 与用 DeepSeek 用户训练的 PPO（v3ext）持平（+2.5pp \[−7.5, +12.5\]）。
1. **但换回 Qwen 用户评测，排名反过来。** Retail 43.1 对 48.8，−5.6pp \[−15.3,
   +3.7\]，单看不显著；两个评测者给出的差值之差（交互）是 +16.9pp \[+2.2, +30.9\]，CI 不含 0。
1. **带剧本 critic 的 EV 更高**：第 101–120 步均值 0.520 对 0.397，差 +0.123 \[+0.070, +0.176\]（按步
   bootstrap）。训练成功率只高 +0.019 \[−0.002, +0.038\]。
1. **Telecom 在 Qwen 用户下训不出东西，已从训练集去掉。** 同一个 step 0 agent，Telecom 在 DeepSeek 用户评测下
   46–49%，在 Qwen 用户评测下 5.6%；Qwen 用户在长对话里不发 STOP，约 45% 的 Telecom episode 撞 200 条消息上限。
1. **critic 在弱用户下能不能看出 agent 的错误，还没有证据。** §5 A 给出了 Qwen 用户下的失败构成和阅读清单；§5 B 在 DeepSeek 用户的已有数据上看到，带剧本 critic 对写错库的失败认得更早，但对用户提前 STOP 两者都看不出；Qwen 用户下的 value 分析（§5 C）等 GPU。

局限：每个 run 只有一个训练种子；DeepSeek 评测每个 checkpoint 只跑了一次，同一 checkpoint 两次评测最多相差约
6pp；上面做了十余次比较，单个 CI 不含 0 不能当定论。

## 1 思路的转变：为什么放弃 $A = a + \epsilon$

10-22 的出发点：一个 agent 时间步上观测到的 advantage $A$ 里，既有 agent 动作的贡献 $a$，也有用户带来的噪声 $\epsilon$；估计出
$\epsilon$（或直接估计 $a$），就能给 agent 它"本应得到"的 advantage。

结论是这个拆分在我们的设定下做不到，理由有三条（详细推导见 [reflection-20261024 §2](reflection-20261024.md) 和
[m2-additive-vs-multiplicative](m2-additive-vs-multiplicative-20261022.md)）。

**(1) 不可辨识。** 只看对话和结局 reward $R$ 时，"用户错了"、"用户难缠但 agent 没应对好"、"agent
先错了、用户顺着走"能产生同样的数据，$\epsilon$ 没有被数据确定。我们试过的四种做法都是在这里失败的：

| 做法                         | 失败方式                                                           |
| ---------------------------- | ------------------------------------------------------------------ |
| 统一 β（对用户段的 TD 打折） | 最后一次用户回复里混着 agent 自己的错误，β 替 agent 打了折         |
| 规则门控                     | "yes + STOP"之前 agent 的方案常常已经错了，反事实 precision 约 58% |
| judge 门控                   | 把 12.8% 的 agent 失败判给用户；judge 本身也面对同样的不可辨识     |
| 按剧本可靠度加权             | 剧本方差主要反映 agent 的时成时败，不反映用户                      |

导师不接受额外的检测器或 judge（10-21）。这样一来，能补上缺失信息的只剩用户剧本 $z$：它写明了用户想要什么、该怎么表现。

**(2) 线性拆分把要学的东西也拆掉了。** 用户 T=0
时，给定对话历史，用户的回复几乎确定，"实现出来的噪声"期望为零、也很小。剩下的"可预测的不可靠"（这个用户在这种情况下就会提前 STOP）就是目标
$J$（在这个用户下的成功率）的一部分，对 $J$ 无偏的估计碰不到它；不用检测器时，加法修正只能由 critic 的 value 变化构成，每回合取一个数就回到了 β。

**(3) 现象在训练分布里很小。** DeepSeek 用户下 U1（用户在确认处提前 STOP）约 2% 的 episode，U3（偏离目标）约 1%，失败约 92% 是
agent 自己的错。用户错误大的地方是换弱模拟器评测，而不是训练。

### 1.1 为什么想到 bootstrap：$A=a+\epsilon$ 本来就藏在 GAE 里

拆分做不到，但 10-22 想要的东西并没有消失。把 advantage 按时间展开（$\gamma=1$，§3.2 有完整式子）：

$$\hat A_t=\sum_{k\ge0}\lambda^k\,\delta_{t+k},\qquad \delta_j=r_j+V_{j+1}-V_j .$$

每个 $\delta_j$ 是对话里某一处的"一步意外"：value 在这一步涨了还是跌了。对话是 agent 回合和用户回复交替进行的，所以这些意外分两类：

- 落在 agent 回合里的 $\delta$：agent 这一步让成功的可能变大还是变小，对应 10-22 的 $a$；
- 跨过一次用户回复的 $\delta$（用户说完之后的 value 减去说之前的 value）：用户这句话带来的变化，对应 10-22 的
  $\epsilon$，而且它有确定的位置。

所以 $A=a+\epsilon$ 不是新的分解，GAE 已经把 advantage 写成"各处意外之和"。问题在于 $\lambda$：

- **$\lambda=1$ 时，这些意外加起来正好消掉，只剩 $\hat A_t=R-V_t$。** 不管 critic 把意外放在哪一步，梯度只看起点和终点，"$a$ 和
  $\epsilon$ 各是多少"对 agent 没有任何影响。这就是 run 2′ 的带剧本 critic 只能降方差的原因。
- **以前的 β 方法**是把跨用户回复的 $\delta$ 乘上 β。它要知道哪一次用户回复是用户的错，所以离不开检测器或 judge。
- \*\*$\lambda<1$（bootstrap）\*\*不判断谁的错，只让每个 agent token 主要对它之后不远处的意外负责：$k$ 步之外的意外权重是
  $\lambda^k$。离它很远的用户回复（包括最后的 STOP）带来的意外，对它的影响自动变小。

也就是说，bootstrap 是一个不估计 $\epsilon$、也不需要 judge，却能让 $\epsilon$ 少算到 agent 头上的办法。代价是 advantage
里出现了 critic 的估计值：critic 错了，错误就进了梯度（§3.2 例子 2 的第 2 条）。

### 1.2 为什么认为"动态 $\lambda$ + 带剧本 critic"可能解决问题

bootstrap 能不能把 credit 放对，取决于 critic 把意外放在了哪里。两个组件各管一半：

1. **带剧本 critic 负责把意外放在对的位置。** 标准 critic $V(h)$ 不知道用户真正要什么，agent
   写错了它常常看不出来，意外一直拖到结算时的最后一次用户回复才出现；这时 bootstrap 会把罚算到结尾附近的 token 上，而不是写错的那一步。$V(h,z)$
   读了剧本，能在 agent 写错时就降下来，用户偏离剧本时在那次回复上降下来。DeepSeek 用户下的已有数据（§5 B）部分支持前半句：486
   条写错库的失败里，到写错后约一个回合，带剧本 critic 已完成总跌幅的 41%，标准 critic 只有 7%；但数据不足以确认是不是恰好降在写错那一步。后半句没有得到支持：用户提前
   STOP 的 60 条失败里，两个 critic 在结算前都几乎不降。Baisero & Amato（AAMAS 2022）还给了理论保证：$V(h,z)$ 这种 history-state critic 用来
   bootstrap 不引入额外偏差，偏差只来自 critic 自身的误差。
1. **动态 $\lambda$ 负责让位置真的影响梯度，同时不丢掉结局信号。** 按 token 计时，一条 episode 有几千个 agent token。固定
   $\lambda=0.95$ 的有效视野只有 20 个 token，1000 个 token 外 $R$ 的权重约 $5\times10^{-23}$，advantage
   几乎全由 critic 决定。SAO 的 $\lambda=1-1/(1.5L)$ 随轨迹长度变化，保证第一个 token 上 $R$ 的权重仍有约
   0.51。critic 的回归目标仍用 $\lambda=1$（$G_t=R$），这样 critic 本身不被自己的估计带偏（VC-PPO 的做法，SAO 论文 §4.1
   同样设 $\lambda_{\text{critic}}=1$）。

合起来的预期：agent 写错导致的失败，罚集中在写错之前和写错那一步，之后的 token 少罚；用户在结尾偏离导致的失败，前面做对的 agent token
只承担一部分。这正是 10-22 想要的"agent 本应得到的 advantage"，只是不靠估计 $\epsilon$，而靠 critic 的信息加上 $\lambda$
的衰减。

这里用"可能"，是因为有两个前提都还没验证：

- **前提 1：Qwen 用户下，带剧本 critic 仍然能把意外放对位置。** 649 是 DeepSeek 用户的数据。弱用户自己会乱说，critic
  可能把用户的乱说也学成了"agent 不行"。§5 C 直接测这一点（GPU，待批准）。
- **前提 2：部分挪动就够了。** SAO 的 $\lambda$ 接近 1，写错之前本来正确的 token 仍要背约 3/4 的罚（§3.2 例子 2 的第 1
  条）。挪动这一部分能不能改变训练结果，要看 run 4 和 run 5 的 2×2
  比较（[run card §13](../experiments/2026-10-24-qwenuser-no-telecom.md)）。如果 run 4 不比 run 5
  好，或两者都不比 $\lambda=1$ 好，这条线就该停。

## 2 现在发现的问题

PPO 和 SAO 训练中已经看到的（来源在括号里）：

1. **Telecom 用 Qwen 用户训不动**（[infra-issues §5](../infra-issues.md)、上面的第 4 条）：问题在用户模拟器，不在
   agent。run 3 带着 Telecom 训练，训练成功率 0.39 → 0.44；去掉后的 run 3′ 是 0.55 → 0.64（第 1–10 步 → 第
   101–120 步均值，两者训练集不同，不能直接比）。
1. **结论依赖评测用户。** 同一对 checkpoint，DeepSeek 用户和 Qwen 用户给出相反的排名（第 0 节第 2 条）。Qwen
   用户评测本身噪声也大：同一 checkpoint 两次评测相差 −5.0 / −2.5 / +6.2pp。
1. **Airline 在所有 run 里都不涨**：训练集里需要拒绝或不写库的题占 15.8%，test 里占
   70%（[train-test-coverage](../train-test-coverage/REPORT.md)，backlog #1，未改数据）。
1. **对话变长、撞上下文上限。** SAO 让超过 32K 的 episode 从 4 增到 76（reflection §5.2）；run 2′ 在 Qwen 评测里有
   4/320 条因上下文超限失败，run 3′ 没有。
1. **合规侵蚀**：标准 PPO 下"同意不足"（B5）+14–17pp，agent
   学会的是问得不完整，不是不问（[compliance-erosion](../compliance-erosion-case-study/REPORT.md)）。
1. **Qwen 用户下的轨迹还没有人读过。** 第 0 节的每条结果都只有分数。

## 3 现在的方法：advantage 和 loss 的完整式子

以下按代码和 run 2′ / run 4
实际加载的配置写（`areal/trainer/ppo/gae.py`、`actor.py`、`lambda_fn.py`、`critic.py`、
`areal/utils/functional/functional.py`）。

### 3.1 符号

- 一条 episode 里 agent 生成的 token 记为 $t=0,\dots,L-1$（$L$ 只算 agent 的 token，含 EOS；用户、工具返回和
  prompt 不算时间步，GAE 递推直接跳过它们）。
- $R\in\{0,1\}$ 是 τ² 的结局 reward，放在最后一个 agent token 上；中间 reward 为 0；$\gamma=1$；KL 系数为 0。
- $V_t$ 是 critic 在 token $t$ 处的 value。标准 critic 读对话 $h_t$；带剧本 critic 另外在开头读用户剧本
  $z$（persona、来电原因、已知/未知信息、行为指示，不含 gold 动作和评测字段，`examples/tau2/critic_scenario.py`）。终止后
  $V_L=0$。

### 3.2 advantage

TD 误差 $\delta_t = r_t + V_{t+1} - V_t$，GAE

$$
\hat A_t=\sum_{k\ge 0}\lambda^k\delta_{t+k}
=\lambda^{L-1-t}R+(1-\lambda)\sum_{j\ge 1}\lambda^{j-1}V_{t+j}-V_t .
$$

第二个等号是把 $\delta$ 展开后合并（$\gamma=1$）：token $t$ 的"目标"是 $R$ 和未来 value 的加权平均，$R$ 的权重是
$\lambda^{L-1-t}$。

- **run 2′ / 3′：$\lambda=1$。** $\hat A_t=R-V_t$，critic 只是 baseline。期望梯度和 critic 无关，带剧本
  critic 只能降方差。
- **run 4 / 5：SAO 长度自适应 $\lambda=1-\dfrac{1}{\alpha L}$，$\alpha=1.5$，每条轨迹一个 $\lambda$。**
  第一个 token 上 $R$ 的权重约 $\lambda^{L}\approx e^{-1/\alpha}=0.51$，剩下 0.49 分给未来的 value；越靠后
  $R$ 的权重越大。

critic 的回归目标单独用 $\lambda_{\text{critic}}=1$ 算：$G_t=R$，与 actor 的 $\lambda$ 无关。

actor 用的 advantage 再做 batch 白化：$\tilde A_t=(\hat A_t-\mu)/(\sigma+10^{-8})$，$\mu,\sigma$
是整个 batch 所有 agent token 上的均值和标准差。SAO 论文没有写是否白化，这是本项目的选择。

AReaL 代码默认不白化（`areal/api/cli_args.py` 里 `adv_norm` 默认 `None`），是
`examples/tau2/config_ppo.yaml` 显式打开的。白化只是整体平移和缩放，不改变同一 batch 里 token 之间 advantage
的相对大小；关掉白化的对照还没跑。

**数值例子（白化之前）。** 两个例子共用一条 episode 和两种 critic：

- **episode**：$L=3000$ 个 agent token（编号 0–2999）。agent 在 token 1000 处写错了库，任务失败，结局 reward
  $R=0$，只给在最后一个 token 上。
- **看出错误的 critic**（带剧本 critic 的理想情况）：token 0–1000 上 $V=0.6$（估计有 60% 会成功），从 token 1001 起
  $V=0.1$。
- **看不出错误的 critic**（标准 critic 常见的情况）：一直 $V=0.6$。

这些 value 是为了说明而设的，不是实测。

**例子 1：$\lambda=1$（run 2′ / 3′）。** 每个 token 的目标就是真实结局 $R$，$\hat A_t=R-V_t=0-V_t$：

- 看出错误：token 0–1000 都是 $0-0.6=-0.60$；token 1001 以后是 $0-0.1=-0.10$。
- 看不出错误：所有 token 都是 $-0.60$。

写错之前的 token 和写错的那个 token 都是 −0.60，梯度分不出哪一步错了。两种 critic 只在写错之后不同：看出错误的 critic 知道这些 token
已经救不回结局，少罚。由于 $V_t$ 只依赖 token $t$ 之前的对话，减掉它在期望上不改变梯度，只改变噪声，这就是"只降方差"。

**例子 2：SAO $\lambda$（run 4 / 5）。**
$\lambda=1-1/(1.5\times3000)=1-1/4500\approx0.99978$。token $t$ 的目标不再是 $R$ 本身，而是 $R$
和它之后各个 token 的 value 的加权平均：还剩 $n$ 个 token 时，$R$ 的权重是
$\lambda^n\approx e^{-n/4500}$，其余权重分给之后的 value，离得近的分得多。$\hat A_t$ = 目标 $-V_t$。

- **token 0**（$n=2999$）：$R$ 的权重 0.51，token 1–1000 的 value 合计 0.20，token 1001–2999 合计
  0.29。
  - 看出错误：目标 $=0.51\times0+0.20\times0.6+0.29\times0.1=0.15$，$\hat A_0=0.15-0.6=-0.45$。
  - 看不出错误：目标 $=0.51\times0+0.49\times0.6=0.29$，$\hat A_0=0.29-0.6=-0.31$。
- **token 1000，写错的那一步**（$n=1999$）：$R$ 的权重 0.64，其余 0.36 全在 token 1001 以后。
  - 看出错误：目标 $=0.36\times0.1=0.04$，$\hat A=0.04-0.6=-0.56$。
  - 看不出错误：目标 $=0.36\times0.6=0.22$，$\hat A=0.22-0.6=-0.38$。
- token 1001 和最后一个 token 2999 按同样的算法得到下表。

| $\lambda$ | critic     | token 0 | token 1000（写错） | token 1001 | token 2999（最后） |
| --------- | ---------- | ------- | ------------------ | ---------- | ------------------ |
| 1         | 看出错误   | −0.60   | −0.60              | −0.10      | −0.10              |
| 1         | 看不出错误 | −0.60   | −0.60              | −0.60      | −0.60              |
| SAO       | 看出错误   | −0.45   | −0.56              | −0.06      | −0.10              |
| SAO       | 看不出错误 | −0.31   | −0.38              | −0.38      | −0.60              |

例子 2 的读法：

1. **SAO + 看出错误的 critic**：罚从 token 0 的 −0.45 逐渐加重，到写错那一步 −0.56 最重，写错之后骤降到 −0.06。和
   $\lambda=1$ 相比，写错的那一步第一次比开头的 token 罚得重，梯度能分出"错在哪里"。但写错之前本来正确的 token 仍要背 −0.45，是
   $\lambda=1$ 时的 3/4：$\lambda$ 接近 1，只能把罚往后挪一部分，不能精确定位到写错的那一步。
1. **SAO + 看不出错误的 critic**：罚从开头的 −0.31 一路加重到结尾的 −0.60，写错的那一步（−0.38）和它之后的 token
   一样重。罚被推向对话结尾，而不是写错的地方：critic 错了，bootstrap 就把错误带进 advantage。

所以"动态 $\lambda$ + 带剧本 critic 可能就够了"取决于一个可检验的前提：**在 Qwen 用户下，带剧本 critic 的 value 会不会在
agent 出错的地方提前下降，而标准 critic 不会。** DeepSeek 用户下看到了一部分：写错库的失败，结算时剩下的意外带剧本 critic
是 −0.26，标准 critic 是 −0.45，带剧本 critic 的下降更早（§5 B）。Qwen 用户下还没测，§5 C 测它。

### 3.3 loss

actor（PPO，decoupled 形式）：

$$
\mathcal L_{\text{actor}}=-\frac{1}{N}\sum_{t}w_t\,\min\!\big(\rho_t\tilde A_t,\ \mathrm{clip}(\rho_t,0.8,1.2)\,\tilde A_t\big),
\qquad
\rho_t=\frac{\pi_\theta(a_t)}{\pi_{\text{prox}}(a_t)},\quad
w_t=\frac{\pi_{\text{prox}}(a_t)}{\pi_{\text{behave}}(a_t)} .
$$

- $\pi_{\text{behave}}$ 是采样时推理引擎的策略（异步 rollout，最多落后 2 个版本），$\pi_{\text{prox}}$
  是训练端更新前重算的 log-prob；$w_t>5$ 的 token 直接剔除（mask）。$N$ 是 batch 里 agent token
  总数（token-mean）。
- 每步只有一个 minibatch、一次梯度更新，更新时 $\pi_\theta=\pi_{\text{prox}}$，$\rho_t=1$，clip 从不触发（日志里
  clip ratio 和 approx KL 都是 0）。实际梯度是
  $-\frac1N\sum_t w_t\tilde A_t\nabla\log\pi_\theta(a_t)$。
- 无 KL、无 entropy 项；actor lr $10^{-6}$。

critic（value clip 0.5）：

$$
\mathcal L_{\text{critic}}=\frac{1}{N}\sum_t\max\!\Big((V_t-G_t)^2,\ \big(V^{\text{old}}_t+\mathrm{clip}(V_t-V^{\text{old}}_t,\pm0.5)-G_t\big)^2\Big),
\qquad G_t=R ,
$$

critic lr $10^{-5}$，每步一个 minibatch。EV 的定义 $1-\mathrm{Var}(V-G)/\mathrm{Var}(G)$，在更新前的同一
batch 上计算。

## 4 结果

![qwenuser_runs](../figures/qwenuser_runs.png)

图：前两行是离线评测的 mean@4（每题多次成功率的平均，agent T=0，用户和 judge T=0），阴影是按题 bootstrap 的 95% CI；第一行用
DeepSeek 官方 `deepseek-flash` 作用户和 judge，第二行用 Qwen3.5-4B 作用户、火山 DeepSeek V4.1 Flash 作
judge（run 3′ / 2′ 两次评测合并，每题 8 次；不评 Telecom）。第三行是训练 batch 的成功率（128 条 episode，T=1）和 critic
EV（更新前），细线每步、粗线 10 步滑动均值。颜色表示 critic（蓝标准、绿带剧本），线型表示 actor $\lambda$（实线 1，虚线 SAO）；灰色 run
3 训练集含 Telecom。黑色星号是用 DeepSeek 用户训练的 PPO（v3ext）第 120 步。run 3 的 Qwen 评测在第 70 步之后每个
checkpoint 只完成了 330–378/400 条（评测被中止），图中只用完成的 episode。run 4 只有训练曲线（进行中）。EV 取日志里的
`critic_explained_variance_pre_update`；每步只有一个 minibatch，它和 10-19 图用的
`critic_explained_variance` 数值相同（run 3′ / 2′ 第 101–120 步均为 0.397 / 0.520），两张图可以直接比。

**表 1　DeepSeek 用户评测，mean@4（%）**

| checkpoint                     | Airline                   | Retail                    | Telecom                   |
| ------------------------------ | ------------------------- | ------------------------- | ------------------------- |
| step 0                         | 13.8                      | 45.0                      | 49.4                      |
| run 3′ 30 / 60 / 90 / 120      | 21.3 / 16.3 / 17.5 / 18.8 | 51.9 / 50.6 / 58.8 / 54.4 | 46.3 / 49.4 / 45.0 / 55.0 |
| run 2′ 30 / 60 / 90 / 120      | 17.5 / 17.5 / 20.0 / 17.5 | 57.5 / 56.9 / 65.6 / 65.6 | 50.6 / 56.3 / 55.6 / 67.5 |
| v3ext 120（DeepSeek 用户训练） | 13.8                      | 63.1                      | 52.5                      |

**表 2　Qwen 用户评测，mean@4（%），两次评测合并**

| checkpoint                | Airline                   | Retail                    |
| ------------------------- | ------------------------- | ------------------------- |
| step 0                    | 16.3                      | 36.3                      |
| run 3′ 30 / 60 / 90 / 120 | 20.0 / 19.4 / 27.5 / 21.3 | 35.6 / 39.7 / 42.5 / 48.8 |
| run 2′ 30 / 60 / 90 / 120 | 15.6 / 25.0 / 15.6 / 23.1 | 33.8 / 43.1 / 44.1 / 43.1 |

**表 3　配对比较（按题配对 bootstrap，20,000 次）**

| 比较                                               | 差值    | 95% CI           |
| -------------------------------------------------- | ------- | ---------------- |
| DeepSeek 评测，Retail：run 2′ − run 3′（step 120） | +11.3pp | \[+1.3, +21.2\]  |
| DeepSeek 评测，Telecom：同上                       | +12.5pp | \[+3.7, +21.9\]  |
| DeepSeek 评测，Airline：同上                       | −1.3pp  | \[−11.3, +10.0\] |
| DeepSeek 评测，Retail：run 2′ − v3ext              | +2.5pp  | \[−7.5, +12.5\]  |
| Qwen 评测，Retail：run 2′ − run 3′                 | −5.6pp  | \[−15.3, +3.7\]  |
| 交互，Retail：(DeepSeek 差值) − (Qwen 差值)        | +16.9pp | \[+2.2, +30.9\]  |

**表 4　训练端（第 101–120 步均值）**

| run                 | 训练成功率 | critic EV（更新前） |
| ------------------- | ---------- | ------------------- |
| run 3（含 Telecom） | 0.443      | 0.486               |
| run 3′              | 0.638      | 0.397               |
| run 2′              | 0.657      | 0.520               |

**结论**

- 观测：带剧本 critic 在 DeepSeek 用户评测下 Retail、Telecom 都明显更好，在 Qwen 用户评测下不更好；EV 后期高 0.12。
- 推断：$\lambda=1$ 时带剧本 critic 不改变期望梯度（§3.2），两个 run 优化的是同一个目标。所以评测者交互要么来自随机种子（两个 Qwen
  下差不多好的策略，在 Qwen 测不到的行为上各自漂向不同方向），要么来自更低的梯度方差让策略少漂向"Qwen
  不罚、强用户会罚"的方向。现有数据分不开这两种，需要换种子复现。
- 未验证：run 2′ 的哪些行为让它在 DeepSeek 用户下更好；带剧本 critic 在 Qwen 用户下是否在 agent 出错处提前降 value。

## 5 case study

### A　轨迹对照（CPU，Codex pueue 704，10-26 已完成，不用 GPU、不调 API）

任务目录 `phai:_artifacts/AReaL/codex-sao-math/case-study-qwenuser-20261026/`（`BRIEF.md`，结果写
`REPORT.md`）。问题：

1. Qwen 用户下失败的构成：第一个 epoch 和最后四个 epoch，run 2′ 和 run 3′ 各自的 stop
   reason、轮数、上下文超限，以及只用现有字段（不用 LLM）的失败分类：agent 写错库、漏了 gold 写操作、写对了但库仍不一致、用户在写完前 STOP
   或离开、上下文/轮数上限、infra。
1. run 2′ 的行为和 run 3′ 有什么不同：对话长度、工具调用、写库前的确认、转人工、结束方式。
1. 交互来自哪些题：step 120 的两份评测里，哪些 Retail 题在 DeepSeek 下 run 2′ 赢、在 Qwen
   下不赢，这些题的失败分类在两个评测者下各是什么。
1. 给你读的 20 对轨迹（同一道题，run 2′ 对 run 3′），每对标出决定结局的消息位置和查看命令。

这一步只给事实和阅读清单，不替你下结论。按 10-24 的约定，先自己读、写下判断，再看 Codex 的总结。

结果：`REPORT.md`（失败构成、行为差异、交互题）和 `READING_LIST.md`（20 对轨迹，每对标了决定结局的消息位置）。在开发机上并排看第 1 对：

```bash
python /data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/case-study-qwenuser-20261026/view_pair.py --pair 01 --spec
```

`--spec` 同时打印用户剧本和 gold，`--around 3` 只看定位点附近。

### B　DeepSeek 用户下，critic 在出错处有没有降（CPU，Codex pueue 706，10-26 已完成）

**结论。** 写错库导致的失败，带剧本 critic 认出得更早：到写错后约一个回合，它已完成总跌幅的四成，标准 critic 不到一成。用户提前 STOP
导致的失败，两个 critic 在结算前都几乎不降。数据里没有写错那一回合结束时的 value，所以"恰好降在写错那一步"既没有证实，也没有否定。这些都是
DeepSeek 用户下的结果，不回答 Qwen 用户下会怎样（那是 C）。

**数据。** 649 case study 已经让两个 critic 在同一批轨迹上前向过，这次只重新分析存下的 value，没有新的前向。

- 轨迹：649 run（带剧本 critic，β=1，用户 `deepseek-v4-1-flash-261017`）第 17–20 个 epoch 的训练 rollout。去掉 9
  条截断后共 2,979 条、746 道题。
- 带剧本 critic 取 649 第 120 步。标准 critic 取 v3ext（`ppo-formal-qwen3-v3-ext`）第 120 步。v3ext 是另一个
  run，用户是 `deepseek-flash`、并发 128，所以这是两个训练经历不同的 critic 在比，差别不能全归给剧本；而且 649 的轨迹不是标准 critic
  训练时见过的分布。
- 失败分类沿用 A 的规则：写库和 gold 不一致（W）486 条，用户在 gold 写完前结束 155 条，撞上限 9 条，其他 52 条，成功 2,286
  条。用户提前 STOP 用 U1 规则标，失败里有 60 条。

**数据缺口。** 当时的前向只在 agent→用户的边界上存 value：agent 回合开始时的 $V_t$、agent 说完时的 $U_b$、用户回复后的
$V_{next}$。写库是一次工具调用，后面接的是工具返回而不是用户，所以写错那一回合结束时的 value 一条也没存（0/486）。能测到的最近一点是写错之后第一个
agent→用户边界，中位数晚 1 个回合（范围 1–18）。

**agent 写错库（W，486 条，全部 $R=0$）。**

| 量                                       | 带剧本 | 标准   | 差 [95% CI]             |
| ---------------------------------------- | ------ | ------ | ----------------------- |
| 包含写错的那段区间里 value 的变化        | −0.125 | −0.024 | −0.100 [−0.121, −0.081] |
| 写错后第一个可测边界处，已完成的总跌幅   | 41%    | 7%     | +33pp [+29, +38]        |
| 结算时剩下的意外 $R-U_{last}$            | −0.26  | −0.45  | +0.20 [+0.17, +0.22]    |
| 写错后的边界上，同题成功 vs 失败的 AUC | 0.86   | 0.61   | +0.26 [+0.18, +0.33]    |

定义：前三行是 486 条 W 的均值。"总跌幅"是每条 episode 的 $R-V_{first}$（带剧本平均 −0.52，标准 −0.59），比例是均值之比。"包含写错的区间"是写错前后最近两个存了
value 的回合起点之间，中位数跨 1 个回合，里面可能还有工具返回和后续对话。AUC 只在能和同题成功轨迹对齐到同一回合位置的样本上算：240 对、181 条
W、147 道题，偏向对话还能继续下去的 episode。CI 是按 domain:task 聚类的 bootstrap，2,000 次。

按写错所在的回合对齐，回合起点 value 的均值（每格的样本不完全相同，只看形状）：

| critic | 写错前 1 回合 | 写错回合 | 后 1 回合 | 后 2 回合 | 结算前最后边界 |
| ------ | ------------- | -------- | --------- | --------- | -------------- |
| 带剧本 | 0.48          | 0.47     | 0.39      | 0.34      | 0.26           |
| 标准   | 0.54          | 0.59     | 0.61      | 0.50      | 0.45           |

读法：

1. [观察] 带剧本 critic 写错后一个回合就开始降，之后一直降。标准 critic 写错后反而微升，再慢慢降，近一半的跌幅留到结算。
1. [推断] 这在 DeepSeek 用户下支持 §1.2 前提 1 的较弱版本："带剧本 critic 更早认出这条要失败"。但写错后一个回合只完成四成，其余六成还在后面；按预登记，这不满足"大部分跌幅在写错回合结束前完成"的强版本。
1. [未验证] 跌幅是否恰好落在写错的那次工具调用上：数据里没有那个点。

**用户提前 STOP（U1，60 条失败，56 道题）。**

- [观察] 从开头到最后一个边界，value 的净变化是带剧本 +0.01 [−0.03, +0.05]、标准 −0.01 [−0.04, +0.02]，都接近
  0。跌幅几乎全在结算：带剧本 −0.73，标准 −0.68。
- [观察] STOP 前的最后一句用户回复上，带剧本 critic 微升 +0.02，标准微降 −0.02，正常回复约为 0。方向和"带剧本 critic
  提前识别用户要乱结束"相反。
- [推断] 两个 critic 都预见不到用户会提前 STOP。这类失败在 SAO $\lambda$ 下前面的 token 能少背罚，靠的是 $\lambda$
  的衰减，不是剧本：value 不降，token 0 的目标里 0.49 的权重落在仍然偏高的未来 value 上（§3.2 例子 2 的算法）。
- 限制：U1 是规则标签。按严格定义（被 STOP 打断的确实是正确的写操作），历史抽检的 precision 只有 31/80 = 38.8%；detector
  报告里常引的 82.6% 是"用户是主因"的宽口径，不能当这里的可信度。643 的归因 judge 和检测器标签与这批轨迹没有重合，中间回合的用户错误没有真值可用。

**SAO 下罚落在哪。** [未验证] 用边界 value 近似 token 级 advantage 时，写错回合缺的 value 往前补还是往后补，会让带剧本 critic
的"写错回合比之前罚得重"反过来（−0.04 对 +0.08）。所以这批数据回答不了 §3.2 例子 2 在真实轨迹上是否成立。

**对 C 的要求。** 要回答"是否恰好降在写错那一步"，必须存每个 agent 回合（包括工具调用回合）开始和结束时的 value。C 的前向已经按这个准备。

完整报告：`phai:_artifacts/AReaL/codex-sao-math/critic-error-response-deepseek-20261026/REPORT.md`。

### C　Qwen 用户下，critic 能不能看出 agent 的错误（GPU，CPU 准备已完成，等你批准）

B 用的是 DeepSeek 用户的数据，A 只读了轨迹文本。Qwen 用户下还没有任何 value，要用保存的 critic 重新前向。

run card：

- **问题**：Qwen 用户下，带剧本 critic 的 value 是否比标准 critic 更早在 agent 出错处下降（§1.2 前提 1）？在评测轨迹上，DeepSeek
  用户和 Qwen 用户的结论是否一样？
- **数据**（Codex pueue 705 已在 CPU 上准备好，共 8,671 条，匹配排除 2 条，0.02%）：
  1. 训练轨迹：run 2′ 和 run 3′ 第 97–120 步的训练 rollout，各 3,016 条，用户是 Qwen。
  1. 评测轨迹：step 0、run 3′ 第 120 步、run 2′ 第 120 步，DeepSeek 用户 1,199 条（airline + retail + telecom），Qwen
     用户 1,440 条（airline + retail，两次评测），包括 A 找出的交互题。
- **做法**：沿用 649 的前向流程，同一批轨迹分别给 run 2′ 和 run 3′ 第 120 步（`globalstep119`）的 critic，只换
  critic。四组数据都存了训练时的原始 token，直接用，不重新 tokenize。与 B 不同，这次存每个 agent token 的 value，包括工具调用回合的开始和结束。
- **唯一的变量**：critic 是否读剧本。两个 critic 的训练经历也不同，差别不能全归给剧本。
- **看什么**：与 B 相同的量，加上 B 测不到的写错回合本身：写错回合内 value 的变化、到写错回合结束已完成的跌幅、写错回合结束时的同题
  AUC，以及用真实 token value 算的 SAO $\lambda$ 下写错前、写错回合、写错后的 advantage。另看 EV、first / mid /
  last-U 的同题 AUC，以及两个 critic 的差别在 DeepSeek 用户和 Qwen 用户的评测轨迹上是否一样。判据已在
  `preregistration.md` 里先登记。
- **预期**：带剧本 critic 对写错库的失败更早下降，并且下降集中在写错回合；对用户提前 STOP 的失败，两者都降得晚（B 里看到的样子）。
- **无信息的结果**：两个 critic 的下降时机和 AUC 相同。那样 run 4 相对 run 5 不应有优势，"动态 $\lambda$ + 带剧本
  critic"缺少机制支持。
- **限制**：评测轨迹是 T=0 采样，DeepSeek 用户的评测轨迹也与 critic 的训练分布不同，测的是迁移，不是 in-sample 能力。截断的 90
  条撞上限轨迹不算结算意外和 SAO 指标。
- **成本**：8 张卡，约 30–80 分钟（按 649 的速度外推；不确定 649 的 11 分钟是一个 critic 还是两个合计），无 API 费用。命令在
  `critic-forward-qwenuser-20261026/RUN_GPU.md`。
- **时间**：GPU 被 run 4、它的评测和 run 5（预计 10-27 04:15 PDT 结束）占满，建议排在 run 5 的评测之后。

### D　之后


run 4 / run 5 的评测出来后，按 [run card §13](../experiments/2026-10-24-qwenuser-no-telecom.md)
预先登记的判断读 2×2 交互，并把评测点补进上面的图。
