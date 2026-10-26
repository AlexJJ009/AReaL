# 反思：τ² 交互式 RL 项目的切入点出了什么问题（2026-10-24）

依据：本目录的 [problem-and-plan.md](problem-and-plan.md)、[paper-roadmap.md](paper-roadmap.md)、[motivation-review-20261013.md](motivation-review-20261013.md)、[m2-additive-vs-multiplicative-20261022.md](m2-additive-vs-multiplicative-20261022.md)，以及
[privileged-critic-case-study](../privileged-critic-case-study/REPORT.md)、[ppo-privileged-649-midrun](../ppo-privileged-649-midrun/REPORT.md)、[compliance-erosion-case-study](../compliance-erosion-case-study/REPORT.md)、[train-test-coverage](../train-test-coverage/REPORT.md)
和 Rajan Agarwal 的博客 [How not to do research](https://www.rajan.sh/multiplayer)。开发机状态读于 2026-10-24：run 3（680，标准
critic + Qwen 用户）训练已结束，离线评测（新 key）还在跑，没有最终分数；682（带剧本 critic + M2）已被 stash，没有在排队（见 §5.1 的 10-24 更新）。

## 0 结论

1. **主要问题不是"太相信 AI"，而是没有执行自己定下的 gate。** 10-13 的动机审查已经指出"用户错误伤害训练"这一环**未证**，并把 X1（受控注入用户缺陷，开/关对比训练）定为核心证据。X1 至今没跑，M4（确认节点离开率）的实验记录 §11–12 仍是空的。10-14 的 roadmap 已经有了论文标题和完整故事线；之后 10 天里先后试了统一 β、规则门控 β、judge 门控、截断 + bootstrap、带剧本 critic、按剧本可靠度加权、加法 M2，七种方法都建在那一环未证的前提上。
2. **"把用户的作用从 agent 的 credit 里分出来"这个切入点，在我们的设定下几乎不成立。** 一是训练分布里这个现象很小：DeepSeek 用户下 U1 约占 2% 的 episode、U3 约 1%，失败约 92% 是 agent 自己的错。二是导师不接受检测器或 judge 之后，这个分解从（对话，结局 reward）里是**不可辨识的**，和博客里"球被当成玩家"是同一类问题。三是 M2 推导自己得出了一堵硬墙：可预测的用户不可靠（伤害 2）就是目标 $J$ 本身，任何对 $J$ 无偏的估计都碰不到它；加法修正落地后又回到 β。
3. **带剧本 critic 的效果是真的，但目前只证明了"EV 更高"，没证明"训练更好"。** 在 $\gamma=\lambda=1$、$\beta=1$ 时，critic 只作为 baseline 进入 advantage，它的作用只能是降方差，不会改变谁被罚。要让剧本里的信息真正改变 credit，需要让 critic 的 value 通过 bootstrap（$\lambda<1$ 或轮级 TD）进入 advantage；Baisero & Amato 的 history-state critic 正好给出这种做法的无偏性保证。这是目前最值得追的一条线，判别实验还差两个 run（§3.3）。
4. **下一步**：先停方法开发，自己读"训练前后同一道题"的轨迹；同时请你决定 682 是否改成"只带剧本、不加 M2"的 run 2（§5.1）。2026-10-24 02:10 PDT 读到：8 张卡都被 run 3 的离线评测占着，`method-m2-priv-beta1-v2/` 自 10-21 之后没有新写入，682 还没开始。

## 1 发生了什么：一条时间线

| 日期  | 事件                                                                                                   | 证据                                                     |
| ----- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| 10-11 | 起点：导师提出"按用户反馈的置信度调整 advantage"                                                       | problem-and-plan §7                                      |
| 10-12 | 逆方差加权草案被数学审稿否掉（核心估计量设计错误），暂停，"先找准现象"                                 | ADR 0010                                                 |
| 10-13 | 动机审查：第 5 环"伤害训练"**未证**，第 6 环"现有做法不够"弱；Codex 指出用户不可靠与任务难"一般不可辨识" | motivation-review §2、§3 第 2 条                         |
| 10-13 | X1 定为"核心证据"，M4 写成实验记录                                                                     | motivation-review §6.2、§7.4                             |
| 10-14 | roadmap 写出标题 *Verifiable but Not Attributable* 和整条故事线，正文"按完整动机链写，语气自信"         | paper-roadmap §1                                         |
| 10-17 → 10-19 | β 拆分、规则门控、judge 门控、截断 + bootstrap；10-19 放弃检测器路线                           | problem-and-plan §3                                      |
| 10-19 → 10-21 | 带剧本 critic（649），EV 提升；对 U3 没有反应；按剧本加权被否定                                | privileged-critic-case-study                             |
| 10-22 | 导师质疑 M2 的独立性假设，建议加法修正；推导发现加法即 β                                               | m2 note §5                                               |
| 10-22 | 合规侵蚀、训练/测试覆盖两个 case study                                                                 | compliance-erosion、train-test-coverage                  |

模式很清楚：**故事线先于证据定下来，之后每个负结果都被当成"方法还不够好"，换一个方法再试，而不是回头问"这个问题本身成不成立"。** 博客作者的说法是 "Every new patch was another attempt to force the model toward my preferred solution without asking whether the problem was well-posed."

## 2 为什么这个切入点站不住

### 2.1 现象在训练分布里很小

- 训练用 DeepSeek 用户时，U1 约 2% 的 episode，分散在 197 个剧本里，没有集中在少数"坏用户"上（privileged-critic-case-study 补充一）；U3 约 1%（problem-and-plan §5.1）。
- Qwen3 PPO 两份独立 case study：失败约 92% 是 agent 的错，约 8% 是用户的错（tau2 README）。
- 用户错误大的地方是**评测时换弱模拟器**（Retail 85.0% → 65.0%），不是训练时。也就是说，我们有一个关于**评测**的强现象，却在为它找一个**训练**问题。

用户 T=0 时，给定历史，用户回复几乎确定，"用户的运气"这部分方差本身就小（problem-and-plan §5.5 第 2 条）。伤害 1（方差）没多少可降的。

10-20 把 2×2 换到自部署 Qwen3.5-4B 用户，理由正是它的用户错误多。在 Qwen 用户下现象确实存在，但它是我们选出来的，而且主要形态是 Telecom 约 45% 的 episode 撞 200 条上限：用户模拟器在长对话里把自己当客服，反复重复、不发 STOP（infra-issues §5）。这是模拟器 prompt / 协议层面的缺陷，属于 10-13 审查说的"最便宜的修补没先做"那一类，不能代表"真实用户不可靠"。无论用哪个用户，下面 §2.2、§2.3 的论证都成立。

### 2.2 不可辨识：博客的第 3 课

博客的核心论点：重建目标下，任何能把下一帧预测好的分解都同样合法，"球拍是玩家"只是众多极小值之一，数据没有指定它。模型落到"球是玩家"不是 bug，是目标允许的解。

对照我们：结局 $R$ 由 agent 和用户共同产生。只看对话和 $R$ 时，"用户错了""用户难缠但 agent 应对不好""agent 先错了、用户顺着走"能产生同样的数据。各方法的失败都是这个问题的表现：

| 方法       | 失败的方式                                                        | 对应的可辨识性问题                         |
| ---------- | ----------------------------------------------------------------- | ------------------------------------------ |
| 统一 β     | 最后一次用户段里混着 agent 的错误，β 替 agent 打了折             | "用户段"≠"用户的作用"                      |
| 规则门控   | 反事实 precision 约 58%，很多"yes + STOP"之前 agent 的方案已经错了 | 表面特征不能区分谁的错                     |
| judge 门控 | 把 12.8% 的 agent 失败判给用户                                    | 引入额外模型也只是把不可辨识移到 judge 上  |
| 按剧本加权 | $\hat\sigma_u^2$ 主要反映 agent 的时成时败，不反映用户            | 方差大 = 学习信号所在，不是用户不可靠      |

10-13 Codex 的审稿意见（"用户不可靠和任务难 / agent 能力不足一般不可辨识"）已经是这个结论。导师 10-21 不接受检测器和 judge 之后，**唯一能补上辨识信息的就是剧本**，这也正是带剧本 critic 是唯一有正结果的方法的原因。

### 2.3 理论上的硬墙

M2 推导（m2 note §3–§5）把用户影响分成三种：(i) 对 agent 的正常反应，(ii) 可预测的不可靠，(iii) 实现出来的噪声。结论是：

- (iii) 期望为零，只贡献方差，而 §2.1 说明它很小；
- (ii) 就是"在这个用户下的成功率"$J$ 本身，对 $J$ 无偏的估计都改不了它，只能改目标，而改目标需要标签；
- 不加检测器时，加法修正只能由 critic 的 value 变化构成，取每回合一个数就回到了 β。

也就是说，在"单条 rollout、无检测器、只用对话和 $R$"这组约束下，问题本身没有留下方法的空间。这不是调参能解决的。

## 3 带剧本 critic 到底起了什么作用

### 3.1 τ² 对 agent 是一个 POMDP，剧本是隐藏状态

agent 看不到用户的真实目标 $z$（要退哪个订单、能接受什么条件），只能从对话里推。标准 critic $V(h)$ 也只能从对话推 $z$；对话前期推不出来，所以它的读数接近"这道题的平均成功率"，agent 写错要等到结算才暴露，失败的意外全部挤在最后一次用户回复上。带剧本的 critic $V(h,z)$ 能在 agent 写库那一轮就判断"这和用户要的是否一致"。

数据支持这个解释（privileged-critic-case-study 结论 1、2）：

| 位置                    | 带剧本题内 pair AUC | 标准题内 pair AUC |
| ----------------------- | ------------------- | ----------------- |
| 第一个 agent token 前   | 0.548               | 0.489             |
| 对话中间                | 0.634               | 0.536             |
| 最后一个 agent→用户边界 | 0.938               | 0.758             |

非 U1 失败（agent 自己错）在终止处的意外：带剧本 −0.262，标准 −0.455，说明错误被提前算进了 value。

但它**不识别用户错误**：U3 那句回复前后，带剧本 critic 的 value 变化是 +0.001 \[−0.023, +0.024\]。原因是训练分布里几乎没有 U3，critic 没学过这种区分。这也对应博客的另一点：模型只学数据里有的结构（Genie 靠的是上千种游戏的多样性）。

另一点要诚实写明：Airline pack 的剧本里，gold 写操作的参数有 74% 直接出现（649-midrun §1 第 2 条），剧本常写出逐步流程。所以它更接近 Le Critique、SWEET-RL 那种"看参考答案的 value"，不是"归因"。

### 3.2 在当前配置下它只能降方差

$\gamma=\lambda=1$、$\beta=1$ 时

$$
\hat A_t = R - V(h_t, z).
$$

critic 只是 baseline。看一个 agent 在第 3 轮写错、最后 $R=0$ 的 episode：

- 第 1–2 轮（正确的 token）：$V$ 还高，$\hat A=0-V$ 是大负数，**仍然被罚**；
- 第 3 轮之后：带剧本 critic 的 $V$ 已经降下来，这些 token 罚得少。

所以带剧本 critic 看出"错在第 3 轮"，并没有让第 1–2 轮少背锅。它带来的是方差缩小：advantage 的方差大约按 $(1-\mathrm{EV})\,\mathrm{Var}(R)$ 缩放，EV 从 0.348 到 0.458，对应 $0.542/0.652\approx0.83$，方差约降 17%。

这和最终结果一致：649 第 120 步 overall 52.5%，标准 critic 基线 v3 同一步 50.5%（两者用户后端不同，不能直接比）。"EV 提升"成立，"训练更好"没有证据。

### 3.3 它真正可能发挥作用的地方

剧本的信息要改变 credit，必须让 critic 的 value 进入 advantage 的**方向**，即 bootstrap：$\lambda<1$，或者按轮做 TD（$\hat A_k=V_{k+1}-V_k$）。这时第 3 轮的 TD 是大负数，第 1–2 轮的 TD 接近 0，credit 落在写错的那一轮。

代价是 bootstrap 会把 critic 的误差带进梯度（problem-and-plan 附录 B.6 反例 2）。这里有一个现成的理论支点：[Baisero & Amato（AAMAS 2022）](https://arxiv.org/abs/2105.11674) 证明，用只看状态的 critic $V(s)$ 训练基于历史的策略是有偏的，而 history-state critic $V(h,s)$ 保持 policy gradient 定理成立、不引入偏差。我们的 $V(h,z)$ 正是 history-state 形式。后续的 [Informed Asymmetric Actor-Critic（2509.26000）](https://arxiv.org/html/2509.26000) 把它推广到状态之外的特权信号；[DCRL（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/d399b67fa017f0f7670102c88507720c-Paper-Conference.pdf) 报告 history-state critic 在一些任务上方差偏大、表现不如普通 critic。这两篇我只读了检索摘要，没读正文。

由此得到一个可以证伪的假说：

> 交互任务里，标准 critic 在 agent 犯错时看不出来，所以 $\lambda$ 越小越偏、越差；带用户剧本的 critic 能看出来，所以可以用更小的 $\lambda$（或轮级 TD）把 credit 定位到出错的那一轮，学得更快。

判别实验：同一用户后端下，{标准, 带剧本} × {$\lambda=1$, bootstrap}，看训练 reward 曲线和固定用户 eval。bootstrap 不能用 token 级固定 $\lambda=0.95$：附录 B.8 算过，1000 个 token 外 $R$ 的权重约 $5\times10^{-23}$，advantage 只剩 critic，比较没有意义。应该用仓库里已有的长度自适应 $\lambda=1-1/(\alpha L)$（`sao_length_adaptive_gae`），或者按轮计的 $\lambda$。预测是带剧本的 critic 在 bootstrap 下不变差甚至变好，标准 critic 变差。两者一样，这条线就该停。

四个格子里，"标准 × 带 bootstrap"已经有数据：SAO 用的就是标准 critic 加长度自适应 $\lambda$（`sao_long120`，以及 SAO 534 在对话变长后 Airline 变差），但用户后端和配置与 run 3 不同，只能作参照。"标准 × $\lambda=1$"是 run 3。缺的是两个 run：带剧本 × $\lambda=1$（run 2，与 run 3 同一 Qwen 用户和配置），以及带剧本 × SAO 的 $\lambda$。

新颖性风险：特权 critic 本身不新（asymmetric actor-critic、Le Critique、SWEET-RL）。能站住的部分是"交互任务中要特权给 critic 的是用户的隐藏意图，以及它让低 $\lambda$ 的 credit assignment 变得可行"，前提是上面的实验成立。

## 4 是太相信 AI，还是自己想得太少

两者都有，但具体机制比这两个词更窄：

1. **停止信号是 AI 给的，也写在仓库里，但没有被执行。** 10-13 的审查明确写了"伤害训练未证"、"不可辨识"、"最便宜的修补没先做"。所以问题不是 AI 只会附和，而是"审查结论 → 改变计划"这一步没有发生。
2. **AI 的产出速度超过了你的消化速度。** 两周里这个目录积累了约 500 KB 的文档。每份都有依据，但你主要读的是摘要和结论表，而不是轨迹本身。结果是你知道"U1 占 2%"这个数，却没有形成"训练时用户根本不是瓶颈"这个判断。判断力要靠自己看原始材料来形成，读摘要形成不了。
3. **AI 擅长把给定方向做得像样，不擅长说"这个方向不该做"。** 你给出"区分用户和 agent 的 credit"，它会推导分解恒等式、设计 β、写 2×2、预登记判据，每一步都严谨，但整体建在一个没有验证的前提上。这一点我也一样：本文给出的方向（§3.3）同样可能是被现有材料锚定的结果，需要你看完轨迹后自己判断。
4. **导师的说法有对的部分。** 方法"歪"的根源不是数学不够，而是问题选得不对：问题不成立时，任何方法都会显得别扭。

也要看到做对的地方：预先登记判据、按题 bootstrap、逐项复算、负结果如实记录。正因为这些记录是诚实的，上面的诊断才能从已有材料里直接读出来，没有浪费的实验。缺的只是一个"gate 不过就不往下走"的规则。

## 5 接下来怎么做

### 5.1 需要你决定的一件事：682

682 是"带剧本 critic + 乘法 M2，β=1"。它把唯一有正结果的组件和刚被推导否定了一半的组件绑在一起：结果好了分不清来自哪个，结果不好也分不清。m2 note §8 自己也写了"如果方法转向加法修正，682 需要重新考虑"。

状态（10-24 02:10 PDT）：GPU 0–7 都被 run 3 的离线评测占用（进程已运行约 7.9 小时），682 的目录自 10-21 之后没有新写入，还没启动；`pueue status` 在 ssh 下没读出来，这个判断来自目录和进程。

建议：评测结束前把 682 换成 2×2 里的 run 2（只带剧本、不加 M2、β=1、与 run 3 同一 Qwen 用户和配置）。它和 run 3 只差 critic 的输入，是检验带剧本 critic 能否帮助训练的唯一干净对照。GPU run 需要你批准，我没有动队列。

更新（10-24 04:50 PDT，用户已同意）：682 和当时排着的方法 run 688 都处于 stash，不会自动启动；687（run 3 的 Qwen 用户评测）已 kill，残留进程组已清理。GPU 现在跑 689：DeepSeek 官方 `deepseek-flash` 同时当用户和 judge，评测 run 3 的 step 0/30/90/120 和 v3ext step 120。run 2 不再按原方案排队；评测出结果后再决定新的一对：去掉 Telecom 的 run 3′，以及去掉 Telecom、只带剧本 critic（β=1，不加 M2）的 run 2′。

### 5.2 自己读轨迹：换一个问题

前面四份 case study 都在找用户错误。还没人认真看过的是**训练改变了 agent 的什么**。博客第 2 课："When your model consistently lands on solutions you don't want, it's telling you something about the problem."

做法：

1. 选标准 PPO 的一个 run（run 3 离线评测跑完后用 `run3-std-beta1-v4/offline-checkpoint-eval-k1023/step-000` 和最后一步；现在可以先用 v3 的 `evidence/async-eval`）。
2. 同一道题，step 0 对最后一步，20–30 对，三个 domain 都要有。
3. **先读、先写，再看任何 AI 总结。** 每对写三行：agent 学会了什么、没学会什么、你猜为什么。
4. 带着已经测出来、但还没解释的现象去读：
   - 同意不足 B5 在标准 PPO 下 +14–17pp（agent 学会了"问得不完整"，不是"不问"）；
   - Airline 不涨，训练里需要拒绝的题 15.8%、test 70%；
   - Telecom 学会转人工、排障变差；
   - SAO 让对话变长，超出 32K 的 episode 从 4 增到 76。

这些都是目标函数在告诉你它奖励什么。它们里面哪一个让你觉得"这是交互式 RL 才有的、而且我能解释机制的"，就是候选问题。

### 5.3 一条机械规则

**没有 gating 实验的结果写进 ADR，就不启动方法 run。** 每个方向开始前写三件事：现象在训练分布里有多大；要学的东西能否从现有信息辨识出来；最便宜的证伪实验是什么。三项任何一项答不上来，先做那一项。

## 6 方向建议

**交互式 RL 这个大方向没有问题，"用户不可靠 → 归因"这个切入点有问题。**

推荐主线是 §3.3 的"带用户剧本的 critic 让低 $\lambda$ 的 credit assignment 可行"。理由：

- 它是唯一有正信号的结果（EV +0.110 \[0.064, 0.156\]，最后边界题内 AUC 0.938 对 0.758）；
- 它有可辨识性：剧本补上了缺的信息；
- 理论有现成支点（history-state critic 无偏）；
- 判别实验便宜：run 2 和"带剧本 × SAO 的 $\lambda$"两个 run。

要盯住的另一个现象是合规侵蚀（B5）：outcome reward 只查 DB 终态，训练让 agent 的确认变得不完整。它是 measurable、交互任务特有的，但目前和用户后端、critic 混在一起，幅度和 judge 误差同一量级，还不够当主线。

哪一条做主线，在你读完 §5.2 的轨迹之后再定。如果读完之后你对两者都没有自己的判断，那本身就说明问题还没找到，应该继续读，而不是开 run。

## 来源

- [How not to do research — Rajan Agarwal](https://www.rajan.sh/multiplayer)
- [Baisero & Amato, Unbiased Asymmetric Reinforcement Learning under Partial Observability（arXiv 2105.11674）](https://arxiv.org/abs/2105.11674)
- [Informed Asymmetric Actor-Critic（arXiv 2509.26000）](https://arxiv.org/html/2509.26000)
- [Dual Critic Reinforcement Learning under Partial Observability（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/d399b67fa017f0f7670102c88507720c-Paper-Conference.pdf)

## 附：DeepSeek 用户下的 run 3 评测（pueue 689，2026-10-24）

设置：用户模拟器和 NL judge 都是 DeepSeek 官方 `deepseek-flash`，温度 0；agent 评测温度 0；airline 20 题、retail 40 题、telecom 40 题，每题 4 次，mean@4 = 4 次成功率的平均；infra 剔除 0 条。响应里返回的模型名只有别名 `deepseek-flash`，无法确认具体版本。耗时 45 分钟，余额 ¥239.29 → ¥211.88（¥27.41）。结果目录 `_artifacts/AReaL/codex-sao-math/tau2-deepseek-eval-20261024/eval/summary.md`。

| checkpoint | airline | retail | telecom | 合计（400 条） |
| --- | --- | --- | --- | --- |
| run 3 step 0 | 11.3 | 46.3 | 46.3 | 39.3 |
| run 3 step 30 | 18.8 | 51.3 | 46.3 | 42.8 |
| run 3 step 90 | 22.5 | 51.9 | 47.5 | 44.3 |
| run 3 step 120 | 20.0 | 53.8 | 43.1 | 42.8 |
| v3ext step 120（DeepSeek 用户训练） | 13.8 | 63.1 | 52.5 | 49.0 |

读法：Qwen 用户评测下 run 3 看起来持平，换成 DeepSeek 用户后 retail +7.5pp、airline +8.8pp，但都在约 1 个标准误内（按题数粗估，未做检验）；v3ext 的 retail +17pp 更明确。Telecom 在 DeepSeek 用户下 step 0 就有 46%，而 Qwen 用户评测只有 5–10%：Telecom 做不动的主要原因是 Qwen 用户，不是 agent。
