# 研究方向：RL 训练中如何对待用户模拟器的质量（2026-10-11 起）

> 来源：原 README 第 16 节开头、16.0、16.1、16.12、16.13。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

起因：与老师讨论的方向是"现在的 RL 没有考虑用户"。设想是：给每个用户的反馈维护一个置信度；置信度高的反馈多学（大优势、小裁剪）， 置信度低的少学；PPO 中 critic
的 value 估计和 GAE 也要考虑用户。实验设计是：找一个弱用户、一个强用户，各训一轮 baseline， 再各训一轮新方法，预期新方法都更高。

本节最初回答的问题是"这个问题是否存在、是否是缺口"。讨论中方向几经调整，§1 给出**当前**的问题定义和决策轨迹；[analysis.md](analysis.md) 与
[related-work.md](related-work.md) 是推导过程的记录， 其中被后来的决策推翻的地方在 §1
的表里注明。还没有做实验，判断来自[user-simulator.md](../user-simulator.md)、[failure-taxonomy.md](../failure-taxonomy.md)、[capability-gaps.md](../capability-gaps.md)已有的数据和文献检索。

## 1 当前问题定义与决策轨迹

**问题陈述**（2026-10-13 重写；方法尚未确定。上一版是 2026-10-11 的"用模拟器训练"表述，对象改为真实用户，见决策 15）

**问题。** 用 RL 训练多轮客服 agent 时，结局 reward 由 agent
和用户共同决定。真实用户不一定可靠：可能在确认的同一句里离开、中途改主意或记错信息、没办完就走。用户单方面造成的失败让 agent 即使每一步都做对也拿 0
分，这次失败还会作为负优势落在 agent 此前的正确行为上。

**两种伤害，性质不同。**

1. 用户出错与 agent 怎么做无关（例如确认后有固定概率离开）：期望 reward 只是整体打折，最优策略不变，伤害是方差变大、同等预算下学得少。
1. 用户出错和 agent 的说法有关（例如某种确认方式更容易让用户离开）：会把策略推向违反 policy 的行为，例如学会少确认、直接写。这是学错方向，比学得慢更严重。

**我们要的是 agent 自身的能力，不是它在某个不可靠用户下的得分。** 所以训练时面对不可靠用户，评测用固定的可靠用户，并直接检查 policy
合规（例如写操作前是否确认）。训练时拿到的 reward 和我们真正想优化的目标不一致，这正是问题所在。

**设定。** 真实用户每次对话只发生一次：不能重采样，没有剧本可查，同一题只有一条 rollout，GRPO 的组内基准无从计算。现有做法都假设能重复或检查用户：丢弃
judge 判为用户出错的轨迹（AutoForge，judge 要看用户剧本）、改进或更换用户模型、对同一题采样多条再丢掉全对或全错的组。单条 rollout 下可比的
baseline 是普通 SAO、REINFORCE++（无 critic，batch 全局归一化）、按用户的滑动平均 baseline，以及只看对话文本的规则审查或
judge（[related-work.md §2](related-work.md)）。

**模拟器的角色：可控的代理。** 用两种方式构造可靠度不同的用户池：换模型或改 prompt /
persona（配合与不配合、好与坏），以及按概率注入某种固定的错误行为（例如确认后离开）。改 prompt 在这里是造用户的手段，不是解决办法，因为真人的 prompt
改不了。每类模拟出来的缺陷都要对应真人行为：U1 对应确认后离开，U3 对应改主意或记错信息，U4b 对应没办完就走。

**证据现状。**

- 已有：同一 actor 在修过的 test 集上换五个用户模拟器，总分 mean@4 从 0.579 到 0.758（最强与最弱之差 17.9pp，按题配对
  bootstrap 的 95% CI 为 \[7.9, 28.3\]；中间几档两两多半分不开），385 个失败里 168
  个归用户（[user-simulator.md §2](../user-simulator.md)）；旧 GRPO 训练中，成功−失败组内对比有 28.6% 是"成功 vs
  用户失败"，Retail 41.8%（[capability-gaps.md §1.3](../capability-gaps.md)，official 旧数据）。
- 没有：用户错误伤害训练结果的直接证据，以及上面第 2 种伤害是否发生。计划见下一步。

**核心困难是可辨识性**：一个用户的结果差，可能是因为他的反馈不可靠（agent 无能为力，应少罚），也可能是因为 agent
还不会应对这类用户（例如用户不主动给订单号，agent 该追问），后者恰恰最该学。方法只能用线上可观测的信息区分这两种情况；做不到的话，降权会把训练推向容易的用户。

**方向。** 单条 rollout、带 critic 的 PPO / SAO，方法有两部分（2026-10-18）：

1. §4：在用户回复前后各读一次 critic，用户回复带来的意外乘 $\beta$。它在 critic 准确时只降方差，管第 1 种伤害；第 2
   种伤害里的系统性部分它管不到，这一限制要写进主张。
1. §5：检测到用户出错的那次回复，把它当作截断而不是终止：出错前的 agent token 用出错前一刻的 critic 读数 bootstrap，出错后的 token
   不参与训练。它管 $\beta$ 管不到的那部分，即用户违反任务、编造事实这类不随机的错误。

**设定要点：**

- 可用的信息：对话文本、结局 reward、同一用户的历史交互；不用用户的隐藏目标。带剧本的 judge 只作为 §5 检测器之一，是模拟环境里才有的
  oracle，与规则、critic 读数这两种部署时可用的检测器并列比较（2026-10-18 修改，原文为"不用带剧本的 judge"）。
- 测试平台：τ² Airline/Retail，fixed 变体，官方 base 划分（训练 99 题、test 60 题），详见
  §3。[analysis.md §7](analysis.md) 的"每个（用户、题目）只交互一次"协议是方法阶段的设定，现阶段先做不同用户模拟器的评测和 case
  study。
- 方法草案见 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)：按用户给 advantage
  乘一个权重，权重取"agent 最后一次行动之后、结果还剩多少不确定性"的倒数。还没有实验；2026-10-12 起暂停，见 §3。
- 必须区分的已有工作：P-GRPO（按用户平移基准）、RGPO（离线标注者可靠度）、AutoForge（带特权信息的整条 mask），见
  [related-work.md §2](related-work.md)。

**决策轨迹：**

| #   | 决策                                                                                                                                                                 | 原因                                                                                                                                                                                                                                                                                                                                                                                | 位置                                                                 |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 1   | 起点：老师提出按用户反馈的置信度调整 advantage、clip，critic 和 GAE 也考虑用户；弱、强用户各训 baseline 和新方法                                                     | 观察到好的用户模拟器下分数高、差的用户下学不会                                                                                                                                                                                                                                                                                                                                      | 本节开头                                                             |
| 2   | 离线真实数据按用户专业程度加权：不做                                                                                                                                 | 离线数据可以先清洗、改写、合成，PPO 的 advantage/clip 也用不上；τ² 没有专业程度这个维度                                                                                                                                                                                                                                                                                             | [analysis.md §1](analysis.md)                                        |
| 3   | 把"用户不好"拆成 A 随机性（方差）和 B 系统性偏离（偏差）                                                                                                             | 两者解法不同：A 有经典降方差方法；B 是优化目标被带偏，样本再多也学错                                                                                                                                                                                                                                                                                                                | [analysis.md §2](analysis.md)                                        |
| 4   | "用户历史成功率"不单独作为置信度，列出 6 种候选信号                                                                                                                  | 成功率混入 agent 能力，粒度只到用户；按成功率给 advantage 加权已有人做（ProxMO 2602.19225）                                                                                                                                                                                                                                                                                         | [analysis.md §3](analysis.md)                                        |
| 5   | 问题 A 不作为贡献                                                                                                                                                    | 自托管用户时可固定用户噪声，经典方法无偏可用；API 用户拿不到随机数，只能统计处理                                                                                                                                                                                                                                                                                                    | [analysis.md §2](analysis.md)、[related-work.md §1](related-work.md) |
| 6   | AutoForge 式整条轨迹 mask（E1）只作 baseline，且不是上界                                                                                                             | AutoForge（2512.22857）已做；整条丢掉会丢掉出错前 agent 做对的部分。可重采样场景的真正上界是重采样到干净数据，同行已做很多                                                                                                                                                                                                                                                          | [analysis.md §4](analysis.md)                                        |
| 7   | 删去 E0（用户反应探针）和 E0′（运气占比）                                                                                                                            | E0 是 benchmark 诊断，不检验方法；E0′ 在 API 用户下分不开运气和反应                                                                                                                                                                                                                                                                                                                 | [analysis.md §5](analysis.md)                                        |
| 8   | **转向不能重采样的在线场景**                                                                                                                                         | 可重采样时评审会问"为什么不直接拿更好的数据"；不能重采样时 E1/E3 不可用，同行工作少。GRPO 不可用，critic 成为必需，与 SAO 工作衔接；用户历史残差成为合理信号                                                                                                                                                                                                                        | [analysis.md §7](analysis.md)                                        |
| 9   | Coverage, Not Targeting（2609.02417）不再引用                                                                                                                        | 证据太弱：τ² 结论只来自一次 LoRA 更新，效应与评测噪声同级                                                                                                                                                                                                                                                                                                                           | [related-work.md §1](related-work.md)                                |
| 10  | 不与 GiGPO 比较                                                                                                                                                      | 它按完全相同的中间状态分组算 step 级优势，需要同一题多条 rollout；τ² 用户回复各不相同，状态几乎不重合；在线场景每题只有一条                                                                                                                                                                                                                                                         | [related-work.md §1](related-work.md)                                |
| 11  | 在线场景不撞车，继续；要正面回应三篇中度相关工作，并修改可靠度的定义                                                                                                 | 检索（[related-work.md §2](related-work.md)）没找到同时做到"单条 rollout + critic、用户反复出现、在线估计用户可靠度并缩放更新"的工作，但各部件都有人做过。$\bar r_u$ 有两个问题：critic 认出用户后残差均值趋于 0；降权"做不好的用户"会把训练推向容易的用户                                                                                                                          | [related-work.md §2](related-work.md)                                |
| 12  | 方法草案：按用户给 advantage 乘权重，权重取 agent 最后一次行动后残差平方的倒数（带收缩、用历史对话算）                                                               | 能改的只有 critic（只降方差）和乘系数（改目标）；残差均值会被 critic 吸收，残差平方不会；在"用户只加噪声"的模型下，乘正权重不改方向，逆方差权重方差最小                                                                                                                                                                                                                             | [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)     |
| 13  | **暂停 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 的方法推导，先换更多用户模拟器和 benchmark、做轨迹 case study，找准现象再定方法**            | 代码改动简单而数学过重，理论没有指导设计，也没拦住核心估计量的设计错误（Codex 审稿）；LLM RL 论文靠"现象 + 简单修正 + 消融"说服人；现象证据目前只有两个模拟器、一个 benchmark                                                                                                                                                                                                       | §3                                                                   |
| 14  | **轻量方法候选：在用户回复前后各读一次 critic，用户回复带来的意外乘 $\beta$；与现象研究并行**                                                                        | 粒度到每次回复、无用户状态、一个超参数；测量位置在线可定；critic 准确时只降方差不改期望；用户反应依赖 agent 时记到 agent 身上                                                                                                                                                                                                                                                       | §4                                                                   |
| 15  | **研究对象定为真实用户的不可靠反馈（2026-10-13）；模拟器是可控的代理**                                                                                               | 真人同样会确认后离开、改主意、不给信息，不能假定用户质量高。真人的 prompt 改不了、不能重来，所以改 prompt 和换模型只用来构造可靠度不同的用户（persona），按概率注入固定错误用来做受控对照；重采样不是可比 baseline。由此要求：训练用不可靠用户、评测用可靠用户并检查 policy 合规；方法必须区分"用户错"和"用户难"                                                                    | §1、`motivation-review-20261013.md`                                  |
| 16  | **方法加第二部分：用户出错处截断 + bootstrap（2026-10-18）**；带剧本的 judge 可作检测器                                                                              | U1 patch 重跑后，确认时同句 STOP（U1）可以用一句 prompt 修掉，修不掉的是违反任务、编造事实（U3），这类错误不随机，$\beta$ 管不到（[experiments/2026-10-18-user-sim-rerun.md](../experiments/2026-10-18-user-sim-rerun.md)）。AutoForge 整条丢弃会丢掉出错前 agent 做对的部分；截断只丢出错之后。检测器可替换，贡献落在边界处怎么分配 credit。不在边界重采样用户回复（偏离研究目标） | §5                                                                   |
| 17  | **转向用户池与用户能力（老师 2026-10-19）**：混合用户池训练，按用户身份修正 advantage（以用户为条件的 baseline / critic，以及能力系数）；β、门控 β 和 judge 留作消融 | 老师判断加 judge 做不通；643 的 judge 在训练分布上把 12.8% 的 agent 失败判给用户；β 只管 U1 类终止边界随机事件，统一 β 还会给结局才暴露的 agent 错误打折（644）。按用户身份条件化恒无偏，按回复内容条件化只在 (C) 下无偏。待修正：老师公式里 $A_2$ 的符号应为减号；PPO 中 critic 可能已认出用户；平移部件 P-GRPO 已做                                                               | [method-derivation.md](method-derivation.md)                         |

**下一步**（2026-10-13 更新，来自 `motivation-review-20261013.md` 第 6
节）：评测侧的现象已确认（[user-simulator.md §2](../user-simulator.md)），下面补"伤害训练"这一环，按顺序：

1. **受控缺陷的 SAO 训练（X1）**：同一个用户模拟器，只开关"确认后按概率离开"，各训一次，在可靠用户上评测并报 policy 合规。
1. **好用户与坏用户各训一次、交叉评测（X2）**：检验能力能否迁移。
1. **"学错方向"的启发式分析（X3）**：训练过程中跟踪写前确认率，以及用户离开率是否随 agent 的确认说法变化。
1. **反事实回放（X4，先内部测）**、**critic 在用户回复边界的校准（X5）**，然后才比较 §4 的方法。

### 最初的想法（用户笔记，2026-10-11 原文）

现在强化学习没有考虑用户。

user-adaptive 的能力估计或行为估计。

可改动：clip，advantage 估计

可以维护对用户的反馈的置信度，可以从过去任务的成功概率上计算；可以从用户给的反馈的专业程度计算。

例如，对置信度高的用户反馈做高优势低裁剪学习，置信度低的则低优势高裁剪学习。

在ppo的场景，critic 的训练需要 user 和 assistant 的 token 都参与到 advantage 里。gae也可以改。

在 critic 训练时候的 value 的估计上改动，需要考虑 user 的置信度。

actor 训练的时候看看 advantage 以及 clip 能怎么改。

需要做出demo

ds v4.1 再找一个弱的 user model。

弱 user 和强 user 各自训一轮作为baseline。

换我们的方法对弱 user 和强 user 各自训一轮，预期要都高于baseline。

## 2 结论

- **问题存在**。训练里约 29% 的组内对比是"agent 做得对，但用户出了错"（Retail
  42%，[capability-gaps.md §1.3](../capability-gaps.md)），换一个用户模型 Retail 就差
  15.6pp（DeepSeek 对 Doubao 2.1 Turbo）到 20.0pp（GLM 5.3 对 Turbo 或 Qwen
  自部署；[user-simulator.md §2](../user-simulator.md)）。
- **"用户不好"其实包含两个性质不同的问题**（[analysis.md §2](analysis.md)）：
  1. 用户的随机性让 reward 变得嘈杂。这是降方差的问题，经典 RL 有成熟方法，不算缺口。
  1. 用户系统性地不按应有的方式表现，导致 agent 学到错误的行为。这是优化目标被带偏的问题，我们有数据支撑，检索中没看到有人直接做。
- **最关键的未知数**：用户噪声到底有没有伤到训练结果。上一轮训练权重几乎没动（[initial-audit.md §2.3](../initial-audit.md)），回答不了。

## 3 转向：先找准现象，方法从轻（2026-10-12）

**决定。** [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)
的方法设计暂停，不再往下推导。下一阶段先换更多用户模拟器、做更多轨迹 case study，确认问题到底是什么，再定方法。

**原因：**

- [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 的代码改动很小（advantage
  乘一个按用户的系数），但为它搭的数学太重：信噪比、Kantorovich 界、Adam 尺度等，没有一条改变过设计决定。Codex
  的审稿（`appendix-math-review-codex.md`）还指出核心估计量有设计错误：最后一次行动的位置是事后选的，提前结束的用户
  $\mathbb{E}[e^2] = \rho(1-\rho)^2$，反而可能被加权。理论没能在动手前拦住这个错误。
- LLM 上的 RL 论文通常的写法是：一个前序工作失败的现象，一个越简单越好的修正，加消融实验。理论最多是一个短命题和一个示意例子。太长的推导审稿人不会细看，也说服不了人。
- 推导越来越重，很可能说明问题切入得还不够准。目前的现象证据是五个用户模拟器、一个 benchmark，主要的失败机制是 Doubao
  在确认的同一句里结束（[user-simulator.md §1](../user-simulator.md)），Qwen 自部署用户则不按任务条件走。

**[appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 保留什么。**
后续压缩成一页时，只留四条改变过设计的规则：改 critic 只降方差、要降权只能乘系数；残差均值会被 critic
吸收；权重只用过去的对话算；测量位置必须在线可定、不能事后选。外加一个命题："用户只加噪声"时，任意正权重不改梯度方向。其余移到附录，或者删除。

**计划的论文结构：**

1. 开头一张图或表揭示核心现象：换用户模拟器后，同一套训练方法的效果差异很大；
1. 解释这个现象为什么存在（来自 case study 的机制）；
1. 相关工作；
1. 方法，尽量轻；
1. 实验：在强、弱各种用户模拟器上训练，每个都稳定超过该模拟器下原模型的水平和已有方法。

**下一步：**

1. **扩充用户模拟器。** 接入更多厂商的模型，按能力梯度选一批（强、中、弱），在同一套题、同一个 agent 上跑 step 0 评测。需要 API 费用，启动前要批准。
1. **轨迹 case study。** 对每个模拟器统计用户侧失败的类型和比例：确认后立即结束、不提供必要信息、偏离目标、违背 persona 等。沿用
   [user-simulator.md §1](../user-simulator.md) 和
   [capability-gaps.md §1](../capability-gaps.md)
   节的分类与脚本（`../scripts/trajectories/lead_confirm_stop.py`、M 编码），得到"模拟器 ×
   失败类型"的矩阵。这个矩阵就是开头那张图的候选。
1. **换 benchmark。** 在其他依赖用户模拟器的 benchmark 上重复，排除"只是 τ² 的问题"。候选要先检索、逐个核实是否真有用户模拟器、reward
   是否由用户和 agent 共同决定。目前想到的有 τ-bench（τ² 之前的版本）、Apple 的 ToolSandbox、SWEET-RL 的
   ColBench、MINT（模拟用户给反馈），都未核实。

**测试平台（2026-10-12 补充）。** 文档里有两个独立的维度，容易混：

- **变体**（[benchmark-variants.md §1](../benchmark-variants.md)）：`official` 是 tau2-bench
  `b7ea907` 原样；`fixed` 是加上上游 PR 和我们补丁之后的版本，当前为 fixed v2。
- **题集**：base 集是 τ² 自带的 Airline 50 + Retail 114 题，按官方划分切成 train / test；AReaL-tau2-data
  是另一份外部数据（[training-data.md §2](../training-data.md) 起），不属于 base 集。

本方向用 **fixed 变体 + base 集的官方划分**：训练 99 题（Airline 30、Retail 69，去重后），评测 test 60 题（Airline
20、Retail 40），与 [training-setup.md §1](../training-setup.md) 相同。正文主表报 fixed，因为 official
有错题和评分盲区（[benchmark-variants.md](../benchmark-variants.md) 第 3、1
节），用户模拟器之间的差异会被这些噪声掺进去；official 只放附录，供和榜单对照。ADR 0010"两个变体都报"的规则不变。不用
AReaL-tau2-data，因为它经过我们的筛选，换用户模拟器后的差异可能混入筛选效应。

已知限制：DeepSeek 用户、policy temperature 0 下，Qwen3.5 在 Retail 训练集 69 题里有 47 题 4
次全对（[training-data.md §1](../training-data.md)、[step0-evaluations.md §4](../step0-evaluations.md)），换弱用户后才有更多失败。

**用户模拟器（2026-10-12 选定）。** actor 固定为 Qwen3.5-4B，先在 test 60 题上各跑 step 0 评测（policy 与用户
temperature 0，每题 4 次），再做 case study。按能力从弱到强：

| 模拟器                   | 档位            | 接入方式                              | 状态                                                                                                                   |
| ------------------------ | --------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Qwen3.5-4B（训练前版本） | 弱              | 本机 SGLang，占 1 张卡（必要时 2 张） | 已有 fixed-v3 test 结果（[user-simulator.md §2](../user-simulator.md)）；演不好用户，case study 里单独归为"模拟器失效" |
| Doubao 2.1 Turbo         | 中下            | Ark API                               | 已有 fixed-v3 test 结果（[user-simulator.md §2](../user-simulator.md)）                                                |
| DeepSeek-Flash           | 中上            | DeepSeek API（`deepseek-flash`）      | 已有；若换成 V4.1，需确认是否同一 endpoint，否则算新模拟器                                                             |
| GLM 5.3                  | 中上 / 强，开源 | 火山订阅接口                          | 已有 fixed-v3 test 结果（[user-simulator.md §2](../user-simulator.md)）                                                |
| Claude Haiku 5.5         | 强，闭源        | 待定                                  | endpoint 待接入，型号名在接入时核实                                                                                    |
| GPT-5.6 Luna             | 强，闭源        | 待定                                  | endpoint 待接入                                                                                                        |

不选最强一档（如 GPT-5.6 Sol）：强用户的偏差方向是替 agent 做判断、把失败救回来，用 Haiku / Luna 这一档就能观察到，没必要多花钱。

**GPU 分配。** 本机 8 张 A100。用户模拟器自己部署时占 1 张卡，其余给 actor：step 0 评测为用户 1 卡 + actor 推理 7 卡；训练在
[training-setup.md §2](../training-setup.md) 的布局（rollout 3、评测 1、actor 4）上从 rollout 里让出 1
张给用户。用户侧会不会成为吞吐瓶颈，先在 smoke test 里测，不够再加到 2 张。

**NL judge 与用户模拟器分开配置（已完成，[decision 0002](../decisions/0002-nl-judge-decoupled.md)）。**
固定为 DeepSeek-Flash、temperature 0，所有模拟器共用。fixed v2 下 test 60 题里有 19 题（Airline 9、Retail
10）的 reward 含非空 NL 断言，所以换用户时 judge 不能跟着换。

**启动前的检查：** 每个新模拟器先跑 5～10 题 smoke test，确认能正常扮演用户、judge 调用走的是固定模型；再从已有评测日志按每条 episode
的用户侧 token 量估算费用，批准后上全量。

**actor 怎么选。** 换 actor 作为第二个消融轴，检验现象和收益是否依赖 agent 的强弱。我的假设是 agent 弱时失败主要来自 agent
自己，用户的影响被掩盖；agent 强时用户的影响才显出来。这个假设本身可以由下面的评测检验。

| actor             | step 0 评测（所有用户模拟器） | 训练       | 理由                                                                                                                                                                                                                                                   |
| ----------------- | ----------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Qwen3.5-4B        | 测                            | 主 actor   | SAO 和 critic 只在它上面跑通过（[infra-issues.md §2](../infra-issues.md)）；用户模拟器的影响已观察到（Retail 0.806 对 0.650）。DeepSeek 下 Retail 提升空间小（fixed 0.844），Airline 0.350 和弱用户下有空间                                            |
| Qwen3-4B-Instruct | 测                            | 第二 actor | 提升空间大（fixed 0.225 / 0.410），组内有对有错的题最多（[capability-gaps.md §2.2](../capability-gaps.md)）。训练前要先处理"工具调用写成正文"（M17），否则提升里混着格式修复；Qwen3 的 critic 路径要先验证（[infra-issues.md §2](../infra-issues.md)） |
| Qwen3-4B-Thinking | 测                            | 不训练     | 训练要求 `enable_thinking=false`（[step0-evaluations.md §6.1](../step0-evaluations.md)）；每轮生成中位数 956 token，是 Instruct 的 11 倍，API 用户下的长对话训练成本高                                                                                 |

step 0 评测只花用户模拟器的 API 费用，三个 actor 都测；训练费用大得多，先在 Qwen3.5-4B 上做"用不同模拟器训练"的 2×2，确认现象后再在
Qwen3-Instruct 上重复。

**要先想清楚的两点：**

- **现象必须出在训练上，不只是评测上**。"换 Doubao 2.1 Turbo 后 Retail 从 0.806 降到 0.650"是评测差异，说明 benchmark
  分数受用户模拟器影响，但还不是 RL 的问题。论文要的现象是：用不同模拟器训练，得到的提升差别很大，甚至变差。这需要训练实验，最小版本是 2×2：用 DeepSeek 和
  Doubao 各训练一次，两个模型都在两个模拟器上评测。
- **"稳定提升"要交叉评测才有说服力**。如果在 Doubao 上训练、也只在 Doubao 上评测，提升可能只是学会了应付 Doubao
  的毛病（例如学会避开会触发提前结束的确认说法），不一定是 agent 变强。评测要包括训练没见过的模拟器，最好固定一个或几个评测用模拟器，报告"训练模拟器 ×
  评测模拟器"的矩阵。

## 4 轻量方法候选：只扣掉用户这次回复带来的意外（2026-10-12；2026-10-17 补 §4.1 token 位置和 §4.3 偏差分析）

**一句话**：在每次用户回复的前后各读一次 critic，把这一步的 TD 误差拆成"agent 这一轮带来的变化"和"用户这次回复带来的意外"，后者乘一个系数
$\beta \in [0, 1]$ 再进 GAE。$\beta = 1$ 就是原来的 SAO。

### 4.1 预备：一段对话在 token 序列上的样子，以及 critic 在哪里读数

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
value\[$b$\]，代价是这个位置现在不训练，要在 critic loss 里补上（见 §4.4 的"代码改动"）。

### 4.2 做法

记 agent 第 $t$ 轮开始前 critic 的预测为 $V_t$（现有，value\[$a-1$\]），agent 第 $t$ 轮说完、用户还没回复时的预测为
$U_t$（新增的读数，value\[$b$\]）。对话结束时 $V_{T+1} = R$。原来跨过用户回复的 TD 误差拆成两段：

$$
\delta_t = V_{t+1} - V_t = \underbrace{(U_t - V_t)}_{\text{agent 这一轮}} + \underbrace{(V_{t+1} - U_t)}_{\text{用户这次回复}},\qquad
\tilde\delta_t = (U_t - V_t) + \beta\,(V_{t+1} - U_t)
$$

用 $\tilde\delta_t$ 代替 $\delta_t$ 算 GAE，其余不变。工具返回是确定的数据库结果，留在 agent 这一段，只在用户回复处拆。

**用 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 的例子看：**

- A 用户：agent 发出确认请求后 $U = 0.5$，A 提前结束、$R = 0$，用户段是 $-0.5$。$\beta = 0$ 时这 $-0.5$ 不再落到
  agent 身上，agent 这一轮按 $U - V$ 计分。
- B 用户：agent 没追问订单号，critic 在这一轮说完时已经能看出要失败，$U$ 低，负分落在 agent 段，照常学。
- agent 的说法影响用户的反应：如果某种确认方式让用户更少提前结束，那么 $U$ 是在看到这句话之后预测的，这种说法的 $U$ 更高，好处记到 agent
  身上。[appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 第 7 步的第 2
  条假设（用户出错与 agent 无关）在这里不需要。

### 4.3 为什么说得通：什么时候无偏，critic 不准时付出什么

**为什么需要这一节**。乘 $\beta<1$ 改动了 advantage。只要改 advantage，就要回答：模型会不会因此学偏，不再是在最大化
$R$？本节给出不学偏的条件，以及条件不满足时偏差有多大。

**第一步：改动了什么**。先看 $\gamma=\lambda=1$ 的情形（PPO 的配置）。对 agent 第 $t$ 轮里的任一动作 token $i$，把它之后所有
$\tilde\delta$ 加起来，agent 段和用户段的读数逐项消去，得到

$$
A_i(\beta) = (R - V_i) - (1-\beta)\sum_{k \ge t} (V_{k+1} - U_k)
$$

$\beta=1$ 时就是原来的 $R - V_i$。$V_i$ 只是 baseline，不依赖动作，所以原来的 $A_i$ 即使 critic 不准也无偏，critic
不准只会增加方差。方法额外引入的只有第二项：从当前这一轮起，每次用户回复的"用户段" $V_{k+1}-U_k$，乘上 $(1-\beta)$ 后减掉。

**第二步：这一项什么时候不带来偏差**。policy gradient 里这一项贡献
$\mathbb{E}[\nabla \log \pi(x_i \mid h_i)\,(V_{k+1}-U_k)]$。记 $h_k^{+}$ 为"对话进行到 agent 第
$k$ 轮说完"的历史；因为 $k \ge t$，动作 $x_i$ 已经包含在 $h_k^{+}$ 里，$\nabla \log \pi(x_i \mid h_i)$ 在给定
$h_k^{+}$ 时是确定的。先对用户回复取条件期望：

$$
\mathbb{E}\big[\nabla \log \pi(x_i \mid h_i)\,(V_{k+1}-U_k)\big]
= \mathbb{E}\big[\nabla \log \pi(x_i \mid h_i)\,\big(\mathbb{E}[V_{k+1} \mid h_k^{+}] - U_k\big)\big]
$$

所以只要

$$
U_k = \mathbb{E}[V_{k+1} \mid h_k^{+}] \tag{C}
$$

这一项就是 0，乘 $\beta$ 不改变梯度的期望，只缩小用户段带来的噪声。

**例子**。agent 问"确认吗？"之后，用户 50% 回答 yes（之后 $V_{t+1}=0.9$），50% 直接 STOP（$V_{t+1}=0.1$）。满足 (C)
的 $U_t = 0.5\times0.9+0.5\times0.1 = 0.5$，用户段是 $+0.4$ 或 $-0.4$，平均为 0。agent 这一轮的好坏由
$U_t - V_t$ 记录：如果换一种问法能让 yes 的比例更高，$U_t$ 就更高，好处记到 agent 身上。

**(C) 是什么意思**。真实的 value 函数天然满足 (C)：用户回复这一步上的 Bellman 一致性就是这个式子，所以"用户段均值为 0"就是 value
的定义。注意 (C) 里的 $V_{k+1}$ 是 critic 自己的读数，不是真实 value，所以 (C) 要求的是 **critic
在用户回复前后自洽**，比"critic 准确"弱：critic 在某道难题上整体高估 0.2，只要 $U_k$ 和用户回复后的 $V_{k+1}$
平均起来一样高估，两者相减就消掉了，不产生偏差。真正产生偏差的是 critic 系统性地看错"用户回复会让局面变好还是变坏"，例如 $U_k$ 认为问完确认后成功率是
0.7，而看到各种用户回复之后的 $V_{k+1}$ 平均只有 0.5。

**第三步：critic 不自洽时，偏差和方差怎么换**。实际的 critic 不会精确满足 (C)。记
$e_k = \mathbb{E}[V_{k+1}\mid h_k^{+}] - U_k$，方法引入的偏差是

$$
-(1-\beta)\sum_{k\ge t}\mathbb{E}\big[\nabla \log \pi(x_i \mid h_i)\, e_k\big]
$$

它随 $(1-\beta)$ 线性增长；同时被缩小的是用户段 $V_{k+1}-U_k$ 里围绕 $e_k$ 的随机波动。这和 GAE 的 $\lambda$
是同一类交换：$\lambda<1$ 用 critic 的读数替换一部分真实回报，critic 不准就有偏，但方差小，PPO 通常就是用 $\lambda=0.95$
接受这个交换。区别是 $\beta$ 只作用在用户回复处，正是我们认为噪声集中的地方。$\beta=1$ 不改变原算法；$\beta=0$ 方差降得最多，也最依赖
critic。偏差小于减掉的方差时，梯度估计的总误差变小，值得用；哪边更大要靠实验，所以计划扫 $\beta \in \{1, 0.5, 0\}$。

SAO 的 $\lambda_{\text{actor}} = 1-1/(1.5L) < 1$，本身已经承担一部分 GAE 偏差。$\lambda<1$ 时用户段进入
$A_i$ 的权重变成 $\lambda$ 的幂次，但上面的推导不变：方法新增的偏差仍然只取决于 $e_k$。

**训练前就能量的两个量**（离线，用训练好 $U_t$ 位置的 critic 和已有 rollout）：

- 偏差的代理：held-out rollout 上用户段 $V_{k+1}-U_k$ 的均值，整体、按 domain、按 agent
  这一轮的动作类型（问确认、执行写操作、拒绝）分组。各组都接近 0 说明 (C) 近似成立。
- 能减掉的方差：用户段在 advantage 方差里的占比（§4 末表"方法 2"）。占比小，这个方法就没有多少可做的。

**和 Mao et al. 的关系**。这与 Mao et al.（[analysis.md §2](analysis.md)）扣掉外部输入的思路相同，区别是：$U_t$
是在看到 agent 这句话之后的预测，所以不要求用户的随机性与 agent 无关，[analysis.md §2](analysis.md) 说的"不能直接套用
Mao"的障碍不存在。

### 4.4 和旧草案的比较、代价和风险

**和 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 比轻在哪里**：

| 项                 | [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)（按用户乘权重） | 本节（按用户回复拆 TD 误差） |
| ------------------ | -------------------------------------------------------------------------------- | ---------------------------- |
| 粒度               | 整条对话、整个用户                                                               | 每次用户回复                 |
| 状态               | 用户 → 历史统计的表、收缩、窗口                                                  | 无                           |
| 超参数             | $k$、$\tau$、上下限、窗口                                                        | $\beta$                      |
| 测量位置           | 事后选的最后一次行动（Codex S1 指出有误）                                        | 用户回复前后，在线即可确定   |
| 需要的理论         | 噪声模型、逆方差、可辨识性论证                                                   | §4.3 的一致性条件 (C)        |
| 用户反应依赖 agent | 降权会少学这部分                                                                 | 记到 agent 身上              |

**代价和风险：**

- **更依赖 critic**。$\beta<1$ 时 critic 在用户回复前后不自洽就有偏（§4.3 的 $e_k$）；$\beta$ 取中间值是折中。critic
  先要有正的 EV（[critic/](../critic/README.md)），这一前提与
  [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 相同。
- **代码改动**（位置定义见 §4.1）：
  1. critic：在每个用户边界的 value\[$b$\]（agent 生成的 `<|im_end|>`）上多训练一个点，target 仍是 $R$（SAO 的
     $\lambda_{\text{critic}}=1$）。用一个独立的 critic mask 加这些点，不改 actor 的 mask；工具边界不加。每个
     episode 约多 4.9 个点，相对约 1,100 个已训练的动作位置可以忽略，forward 也已经算出了这个位置的 value。
  1. GAE：token 级实现里，跨过用户回复的 TD 误差落在位置 $b-1$ 上，原来是 $\delta_{b-1} = W - V[b-1]$（$W$ 是下一轮的
     $V_{t+1}$，最后一轮是 $R$）。改成 $\tilde\delta_{b-1} = (U - V[b-1]) + \beta\,(W - U)$，$U$ 取
     value\[$b$\]。$\beta=1$ 时逐 token 还原原算法。$\lambda_{\text{actor}}<1$ 时，按 token 加权求和的结果和
     §4.2 按轮写的公式不严格相等，实现以这里的 token 级修正为准。
  1. 识别用户边界要看 `<tool_response>` 或原始消息 role，不能只看 `user` 角色 token（工具返回也是 `user`），也不能只看
     `</tool_call>` 后缀（数据里有 1 段 tool tag 损坏、后面接的是真实用户回复）。
- **撞车风险，待核实**：ArCHer（Zhou et al. 2024）在 utterance 级训练 Q 和 V，用 Q − V 作为 token 级策略的
  advantage，结构接近；SWEET-RL 也用 turn 级 advantage，但 critic 用特权信息训练。要读原文确认区别。
- **新颖性薄**。控制变量本身是经典方法，所以贡献要落在现象（§3）和"一个旋钮在强、弱用户模拟器上都稳定提升"的实验上，与 §3 的论文结构一致。
- **不处理系统性偏离**。用户总是以同一方式偏离目标（[analysis.md §2](analysis.md) 问题 B）时，偏离在给定 agent
  说法下不是随机的，$U$ 会把它预测进去，用户段很小，本方法不起作用。

[appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 的按用户加权作为可能的扩展保留（例如按用户调
$\beta$），第一版不用。

**与 §3 并行**。两条线只在 critic 上有共同前提，可以同时做：

| 线         | 内容                                                                                               | 花费              |
| ---------- | -------------------------------------------------------------------------------------------------- | ----------------- |
| 现象（§3） | 更多用户模拟器的 step 0 评测、case study                                                           | API               |
| 方法 1     | 在现有 DeepSeek、Doubao rollout 上离线训练 critic，补上用户回复前的位置，先把 EV 做到正            | GPU               |
| 方法 2     | 离线诊断：advantage 方差里用户段占多少，按模拟器分开报。预期 Doubao 明显更高；结果也可以放进现象图 | GPU               |
| 方法 3     | 前两步通过后再训练：$\beta \in \{1, 0.5, 0\}$，DeepSeek、Doubao 各一次                             | GPU + API，需批准 |

**汇合点**：case study 出来后，看主要的用户侧失败是"给定 agent
的说法仍然随机"（本方法管得到），还是"系统性偏离"（管不到）。后者占主导时改方法；代码改动小，返工成本低。

## 5 用户出错处截断 + bootstrap（2026-10-18）

### 5.1 为什么还要第二部分

§4 的 $\beta$ 只缩小"给定 agent 这句话，用户回复仍然随机"的那部分波动。重跑和 case study v4
（[experiments/2026-10-18-user-sim-rerun.md](../experiments/2026-10-18-user-sim-rerun.md)、[user-sim-case-study-v4](../user-sim-case-study-v4/REPORT.md)）显示：

- 确认时同句 STOP（U1）加一句 guideline 就基本消失：Lite 20→0，Turbo 42→16，DeepSeek 3→0。
- 修不掉的是违反任务、编造事实（U3）：Lite 打 patch 后 U3 从 8 条变为 13 条，Qwen3.5-4B 用户 45 条。这类错误在同一种 agent
  说法下会反复出现，$U_k$ 会把它预测进去，用户段 $V_{k+1}-U_k$ 很小，乘 $\beta$ 不起作用（§4.4 最后一条）。

所以对这类错误要直接处理：先认出用户在哪次回复出错，再决定出错前后的 token 怎么计分。

### 5.2 做法：用户出错是截断，不是终止

记 episode 中第一次被检测为用户出错的回复紧跟在 agent 第 $k$ 轮之后。位置沿用 §4.1：agent 第 $k$ 轮的 token 为
$a_k..b_k$，$U_k = \text{value}[b_k]$。

- **出错前**（agent 第 $1..k$ 轮）：把 $U_k$ 当作截断处的 bootstrap 值，GAE 在 $b_k$
  处结束。$\gamma=\lambda=1$ 时 $A_i = U_k - V_i$。
- **出错后**（agent 第 $k+1$ 轮起）：actor loss mask 置 0，不参与训练。
- **critic**：被标记的 episode 不进 critic loss。结局 $R$ 混入了用户错误，用它当 target 会让 critic
  学到"这个局面会失败"。
- **没被标记的 episode**：标准 PPO / SAO，不受影响。

和 §4 的关系：等价于只在被标记的那一次用户回复上取 $\beta=0$，并丢掉之后的 token；其余用户回复仍按 §4 的 $\beta$ 处理（$\beta=1$
即原算法）。

**例子**。agent 问"确认取消订单吗？"之后 $U_k=0.5$；用户回答"Yes, proceed ###STOP###"，写操作来不及执行，$R=0$。

- PPO：agent 前面每个 token 都拿到 $0-V_i<0$，包括核对身份、查订单这些做对的步骤。
- AutoForge：整条 mask，这些做对的步骤也不学。
- 截断：前面的 token 拿到 $0.5-V_i$，按 critic 认为"走到这一步值多少"计分；STOP 之后没有 agent token。

### 5.3 检测器（可替换，作消融）

| 检测器         | 能认出的错误                                                       | 需要的信息                  | 部署时可用   |
| -------------- | ------------------------------------------------------------------ | --------------------------- | ------------ |
| 规则           | U1（授权与 STOP 同句）、U7a（agent 调用转接工具前用户发 TRANSFER） | 对话文本                    | 是           |
| 带剧本的 judge | U3（违反任务、编造事实）、U4b（没办完就走），也覆盖 U1、U7a        | 用户剧本 + guideline + 对话 | 否（oracle） |
| critic 读数    | 用户回复后 value 大幅下跌（$V_{k+1}-U_k$ 很负）                    | critic                      | 是           |

judge 的用法和 AutoForge 不同：AutoForge
用它决定整条丢不丢，这里用它定位第一次出错的回复，出错前的部分照常学。不在边界处重采样用户回复：那等于换一个更好的用户，偏离"区分用户错误和 agent 错误"这个研究目标。

离线验证（2026-10-18 启动）：

- critic 读数：用 PPO、SAO 训练好的 critic 在已有 rollout 上做前向，看
  $U_k$（value\[$b$\]、value\[$b-1$\]、最后 8/32 个 token 平均、整段回复平均）能否区分"用户确认时同句
  STOP"和"正常确认后成功"（`tau2-critic-boundary-20261018/`）。
- judge：在 v3、v4 的人工标注上测 precision、recall 和出错位置是否对得上；通用 prompt 与"附 U 码定义"的 prompt
  对比；规则作对照（`tau2-user-error-judge-20261018/`）。

### 5.4 风险

- **误报**：把 agent 引起的用户偏离标成用户错误，agent 的错就不学了。这是 §1 可辨识性问题在检测器上的体现；judge 的误报率要先在离线验证里量出来。
- **漏报**：退回标准 PPO，不会比 baseline 差。
- **bootstrap 值的偏差**：$U_k$ 不准时出错前的 advantage 跟着偏。$U_k$ 是 agent 生成的 `<|im_end|>` 位置，只有打开
  §4 的 `train_user_boundary_values` 才有训练，因此两部分要一起开。
- **critic 的样本选择**：被标记的 episode 不进 critic loss，critic 看到的失败变少，value 可能整体偏高。要报告被标记的比例，并检查
  critic 在未标记 episode 上的校准。

### 5.5 实验矩阵与代码改动

同一算法开关开与关比较（同 [user-td-split-plan](../experiments/2026-10-17-user-td-split-plan.md) §1）：

| 组                   | 说明                                            |
| -------------------- | ----------------------------------------------- |
| PPO                  | baseline（已有，120 步）                        |
| PPO + $\beta=0.5$    | §4，所有用户回复统一乘 0.5（pueue 631，已完成） |
| PPO + 整条 mask      | AutoForge 式 baseline，与截断用同一个检测器     |
| PPO + 截断           | 本节；检测器分别用规则、judge、critic 读数      |
| PPO + 截断 + $\beta$ | 两部分合用                                      |

代码：现有实现已支持终止边界（类型 3）；需要新增按 episode 输入的出错边界编号，$b_k$ 之后的 actor mask，以及被标记 episode 的 critic
loss mask。由 Codex 实现、Claude review，测试先红后绿。

### 5.6 进展（2026-10-19）

- **检测器**：离线评估后（[user-error-detectors/REPORT.md](../user-error-detectors/REPORT.md)），训练只用失败门控的
  U1 规则。U7a 去掉（训练轨迹上主要是在给 agent 免责）；gold 不进检测器（用户决定）；judge 在 DeepSeek 用户的训练轨迹上几乎没有
  增量，暂不接入；**critic 读数否决**：用户段下跌最大的边界全是终止边界，U1 只占约三分之一，其余是 agent 结算失败
  （[critic/user-boundary-forward.md](../critic/user-boundary-forward.md)）。5.3
  表里"部署时可用"的检测器因此只剩规则。
- **$U_k$ 的位置**：value\[$b$\]、value\[$b-1$\]、末 8/32 token 均值、整段均值的 AUC 相差不超过 0.02，保留
  value\[$b$\]； 但几种读法都分不出 agent 提议的对错，截断前的 $U_k-V_i$ 只是 critic 读数差，效果接近把结局 mask 掉。
- **训练**：完整的截断 + bootstrap 还没跑。pueue 636 只在 U1 标记的边界取 $\beta=0$：U1 在终止边界，actor 一侧与截断等价；
  critic 仍在被标记 episode 上训练，与 5.2 的设计不同。631（统一 $\beta=0.5$）在 checkpoint 70 后训练域低于 PPO 基线
  6–12pp， 但用户后端、并发、timeout 都不同，不能归因于
  $\beta$（[ppo-beta05-eval-drop/REPORT.md](../ppo-beta05-eval-drop/REPORT.md)）。
- **与动机的冲突**：U3 只在弱用户上多，DeepSeek 用户的训练轨迹里几乎没有。"重点转到 U3"要求训练用弱用户， 即"弱用户 /
  强用户各训一轮、再各加方法"的实验，尚未开始。
