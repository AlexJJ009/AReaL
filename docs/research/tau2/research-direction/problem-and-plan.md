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

- 已有：同一 actor 在修过的 test 集上换五个用户模拟器，总分 mean@4 从 0.583 到 0.758（最强与最弱之差的 95% CI 为 \[7.1,
  27.9\] 个百分点；中间几档两两分不开），385 个失败里 168
  个归用户（[user-simulator.md §2](../user-simulator.md)）；旧 GRPO 训练中，成功−失败组内对比有 28.6% 是"成功 vs
  用户失败"，Retail 41.8%（[capability-gaps.md §1.3](../capability-gaps.md)，official 旧数据）。
- 没有：用户错误伤害训练结果的直接证据，以及上面第 2 种伤害是否发生。计划见下一步。

**核心困难是可辨识性**：一个用户的结果差，可能是因为他的反馈不可靠（agent 无能为力，应少罚），也可能是因为 agent
还不会应对这类用户（例如用户不主动给订单号，agent 该追问），后者恰恰最该学。方法只能用线上可观测的信息区分这两种情况；做不到的话，降权会把训练推向容易的用户。

**方向。** 单条 rollout、带 critic 的 SAO。轻量候选是 §4：在用户回复前后各读一次 critic，用户回复带来的意外乘 $\beta$。它在
critic 准确时只降方差，管第 1 种伤害；第 2 种伤害里的系统性部分它管不到，这一限制要写进主张。

**设定要点：**

- 可用的信息：对话文本、结局 reward、同一用户的历史交互；不用用户的隐藏目标，不用带剧本的 judge。
- 测试平台：τ² Airline/Retail，fixed 变体，官方 base 划分（训练 99 题、test 60 题），详见
  §3。[analysis.md §7](analysis.md) 的"每个（用户、题目）只交互一次"协议是方法阶段的设定，现阶段先做不同用户模拟器的评测和 case
  study。
- 方法草案见 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)：按用户给 advantage
  乘一个权重，权重取"agent 最后一次行动之后、结果还剩多少不确定性"的倒数。还没有实验；2026-10-12 起暂停，见 §3。
- 必须区分的已有工作：P-GRPO（按用户平移基准）、RGPO（离线标注者可靠度）、AutoForge（带特权信息的整条 mask），见
  [related-work.md §2](related-work.md)。

**决策轨迹：**

| #   | 决策                                                                                                                                                      | 原因                                                                                                                                                                                                                                                                                                             | 位置                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 1   | 起点：老师提出按用户反馈的置信度调整 advantage、clip，critic 和 GAE 也考虑用户；弱、强用户各训 baseline 和新方法                                          | 观察到好的用户模拟器下分数高、差的用户下学不会                                                                                                                                                                                                                                                                   | 本节开头                                                             |
| 2   | 离线真实数据按用户专业程度加权：不做                                                                                                                      | 离线数据可以先清洗、改写、合成，PPO 的 advantage/clip 也用不上；τ² 没有专业程度这个维度                                                                                                                                                                                                                          | [analysis.md §1](analysis.md)                                        |
| 3   | 把"用户不好"拆成 A 随机性（方差）和 B 系统性偏离（偏差）                                                                                                  | 两者解法不同：A 有经典降方差方法；B 是优化目标被带偏，样本再多也学错                                                                                                                                                                                                                                             | [analysis.md §2](analysis.md)                                        |
| 4   | "用户历史成功率"不单独作为置信度，列出 6 种候选信号                                                                                                       | 成功率混入 agent 能力，粒度只到用户；按成功率给 advantage 加权已有人做（ProxMO 2602.19225）                                                                                                                                                                                                                      | [analysis.md §3](analysis.md)                                        |
| 5   | 问题 A 不作为贡献                                                                                                                                         | 自托管用户时可固定用户噪声，经典方法无偏可用；API 用户拿不到随机数，只能统计处理                                                                                                                                                                                                                                 | [analysis.md §2](analysis.md)、[related-work.md §1](related-work.md) |
| 6   | AutoForge 式整条轨迹 mask（E1）只作 baseline，且不是上界                                                                                                  | AutoForge（2512.22857）已做；整条丢掉会丢掉出错前 agent 做对的部分。可重采样场景的真正上界是重采样到干净数据，同行已做很多                                                                                                                                                                                       | [analysis.md §4](analysis.md)                                        |
| 7   | 删去 E0（用户反应探针）和 E0′（运气占比）                                                                                                                 | E0 是 benchmark 诊断，不检验方法；E0′ 在 API 用户下分不开运气和反应                                                                                                                                                                                                                                              | [analysis.md §5](analysis.md)                                        |
| 8   | **转向不能重采样的在线场景**                                                                                                                              | 可重采样时评审会问"为什么不直接拿更好的数据"；不能重采样时 E1/E3 不可用，同行工作少。GRPO 不可用，critic 成为必需，与 SAO 工作衔接；用户历史残差成为合理信号                                                                                                                                                     | [analysis.md §7](analysis.md)                                        |
| 9   | Coverage, Not Targeting（2609.02417）不再引用                                                                                                             | 证据太弱：τ² 结论只来自一次 LoRA 更新，效应与评测噪声同级                                                                                                                                                                                                                                                        | [related-work.md §1](related-work.md)                                |
| 10  | 不与 GiGPO 比较                                                                                                                                           | 它按完全相同的中间状态分组算 step 级优势，需要同一题多条 rollout；τ² 用户回复各不相同，状态几乎不重合；在线场景每题只有一条                                                                                                                                                                                      | [related-work.md §1](related-work.md)                                |
| 11  | 在线场景不撞车，继续；要正面回应三篇中度相关工作，并修改可靠度的定义                                                                                      | 检索（[related-work.md §2](related-work.md)）没找到同时做到"单条 rollout + critic、用户反复出现、在线估计用户可靠度并缩放更新"的工作，但各部件都有人做过。$\bar r_u$ 有两个问题：critic 认出用户后残差均值趋于 0；降权"做不好的用户"会把训练推向容易的用户                                                       | [related-work.md §2](related-work.md)                                |
| 12  | 方法草案：按用户给 advantage 乘权重，权重取 agent 最后一次行动后残差平方的倒数（带收缩、用历史对话算）                                                    | 能改的只有 critic（只降方差）和乘系数（改目标）；残差均值会被 critic 吸收，残差平方不会；在"用户只加噪声"的模型下，乘正权重不改方向，逆方差权重方差最小                                                                                                                                                          | [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)     |
| 13  | **暂停 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 的方法推导，先换更多用户模拟器和 benchmark、做轨迹 case study，找准现象再定方法** | 代码改动简单而数学过重，理论没有指导设计，也没拦住核心估计量的设计错误（Codex 审稿）；LLM RL 论文靠"现象 + 简单修正 + 消融"说服人；现象证据目前只有两个模拟器、一个 benchmark                                                                                                                                    | §3                                                                   |
| 14  | **轻量方法候选：在用户回复前后各读一次 critic，用户回复带来的意外乘 $\beta$；与现象研究并行**                                                             | 粒度到每次回复、无用户状态、一个超参数；测量位置在线可定；critic 准确时只降方差不改期望；用户反应依赖 agent 时记到 agent 身上                                                                                                                                                                                    | §4                                                                   |
| 15  | **研究对象定为真实用户的不可靠反馈（2026-10-13）；模拟器是可控的代理**                                                                                    | 真人同样会确认后离开、改主意、不给信息，不能假定用户质量高。真人的 prompt 改不了、不能重来，所以改 prompt 和换模型只用来构造可靠度不同的用户（persona），按概率注入固定错误用来做受控对照；重采样不是可比 baseline。由此要求：训练用不可靠用户、评测用可靠用户并检查 policy 合规；方法必须区分"用户错"和"用户难" | §1、`motivation-review-20261013.md`                                  |

**下一步**（2026-10-13 更新，来自 `motivation-review-20261013.md` 第 6
节）：评测侧的现象已确认（[user-simulator.md §2](../user-simulator.md)），下面补"伤害训练"这一环，按顺序：

1. **受控缺陷的 SAO 训练（X1）**：同一个用户模拟器，只开关"确认后按概率离开"，各训一次，在可靠用户上评测并报 policy 合规。
1. **好用户与坏用户各训一次、交叉评测（X2）**：检验能力能否迁移。
1. **"学错方向"的启发式分析（X3）**：训练过程中跟踪写前确认率，以及用户离开率是否随 agent 的确认说法变化。
1. **反事实回放（X4，先内部测）**、**critic 在用户回复边界的校准（X5）**，然后才比较 §4 的方法。

## 2 结论

- **问题存在**。训练里约 29% 的组内对比是"agent 做得对，但用户出了错"（Retail
  42%，[capability-gaps.md §1.3](../capability-gaps.md)），换一个用户模型 Retail 就差
  20pp（[user-simulator.md §1](../user-simulator.md)）。
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
- 推导越来越重，很可能说明问题切入得还不够准。目前的现象证据只有两个用户模拟器（DeepSeek、Doubao）、一个
  benchmark，主要的失败机制只有一种（Doubao 在确认的同一句里结束，[user-simulator.md §1](../user-simulator.md)）。

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

| 模拟器                   | 档位            | 接入方式                              | 状态                                                                             |
| ------------------------ | --------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| Qwen3.5-4B（训练前版本） | 弱              | 本机 SGLang，占 1 张卡（必要时 2 张） | 待部署；4B 可能演不好用户，case study 里单独归为"模拟器失效"                     |
| Doubao 2.1 Turbo         | 中下            | Ark API                               | 已有 base 集结果（[step0-evaluations.md §2](../step0-evaluations.md)，official） |
| DeepSeek-Flash           | 中上            | DeepSeek API（`deepseek-flash`）      | 已有；若换成 V4.1，需确认是否同一 endpoint，否则算新模拟器                       |
| GLM 5.3                  | 中上 / 强，开源 | 待定                                  | endpoint 待接入                                                                  |
| Claude Haiku 5.5         | 强，闭源        | 待定                                  | endpoint 待接入，型号名在接入时核实                                              |
| GPT-5.6 Luna             | 强，闭源        | 待定                                  | endpoint 待接入                                                                  |

不选最强一档（如 GPT-5.6 Sol）：强用户的偏差方向是替 agent 做判断、把失败救回来，用 Haiku / Luna 这一档就能观察到，没必要多花钱。

**GPU 分配。** 本机 8 张 A100。用户模拟器自己部署时占 1 张卡，其余给 actor：step 0 评测为用户 1 卡 + actor 推理 7 卡；训练在
[training-setup.md §2](../training-setup.md) 的布局（rollout 3、评测 1、actor 4）上从 rollout 里让出 1
张给用户。用户侧会不会成为吞吐瓶颈，先在 smoke test 里测，不够再加到 2 张。

**启动前必须先解决：NL judge 和用户模拟器绑在一起。** `examples/tau2/agent.py` 的
`_bind_nl_assertion_evaluator` 把 NL 断言的 judge 设成 `user_llm`，换用户就等于同时换 judge。fixed v2 下
test 60 题里有 19 题（Airline 9、Retail 10）的 reward 含非空 NL
断言；[user-simulator.md §1](../user-simulator.md)估计 Doubao 下 judge 最多影响约 1pp，但那是 official
变体（Airline 不计 NL）、用户还是大模型的结果。用 4B 做用户时，judge 也变成 4B，结果不可比。需要加一个独立的 judge 配置，固定为
DeepSeek-Flash、temperature 0，所有模拟器共用。

**启动前的检查：** 每个新模拟器先跑 5～10 题 smoke test，确认能正常扮演用户、judge 调用走的是固定模型；再从已有评测日志按每条 episode
的用户侧 token 量估算费用，批准后上全量。

**actor 怎么选。** 换 actor 作为第二个消融轴，检验现象和收益是否依赖 agent 的强弱。我的假设是 agent 弱时失败主要来自 agent
自己，用户的影响被掩盖；agent 强时用户的影响才显出来。这个假设本身可以由下面的评测检验。

| actor             | step 0 评测（所有用户模拟器） | 训练       | 理由                                                                                                                                                                                                                                                   |
| ----------------- | ----------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Qwen3.5-4B        | 测                            | 主 actor   | SAO 和 critic 只在它上面跑通过（[infra-issues.md §2](../infra-issues.md)）；用户模拟器的影响已观察到（Retail 0.851 对 0.651）。DeepSeek 下 Retail 提升空间小（fixed 0.849），Airline 0.388 和弱用户下有空间                                            |
| Qwen3-4B-Instruct | 测                            | 第二 actor | 提升空间大（fixed 0.225 / 0.410），组内有对有错的题最多（[capability-gaps.md §2.2](../capability-gaps.md)）。训练前要先处理"工具调用写成正文"（M17），否则提升里混着格式修复；Qwen3 的 critic 路径要先验证（[infra-issues.md §2](../infra-issues.md)） |
| Qwen3-4B-Thinking | 测                            | 不训练     | 训练要求 `enable_thinking=false`（[step0-evaluations.md §6.1](../step0-evaluations.md)）；每轮生成中位数 956 token，是 Instruct 的 11 倍，API 用户下的长对话训练成本高                                                                                 |

step 0 评测只花用户模拟器的 API 费用，三个 actor 都测；训练费用大得多，先在 Qwen3.5-4B 上做"用不同模拟器训练"的 2×2，确认现象后再在
Qwen3-Instruct 上重复。

**要先想清楚的两点：**

- **现象必须出在训练上，不只是评测上**。"换 Doubao 后 Retail 从 0.851 降到 0.651"是评测差异，说明 benchmark
  分数受用户模拟器影响，但还不是 RL 的问题。论文要的现象是：用不同模拟器训练，得到的提升差别很大，甚至变差。这需要训练实验，最小版本是 2×2：用 DeepSeek 和
  Doubao 各训练一次，两个模型都在两个模拟器上评测。
- **"稳定提升"要交叉评测才有说服力**。如果在 Doubao 上训练、也只在 Doubao 上评测，提升可能只是学会了应付 Doubao
  的毛病（例如学会避开会触发提前结束的确认说法），不一定是 agent 变强。评测要包括训练没见过的模拟器，最好固定一个或几个评测用模拟器，报告"训练模拟器 ×
  评测模拟器"的矩阵。

## 4 轻量方法候选：只扣掉用户这次回复带来的意外（2026-10-12）

**一句话**：在每次用户回复的前后各读一次 critic，把这一步的 TD 误差拆成"agent 这一轮带来的变化"和"用户这次回复带来的意外"，后者乘一个系数
$\beta \in [0, 1]$ 再进 GAE。$\beta = 1$ 就是原来的 SAO。

**做法**。记 agent 第 $t$ 轮开始前 critic 的预测为 $V_t$（现有），agent 第 $t$ 轮说完、用户还没回复时的预测为
$U_t$（新增的读数）。对话结束时 $V_{T+1} = R$。原来跨过用户回复的 TD 误差拆成两段：

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

**为什么说得通（唯一需要的命题）**。若 $U_t$ 准确，即
$U_t = \mathbb{E}[V_{t+1} \mid \text{对话到 agent 第 } t \text{ 轮说完为止}]$，则用户段的条件期望为 0，乘
$\beta$ 不改变梯度的期望，只降低方差。这与 Mao et al.（[analysis.md §2](analysis.md)）扣掉外部输入的思路相同，区别是：$U_t$
是在看到 agent 这句话之后的预测，所以不要求用户的随机性与 agent 无关，[analysis.md §2](analysis.md) 说的"不能直接套用
Mao"的障碍不存在。

**和 [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 比轻在哪里**：

| 项                 | [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)（按用户乘权重） | 本节（按用户回复拆 TD 误差） |
| ------------------ | -------------------------------------------------------------------------------- | ---------------------------- |
| 粒度               | 整条对话、整个用户                                                               | 每次用户回复                 |
| 状态               | 用户 → 历史统计的表、收缩、窗口                                                  | 无                           |
| 超参数             | $k$、$\tau$、上下限、窗口                                                        | $\beta$                      |
| 测量位置           | 事后选的最后一次行动（Codex S1 指出有误）                                        | 用户回复前后，在线即可确定   |
| 需要的理论         | 噪声模型、逆方差、可辨识性论证                                                   | 上面一条命题                 |
| 用户反应依赖 agent | 降权会少学这部分                                                                 | 记到 agent 身上              |

**代价和风险：**

- **更依赖 critic**。$\beta = 0$ 时 advantage 全由 critic 的差分构成，critic 不准就有偏；$\beta$
  取中间值是折中。critic 先要有正的 EV（[initial-audit.md §2.4](../initial-audit.md)），这一前提与
  [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md) 相同。
- **代码改两处**：critic 在每个 agent 轮结束、用户回复前的位置多训练一个点（目标仍是 $R$）；GAE 在用户回复处换成
  $\tilde\delta$。现在的 value 是"看到这个 token 之前"的预测（`actor.py:226` 把 mask 左移一位），所以 $U_t$ 要取
  agent 最后一个 token 之后那个位置，现在那里没有 loss，要补上。
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
