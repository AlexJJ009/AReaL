# τ² 基线审计与研究记录

**问题**：Qwen3.5-4B 在 τ² Airline / Retail 上初始分数很高（约 80%），GRPO 和 SAO
训练却看不出效果。分数高在哪里、训练为什么无效、用户模拟器的质量对训练有多大影响。对象：GRPO `formal-v2`、SAO
`mixed-v1`/`mixed-v4-disk`，以及之后的 fixed 变体上的 step 0 评测、critic 和 SAO 实验。

**状态（2026-10-14）**：初始分数、失败分类、benchmark 变体、训练数据筛选和用户模拟器影响的核查已完成；critic 的 EV 一直在 0 附近，原因见
[critic/](critic/README.md)；SAO 正式运行（mix pack + critic 510 第 55 步）见
[sao-runs.md](sao-runs.md)。研究方向暂停方法推导，先找准现象（[research-direction/](research-direction/README.md)）。

## 结论

**初始分数高（Retail/Airline 约 80%，Telecom 100%）的主因是模型本身能完成多数任务，加上 benchmark
自身的评分盲区；不是数据或评测代码被改过，也不是用户模拟器代劳。**

**2026-10-10 更正**：[initial-audit.md §1](initial-audit.md)
"用户模拟器过度配合：否"只说明用户没有代劳；用户模拟器的选择本身决定 Retail 分数。 同一份代码下，DeepSeek 用户 Retail 0.851、Doubao 用户
0.651，差距主要来自 Doubao
在确认的同一句话里结束对话（[recuris-and-leaderboard.md](recuris-and-leaderboard.md)）。

**2026-10-13 更新**：在修过的 fixed-v3 test
集上换了五个用户模拟器（[user-simulator.md §2](user-simulator.md)），actor 不变，总分 mean@4 从
0.583（Qwen3.5-4B 自部署）到 0.758（GLM 5.3）。385 个失败里 168 个由用户造成；Doubao 的头号问题仍是确认的同一句里结束（Turbo
41 个），Qwen 用户主要是不按任务条件走。早先记录的"Doubao 开场第一句就结束"是方舟服务异常，v3 五组运行里都是 0。

**训练看不出效果，原因是评测噪声大于训练带来的变化、有效学习信号少、权重几乎没有朝一致方向移动**。现有证据只能说"无法判断训练是否有效"，不能说"训练变差"。

**2026-10-10 补充：航空的高分里约 30pp 是转人工得来的**。航空任务按 DB + COMMUNICATE 打分，在"应该拒绝"的任务上，agent
转人工也能拿满分（[failure-taxonomy.md §3](failure-taxonomy.md)）。这是官方评分的漏洞，同样的 DB reward
用于训练时会奖励转人工。

## 仍未完成

- 用 gpt-5.2（reasoning low）做用户跑一次 base 集，与榜单定标。需要 OpenAI 渠道和费用，需要批准。
- 按题统计 infra 失败率（[training-setup.md §3](training-setup.md)）。

## 目录

数据与脚本：原始数据在 `/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/`
下各运行目录，核查脚本在 [scripts/](scripts/)。

### 核查与评测

- [initial-audit.md](initial-audit.md)：初始分数为什么高、训练为什么看不出效果，以及对应的代码改动记录。
- [benchmark-variants.md](benchmark-variants.md)：official / fixed 变体的定义、转人工规则、与官方
  `tau2 run` 的差异。
- [step0-evaluations.md](step0-evaluations.md)：训练前（step 0）评测，含 Qwen3.5-4B 和
  Qwen3-4B-2507。
- [failure-taxonomy.md](failure-taxonomy.md)：两组 step 0 评测的失败分类。
- [user-simulator.md](user-simulator.md)：用户模拟器对分数和训练信号的影响，五个用户模拟器的对比。
- [recuris-and-leaderboard.md](recuris-and-leaderboard.md)：与 Recuris 论文和官方榜单的对比。
- [capability-gaps.md](capability-gaps.md)：模型能力缺口与训练期问题的 case study。
- [user-sim-case-study-20261012/](user-sim-case-study-20261012/REPORT.md)、[user-sim-case-study-v3/](user-sim-case-study-v3/REPORT.md)：用户模拟器
  case study 报告及标注。

### 训练

- [training-setup.md](training-setup.md)：题目划分、运行约定、训练侧对丢失 episode 的处理。
- [training-data.md](training-data.md)：AReaL-tau2-data 的核查、筛选、pack 和 Telecom pack。
- [critic/](critic/README.md)：critic 实验（扩数据、value head、EV 上限、数据配比、case study）。
- [sao-runs.md](sao-runs.md)：SAO mix pack + critic 510 第 55 步。
- [infra-issues.md](infra-issues.md)：proxy 容量泄漏、Qwen3 加速配置、SAO 推权重崩溃。
- [issues/](issues/)：给 Codex 的 issue 记录。

### 研究方向与决策

- [research-direction/](research-direction/README.md)：问题定义、分析、相关工作、方法草案附录、审稿。
- [decisions/](decisions/README.md)：决策记录（ADR）。
- [experiments/](experiments/README.md)：实验记录：每个实验一份，含假设、设置、运行记录和结果。
