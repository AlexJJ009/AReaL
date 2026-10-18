# τ² 基线审计与研究记录

**问题**：Qwen3.5-4B 在 τ² Airline / Retail 上初始分数很高（约 80%），GRPO 和 SAO
训练却看不出效果。分数高在哪里、训练为什么无效、用户模拟器的质量对训练有多大影响。对象：GRPO `formal-v2`、SAO
`mixed-v1`/`mixed-v4-disk`，以及之后的 fixed 变体上的 step 0 评测、critic 和 SAO 实验。

**状态（2026-10-15）**：初始分数、失败分类、benchmark 变体、训练数据筛选和用户模拟器影响的核查已完成；评测对超出 32K 上下文的 episode
首次超出即记 0
分，已有评测已按此重算（[issues/2026-10-15-eval-context-overflow-excluded.md](issues/2026-10-15-eval-context-overflow-excluded.md)）；critic
的 EV 一直在 0 附近，原因见 [critic/](critic/README.md)；SAO 正式运行（mix pack + critic 510 第 55 步）见
[sao-runs.md](sao-runs.md)。研究方向暂停方法推导，先找准现象（[research-direction/](research-direction/README.md)）。

**状态（2026-10-16）**：baseline 换到 Qwen3-4B-Instruct-2507，PPO / GRPO / SAO 在同一题池和评测上对比，记录见
[experiments/2026-10-15-qwen3-ppo-grpo-sao.md](experiments/2026-10-15-qwen3-ppo-grpo-sao.md)。PPO
48 步、GRPO 47 步已完成；PPO 续训到 120 步运行中，SAO 120 步排队，Telecom 全量评测（D1）在最后。

## 结论

**初始分数高（Retail/Airline 约 80%，Telecom 100%）的主因是模型本身能完成多数任务，加上 benchmark
自身的评分盲区；不是数据或评测代码被改过，也不是用户模拟器代劳。** Qwen3.5-4B step 0，DeepSeek 用户，官方 base 集：Airline
0.765，Retail 0.840（[step0-evaluations.md §2](step0-evaluations.md)）。

**航空的高分里约 36.5pp 是转人工得来的。** 航空任务按 DB + COMMUNICATE 打分，在"应该拒绝"的任务上，agent
转人工也能拿满分（[failure-taxonomy.md §3](failure-taxonomy.md)）。同一模型、同一用户，Airline 在 official 上
0.745，在 fixed v2 上 0.350，差 39pp，其中转人工规则清零的占
36.5pp（[benchmark-variants.md §1.3](benchmark-variants.md)）。这是官方评分的漏洞，同样的 DB reward
用于训练时会奖励转人工。

**用户模拟器的选择本身决定 Retail 分数。** [initial-audit.md §1](initial-audit.md)
"用户模拟器过度配合：否"只说明用户没有代劳。在修过的 fixed-v3 test
集上换五个用户模拟器（[user-simulator.md §2](user-simulator.md)），actor 不变，总分 mean@4 从
0.579（Qwen3.5-4B 自部署）到 0.758（GLM 5.3），差 17.9pp；Retail 上 Doubao 2.1 Turbo 比
DeepSeek-Flash 低 15.6pp。385 个失败里 168 个由用户造成；Doubao 的头号问题是确认的同一句里结束对话（Turbo 41 个），Qwen
用户主要是不按任务条件走。

**训练看不出效果，原因是评测噪声大于训练带来的变化、有效学习信号少、权重几乎没有朝一致方向移动**：同配置重测的总分波动 2–3pp，Airline 约 5pp。GRPO
`formal-v2` 和早期 SAO 的证据只能说"无法判断训练是否有效"，不能说"训练变差"。SAO 534（mix pack + critic 510）的评测在 32K
上下文限制下显示：训练让对话变长，超出 32K 的 episode 从第 0 步的 4 条增到第 40 步的 76 条（共 240 条），Airline mean@4 从
0.475 降到 0.263；第 10 步 Airline 比第 0 步高 0.113，但区间包含 0（[sao-runs.md](sao-runs.md)）。

**Qwen3 PPO baseline
提升小，目前不能归因到用户模拟器**（[experiments/2026-10-15-qwen3-ppo-grpo-sao.md §4](experiments/2026-10-15-qwen3-ppo-grpo-sao.md)）：评测
起点到 48 步 +3.5pp，CI \[−2.0, +9.0\]，而这套评测只能分辨约 8pp；同一批训练题上 reward 配对 +6.6pp，policy 在学。两份独立
case study 都发现失败约 92% 是 agent 的错，约 8% 是用户的错，且全是"确认后同一句 STOP"；用户的随机性主要放大评测噪声。48 步 × lr
1e-6 是可比工作里训练量最少的（[ppo-training-budget-survey.md](ppo-training-budget-survey.md)）。

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
- [telecom-case-study-d1/](telecom-case-study-d1/REPORT.md)：Qwen3-4B-2507 在官方 Telecom
  full 上的失败来源（随机失败样本中 agent 59/60、用户 0/60）、"恰好 2 GB"题错和 249 道训练题的筛选。
- [user-error-judge-validation/](user-error-judge-validation/REPORT.md)：带剧本的用户错误 judge 在 v3/v4 标注上的离线验证
  （G+T recall 约 80%，U3 约 57%；约 30% 的 agent 致败被误判为用户错误）。

### 训练

- [training-setup.md](training-setup.md)：题目划分、运行约定、训练侧对丢失 episode 的处理。
- [training-data.md](training-data.md)：AReaL-tau2-data 的核查、筛选、pack 和 Telecom pack。
- [critic/](critic/README.md)：critic 实验（扩数据、value head、EV 上限、数据配比、case study）。
- [sao-runs.md](sao-runs.md)：SAO mix pack + critic 510 第 55 步。
- [experiments/2026-10-15-qwen3-ppo-grpo-sao.md](experiments/2026-10-15-qwen3-ppo-grpo-sao.md)：Qwen3
  上 PPO / GRPO / SAO 对比、OOM 和 SAO 崩溃的修复、两份 case study 的结论。
- [ppo-training-budget-survey.md](ppo-training-budget-survey.md)：相关工作的训练步数、batch、lr
  与我们的对比。
- [infra-issues.md](infra-issues.md)：proxy 容量泄漏、Qwen3 加速配置、SAO 推权重崩溃。
- [issues/](issues/)：给 Codex 的 issue 记录。
- [reports/](reports/README.md)：Codex / Claude 的最终报告（xccl、critic recover、SAO mix），critic
  的报告在 [critic/reports/](critic/reports/ev-case-study-codex.md)。

### 研究方向与决策

- [research-direction/](research-direction/README.md)：问题定义、分析、相关工作、方法草案附录、审稿。
- [decisions/](decisions/README.md)：决策记录（ADR）。
- [experiments/](experiments/README.md)：实验记录：每个实验一份，含假设、设置、运行记录和结果。
