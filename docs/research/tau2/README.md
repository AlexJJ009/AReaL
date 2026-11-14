# τ² 研究记录

本目录保存 τ²-bench 上的基线核查、PPO / GRPO / SAO 实验、critic 分析和研究决策。 结果和运行状态按对应实验的日期、模型、数据与设置阅读。

优先更新已有实验记录和索引，避免在不同文档中重复维护同一份结果。 旧环境的操作说明见 [工作区归档](archive/workspaces.md)。

## 历史记录索引

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
- [user-error-judge-validation/](user-error-judge-validation/REPORT.md)：带剧本的用户错误 judge 在
  v3/v4 标注上的离线验证 （G+T recall 约 80%，U3 约 57%；约 30% 的 agent 致败被误判为用户错误）。
- [user-error-detectors/](user-error-detectors/REPORT.md)：规则、gold、因果 judge 三类检测器在标注集和
  critic 前向边界上的离线比较。训练轨迹上用户致败几乎都是 U1，失败门控的 U1 规则 precision 约 82%，judge 没有增量；
  这是当时的检测器选择；10-19 后已放弃检测器主线，见[问题与计划 §3](research-direction/problem-and-plan.md)。

### 训练

- [training-setup.md](training-setup.md)：早期 99/60 题设置、后续 PPO 范围差异，以及训练侧对丢失 episode 的处理。
- [experiments/2026-10-24-qwenuser-no-telecom.md](experiments/2026-10-24-qwenuser-no-telecom.md)：Qwen
  用户、无 Telecom 的 2×2 和 seed-43 复现，结果截至 10-30。
- [training-data.md](training-data.md)：AReaL-tau2-data 的核查、筛选、pack 和 Telecom pack。
- [critic/](critic/README.md)：critic 实验（扩数据、value head、EV 上限、数据配比、case study）。
- [sao-runs.md](sao-runs.md)：SAO mix pack + critic 510 第 55 步。
- [experiments/2026-10-15-qwen3-ppo-grpo-sao.md](experiments/2026-10-15-qwen3-ppo-grpo-sao.md)：Qwen3
  上 PPO / GRPO / SAO 对比、OOM 和 SAO 崩溃的修复、两份 case study 的结论。
- [ppo-beta05-eval-drop/](ppo-beta05-eval-drop/REPORT.md)：631（PPO + 全局 β=0.5）后期 eval
  比基线低约 8.6pp 的诊断： 训练没崩、无 infra 污染，Retail 有策略退化迹象，但 user/judge 后端不同，不能归因到 β。
- [ppo-privileged-649-midrun/](ppo-privileged-649-midrun/REPORT.md)：649（带剧本 critic，β=1）第
  100 步的中途诊断：Retail +17.5pp；Airline 的收益只在合成 pack 上；Telecom 转人工题变好、排障题变差，净变化≈0。基线的
  Telecom 涨过（+9.4pp），差别在 agent 的账户操作，不在用户后端（651）。
- [privileged-critic-case-study/](privileged-critic-case-study/REPORT.md)：649 带剧本 critic
  的 EV 和 β 证据：晚期 EV 0.458 对 0.348；同一批 rollout 上题内 AUC 0.938 对 0.758，提升来自对对话状态的判断；中间用户回复处
  value 没有下降；终止处的意外更集中在 U1 上，统一 β 的误伤比标准 critic 小（292 对 692）。按用户剧本的可靠度加权不成立：剧本噪声稳定，但主要反映
  agent 的成败（与非 U1 失败的相关强于与 U1 的相关）。
- [ppo-training-budget-survey.md](ppo-training-budget-survey.md)：相关工作的训练步数、batch、lr
  与我们的对比。
- [infra-issues.md](infra-issues.md)：proxy 容量泄漏、Qwen3 加速配置、SAO 推权重崩溃。
- [issues/](issues/)：给 Codex 的 issue 记录。
  - [issues/2026-10-19-volcano-deepseek-model-switch.md](issues/2026-10-19-volcano-deepseek-model-switch.md)：火山
    plan 上 260910 在 10-19 03:26 被静默换成 261017，636 作废，训练用户改为豆包 Turbo（642）。
  - [issues/2026-10-19-volcano-weekly-quota-exhausted.md](issues/2026-10-19-volcano-weekly-quota-exhausted.md)：火山
    plan 周额度在 10-19 14:02 用尽，642 停在 step 24 后被 kill（recover 在 step 20），后续实验暂停到换接口。
- [reports/](reports/README.md)：Codex / Claude 的最终报告（xccl、critic recover、SAO mix），critic
  的报告在 [critic/reports/](critic/reports/ev-case-study-codex.md)。

### 研究方向与决策

- [compliance-erosion-case-study/](compliance-erosion-case-study/REPORT.md)：训练前后 policy
  合规变化，保留不同标注版本的定义与校准。

- [train-test-coverage/](train-test-coverage/REPORT.md)：训练与测试的能力覆盖差异及人工检查。

- [injection/DESIGN.md](injection/DESIGN.md)：用户错误注入器设计（2026-11-09 草案，待讨论；未实现、未运行）。

- [research-direction/](research-direction/README.md)：问题定义、分析、相关工作、方法草案附录、审稿。

- [decisions/](decisions/README.md)：决策记录（ADR）。

- [experiments/](experiments/README.md)：实验记录：每个实验一份，含假设、设置、运行记录和结果。
