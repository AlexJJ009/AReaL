# τ² 初始分数偏高与训练无效：核查记录（2026-10-09）

短期核对用。对象：GRPO `formal-v2` 与 SAO
`mixed-v1`/`mixed-v4-disk`（Qwen3.5-4B，non-thinking，DeepSeek Flash 用户模拟器与 NL judge，评测
temperature=1、每题 8 次）。全部结论来自已有记录的重算，没有新增采样、训练或付费模型调用。

各节内容：

- 第 1–5 节：核查 2026-09-27 的训练记录。
- 第 7–8 节：评测调度与官方对齐，以及新口径下的 step 0 评测。
- 第 9 节：与 Recuris、官方榜单的对比，以及 Doubao 用户失败精读。
- 第 10 节：失败分类。
- 第 11 节：当前 benchmark 变体的定义和校验步骤。
- 第 12 节：训练计划，以及额外训练数据的调研与核查（12.7–12.8）。
- 第 13 节：两轮 step 0 评测。
- 第 14 节：模型能力缺口与交互式 RL 训练期问题的 case study（2026-10-11）。
- 第 15 节：Qwen3-4B-2507 两个模型的 step 0 评测、能力缺口与加速配置核查（2026-10-11）。
- 第 16 节：研究方向讨论——RL 训练中如何对待用户模拟器的质量（2026-10-11）。

第 11–12 节只保留最新的决策和执行步骤。τ² 的调研、对比、找 gap 都记在本文，不另开文档。

原始数据根目录：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking`。
核查脚本在本目录 `scripts/`，大体积中间结果保留在
`/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/`（不入库）。
第 7.1 节的 Doubao 首轮探测、第 9.5 节的高分核查证据归档在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-scratch-archive-20261010/`。

## 1. 结论

**初始分数高（Retail/Airline 约 80%，Telecom 100%）的主因是模型本身能完成多数任务，加上 benchmark
自身的评分盲区；不是数据或评测代码被改过，也不是用户模拟器代劳。**

**2026-10-10 更正**：第 2 节"用户模拟器过度配合：否"只说明用户没有代劳；用户模拟器的选择本身决定 Retail 分数。 同一份代码下，DeepSeek 用户
Retail 0.851、Doubao 用户 0.651，差距主要来自 Doubao 在确认的同一句话里结束对话（第 9 节）。

**训练看不出效果，原因是评测噪声大于训练带来的变化、有效学习信号少、权重几乎没有朝一致方向移动**。现有证据只能说"无法判断训练是否有效"，不能说"训练变差"。

**2026-10-10 补充：航空的高分里约 30pp 是转人工得来的**。航空任务按 DB + COMMUNICATE 打分，在"应该拒绝"的任务上，agent
转人工也能拿满分（第 10.3 节）。这是官方评分的漏洞，同样的 DB reward 用于训练时会奖励转人工。

## 2. 初始分数为什么高

| 怀疑方向             | 结论             | 依据                                                                                                                                                                                                                                 |
| -------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 改过数据集或评测代码 | 否               | 1,098 条日志记录的数据目录都是 `_cache/tau2-bench-b7ea907…/data`；与 GitHub 官方 b7ea907 `diff -rq` 无差异。site-packages 中 tau2 源码与官方逐字节一致。运行时只改了 NL judge 模型（`examples/tau2/agent.py:199-207`）和关闭费用统计 |
| agent 看到答案       | 否               | 3,088 条 GRPO rollout 的 system 部分只有官方 policy 与工具定义，没有 `task_instructions`/`nl_assertions`                                                                                                                             |
| 用户模拟器过度配合   | 否（抽样内）     | 用户 prompt 为官方 `simulation_guidelines.md` 原文；精读 30 条轨迹，0 条成功是用户代劳。反而 Retail 39% 的失败停在 agent "请确认"后用户无回复（由日志推断），这压低分数                                                              |
| NL judge 判错        | 影响小           | Airline 20/20 题 `reward_basis` 不含 NL（官方数据如此）；Retail 39 题含 NL 标记，只有 11 题断言非空、真正调用 judge                                                                                                                  |
| 评分盲区             | 存在，解释一部分 | 只说 "Done." 不调工具也得 1：Airline test 7/20（2、6、13、26、31、45、48），Retail test 2/40（12、65），Airline train 16/30，Telecom 0。原因：这些题正确做法是拒绝或只查询，DB 终态不变，而"应当拒绝"写在不计分的 NL 断言里          |
| 模型本身能力         | 主因             | 抽样成功中 60–65% 为干净完成，约 20% 属评分宽松，约 20% 依赖运气或有歧义的 gold                                                                                                                                                      |

另有两处标注问题：Retail 105 的 gold 换货操作本身报"礼品卡余额不足"（\$17 \< \$21.10），NL 却要求完成换货；Retail 38 的取消理由
gold 固定为 "no longer needed"，用户多数时候选 "ordered by mistake"。

官方 NL 实现有一个隐患：judge 返回空 `results` 时 `all([])`
为真，直接判通过（`tau2/evaluator/evaluator_nl_assertions.py:51,134`）。历史记录未保存 judge
原始返回，无法确认是否发生过。

### 与 Recuris 论文（Qwen3.5-4B：Retail 68.0 / Airline 75.3）的差距

已在第 9 节查清：两边都用 Doubao 用户时，差距在抽样误差内；Recuris 的 harness 对这一行没有影响。下面是 2026-10-09
的原始判断，保留作记录。

- **不是题集差异**。未训练模型在 train split（epoch 1 覆盖全部任务，权重相对变化约 1e-4）上为 Retail 84.8%、Airline
  78.3%；与 test 合并估算全量 114/50 题约 82%/79%。
- **论文 Table 1 的用户模拟器不是 4B 自己**。Recuris 源码 `src/recuris/adapters/tau2/treatment.py` 的
  transfer 配置固定 Doubao Seed 2.0 Pro 作为用户模拟器和 NL
  judge（`reasoning_effort=medium`，temperature=0），被测 agent temperature 强制为 0。

## 3. 训练为什么看不出效果

### 3.1 评测噪声

| 项目                                                   | 数值                                                                        |
| ------------------------------------------------------ | --------------------------------------------------------------------------- |
| 同一初始模型两次 step 0（GRPO vs SAO，Retail+Airline） | 81.88% vs 77.29%，差 4.58 pp，按题配对 95% CI \[+0.4, +9.0\]                |
| 仅采样噪声的零假设 95% 半宽                            | 合并 ±4.1 pp；Retail ±5.3；Airline ±6.4                                     |
| step 0 每题结果（60 题）                               | GRPO：29 题 8/8、1 题 0/8、30 题混合；SAO：21/1/38；两次都 8/8 的只有 16 题 |
| SAO step 5 重复记录去重规则影响                        | 保留首条或末条，Airline 差 3.75 pp                                          |

所有 checkpoint 相对自身 step 0 的变化都在噪声带内：GRPO −2.3 至 −0.8 pp，SAO +0.8 至 +6.3 pp（SAO step 0
恰好是低的那次）。

### 3.2 有效学习信号少

- GRPO 导出的 386 个八轨迹组中 280 个（72.5%）奖励完全相同，无组内优势；实际训练只消费 368 组，比例略有偏差。
- 46 步中 5 步（9、19、33、37、40）整批全对，actor 梯度为零。
- Telecom 训练前就 320/320，却占每 epoch 41.6%（592/1424）rollout 预算；导出组中 160/166 全对。
- 训练期分场景奖励：Telecom 0.996、Retail 0.868、Airline 0.800。

### 3.3 权重几乎没动，且方向随机

- 训练确实保留 fp32 master weights（`optimizer_dtype: float32`；actor 以 fp32 实例化，日志有对应
  flash-attn 警告），精度设置没有问题。
- lr=1e-6、无 warmup、恒定；典型权重（|w|≈0.0076）的 bf16 ulp≈3.05e-5，单步 Adam 位移约 1/30 ulp。
- step 46 相对基座只有约 13% 的 bf16 权重改变取值，中位数改变 1 ulp。46 步方向一致时预期约 79% 翻转，随机游走预期约
  15%，实测与随机游走吻合。
- GRPO actor 梯度只在 step 46 超过裁剪阈值 1.0。

### 3.4 SAO critic

- explained variance 46 步中 28 步为负，中位数 −0.068；epoch 1 中 20/23 步为负。
- critic 梯度范数 46 步全部超过 1.0，每步都被裁剪（critic lr 5e-6）。

## 4. 对 Codex 先前调研的勘误

- 数字（成功率、组内饱和、什么都不做得分、Retail 105、critic EV）全部可复现。
- "题集不同是与论文比较的最明确障碍"：已用 train split 排除。
- "论文正文与源码不一致"：混淆了 frozen 与 transfer 两种配置。
- "NL judge 影响 39/40 题"：实际 11/40（Codex 后来已自行更正）。
- 漏掉：step 0 噪声水平、权重几乎未动、用户提前结束对话造成的假失败。

## 5. 据此做的代码修改（同日）

| 修改                                                             | 位置                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 评测 temperature=0（训练保持 1）                                 | `examples/tau2/config_grpo.yaml` `eval_gconfig`；`scripts/tau2/evaluate.py --temperature` 默认 0                                                                                                                                                                                                                |
| 评测每题 4 次（原 8 次）；评测并发改为 8 组×4 次，保持 32 条在途 | `eval_gconfig.n_samples: 4`、`evaluation_rollout.max_concurrent_rollouts: 8`；`evaluate.py --repeats` 默认 4                                                                                                                                                                                                    |
| 训练与评测去掉 Telecom                                           | `domains: [airline, retail]`（GRPO，SAO 继承）；SAO 预算 240+592=832/epoch，共 1664 episode、26 步；GRPO 也是 26 步                                                                                                                                                                                             |
| critic 改为 airline+retail 重训                                  | `config_critic_production.yaml`；`train_critic.py` 不再强制三场景，验证集行数按场景计算（formal 60、tune 21），共 14 步                                                                                                                                                                                         |
| 评测脚本按配置的场景计算覆盖                                     | `evaluate.py` 期望行数 = 各场景 test 题数 × repeats，manifest 记录 `domains`；`upload_metrics.py` 读 manifest 场景（旧 manifest 回退三场景）                                                                                                                                                                    |
| 评测题集二选一                                                   | `evaluate.py --task-set test`（默认，official test：Airline 20、Retail 40，每题 4 次共 240 条）或 `--task-set base`（官方榜单全量 train+test：Airline 50、Retail 114，共 656 条）；base 运行前核对题目 ID 等于官方 `base` 列表，summary 的 `by_domain_split` 分开给出 train/test 部分；manifest 记录 `task_set` |
| SAO 单场景入口                                                   | `run.sh policy {airline,retail} sao` 不再删除已不存在的 telecom 预算；`policy telecom sao` 明确报错                                                                                                                                                                                                             |

未改：GRPO/SAO 的 lr、训练步数、用户模拟器、judge。历史结果保持原口径（temperature=1、N=8、含 Telecom），不与新口径混比。

验证：tau2 相关 6 个测试文件 117 项通过；设置 `TAU2_DATA_DIR` 后数据依赖测试 52
项通过；`run.sh policy mixed grpo|sao`、`policy airline sao`、`data critic` 与 `evaluate.py` 均
`--check-config` 通过（GRPO/SAO 26 步，critic 14 步，评测 240 episode、temperature 0）。

## 6. 下一步

已执行或并入第 12 节。

## 7. 当前实现与官方 `tau2 run` 的差异（2026-10-10 复核）

### 7.1 起因

2026-10-09 的 `step0-base-t0-doubao21turbo` 跑到 74/656 个 episode 时中止。原因是 Doubao 用户第一句就
stop，policy 一次都没被调用，旧代码把这种 episode 当导出错误处理，连续重试失败后中止整轮。官方 tau2 会正常给这个 episode
打分。修复后整个调度层改为照搬官方 runner，见
[ADR 0009](../../decisions/0009-official-tau2-runner-semantics.md) 和
[PRD](../../prd/2026-10-09-tau2-official-runner.md)。

### 7.2 当前对照（HEAD `c70a9800`，官方 pinned `b7ea907`）

逐项对照官方源码。官方路径相对 `site-packages/tau2`。

**已一致**：

- **重试**：任何异常都整段重跑，最多再试 3 次，间隔 1 s（`runner/progress.py`）。
- **infra 排除**：重试用尽记 `INFRASTRUCTURE_ERROR`，不进指标，并报数量和原因。
- **打分**：只有 AGENT_STOP / USER_STOP 用官方 evaluator 打分，其余终止原因记 0；用户首句就 stop 也正常打分。
- **max_steps / max_errors**：200 / 10。
- **orchestrator 行为**：首条 agent 消息、非 solo、不做通信校验，都与官方相同。
- **用户模拟器**：直接调用官方实现，空回复时整段重跑，不在原地重试。
- **评测回复长度**：policy 单次回复不设上限，只受剩余上下文限制。

仍不同的项：

| #   | 维度                 | 官方                                                | 我们（评测）                                                        | 我们（训练）                                                                | 状态               | 影响                                                                |
| --- | -------------------- | --------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------ | ------------------------------------------------------------------- |
| 1   | 用户模拟器模型       | gpt-4.1，temperature 0；榜单推荐 gpt-5.2            | DeepSeek-Flash（默认）或 Doubao，temperature 0，top_p 1，超时 120 s | 同评测                                                                      | 未对齐             | 最大的一项：只换用户，Retail 差约 20pp（9.1）；与榜单数字不可直接比 |
| 2   | NL judge             | gpt-4.1，temperature 0                              | 与用户模拟器同一模型和参数                                          | 同评测                                                                      | 未对齐（暂不拆）   | 只影响含 NL_ASSERTION 的任务，约 1pp（9.1）                         |
| 3   | seed                 | `random.seed(300)` 为每个 trial 生成 seed           | 不传                                                                | 不传                                                                        | 未对齐（决定不改） | 只影响可复现性；provider 是否支持 seed 未核实                       |
| 4   | policy `num_retries` | 3                                                   | 0，重试放在 episode 层                                              | 同评测                                                                      | 未对齐（决定不改） | 一次传输错误会让整段 episode 重跑                                   |
| 5   | 上下文               | 不设窗口                                            | 32K；溢出抛异常，重试后排除                                         | 溢出记 `context_limit`，0 分并计入                                          | 有意不同           | 只有极长 episode 会被排除；两组 step 0 评测都没有发生               |
| 6   | 墙钟超时             | 默认不设                                            | 1800 s，超时按官方 TIMEOUT 记 0 分                                  | 1200 s，超时记 truncated、0 分                                              | 有意不同           | 实测最长 651 s，评测基本不触发                                      |
| 7   | policy 生成设置      | temperature 0                                       | temperature 0，thinking 关闭，concat 模板加 qwen3_coder 解析        | temperature 1，单次回复 4096 上限                                           | 有意不同           | concat 拼接与服务端完整渲染的 token 差异未量化                      |
| 8   | 领域                 | 榜单覆盖全部领域                                    | 只跑 Airline + Retail                                               | 同评测                                                                      | 有意不同           | 不能和榜单总分比，只能按领域比                                      |
| 9   | `pass^k` 的 k        | k 裁到每题最少有效 trial 数，排除 1 条就没有 pass^4 | 只报 k=1/2/4，每个 k 在有效 trial 数 ≥ k 的题上平均，并报题数       | 同评测                                                                      | 有意不同           | 无排除时两种算法结果相同；有排除时多报出一个 pass^4                 |
| 10  | 失败 episode 处理    | —                                                   | 同官方                                                              | 丢弃重试用尽的成员，存活 ≥4 的组照常训练，否则 epoch 末重跑一次（ADR 0011） | 训练有意不同       | 存活成员可能偏向短对话；按题失败率尚未统计                          |
| 11  | 用户工具             | `get_user_tools(include=task.user_tools)`           | `get_user_tools()`，不按任务过滤                                    | 同评测                                                                      | 未对齐（新发现）   | Airline/Retail 没有用户工具，无影响；重新加 Telecom 前要改          |
| 12  | `evaluate.py` 题集   | 榜单用 base 全量，≥4 trials                         | `--task-set` 默认 test，需显式传 `base`                             | 训练中评测用 test（formal）或 dev                                           | 默认值不同         | 忘传 `--task-set base` 的结果和榜单不可比                           |
| 13  | 题目与评分           | `b7ea907` 原样                                      | official 与 fixed 两个变体都报（第 11 节）                          | fixed                                                                       | 有意不同           | fixed 的数字不能和榜单比                                            |

10-09 旧对照表中已经解决的项：重试、最终失败排除、评测时的上下文溢出与墙钟、原地重试、用户 512 上限 （`afb42082`），评测去掉 4096
回复上限（`31634dee`），汇总改为官方 pass^k（`afb42082`、`c70a9800`）。

**未核实**：

- 官方遇到上下文溢出时的行为是从代码推断的。
- 文本模式下官方的 `persona_config` 是否为 None。
- concat 模板与服务端完整渲染的 token 差异。

### 7.3 指标口径（2026-10-10 起）

所有评测都报同一组指标：离线 `evaluate.py`、训练中的 validation 和 critic 之后的 policy 训练。指标是
mean@4、pass^1、pass^2、pass^4，按合计和领域分开报，不再报 pass@k。

- **mean@4**：每题 4 次的成功率先在题内平均，再在题间平均。没有排除时等于 pass^1 和 `avg_reward`。
- **pass^k**：官方 $\binom{c}{k}/\binom{n}{k}$，即同一题 k 次全部成功的概率，再在题间平均。

训练中评测的 `eval_gconfig.n_samples` 是 4，所以 pass^4 有定义。实现见 `examples/tau2/official_runner.py`
的 `REPORTED_PASS_HAT_KS`。

## 8. 官方 runner 口径下的 step 0 评测（同日补充）

两组都用 commit `31634dee`：未训练的 Qwen3.5-4B（non-thinking），base 集（Airline 50 + Retail 114 题，不含
Telecom），policy temperature 0，每题 4 次，用户模拟器 temperature 0、无回复上限，32K context，评测不设单次回复上限。

| 用户模拟器       | 领域    | episode（排除） | mean@4 = pass^1 | pass^2 | pass^4  |
| ---------------- | ------- | --------------- | --------------- | ------ | ------- |
| DeepSeek-Flash   | 合计    | 656（0）        | 0.838           | 0.752  | 0.646   |
| DeepSeek-Flash   | Airline | 200（0）        | 0.810           | 0.733  | 0.660   |
| DeepSeek-Flash   | Retail  | 456（0）        | 0.851           | 0.760  | 0.640   |
| Doubao 2.1 Turbo | 合计    | 656（2）        | 0.691           | 0.548  | 0.426\* |
| Doubao 2.1 Turbo | Airline | 200（1）        | 0.783           | 0.703  | 0.633\* |
| Doubao 2.1 Turbo | Retail  | 456（1）        | 0.651           | 0.480  | 0.336\* |

\* Doubao 有 2 个 episode 在重试时遇到 Ark 账户欠费，被排除；pass^4 只在其余 162 题上算（Airline 49、Retail 113）。

- 只换用户模拟器，Retail 从 0.851 降到 0.651，Airline 只差 0.03。Doubao 用户下的 Retail（0.651）已经接近 Recuris
  报告的 68.0；Airline（0.783）仍高于 75.3。
- 旧代码下同配置的 DeepSeek 评测是 0.794，新代码 0.838。区别是评测不再截断 4096 token 的回复（旧代码有 32 个 episode
  被截断）和重试规则；本次无排除。两者和用户模拟器本身的随机性各占多少，没有逐题核对。
- Doubao 评测里 3 个 episode 是用户第一句就 stop，按官方打分，没有中止评测。
- 第 6 节第 2 条的 Doubao 对照已完成（2.0 Pro 已下线，用 2.1 Turbo），分析见第 9 节。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261009/step0-base-t0-deepseek-official/`
和
`.../step0-base-t0-doubao21turbo-official/`（`summary.json`、`episodes.jsonl`、`trajectories/`）。

## 9. 用户模拟器、Recuris 与官方榜单（2026-10-10）

### 9.1 Doubao 用户为什么拉低 Retail

同一题、同一 trial 配对比较（第 8 节两组）：Airline 0.810 → 0.780，差 0.03 ± 0.03，不显著；Retail 0.851 →
0.649，差 0.20 ± 0.03（约 7σ），62 题变差、11 题变好。Retail 失败里 DB 不对的从 43 条增到 106 条，只挂 NL 断言的从 4 条增到
9 条， 失败的 `stop_reason` 都是 `user_stop`。NL judge
和用户绑定同一模型（`examples/tau2/agent.py:293-301`），最多解释约 1pp。

精读 Doubao Retail 失败。候选条件：最后一条用户消息带 `###STOP###` 且含确认词，上一条 agent 消息在请求确认，至少一个 gold
写操作没执行。共 90 条（DeepSeek 同条件 20 条，全部失败），人工逐条读 42 条，其余 48 条用脚本核对 agent 的确认 消息是否已带齐 gold
写操作的全部参数，并抽查 27 条一致：

| 类别                                                                             | 条数 | DeepSeek 同 trial 通过 |
| -------------------------------------------------------------------------------- | ---- | ---------------------- |
| 用户提前结束：agent 已按 policy 给出完整、正确的确认，用户说 yes 的同一句就 STOP | 79   | 66                     |
| agent 在 STOP 前已出错（错的退款方式、换成同一件商品、选错卡）                   | 7    | 4                      |
| 用户自己偏离任务说明（选了 gold 以外的颜色/型号）                                | 4    | 1                      |
| 用户合理结束 / agent 确认方式不当                                                | 0    | –                      |

用户提前结束能解释 Retail 下降中的 14.5–17.4pp（66/456 到 79/455；补上写操作后仍可能在别的检查上失败，所以 17.4 是上界）。
例：`retail:6:trial1`，agent 列出 item、新 item、支付方式并请用户回复 yes，用户回 "Yes, I confirm I want to
proceed with this exchange. ###STOP###"；`retail:109:trial1` 用户确认第 1 步就结束，另两个写操作没开始。

这是 Doubao 违反官方用户指令，不是 agent 的错：文本版
`data/tau2/user_simulator/simulation_guidelines.md:13-14` 要求 "continue the conversation
until the task is complete"，目标满足才发 `###STOP###`；agent 先确认再写是 retail `policy.md:16`
的要求。语音版 guideline 多一句 "Agreeing to an action is not the same as the action being
completed" （`simulation_guidelines_voice.md:42`），文本版没有，措辞有歧义。DeepSeek 在同一 prompt 下也犯 20
次，Doubao 约是它的 4.5 倍。

对训练的影响：训练 reward 来自同一个用户模拟器。换 Doubao 训练时，这类 episode 会把正确的确认行为记成 0 分。

### 9.2 与 Recuris 的差距

Recuris Table 1 里 Qwen3.5-4B 的 68.0 / 75.3 是不加 harness 的基线（stock
`llm_agent`，`README.md:71-73`）；+Recuris 是 68.3 / 79.0。他们的 tau2 是上游 `8ebb749` 加
`third_party/tau2/recuris.patch`：`llm_agent.py`（含 prompt）、 `user_simulator.py`、NL/action
evaluator、retail/airline environment 和我们 pinned 的 b7ea907 逐字节相同，用户 guideline 和数据 `diff`
为空；口径同为 base 全量、4 trials、max_steps 200、max_errors 10、失败 episode 排除。

| 我们的用户       | 领域    | 我们 | 按题 bootstrap 95% | Recuris | 差    |
| ---------------- | ------- | ---- | ------------------ | ------- | ----- |
| Doubao 2.1 Turbo | Retail  | 65.1 | \[59.2, 70.8\]     | 68.0    | −2.9  |
| Doubao 2.1 Turbo | Airline | 78.3 | \[68.8, 87.0\]     | 75.3    | +3.0  |
| DeepSeek-Flash   | Retail  | 85.1 | \[80.5, 89.3\]     | 68.0    | +17.1 |
| DeepSeek-Flash   | Airline | 81.0 | \[72.0, 89.5\]     | 75.3    | +5.7  |

结论：两边都用 Doubao 时差距在误差内，看不出谁高；我们"偏高"只发生在 DeepSeek 用户的 Retail，原因是 9.1 的用户行为差异。
剩余不确定：Doubao 2.0 Pro（Recuris）与 2.1 Turbo（我们）的差别无法复现；Recuris 没公开 4B 的部署参数（推理框架、tool
parser、thinking）。

### 9.3 Recuris 的 Mode A 与 terminal gate

- **是什么**。Mode A 指"用户的确认里带着 `###STOP###`，之后没有 agent 回合"，也就是 9.1 的模式（RT2
  `orchestrator.py:303`）。代码只在这一处定义它。
- **怎么处理。**`TAU2_GATE_TERM=1` 时，在 `USER_STOP` 之后、`_finalize()`
  之前检查：全程没执行过写工具，且任意一条用户消息命中 确认词正则（yes / sure / go ahead / confirm / ok / perfect
  等）。命中就往轨迹末尾追加一条 `role="user"` 的
  `[SYSTEM CHECK - not from the customer] ... execute the confirmed change now` 消息，让
  agent 再跑最多 4 轮并真实执行工具。 用户的 STOP 不删不改，`termination_reason` 仍是
  `USER_STOP`；评分重放整条轨迹，所以补执行的写操作会计入 DB 分。 变体 `_V5` 处理部分执行，`_A2` 看 agent 是否在请求确认，`_WM` 看
  Recuris agent 的待办账本。开关默认全关。
- **解决了没有**。仓库里没有可复核的效果数据：唯一的数字是代码注释里的自述（在 977 条 retail 仿真上调参，对通过的仿真触发 2%、失败的
  47%），没有对应的结果文件或日志。从正则看有明显的误报（开场 "Yes, I need help"、告别 "Perfect, thanks"、 否定句 "that's
  not correct" 都会命中，可能把应当不写的任务强行写坏）和漏报（确认措辞不在词表里；已有任意写操作时默认路径
  直接跳过）。它改变了评分对象（多了一条伪用户消息和用户看不到的 agent 回合），结果和官方 `tau2 run` 不可比。
- **发表的 68.0 / 75.3 开了没有**。base config 和 README 命令三个开关都是开的，但 `TAU2_GATE_TERM_WM=1` 时
  gate 需要 agent 实现 `wm_pending_lines`，`llm_agent` 没有，直接返回。所以这一行名义上开、实际没生效；+Recuris
  那一行大概率生效。发表运行的 参数文件不在仓库里，这是按配置推断。

我们不移植这个 gate：它不是官方行为。要在官方口径下避免 Mode A，只能在用户侧（换用户模拟器）或 policy 侧解决。

### 9.4 官方榜单

榜单数据在 pinned clone 的 `web/leaderboard/public/submissions/`（线上页面从 S3 拉同样的 JSON；本机连不上
taubench.com，线上内容没有直接核对）。当前 standard 行由 Sierra 自跑：tau2 1.0.1（即 b7ea907）、用户 gpt-5.2
（reasoning low）、NL judge 默认 gpt-4.1、base split、至少 4 trials、max_steps 200；pass^1 与我们的
mean@4 同口径。

| 模型（standard 行）      | Airline pass^1 | Retail pass^1 |
| ------------------------ | -------------- | ------------- |
| GPT-5.2 (high)           | 83.0           | 81.6          |
| Claude Opus 4.5          | 84.0           | 79.6          |
| Gemini 3 Flash           | 82.5           | 76.8          |
| GLM-5 think              | 82.5           | 73.7          |
| Gemini 3 Pro             | 80.5           | 75.9          |
| Qwen3.5-397B-A17B think  | 81.5           | 84.4          |
| Claude Sonnet 4.5        | 72.0           | 72.4          |
| GPT-5.2 (reasoning none) | 52.5           | 75.0          |

- "榜单上 Retail 普遍高于 Airline"只对 Legacy 行成立（16/16，中位数高 16.8pp）。这些是 tau2 0.1.x 或原版 τ-bench，
  v1.0.0 修了 airline 27 题、retail 26 题（`CHANGELOG.md:175-237`），前端标明不可直接比较。当前 standard 行只有
  3/8 是 Retail 更高，中位数 Retail 反而低 2.9pp。
- 所以 Airline > Retail 本身不算倒挂。我们的问题是 Doubao 用户让 Retail 掉了 20pp；DeepSeek 用户下的排序（Retail 更高）
  和 GPT-5.2 (none)、Qwen3.5-397B 一致。
- 榜上没有 ≤8B 的模型。4B non-thinking 在 Airline 上 0.81，接近 Opus 4.5 的 84、远高于 GPT-5.2 (none) 的
  52.5，说明我们的 用户模拟器可能比 gpt-5.2 更好说话。要和榜单对齐，需要用 gpt-5.2（reasoning low）做一次用户。

### 9.5 仍有效的早期核查

- 本地 Recuris checkout 的 Retail/Airline tasks、db、split、policy 与我们哈希一致。
- 960 条初始 validation 严格离线重放：Airline 全部一致；Retail 5 条差异都在需要 NL 的 task 68，没有本地聚合把失败改成成功。
- 评分覆盖不足的候选（未逐条验证）：Retail 206 条 reward=1 的记录先读订单、后认证；Airline 38 投诉的是延误航班，agent 查的是另一张
  reservation 仍得 1；Airline 7 的 COMM 金额有歧义。
- 每个训练 run 只有 4 条 episode 触到上下文或步数上限，这不是失败主因。注意这不是单次回复 4096 上限的截断：上一轮 GRPO 有 109/3088 条
  episode（Airline 12%）至少一次回复被 4096 截断，SAO 是 117/3296（Airline 17%）。抽查 119
  条被截的回复，都是被截断的长推理，不是重复循环。SAO 长度自适应 GAE 让终局信号衰减约 0.513，不会指数消失。
- 训练过 train 部分后，base 集分数不是完全 held-out 的泛化结果。MUA-RL、RWML、AReaL-SEA
  等同类工作的题集和用户模拟器都不同，分数不能直接比。
- 逐 step 的训练中 validation 表（mean@4 / mean@8）以飞书文档为准。

### 9.6 下一步

仍未完成的项目已并入 12.6。

复现脚本：`scripts/official-runner/fault_user_proxy.py`（PRD 第 6 节的故障注入
smoke）、`scripts/official-runner/metrics_table.py` （从 `episodes.jsonl` 重算第 8 节的 mean@4 /
pass^1/2/4，在仓库根目录运行）。

## 10. 两组 step 0 评测的失败分类（2026-10-10）

对象是第 8 节的两组评测，共 308 条失败：DeepSeek 106 条，Doubao 202 条。处理方式：

- 全部失败先按规则自动分类；
- 人工精读约 90 条（DeepSeek 约 40、Doubao 约 50）；
- 只因 NL 断言失败的 episode 逐条核对；
- 另读了约 20 条成功样本。

NL judge 与用户模拟器是同一个模型、同一套配置（7.2 第 2 项），所以下表放在同一大类里，但分开计数。

### 10.1 infra

数字精确，来自逐次尝试记录和 run.log。

| run × 领域    | 完成/计划 | 失败 | 排除 | 失败的尝试（按原因）                            | 重试后救回（其中成功） |
| ------------- | --------- | ---- | ---- | ----------------------------------------------- | ---------------------- |
| DeepSeek 航空 | 200/200   | 38   | 0    | 上下文超限 13                                   | 11（9）                |
| DeepSeek 零售 | 456/456   | 68   | 0    | 上下文超限 6，连接 1                            | 7（6）                 |
| Doubao 航空   | 199/200   | 43   | 1    | 上下文超限 12，Ark 欠费 2，连接 1               | 9（8）                 |
| Doubao 零售   | 455/456   | 159  | 1    | 上下文超限 11，用户空回复 2，Ark 欠费 2，连接 1 | 10（6）                |

- 两条被排除的 episode 是 `airline:10:trial0` 和 `retail:99:trial2`：都先上下文超限两次，再遇上 Ark 欠费两次。
- 上下文超限共 42 次，prompt 在 32,791–33,178 token，都比 32,768 多一点，全部被重试吸收。
- TIMEOUT 和 MAX_STEPS 都是 0。所有完成的 episode 都以 `user_stop` 结束，包括 `###TRANSFER###` 和
  `###OUT-OF-SCOPE###`。
- 完成集里的失败没有一条是 infra 造成的。
- 重试会偏向短轨迹：救回的 episode 里 DeepSeek 18 条成功 15 条，Doubao 19 条成功 14 条。估计 pass^1 因此偏高不到 1pp。

### 10.2 主分类

"精确"是逐条确认的；"估计"是规则分类，再用精读样本校正。

| 类别       | 子类型                           | DeepSeek | Doubao | 口径 | 例子                                                                                                   | 改进途径                          |
| ---------- | -------------------------------- | -------- | ------ | ---- | ------------------------------------------------------------------------------------------------------ | --------------------------------- |
| benchmark  | gold 不可行                      | 4        | 4      | 精确 | retail:105：礼品卡 \$17 付不了 \$21.10，agent 正确拒绝，NL 断言却要求完成换货                          | 无，记为上限损失                  |
| benchmark  | gold 写死取消原因                | 3        | 1      | 精确 | retail:38：用户说 "ordered by mistake"，gold 是 "no longer needed"                                     | 同上                              |
| benchmark  | gold 动作顺序错                  | 2        | 3      | 精确 | retail:64                                                                                              | 同上                              |
| benchmark  | 任务有歧义                       | 3        | 4      | 精确 | airline:7：问的是 "other upcoming flights"，gold 把要取消的两单也算进金额（\$1628 vs \$708），8 条全挂 | 同上                              |
| benchmark  | 未指定支付方式                   | 5        | 3      | 估计 | retail:60：任务没写支付方式，用户自己选了礼品卡，gold 是 paypal                                        | 同上；归 benchmark 还是用户有争议 |
| benchmark  | 环境实现 bug                     | 2        | 0      | 精确 | retail:100：`modify_pending_order_items` 把最后一个 variant 的价格写给所有 item，DB hash 对顺序敏感    | 同上                              |
| benchmark  | NL 断言要求空操作                | 0        | 2      | 精确 | retail:110：默认地址已经是新地址，断言仍要求 "modify user address"                                     | 同上                              |
| 用户模拟器 | 确认后立刻 STOP                  | 21       | 88     | 估计 | retail:17："Yes, I confirm… ###STOP###"，agent 还没执行写操作                                          | 换用户模型，或在 prompt 层修      |
| 用户模拟器 | 首轮即 OUT-OF-SCOPE              | 0        | 3      | 精确 | retail:109：第一条用户消息就是 `###OUT-OF-SCOPE###`                                                    | 换用户模型                        |
| 用户模拟器 | 中途 OUT-OF-SCOPE                | 0        | 5      | 估计 | airline:29                                                                                             | 换用户模型                        |
| 用户模拟器 | 编造或偏离指令                   | 3        | 4      | 估计 | retail:79（Doubao）编造"另一只瓶子是黑色"；retail:5（DeepSeek）known_info 里有姓名却拒绝提供           | 换用户模型                        |
| NL judge   | 判错                             | 0        | 0      | 精确 | DeepSeek 的 4 条 NL-only 失败都是 retail:105；Doubao 的 9 条 judge 都按字面判对                        | 当前不是瓶颈                      |
| 模型       | 参数、物品或支付方式错           | 25       | 46     | 估计 | airline:18：5 张票的退款全打到一张卡，应原路退回；retail:49 选错 variant                               | 训练                              |
| 模型       | 违反政策的写操作，或该转人工没转 | 13       | 15     | 估计 | airline:47：不符合条件的 basic economy 照样取消                                                        | 训练                              |
| 模型       | 不该转人工却转了                 | 21       | 17     | 估计 | airline:44：本该升舱，却转了人工                                                                       | 训练，但见 10.3                   |
| 模型       | 漏步骤、漏传达、错误拒绝、幻觉   | 4        | 7      | 估计 | retail:110：错称 pending 订单不能修改；retail:39 编造地址                                              | 训练                              |
| infra      | —                                | 0        | 0      | 精确 | 见 10.1                                                                                                | 充值 Ark；32K 是有意设定          |

按类别折算的 pass^1 损失（pp；合计 = 100 − pass^1）：

| run × 领域    | benchmark | 用户模拟器 | 模型 | 合计 |
| ------------- | --------- | ---------- | ---- | ---- |
| DeepSeek 航空 | 1.5       | 0.5        | 17.0 | 19.0 |
| DeepSeek 零售 | 3.5       | 5.0        | 6.4  | 14.9 |
| Doubao 航空   | 2.0       | 2.0        | 17.6 | 21.6 |
| Doubao 零售   | 2.9       | 21.1       | 11.0 | 34.9 |

- 两种用户下模型自己的失败差不多：航空约 17pp，零售 6–11pp。
- 两组评测的差距几乎全部来自 Doubao 确认后立刻 STOP，与 9.1 的精读结论（90 条中 79 条）一致。
- "确认后立刻 STOP"的判定条件是：agent 最后一次确认里列出了 gold 的全部 id。抽检 8 条有 1 条误判，所以 21 和 88 可能各偏高
  5–10%。即使用户不提前结束，agent 后面的步骤也未必都对，所以这部分 pp 是可追回的上限。
- 每条失败只标一个原因，取最先导致失败的那个。例如 retail:98 同时有支付方式错和提前 STOP。
- Doubao 有 14 条 episode 的用户中途改说中文，其中 5 条失败，已计入上面各类。

### 10.3 假阳性：转人工骗过 DB 评分

这是这次发现的最大问题，比任何一类失败都大。

**机制**：航空任务的 `reward_basis` 是 DB + COMMUNICATE，NL 断言不计分。以 airline:0 为例，它没有 gold 写操作，唯一的
NL 断言是 "Agent should refuse to proceed with the cancellation"。在这类"应该拒绝"的任务上，agent
只要转人工，DB 不变，就判成功。

**规模**：

- DeepSeek 航空无写操作任务上的成功有 86 条，其中 63 条转了人工，约占航空 pass^1 的 31.5pp。
- Doubao 航空是 55 条，约 27.6pp。
- 零售两组各 5 条。

**样本**：抽读 10 条（airline 0、5、6、26、27、34、41、45、48 和 retail 65），agent
都是在回避，没有按政策拒绝。airline:45 的用户明确说了不要转人工。直接计数：DeepSeek 航空 162 条成功里有 75 条调用过
`transfer_to_human_agents`，Doubao 是 156 条里 63 条。

**影响**：

- 这是官方评分规则本身的漏洞，榜单上的数字同样受影响，所以不算我们与官方的差异。
- 但它直接影响训练：训练 reward 用的是同一套 DB 评分，会强化"不确定就转人工"。10.2 里的"不该转人工却转了"就是同一个倾向的另一面。
- 从航空 0.81 里扣掉这部分，实际按政策解决的比例约 0.5。这也解释了为什么 4B 模型的航空分数接近 Opus 4.5（9.4）。

**其他检查**：

- 已知的"回一句 Done. 就得分"盲点，在这两组评测里没有出现：无写操作任务上的成功都有工具调用，用户轮数都大于 2。
- 没看到 NL judge 放水：成功样本里带数字的 NL 断言，数字都能在 agent 回复里找到。

### 10.4 可提升的方向

| 方向                           | 涉及的失败                      | 量级（pp，DeepSeek / Doubao）     | 做法                                                  | 是否改变与官方的可比性    |
| ------------------------------ | ------------------------------- | --------------------------------- | ----------------------------------------------------- | ------------------------- |
| 训练用的用户模拟器             | 确认后 STOP、OUT-OF-SCOPE、编造 | 零售 5 / 21                       | 训练用 DeepSeek（已定），不用 Doubao                  | 不改评测                  |
| 训练 reward 罚转人工           | 转人工假阳性，以及不该转却转    | 航空约 30 的假阳性 + 约 10 的失败 | 已做：fixed 变体的转人工规则（11.3），训练用 fixed    | 只改 fixed，official 不变 |
| 训练（模型能力）               | 参数、物品、支付、政策违规      | 航空约 17，零售 6–11              | GRPO / SAO（第 12 节）                                | —                         |
| 评测时附报一个去掉转人工的视图 | 转人工假阳性                    | 航空约 30                         | 由 fixed 变体代替：评测同时报 official 与 fixed（13） | 不改官方指标              |
| benchmark 本身                 | gold 不可行、歧义、环境 bug     | 1.5–3.5                           | fixed 已修大部分（11.1–11.2）；official 不改          | 只改 fixed                |
| infra                          | Ark 欠费；32K 边缘超限          | 排除 2 条                         | Ark 充值并换 key；32K 不改                            | —                         |

### 10.5 官方和社区是否已经修过（2026-10-10 调研）

**官方没有修。**

- 我们 pin 的 `b7ea907` 与 2026-10-10 的 main 相比，`src` 和 `data` 的 diff 为空。这一条是调研 agent clone
  后 diff 得到的，我没有独立复核。
- Airline 和 Retail 的 `tasks.json` 只在 v1.0.0 改过一次，即官方的
  [τ³ task fixes](https://taubench.com/blog/tau3-task-fixes.html)（Airline 27 题、Retail 26
  题，来自 Amazon 的 τ-Bench Verified），这批修复已经包含在 pin 里。
- Airline 的 `reward_basis` 从来没有包含 NL_ASSERTION。维护者在
  [#224](https://github.com/sierra-research/tau2-bench/issues/224)
  表示会处理，[RFC #129](https://github.com/sierra-research/tau2-bench/issues/129) 计划让
  no-action agent 得 0 分，两者都没有实现。

**社区修复**：以下 PR 全部 open，未合并。

| 来源                                                                               | 修什么                                                                                       | 覆盖 10.2 / 10.3 的哪项  | 核实情况                                                                        |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------- |
| [PR #411](https://github.com/sierra-research/tau2-bench/pull/411)                  | Airline 23 道不写 DB 的拒绝题改为 `[DB, NL_ASSERTION]`，按实际 criteria 修剪 Retail 的 basis | 转人工假阳性，间接覆盖   | 改动已读；断言里没写"不得转人工"，judge 能否拦住没人测过                        |
| [PR #322](https://github.com/sierra-research/tau2-bench/pull/322)                  | `modify_pending_order_items` 按每个 item 自己的 variant 写价格                               | retail:100 环境 bug      | 本地验证：打补丁后两种顺序 hash 相同、价格正确                                  |
| [PR #571](https://github.com/sierra-research/tau2-bench/pull/571)                  | retail:105 换成礼品卡付得起的商品；删去 retail:64 中执行不了的一步                           | retail:105、retail:64    | 本地验证：原 gold 执行抛错，新 gold 执行成功                                    |
| [PR #580](https://github.com/sierra-research/tau2-bench/pull/580)                  | gold 的写操作执行抛错时，不再给 DB=1                                                         | 防止同类 gold 不可行的题 | 改动已读                                                                        |
| [KnowWhenToHandOff](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2) | 训练 reward 对不必要的转人工 −0.5、漏转 −1.0，基于 tau2 train split                          | 转人工（只改训练）       | 只读了 model card：Qwen3-4B 过度转人工从 46% 降到 10%，成功率持平；代码仓库 404 |
| [ABC](https://arxiv.org/abs/2507.02825) patch                                      | 原版 tau-bench：不写 DB 的题要求调用指定的读工具                                             | do-nothing agent         | 没有移植到 tau2；拦不住"先读再转人工"                                           |

Amazon 的 [tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified) 比我们的
pin 旧，内容已经被官方吸收。AGI-Eval 的
[tau2-bench-revised](https://github.com/AGI-Eval-Official/tau2-bench-revised) 只改了用户
prompt。两者都没有修转人工问题。

**调研时没有任何来源修过的**（除 airline:7 和 #514 外，现已在 fixed 变体中由我们修复，见第 11 节）：

- 评测端针对转人工的直接规则；
- airline:7 的金额歧义（只有 issue
  [#495](https://github.com/sierra-research/tau2-bench/issues/495)）；
- retail:38 的取消原因、retail:60 的支付方式、retail:110 的空操作断言；
- Airline `payment_history` 的列表顺序
  hash（[#514](https://github.com/sierra-research/tau2-bench/issues/514)）。

另外 NL judge 本身有已报告的缺陷，例如
[#554](https://github.com/sierra-research/tau2-bench/issues/554) 中空回复会被判通过。把
NL_ASSERTION 放进训练 reward，这些缺陷会直接变成 reward 噪声。

**转人工规则的边界**（official 数据）：gold 动作含 `transfer_to_human_agents` 的只有 Airline 1 题（13）和
Retail 4 题（10、12、26、50）。Airline 有 23 题、Retail 有 7 题没有 gold 写操作。fixed 的现状见 11.3。

分类脚本和逐条结果：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-failure-taxonomy-20261010/`。其中
`auto.json` 的 `final` 字段是逐条分类，`table.py` 的 `MAN` 字典记录精读时的人工改判。

## 11. benchmark 变体：official 与 fixed（当前定义）

决策记录见 [ADR 0010](../../decisions/0010-tau2-official-and-fixed-benchmark-variants.md)。

- `official` 是 tau2-bench `b7ea907` 原样，用于和榜单比较。
- `fixed` 用于训练。

评测时两个变体都报，fixed 的数字不单独报。用 `econfig.benchmark_variant` 选择变体，所有配置的默认值都是 `official`。

### 11.1 fixed 包含什么

| 层           | 内容                                                             | 来源                                                                        |
| ------------ | ---------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 数据（上游） | Airline 44 题、Retail 105 题的 reward_basis；替换 retail 64、105 | PR #411、#571，pin 的 head commit 记在 `examples/tau2/benchmark_fixes.json` |
| 数据（我们） | 20 道 Retail 题的补丁，见 11.2                                   | 2026-10-10 的 Retail 核查                                                   |
| 环境         | `modify_pending_order_items` 按每个 item 自己的 variant 定价     | PR #322                                                                     |
| 评分         | 两条转人工规则，见 11.3                                          | 我们                                                                        |
| 训练集       | Retail train 去掉与 test 重复的 5 题，见 11.4                    | 我们                                                                        |

- **官方没有修。**`b7ea907` 之后，main 没有改过 Airline 和 Retail 的数据、工具或评估器（10.5）。
- **#580 没有作为代码合入**。它的作用是：gold 写操作执行抛错时，DB 判定不算通过。fixed 下，`check_gold_replay.py` 没有查出抛错的
  gold 写操作，合入它不会改变任何分数。
- **社区里不采用的修改**：
  - Amazon verified 对 49、18、91、21 的改动：49 的新 gold 不在订单里，18、91 的目标商品缺货，21 把 product id 当成了
    item id；
  - #561：它会替换 `pyproject.toml`；
  - #392 中把用户模型默认换成 gpt-5.5、修改转人工停止条件的部分；
  - AGI-Eval revised：只改了用户 prompt。

### 11.2 我们自己的 Retail 补丁

依据是两次 step 0（official 和 fixed v1，各 456 条 Retail episode）的失败轨迹：逐条比较 agent 实际的写操作和 gold
的写操作。补丁和每题的理由记在 `benchmark_fixes.json` 的 `patches` 里。社区调研的中间结果存档在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-retail-audit-20261010/`。

| 问题                                                                                             | 题号（划分）                                 | 补丁                                                                                                                                | 两次运行中直接因此失败的 episode |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 指令没写支付或退款方式，gold 用的是原支付方式 PayPal                                             | 16、63（train）；60、61（test）              | 用户指令加一句：被问到时选 PayPal                                                                                                   | 3                                |
| 指令里的 "the other card" 指向 gold 没用的那张卡                                                 | 98（train）                                  | 改为"尾号 8484 的 Mastercard"                                                                                                       | 4                                |
| 指令没写取消原因，gold 是 "no longer needed"                                                     | 59、66（train）；38（test）                  | 用户指令加一句：被问到原因时说 no longer needed                                                                                     | 10（38 题 8 条全挂）             |
| 64（#571 的替换题）：预算内有两款 4K 防水相机，gold 选了较便宜的黑色，agent 选了与原机同色的银色 | 64（test）                                   | 指令加一句：多款符合时选最便宜的                                                                                                    | 4（fixed v1 中 4/4）             |
| 什么都不做也得 1 分                                                                              | 25、57（train）；65（test）                  | 25：gold 加上转人工（用户要求退到 Amex，账户里没有 Amex），并加 COMMUNICATE 检查追踪号；57、65：改为 DB+ACTION，要求 agent 查过订单 | 25 有 2 条转人工被清零           |
| gold 里的转人工是从 25 抄来的：summary 写的是 Amex 退款，26 的用户根本没提                       | 26（test）                                   | 删掉这个 gold 转人工                                                                                                                | —                                |
| NL 断言只是重述写操作，而 judge 只看对话文本                                                     | 105–107、109、110、113（train）；111（test） | reward_basis 只保留 DB                                                                                                              | 2（110 只因这条断言失败）        |

**对已删除的旧 12.7 的两处更正**：

- 82 不是指令矛盾。用户在两个订单里各有一台平板，较贵的那台是用 gift card 付的，退款回不到信用卡。按指令，这时应整单退到 gift card，这正是 gold
  的做法。
- 在 fixed v1 中，10、12、50 已经按 DB+ACTION 计分，必须转人工才能得分，不属于"什么都不做也得分"。

**不修的**：

- 5–9 等 23 道题的 known_info 只给了 user id 和邮编（#337/#356）。这些题只按 DB 计分，失败轨迹里也没有卡在认证上的。
- 20、18、79 等题是 agent 选错了变体，gold 是对的，属于正常难度。
- 30–32、54、55、69 的取消原因能从指令推出来。
- 40（test）和 99（train）的指令只说"信用卡"，用户却有两张。两次运行中没有因此失败的 episode。
- airline:7 的金额有歧义（#495），Airline `payment_history` 的 hash 对列表顺序敏感（#514）。

### 11.3 转人工规则

两条规则都在官方 evaluator 之后执行（`benchmark_variant.apply_transfer_rule`）：

1. agent 转了人工而 gold 没有转，记 0 分。例外：Retail 中 gold 有写操作的题。
1. gold 有转人工而 agent 没有转，记 0 分。gold 含转人工的题是 Airline 13，Retail 10、12、25、50。

**为什么只在 Retail 设例外**（依据 fixed v1 的 step 0）：

- Retail 有 17 条原本成功的 episode 被转人工规则清零，其中 13 条在有 gold
  写操作的题上（1、32、43、46、47、51、53、93、94、101）。 抽读到的都是写操作已经完成后，用户提出 agent
  做不了的要求，比如"收不到邮件，拿不到退货单""能不能加急退款"，agent 才转人工。这符合 Retail policy 的"超出能力范围时转人工"。
- Airline 有 63 条被清零，其中 14 条在有写操作的题上（24、32、35、37、39）。抽读到的是在回避拒绝：35 的用户明确说 "I don't want
  to be transferred"，agent 应该拒绝全额退款，却转了人工；37、39 同类。所以 Airline 不设例外。
- 代价：两个领域的规则不一致；例外的依据是抽读，不是 evaluator 能检查的条件。

### 11.4 训练集去重

Retail train 的 72、96、37、41、67 与 test 的 71、97、36、42、68 是同一用户，gold 写操作完全相同（逐条核对），因此从训练中去掉。

- test 不变。去重后 train 是 Airline 30 + Retail 69 = 99 题。
- 实现方式是 `get_tau2_dataset(variant="fixed")` 在训练集里去掉这些题。benchmark 本身不变，`--task-set base`
  仍然评全部 164 题；报告中另给 `retail/train_pool`（69 题）。
- 近似重复的题没有去掉，因为 gold 不同：31↔32、93↔94、6/7/8↔5/9、63↔62、13/14↔12、34↔33、25↔26。

### 11.5 生成与校验步骤

1. **生成数据目录**：`python scripts/tau2/build_fixed_data.py <official data> <out>`，生成的 digest
   必须等于 `benchmark_variant.TASK_FILE_SHA256["fixed"]`。

   - 当前目录：`_cache/tau2-bench-b7ea9074…-fixed-v2/data`。Retail `tasks.json` 的 sha256 是
     `03a79539…`，Airline 不变（`ec04aec0…`）。
   - 上一版 `…-fixed-pr411-pr571/data` 保留，只用于复现 13.1。

1. **gold
   重放**：`TAU2_DATA_DIR=<fixed> python scripts/tau2/check_gold_replay.py fixed`，要求没有抛错的写操作。当前是
   0 条；另有 16 条读操作失败，读操作不影响 DB。

1. **补丁题打分检查**：运行本目录的 `scripts/official-runner/check_patched_scoring.py`。

   - 做法：对 20 道补丁题各构造三条合成轨迹（空轨迹、gold、去掉转人工的 gold），用真实 evaluator 打分，NL 断言打桩为通过。
   - 要求：gold 得 1 分，空轨迹得 0 分。当前全部满足。
   - 这个检查能失败：换到上一版数据上运行，会报出 25、57、65。

1. **单元测试**：

   - `tests/test_tau2_benchmark_variant.py`：补丁操作、两条转人工规则、Retail 例外、固定的 digest；
   - `tests/test_tau2_official_runtime.py::test_fixed_variant_trains_without_the_test_twins`。

   分别破坏反向转人工规则、Retail 例外和训练集去重，对应的测试都会失败。

1. **启动校验**：评测、训练和 critic 启动时都核对 `TAU2_DATA_DIR` 的 digest，与变体不符就直接报错。

______________________________________________________________________

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

______________________________________________________________________

## 12. 训练计划（待批准启动）

### 12.1 题目划分

不用 Telecom，用 fixed 变体去重后的训练集（11.4）。

| 模式           | 训练集                                      | 训练中评测                                      |
| -------------- | ------------------------------------------- | ----------------------------------------------- |
| formal（本轮） | 99 题：Airline 30、Retail 69                | 官方 test 60 题：Airline 20、Retail 40          |
| tune           | 从这 99 题切出 79 题：Airline 24、Retail 55 | dev 20 题：Airline 6、Retail 14，同样来自训练集 |

- **tune 的 dev 切法**：每个领域先排序，再用 `Random("42:<domain>")` 打乱，取 `round(n × 0.2)` 题。代码会检查 dev
  与训练集不重叠。
- **划分校验**：官方划分的题数和 sha256 必须与 `examples/tau2/contracts.py` 的 `OFFICIAL_SPLIT_SNAPSHOTS`
  一致，否则拒绝启动。去重在这一步之后执行。
- **为什么不用 Telecom**：训练前 Telecom 就是 320/320 全对，166 个 GRPO 组里 160 个全对，没有组内对比，却占了约 42% 的
  rollout（[ADR 0003](../../decisions/0003-tau2-data-and-evaluation.md)）。
- **哪些结果是 held-out**：训练只在 train 上做，test 只记录，不用来选 checkpoint。base 集包含 train，所以训练后的 base
  分数不是 held-out 结果，test 60 题才是。

### 12.2 运行顺序与配置

都用 DeepSeek-Flash 做用户模拟器和 NL judge，temperature 0，不设回复上限，key 从 `TAU2_DEEPSEEK_ENV_FILE`
读取。每次运行都要传 `econfig.benchmark_variant=fixed`，并把 `TAU2_DATA_DIR` 指向 fixed-v2 数据目录。配置的默认值是
official，目录与变体不符时会拒绝启动。

| 步骤            | 配置                                            | 步数与 episode                                                        | 存档与评测                                  | 评测内容                                                |
| --------------- | ----------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| 1. critic       | `scripts/tau2/train_critic.sh`，formal          | 2 epoch：$\lceil 99/16 \rceil \times 2 = 14$ 步，198 条训练 episode   | 第 5、10、14 步存档；第 0、5、10、14 步验证 | critic 在 60 条固定 test 轨迹上的 MSE、EV（按领域分开） |
| 2. SAO          | `run.sh policy mixed sao`，加载第 1 步的 critic | 2 epoch，26 步；每 epoch Airline 240 + Retail 552 条，共 1584 条      | 每 5 步存档并评测，训练前先评一次           | test 60 题 × 4 次，temperature 0：mean@4、pass^1/2/4    |
| 3. GRPO（对照） | `run.sh policy mixed grpo`                      | 2 epoch：$\lceil 99/8 \rceil \times 2 = 26$ 步，每题 8 条，共 1584 条 | 同上                                        | 同上                                                    |

- **SAO 预算**：每个训练题 8 条，与 GRPO 相同。`config_sao.yaml` 已从 Retail 592（74 × 8）改为 552（69 × 8）。

- **critic 的 epoch**（ADR 0005）：actor 固定（lr 0）。每个 epoch 给每道训练题现采一条 temperature 1 的
  episode，只更新 critic。每个 epoch 的最后一个 batch 只有 3 题（`drop_last: false`）。

- **critic 的验证**：开头用训练的 `gconfig`（temperature 1，每题 1 次）采一次 60 条固定轨迹，之后每次验证都复用这批轨迹。验证测的是
  critic，不是 policy，所以没有 mean@4 / pass^k。

- **资源**：8 张 A100-80GB。

  - critic：rollout 用 4 卡，actor 和 critic 共用另外 4 卡，阶段之间 offload。
  - GRPO / SAO：rollout 3 卡，独立评测 1 卡，actor 4 卡。

- **critic 耗时与用量（估计）**：上一次 critic 跑 24 步用了 1h44m，按步数比例，本次约 1–1.5 h。DeepSeek 用户侧约 3M
  prompt token、0.7M completion token，已按 1.3 倍留了余量。这个用量里是否包含 NL judge 的调用，没有核实。

- **运行目录**：每次运行都要用新的 `TAU2_RUN_ROOT`，因为 `recover.mode=auto` 会续跑同一目录。

- **启动前检查（2026-10-10 已跑）**：三份配置加上 `econfig.benchmark_variant=fixed` 跑
  `--check-config`，都退出 0。

  - critic：train 99、validation 60，14 步；
  - GRPO：train 99，26 步；
  - SAO：调度 792 条，26 步。

  不加变体参数时启动被拒绝：数据目录的 digest 与 official 不符。

### 12.3 训练侧的处理（已实现）

- **单次回复上限**：保留 4096，以节省资源（用户决定）。已知代价：上一轮 12% 到 17% 的 Airline 训练 episode
  至少有一次回复被截断（9.5）。
- **训练墙钟超时**：保留这条 episode，标记为 truncated，reward 0，不重跑（ADR 0009 修订）。如果重跑，只会保留按时跑完的
  episode，会系统性地去掉慢的长轨迹。
- **组内成员丢失**（[ADR 0011](../../decisions/0011-tau2-lost-episodes-in-training.md)）：
  - 上下文超限和墙钟超时记 0 分，照常训练。
  - 其他异常对单个成员从头重跑，最多 3 次。仍然失败的成员从组里去掉，不计入 baseline，也不训练。
  - 存活成员 ≥ $\lceil n/2 \rceil = 4$ 时，在存活成员上归一化。不足 4 个时，这道题放到本 epoch 末尾重跑一次；再被拒就记为
    skipped。每个 epoch 超过 2 题 skipped，训练中止。
  - 用户第一句就结束、policy 一次都没被调用的 episode，按同样的方式丢弃。
  - critic 每题只有一条 episode，所以在同一步内重跑一次，预算相同。
  - 新增训练指标：`tau2_batch/epoch_deferred_tasks`、`tau2_batch/epoch_skipped_tasks`。
- **还没实现**：按题统计 infra 失败率，以及失败与轮数的相关性。这是判断上述处理是否偏向短对话的依据。

同类框架的做法（定阈值 4、预算 2 的依据）：

调研了 AReaL upstream、verl、OpenRLHF、slime、ROLL、SkyRL、rLLM、Agent Lightning、TRL、NeMo-RL、ART 和
DAPO。下表的结论来自代码，不是文档：

| 做法                                              | 谁在用                                                                              | 偏差                                                            |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 先分类：模型造成的失败记 0 分，infra 失败另行处理 | AReaL upstream（`_classify_agent_failure`）、NeMo-RL、SkyRL harbor                  | 依赖分类是否准确                                                |
| infra 失败对单个成员做有上限的重跑                | NeMo-RL（行级 3 次，再整组）、SkyRL（每条 2 次）、rLLM（3 次）                      | 对与轨迹无关的瞬时故障无偏；对随轮数增长的故障，仍偏向短对话    |
| 在剩余成员上算 baseline                           | AReaL upstream（`min_usable_group_size` 默认 1）、verl v1、ART、TRL（NaN 不计入）   | 失败与轨迹有关时会有幸存者偏差                                  |
| 丢掉整组，用别的 prompt 补                        | AReaL upstream 系统错误、OpenRLHF、ROLL、DAPO 动态采样                              | 难任务被少采；破坏有限 epoch 的"每题一次"                       |
| infra 失败记 0 分照常训练                         | AReaL upstream tau2 示例、Agent Lightning、slime 示例                               | 把 infra 失败当成策略失败，惩罚失败前的动作，并推动策略缩短对话 |
| 预算用尽就中止训练                                | NeMo-RL 默认 `max_skipped_prompts=0`、ART `max_exceptions=0`、rLLM `raise_on_error` | 不静默吞掉问题，代价是训练会停                                  |

**"至少剩 2 个就保留"的问题**：

- 没有一个框架用这个阈值作为默认值，最接近的是 rLLM 的 `min_trajs_per_group=2`，默认关闭。
- 2 个成员、二值 reward 时，advantage 只有两种情况：全 0，或者 ±1，噪声很大。
- 一个组丢了 6 个成员，本身就说明任务或环境有系统性问题；这个阈值会把这个信号掩盖掉。

### 12.4 这轮能验证什么

- **critic 能否在两个领域上学到价值**：60 条 test 轨迹上 MSE 下降、EV 上升，按领域分开看。
- **SAO / GRPO 是否提高 test 成功率**：每 5 步评测一次 mean@4 和 pass^1/2/4，fixed 和 official
  两个变体都报。step 0 的同口径基线由训练启动时自动产出。
- **提升是不是靠转人工**：在 fixed 上，不该转却转、该转却没转都会记 0 分，所以 fixed 上的提升不会来自转人工。official 上的提升如果明显大于
  fixed， 多出来的部分可能来自转人工。
- **SAO 与 GRPO 的对比**：两者的 rollout 预算、题集和评测都相同。
- **稳定性**：看 pass^4 与 pass^1 的差，判断训练是否让模型在同一题上表现更一致。

### 12.5 这轮验证不了什么

- **与 Recuris 或榜单比绝对分数**：用户模拟器不同（7.2 第 1 项），也没有 Telecom。fixed 的分数本来就不能与榜单比。
- **小于噪声的提升**：test 只有 60 题。同一配置两次重测相差约 3pp（13.1）；采样噪声的 95% 半宽，合计约 ±4.1pp，Airline 约
  ±6.4pp（3.1）。 几个点的变化不能当作训练有效。
- **对 Doubao 或 gpt-5.2 用户的泛化**：训练和评测都用 DeepSeek 用户，换用户后的效果需要另跑评测。
- **critic 对 SAO 的具体贡献**：MSE / EV 只说明 critic 拟合了 temperature 1 下的回报。它对 SAO 有没有帮助，只能从 SAO
  与 GRPO 的对比里间接看出，而这个对比还混着两种算法的其他差异。
- **NL judge 带来的 reward 噪声有多大**：Airline 有 26 道题按 NL 断言计分，judge 就是用户模拟器所用的模型。

### 12.6 未决事项

1. 启动 critic、SAO、GRPO，需要批准。
1. 用 gpt-5.2（reasoning low）做用户跑一次 base 集，与榜单定标。需要 OpenAI 渠道和费用，需要批准。
1. NL judge 与用户模拟器分开配置：暂不做（用户决定）。
1. 按题统计 infra 失败率（12.3）。
1. Retail 训练信号不足的处理（12.7 的两条建议）。

### 12.7 Retail 训练信号不足：社区的做法（2026-10-10 调研）

**问题**：fixed v2 下，Retail 训练池 69 题里有 47 题在 temperature 0 下 4 次全对（13.2）。

- 对 GRPO 来说，全对的组 advantage 为 0，没有梯度。
- 对 PPO / SAO 来说，critic 能给全对轨迹的中间步骤非零的 advantage。但这些题上从来没有失败样本，critic 学不到"哪里会失败"，信号很弱。所以
  PPO 也只是比 GRPO 好一些，题太简单的问题仍然存在。

下面的结论来自调研 agent，标注说明来源：\[代码/论文\] 表示读过原文，\[摘要\] 表示只读了摘要或 README。两个 HF 数据集的存在、license 和 DB
文件，我用 HF API 独立核对过；其余没有独立核对。

**现成的、能直接当 RL 任务用的数据**：

| 数据                                                                                                         | 规模                                           | 内容                                                               | License     | 用于我们的代价                                                                                     |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------------------------- |
| [inclusionAI/AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)（AReaL-SEA 论文） | RL 任务：Airline 1148、Retail 563、Telecom 271 | 每题带 gold actions 和 `db_path`。Retail 用 3–4 个新合成的 DB      | Apache-2.0  | 要能按任务加载各自的 DB；现在 `agent.py` 只用默认 DB。数据质量见 12.8                              |
| [wwwenlin/skillscaffold-tau-rl](https://huggingface.co/datasets/wwwenlin/skillscaffold-tau-rl)               | 每个领域 300 题                                | 上面那份数据的子集：用 Qwen3-8B 每题跑 8 次，只保留通过 1–6 次的题 | Apache-2.0  | 格式与上面相同。它是按 8B 筛的，要用我们的 4B 重新筛                                               |
| tau-bench v1 Retail train                                                                                    | 500 题                                         | 与 tau2 Retail 共用同一个 DB                                       | MIT         | **不推荐**：调研 agent 实测，tau2 Retail test 的 41 个写操作 order_id 中有 26 个出现在 v1 train 里 |
| APIGen-MT-5k、CoVe-12k                                                                                       | 5k / 12k 条轨迹                                | 只有轨迹，没有 gold 或 DB                                          | NC / Apache | 只能用于 SFT                                                                                       |

**不换数据也能做的**：

- **训练时丢掉全对或全错的组**（DAPO 的 dynamic sampling）。AReaL-SEA 在 Airline 30B 上做过消融：pass^1 65.0 →
  70.5，pass^4 40 → 52 \[论文\]。
- **离线按 pass-rate 筛题**：用当前模型每题跑 8 次，只留通过率在 1/8 到 6/8 之间的题，即 Skywork-OR1、POLARIS 的做法
  \[摘要\]。
- **训练时用户模拟器用 temperature 1**：AReaL-SEA 训练时 agent 和用户都用 1.0 \[论文\]。我们的训练配置里，用户是
  temperature 0。
- tau2 自带的 `PersonaConfig` 只有 verbosity 和 interrupt 两项，提不了多少难度 \[代码\]。

**同类工作**：

| 工作        | 模型    | 训练数据                           | Retail / Airline                   |
| ----------- | ------- | ---------------------------------- | ---------------------------------- |
| AReaL-SEA   | 30B-A3B | 自己合成的任务和新 DB              | 54.2 → 75.0 / 56.0 → 70.5 \[论文\] |
| MUA-RL      | 32B     | tau1 Retail 115 题 + Airline 50 题 | 67.3 / 45.4 \[摘要\]               |
| AgentScaler | 4B      | 合成环境，只做 SFT                 | 56.1 → 62.3 / 52.0 → 56.0 \[论文\] |

这些工作的训练数据都不只是官方的 train split。

**建议**（未实施，需要决定）：

1. 先做零成本的两项：GRPO 丢掉全对或全错的组，训练时用户模拟器用 temperature 1（评测仍用 0）。
   - 代价：每步有效样本变少，可能要加大 batch。
   - 与已经批准的训练计划不同，需要批准。
1. 主方案：接入 skillscaffold 或 AReaL-tau2-data 的 Retail / Airline 任务，用我们的 4B 重新按 pass-rate
   筛一遍，评测仍用官方 test 60 题。
   - 工程量估计 1–2 天：按任务加载 DB，并按 12.8 先过滤掉有问题的题。
   - API 费用：每道候选题要用 DeepSeek 用户跑 8 次。

### 12.8 AReaL-tau2-data 的数据质量（2026-10-10 核查）

**结论**：这份数据可以用，但不能整份直接用。官方没有披露数据问题或标注误差；我们的机械检查找到约 2% 的题 gold 确定是错的，算上有歧义的题至少
6.8%，Retail 偏高。接入前要先按下面的检查过滤，并按 pass-rate 重新筛题。

**来源与披露**（调研 agent 读了 HF 数据卡、HF 讨论区、[AReaL 仓库](https://github.com/inclusionAI/AReaL)和
AReaL-SEA 论文）：

- 合成流水线没有开源。有人在 AReaL issue #915 里问过，没有公开代码。
- 论文没有给出噪声率、和官方 test 的去重方法，也没有局限性一节。HF 讨论区没有人报告数据问题。
- 所以下面的问题都是我们自己查出来的，不是官方披露的。

**结构**（我核对过）：

- RL 任务共 1982 题：Airline 1148、Retail 563、Telecom 271。`evaluation_criteria` 存成 JSON
  字符串，每题有自己的 `db_path`。
- Airline 有 543 题用官方 DB，其余用 `new_db_1/2/3`；Retail 全部用 `new_db_1–4`。
- Airline 和 Retail 的 1711 题全部用默认 reward_basis（DB + COMMUNICATE），而 `communicate_info`
  都是空的，`nl_assertions` 都是 null。也就是说，实际只检查最终 DB 的 hash。

**机械检查**（脚本与结果见下面的归档；除标注"我核对过"的，数字来自调研 agent）：

| 检查                | 结果                                                                             | 含义                                            |
| ------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------- |
| gold 写操作重放     | 0 条抛错。故意破坏一条 gold 后，检查会报错                                       | gold 都能执行                                   |
| 什么都不做也得 1 分 | Airline 55 题、Retail 15 题                                                      | 和官方 Retail 25、57、65 同类，训练时奖励不作为 |
| 重复题              | Airline 有 228 道多余的重复题，其中 Mia Li 的 HKEG34 一个预订就有 152 题         | 训练分布严重偏向少数几个用户                    |
| 与官方 test 重叠    | 3 道 Airline 题与某道 test 题写同一个预订；14 道与 test 同一个用户。没有文本重叠 | 这 17 道要去掉                                  |
| 支付方式没写        | Retail 61 题：用户有多种支付方式，指令没说用哪一种，gold 却固定了一种            | 和我们 11.2 修过的问题同类                      |
| 新 DB 内部不一致    | `new_db_1/2` 有 668 / 777 个预订的航线与航班表不符；55 道 gold 用到这些坏记录    | agent 按航班表推理会与 gold 不一致              |

**case study**：

| 题                                                | 问题                                                                                                                                                                                                       | 判断                         |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| airline_56（同类共 30 题，含 63、664、810、1045） | 指令三次要求用 `credit_card_4421486`（Visa 尾号 7447）付款，gold 用的是 `credit_card_1955700`（尾号 1907）。tau2 信用卡没有额度限制，不存在换卡的理由。**我核对过**                                        | gold 确定错误                |
| airline_718（同类共 5 题，含 781、844）           | 用户说"昨天订的"，要求按 24 小时免费取消。gold 取消的 3 个预订创建于 2024-05-14 14:00–14:30，policy 的当前时间是 2024-05-15 15:00，已超过 24 小时；都是 economy、没买保险，policy 不允许取消。**我核对过** | gold 违反 policy，确定错误   |
| airline_54                                        | 要求最早的航班。最早的 10:15 航班没有座位；有座位的最早时刻是 10:45，有两个航班并列，gold 只认 `CSPD2YO`                                                                                                   | 有歧义                       |
| retail_18（同类含 98、141）                       | 用户有信用卡和 PayPal，指令没提支付方式，gold 用信用卡补差价。**我核对过**                                                                                                                                 | 有歧义，agent 问用户也问不出 |

**误差估计**：

- 确定错误：35 / 1711 = 2.0%（Airline 3.0%）。
- 确定错误加上有歧义：6.8%（Retail 约 11%，Airline 4.6%）。这是下限，只计入了机械检查能发现的类型。
- 推测的真实比例：Retail 10–20%，Airline 5–10%。这是调研 agent 的推测，没有抽样验证。
- 对比：官方 Retail 我们修了 20 题（114 题中的 18%），所以官方数据并不比它干净多少；差别是官方的问题我们已经逐题查过。

**与 fixed 变体的关系**：#322 会改变 13 道 Retail 题的 gold hash。但 gold 是在线重放出来的，不是存好的 hash，所以 fixed
变体可以直接用在这份数据上。

归档：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/areal-tau2-data-audit-20261010/`（`scripts/`
是检查脚本，`out/` 是每题的检查结果，原始数据从 HF 下载）。

### 12.9 AReaL-tau2-data 的筛选与修复：v1（2026-10-10 → 10-11）

**决定**（用户，2026-10-10）：从 Airline / Retail 的 1711 题里筛出一个 reward 正确的子集用于训练（按 12.1 不用
Telecom）。

- 能修的自己修，修不好或超出修复范围的直接丢掉。
- 留下的题，不论原样保留还是修过，都要用评测证明可用。
- 评测时 agent 用本机 4B，用户模拟器仍用 DeepSeek。

**第一步：机械过滤**（脚本，零成本），结果在 `out/bucket_mechanical.json`：

| 类别               | Airline | Retail | 处理                                                                                                               |
| ------------------ | ------: | -----: | ------------------------------------------------------------------------------------------------------------------ |
| 直接丢弃           |     353 |     17 | 重复题（每组留一道）、什么都不做也得分、gold 只有转人工、用到 DB 坏记录、与官方 test 同用户或同写入目标            |
| 修复候选           |      35 |     63 | 支付方式与指令矛盾（18）、24 h 取消违规（5）、"最早航班"有歧义（12）；Retail 支付方式没写（61）、取消原因没写（2） |
| 机械检查未发现问题 |     760 |    483 | 机械检查只能查出已知类型的问题，仍要逐题人工审                                                                     |

后两类共 1341 题进入第二步。一题命中多条规则时，丢弃优先。

**第二步：逐题 case study**（本机 Codex 订阅，7 个 `codex exec` 并行，按 DB 分片，每片 146–251 题）：

- 每题的结论只能是 keep、fix、drop 之一，并附证据（记录 id、字段值、policy 条款）。

- fix 只允许两种改法：给用户指令补一句用户本来就知道或能决定的信息，或者在指令明确时替换 gold。超过约两处小改动就 drop； 没有把握时也 drop。

- 每个 keep 和 fix 都要用工具 `score.py` 在该题自己的 DB 上重放，满足：

  - gold 得 1 分、空轨迹得 0 分、gold 写操作不报错；
  - 去掉任何一个 gold 写操作都得 0 分。

- 不允许把题修成 gold 不写 DB 的拒绝题，因为那样什么都不做也能得分。

- **试跑**（2026-10-10，6 题，约 8 分钟）：1 keep（airline_2）、1 fix（airline_56）、4
  drop（airline_718、54，retail_18、549）。

  - 已知的三类错误都查出来了：换卡、24 h 取消违规、"最早航班"有歧义。
  - 另外查出两处 DB 引用断裂，重放时不会报错：retail_18 订单里的商品 `3949674330` 不在该产品的 variant
    列表里（**我核对过**）；airline_54 的预订引用了航班表里没有的 `VNOB4BW`。
  - airline_56 的修法：gold 改用指令要求的卡；指令里写死的 `HAT084` 是 04:00
    起飞，不在用户要求的时间段内，改成"最便宜的直飞商务舱"，唯一解是 HAT229。
  - 试跑还发现 `score.py` 的一个 bug：retail 的 #322 定价补丁丢了 WRITE 标记，导致"去掉单个写操作"的检查漏掉了
    `modify_pending_order_items`。已修复，并在审题规则里加了一条：重放成功不等于引用的记录都存在，要自己核对。

- **全量结果**（7 个分片，2026-10-10 23:46 → 10-11 00:07，模型 gpt-6-astra）：1341 题全部有结论，逐题结论在
  `out/s*.jsonl`。

  | 领域    | 进入第二步 | keep | fix | drop |
  | ------- | ---------: | ---: | --: | ---: |
  | Airline |        795 |  305 | 227 |  263 |
  | Retail  |        546 |  119 | 250 |  177 |

  - drop 的主要原因：Airline 是 DB 引用断裂（178）和 gold 与指令矛盾（43）；Retail 是 reward 太弱（56，含不作为得分）、gold
    与指令矛盾（48）和 DB 不一致（44）。
  - 机械检查"没发现问题"的 1243 题里，只有 424 题原样 keep，其余 819 题被 fix 或 drop。机械检查漏掉的问题远多于查出的。
  - fix 的改法：320 题只在指令里补一句（主要是指定支付方式、变体、航班），156 题替换了 gold。
  - **我的独立重放**：用 `score.py` 重放全部 424 道 keep 和 477 道 fix，都满足 gold 1 分、空轨迹 0 分、写操作不报错。有 4
    道 keep 的 gold 包含一次不改变 DB 的行李更新（airline_90、259、414、514），不影响 reward。

- **抽查发现的风险**：我抽读了 8 道 fix。只补指令的改法都合理；但有几道替换 gold 的改法，依据的是审题 agent 自己推断的 policy 规则，官方
  gold 从不这么做。例如 airline_533、873：降舱后，对超出新额度的托运行李补收费用；airline_225：先升舱，再借此改航班。这类 gold
  放进训练，会奖励与官方评测不一致的行为。

- **第二轮对抗复核**（2026-10-11 00:12 → 00:29，5 个分片，模型 gpt-6.1-sol，规则见 `REVIEW.md`）：

  - 复核范围是全部 477 道 fix，外加随机抽样的 100 道 keep。
  - 复核者先自己判断，再看第一轮的证据。依据 policy 的结论要逐字引用原文，复核结束时还会校验引用确实是原文的子串。没把握就拒绝。

  | 对象                               | 题数 | 通过 | 拒绝 | 主要拒绝原因                                                                                 |
  | ---------------------------------- | ---: | ---: | ---: | -------------------------------------------------------------------------------------------- |
  | Airline fix，只改指令              |  143 |  132 |   11 | 补丁文本不合格、gold 不唯一                                                                  |
  | Airline fix，替换 gold             |   84 |   53 |   31 | 依据推断出的 policy（Airline 全部 fix 中因此被拒 32 道，含我抽查到的 airline_533、873、225） |
  | Retail fix，只改指令               |  222 |  195 |   27 | 补丁文本不合格（Retail 全部 fix 中 23 道），比如替 agent 规定怎么处理                        |
  | Retail fix，替换 gold              |   28 |   20 |    8 | DB 不一致、gold 不唯一                                                                       |
  | keep 抽样（Airline 79、Retail 21） |  100 |   98 |    2 | 都是"商品不存在"那一类，见下                                                                 |

  - 结论：替换 gold 的修法拒绝率是 35%，只改指令的是 10%。keep 抽样的错误率约 2%，按二项分布的 95% 区间是 0.2%–7%。

- **"商品不存在"类**：retail_97、127 的指令要求替换用户订单里根本没有的拼图。按这个特征机械扫描 keep 和 fix，共找到 24 道。

  - 第一轮把它们当作"用户记错了"：agent 查无此物，只做地址修改，gold 仍然唯一。复核拒绝了其中 15 道，理由是指令说的目标做不到。
  - **处理**：24 道全部去掉。这类题的目标本来就有争议，去掉最稳妥；以后想要，可以从 `out/` 里的结论恢复。

- **进入评测的候选集**（当时写在 `out/candidates_after_review.json`，现已由 12.11 的 decisions 文件取代并删除）：

  | 领域    | 原样 keep，未复核 | 原样 keep，已复核 | fix，已复核 | 合计 |
  | ------- | ----------------: | ----------------: | ----------: | ---: |
  | Airline |               226 |                79 |         185 |  490 |
  | Retail  |                93 |                19 |         211 |  323 |

  - 进入的条件：两轮都没被拒绝，且不属于"商品不存在"类。
  - 共 813 题，是官方 train 99 题的 8 倍多。

**第三步：评测证明**（2026-10-11，用户批准）

1. **按任务加载 DB**（`1b17bc3f`）：

   - `examples/tau2/task_pack.py` 定义 task pack：每个领域一份 `tasks.json` 和 `task_db.json`，加上
     DB 文件和带每个文件 sha256 的 manifest。
   - `task_pack.install` 包装 registry 里的环境构造函数，让它接受 `task_db`。episode 和官方 evaluator 的
     gold 重放都通过它建环境，每次都拿到一份 新的 DB。
   - `scripts/tau2/evaluate.py --task-pack <pack>` 评测 pack 里的全部题。结果里的 `split` 记为 keep /
     fix，方便分开看分数。

1. **生成 pack**：`scripts/tau2/build_areal_task_pack.py` 按候选集生成 pack，并应用每道 fix 的指令修改和 gold
   替换。

   - 当前 pack：`_cache/tau2-areal-pack-20261011-v1`，Airline 490 题、Retail 323 题。

1. **打分检查**：`scripts/tau2/check_task_pack.py` 对每道题，在它自己的 DB 上构造 gold 和空轨迹，用官方
   `evaluate_simulation` 加上 fixed 的转人工规则打分。

   - 结果：813 题全部是 gold 1 分、空轨迹 0 分。
   - 这个检查能失败：改用默认 DB 打分（`--default-db`），402 题不通过，多数是 gold 重放报错。仍然通过的 411 题里，340 道
     Airline 题用的 DB 与官方 DB 逐字节相同，其余的新 DB 题碰到的记录与官方 DB 相同（抽查过）。
   - 单元测试 `tests/test_tau2_task_pack.py`：让 `task_db` 失效，测试就会失败。

1. **评测**（2026-10-11 03:11 → 04:16，pueue 任务 437）：口径与 13.2 相同。本机 Qwen3.5-4B 用 temperature
   0，DeepSeek-Flash 做用户和 NL judge，每题 4 次， `benchmark_variant=fixed`，`sglang:d8p1t1`，并发
   256，代码固定在 `1b17bc3f`。

   - 3252 条 episode 中 3251 条有分。排除的 1 条是 `retail_524` 的上下文超过 32K。DeepSeek 用户侧实际用了 33M
     prompt token、5.0M completion token。
   - 产物：`tau2-eval-20261011-areal-pack/step0-areal-pack-v1-t0-deepseek/`。

   | 范围         | 题数 | mean@4 = pass^1 | pass^2 | pass^4 | 4 次全对 | 有对有错 | 4 次全错 |
   | ------------ | ---: | --------------: | -----: | -----: | -------: | -------: | -------: |
   | 合计         |  813 |           0.912 |  0.859 |  0.783 |      637 |      161 |       15 |
   | Airline      |  490 |           0.896 |  0.837 |  0.755 |      370 |      106 |       14 |
   | Airline keep |  305 |           0.932 |  0.891 |  0.830 |      253 |       46 |        6 |
   | Airline fix  |  185 |           0.838 |  0.749 |  0.630 |      117 |       60 |        8 |
   | Retail       |  323 |           0.936 |  0.891 |  0.827 |      267 |       55 |        1 |
   | Retail keep  |  112 |           0.902 |  0.839 |  0.759 |       85 |       26 |        1 |
   | Retail fix   |  211 |           0.954 |  0.918 |  0.863 |      182 |       29 |        0 |

   - **可用性**：798 / 813 题至少通过一次，说明 gold 可以达到。4 次全错的 15 题（Airline 14 道，Retail 只有
     retail_353）不能直接认定为题错，要看轨迹。
   - **这份数据对当前模型偏容易**：同一模型、同一口径下，官方 fixed v2 base 集是 0.708（Airline 0.388，Retail 0.849，见
     13.2），这里是 0.912。 temperature 0 下，78% 的题 4 次全对。
   - **Retail 的 fix 比 keep 还高**（0.954 对 0.902）：fix
     往往是在指令里写明支付方式、变体、航班，让用户能直接说出唯一答案。这正是修复要做的，但也让题变容易了。 Airline 的 fix 反而低于 keep（0.838 对
     0.932），因为替换后的 gold 多是多步的改签、降舱。

1. **失败轨迹复核**（pueue 任务 439，Codex gpt-6.1-sol，规则见 `EVAL_REVIEW.md`）。任务 438 因为 codex 等待
   stdin 挂了 4 小时，没有产出，入口加了 `< /dev/null` 后重跑。

   - 4 次全错的 15 题：9 道是 agent 的错，题本身没问题；6 道是题的错。
   - 从"有对有错"的 161 题里随机抽的 40 题：15 道是 agent 的错，25 道是题或用户模拟器的错。

### 12.10 v1 pack 审查（2026-10-11，结论：不能直接训练）

用户的标准：题可以简单，但不能是错题，也不能是做不完的题。按这个标准，v1 的 813 题**还不能用**。证据有三部分：

**1. 独立抽审**（Claude，与 Codex 不是同一个模型；分层抽 32 题，结果在 `scratch/claude_audit.jsonl`）：

| 类别                  | 抽查数 | 正确 | 错误                                        | 存疑                                                             |
| --------------------- | -----: | ---: | ------------------------------------------- | ---------------------------------------------------------------- |
| gold 被替换的 fix     |     12 |   11 | 0                                           | 1（retail_104：gold 只有在先完成取消、礼品卡余额够了之后才成立） |
| 只改指令的 fix        |     10 |    9 | 1（retail_375：指令里的手表在 DB 里不存在） | 0                                                                |
| 原样 keep、只审过一轮 |     10 |   10 | 0                                           | 0                                                                |

**2. 失败轨迹复核**（12.9 第 5 步）找出了几类系统性的问题。下面按根因归类，并用脚本统计了每类在整个 pack 里有多少题：

| 根因                                                                                                          | pack 内题数 | 处理                                                 |
| ------------------------------------------------------------------------------------------------------------- | ----------: | ---------------------------------------------------- |
| HKEG34 / Mia Li 题族：指令写的是 Mia 本人的行程，DB 里的乘客却是 Amelia，用户模拟器会要求改乘客               |          54 | 整族丢弃                                             |
| 模板套用：指令自己承认"这里的手表其实是相机"之类                                                              |          19 | 丢弃                                                 |
| 复核判为题错的其他题，加上 airline_654（gold 改了 basic economy 的航班，违反 policy）、retail_104、retail_375 |  合计约 100 | 丢弃                                                 |
| 国家字段：gold 写的是 "USA"，指令没写，用户会说 "United States"，DB hash 因此不同                             |          99 | 补一句"国家写 USA"，或丢弃                           |
| 乘客列表顺序：gold 的顺序固定，指令没写                                                                       |          43 | 补一句列出顺序，或丢弃                               |
| 用户模拟器在 agent 执行最后一步之前就发 `###STOP###`                                                          |           — | 是 DeepSeek 用户的行为噪声，不是数据错误，官方题也有 |

"整族丢弃""模板套用"和"其他题错"三行合起来去重后是 100 题。

**3. 机械 policy 检查**（全 pack）：取消资格（24 h、商务舱、保险及理由、航司取消）、basic economy 不能改航班、新航班存在且可订。只查出
airline_654 一道违规。其余被标出的都是误报：gold 里先订再改的新预订；6 道靠保险取消的题都给了健康理由。

**结论**：

- **评测通过不等于题对**。模板套用的 19 题里有 13 题 4 次全对：用户模拟器把错误的描述"翻译"回来了。所以 4/4 只能说明 gold 可达。
- **问题主要出在两处：**
  - 只审过一轮的部分：第一轮的 keep，以及"有对有错"里没被抽到的 121 题；
  - 只补一句指令、但没处理场景其余部分矛盾的 fix。 替换 gold 的修法经过第二轮的逐字引用规则筛选后，抽查是可靠的。
- **"100 题"的构成**：HKEG34 题族 54 + 模板套用 19 + 失败轨迹复核判为 drop / 重修、且不在前两类里的 26 + retail_104 =
  100（airline_654、retail_375 已在 26 题里）。后来做了两处修正：
  - 26 题里有 11 题是用户模拟器提前 `###STOP###`，不算题错，放回保留；
  - 国家字段（99）和乘客顺序两类改为补一句事实，不丢弃。

**v2 的做法与结果**（2026-10-11，用户确认：国家、乘客顺序补一句事实；用 Codex gpt-6.1-sol 复核）：

1. 机械修正（`tools/decisions.py`）：丢掉 HKEG34 题族、模板套用题和复核判错的题；凡是 gold 写地址的，指令补"Whenever you
   give an address, the country is USA (say exactly "USA")."；凡是 gold 改或订多名乘客的，补"When you
   list the passengers, list them in this order: …"。按 gold 的写操作判定，所以补顺序的是 106
   题，比上表按失败轨迹统计的 43 题多。得到 724 题的 staging 版本，全部 gold 1 分、空轨迹 0 分。
1. 第三轮复核（`REVIEW3.md`，7 个分片，每题附 v1 的 4 次轨迹作证据）：724 题全部有结论，keep 525、fix 123、drop 76。
   - drop 里 64 道是"场景与 DB 不符"，在 HKEG34 之外又找到同类：例如 retail_489 的指令一直把订单里的鼠标叫作手表，airline_137
     的指令说 Ivan 改"我的行程"，预订上却没有他。
   - 76 道 drop 里有 61 道在 v1 评测中 4 次全对，再次说明 4/4 只证明 gold 可达。我抽查了 6 道 4/4 的 drop、5 道
     fix，判断都成立。
   - 123 道 fix 都只改指令（替换 74、追加 43、两者都有 6），没有替换 gold。
1. **v2 pack**：`_cache/tau2-areal-pack-20261011-v2`，Airline 380（keep 191、fix 189）、Retail
   268（keep 48、fix 220），共 648 题；`check_task_pack.py` 结果是全部 gold 1 分、空轨迹 0 分。v2 还没有重新评测。

### 12.11 筛选流程（现行做法）

这一节记录现在怎么筛题、为什么这么做、用哪些脚本。12.8–12.10 是过程记录；以后筛新数据（包括 Telecom）照这一节做。

**原则**：

- **标准**：题可以简单，但不能是错题，也不能是做不完的题（infra 原因或题本身的原因都算）。没有把握就丢，错留一道题会把错误的 reward 带进 RL
  训练，错丢一道只少一道题。
- **评测通过不等于题对**。gold 可达只说明 reward 能拿到，不说明题面、DB 和 gold 互相一致。v1 里模板套用的 19 题有 13 题 4/4，v2
  第三轮丢掉的 76 题有 61 题 4/4。评测失败轨迹的用处是发现问题类别。
- **逐题审查看不出跨题的模板问题**。HKEG34 题族两轮 Codex 都没发现，是从失败轨迹里看到个例后，用脚本扫全量才找出来的。发现一个个例，就写规则扫整个
  pack，再把这一类写进 rubric。
- **修复只改指令，尽量不改 gold**。补一句用户本来就知道的事实（国家写法、乘客顺序、支付方式），或替换一个写错的名称。v1 第二轮里，替换 gold 的修法被拒
  35%，只改指令的被拒 10%，多数是 agent 自己推断出 policy 里没有的规则。
- **每次修改都要重放验证**：gold 1 分、空轨迹 0 分、去掉任一 gold 写操作得 0
  分。检查本身也要能失败：`check_task_pack.py --default-db` 和单元测试里有意破坏的检查都能变红。
- **用户模拟器提前 `###STOP###` 不算题错**，除非是场景本身诱发的。
- **先拿官方题校准"什么算错"**。官方题本来就有的写法（通用模板的 ticket、persona 年龄、工具名的叫法），不能当作题错。Telecom 第一轮因为
  rubric 没写这一条，97 道 drop 里绝大多数是这类误判（13.4）。写 rubric 前先在官方题里数一数：某个特征官方题里有多少。
- **rubric 不能跨领域照搬**。Airline 的"gold 没有转人工、用户却要求转人工就丢"搬到 Telecom 就错了：Telecom 的 policy
  规定问题解决不了时要转。

**步骤与脚本**：

| 步骤              | 做什么                                                                               | 脚本或规则                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| 1. 机械过滤       | 重复题、什么都不做也得分、gold 只有转人工、DB 坏记录、与官方 test 同用户或同写入目标 | 工作目录 `tools/common.py`、`tools/mech_check.py`（policy 规则：取消资格、basic economy 改航班等）  |
| 2. Codex 逐题审   | keep / fix / drop，附证据；fix 用 `score.py` 自查                                    | `PROMPT.md`；`tools/dossier.py`（打印题目和相关 DB 记录）、`tools/score.py`（在该题 DB 上重放打分） |
| 3. Codex 对抗复核 | 先独立判断再看上一轮证据；policy 结论必须逐字引用                                    | `REVIEW.md`（第二轮）、`REVIEW3.md`（第三轮，列出已知故障类别并附评测轨迹）                         |
| 4. 评测           | 口径同 13.2：本机 4B、temperature 0、DeepSeek 用户、每题 4 次                        | `scripts/tau2/evaluate.py --task-pack`                                                              |
| 5. 失败轨迹复核   | 找到第一个偏离 gold 的写操作，判断是 agent 错还是题错                                | `EVAL_REVIEW.md`                                                                                    |
| 6. 汇总与构建     | 把各轮结论合成一份 decisions 文件，构建 pack，检查 gold 1 / 空轨迹 0                 | `tools/decisions.py`，`scripts/tau2/build_areal_task_pack.py`，`scripts/tau2/check_task_pack.py`    |

**派发 Codex**：

- `tools/run_codex.sh <rubric> <shard> <输出前缀>`，由 pueue 运行（`codex-review` 组）。分片结论缺 id
  时自动续跑，最多 6 轮。
- 模型 `gpt-6.1-sol`，reasoning effort high。必须显式写 `-m`：本机 Codex 的默认模型是
  `gpt-6-astra`，第一轮就是因为没写 `-m` 才用了它。
- stdin 必须接 `/dev/null`，否则 `codex exec` 会一直等输入（v1 的 pueue 任务 438 因此挂了 4 小时）。
- 在 prompt 里明确禁止开 subagent。全局 `~/.codex/AGENTS.md` 要求 Codex 主动开 subagent，第三轮 7 个分片因此额外开了
  20 个；`--disable multi_agent` 和 `agents.max_threads` 都去不掉这个工具（codex-cli 0.162.1）。
- `tools/supervise.sh <输出前缀> <分片>…` 放在后台跑，监督整轮：等 pueue
  任务结束，检查每个任务都成功、每片结论不缺不重、格式合法，然后汇总、构建候选 pack 并打分。只有全部通过才返回 0，主 agent 只需要等它结束。

**版本**：

- `_cache/tau2-areal-pack-<日期>-v<N>` 每个版本只读保留，里面带
  `decisions.json`，记着每道题为什么保留、改了什么、为什么丢，供追溯和写 ADR 用。
- `_cache/tau2-areal-pack-latest` 是指向当前版本的软链（现在指向 v2），脚本默认读它。出新版本只改这个软链。
- Telecom 单独建 pack：`_cache/tau2-areal-telecom-pack-<日期>-v<N>`，当前版本的软链是
  `tau2-areal-telecom-pack-latest`（现在指向 v1）。

**工作目录**
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/areal-tau2-data-screen-20261010/`：`data/`（原始数据），四份
rubric 加 Telecom 的 `TELECOM_REVIEW.md`、`TELECOM_REVIEW2.md`，`tools/`（上表的 7 个脚本，加 Telecom
的 `telecom_mech.py`、`score_telecom.py`、`decisions_telecom.py`），`shards/`，`out/`（各轮逐题结论和
decisions 文件），`scratch/`（Codex 的验证文件和 Claude 抽审），`logs/`。旧的启动脚本和中间产物已经删掉。

## 13. step 0 评测

### 13.1 上一轮：official 与 fixed v1（2026-10-10）

fixed v1 是加入 11.2–11.4 之前的版本，只有 #411、#571、#322 和第一条转人工规则（没有 Retail 例外）。

配置与第 8 节相同：未训练的 Qwen3.5-4B，base 集 164 题，每题 4 次，policy 和 DeepSeek-Flash 用户都是 temperature
0。代码是 `1d14d8dc`，从一个单独固定在这个提交的 worktree 运行。两个变体先后在同一台 8 卡机上跑完，各用约 26 分钟和 31 分钟。

| 变体     | 领域 / 划分   | 题数 | 排除 | mean@4 = pass^1 | pass^2 | pass^4 |
| -------- | ------------- | ---- | ---- | --------------- | ------ | ------ |
| official | 合计          | 164  | 0    | 0.808           | 0.705  | 0.561  |
| official | Airline       | 50   | 0    | 0.780           | 0.710  | 0.620  |
| official | Airline train | 30   | 0    | 0.775           | 0.717  | 0.667  |
| official | Airline test  | 20   | 0    | 0.787           | 0.700  | 0.550  |
| official | Retail        | 114  | 0    | 0.820           | 0.703  | 0.535  |
| official | Retail train  | 74   | 0    | 0.848           | 0.739  | 0.568  |
| official | Retail test   | 40   | 0    | 0.769           | 0.637  | 0.475  |
| fixed v1 | 合计          | 164  | 1    | 0.690           | 0.571  | 0.460  |
| fixed v1 | Airline       | 50   | 1    | 0.428           | 0.297  | 0.224  |
| fixed v1 | Airline train | 30   | 1    | 0.364           | 0.228  | 0.138  |
| fixed v1 | Airline test  | 20   | 0    | 0.525           | 0.400  | 0.350  |
| fixed v1 | Retail        | 114  | 0    | 0.805           | 0.692  | 0.561  |
| fixed v1 | Retail train  | 74   | 0    | 0.845           | 0.750  | 0.635  |
| fixed v1 | Retail test   | 40   | 0    | 0.731           | 0.583  | 0.425  |

fixed v1 排除了 1 条 episode，`airline:14:trial1`：4 次尝试都超过了 32K 上下文。它所在题的 pass^4 只按 3 次有效
trial 无法计算，所以 Airline 的 pass^4 只在 49 题上算。

**fixed v1 的下降几乎全在 Airline，主要来自转人工规则**：

- Airline 从 0.780 降到 0.428。fixed v1 运行中，转人工规则之前的分数是 0.744，规则把 199 条 episode 里 63
  条原本成功的记为 0，扣了 31.7pp。剩下约 3.6pp 是 #411 新增的 NL 断言计分，这个量与下面的运行间噪声同级，不能单独确认。
- Retail 几乎不变（0.820 → 0.805）。转人工规则扣了 17 条成功（3.7pp），规则之前的分数是 0.842，高于 official。
- 这和 10.3 的估计（Airline 约 30pp 来自转人工）一致。也就是说，修复后这个 4B 模型的 Airline 真实水平约为 0.43，Airline
  train 只有 0.36。

**训练信号的变化**：Airline train 的 30 题从 0.775 降到 0.364，GRPO 组内会出现更多有对比的组，训练有更多可学的东西。Retail
train 基本不变。

**同一配置两次运行的差异**：

- official 的 0.808 比 10-09 同配置的 0.838 低 3.0pp。按题配对 bootstrap 的 95% 区间是 \[−5.9,
  −0.3\]pp（Airline \[−8.0, +1.5\]，Retail \[−6.4, +0.4\]）。
- 两次之间评测路径的代码没有改动：只增加了只对 fixed 生效的转人工规则、只对训练生效的超时处理，以及变体校验。
- 164 题中有 64 题的 4 次平均分变了。temperature 0 仍然不是确定的：SGLang 的批处理和 DeepSeek API 都会引入差异。
- 所以同一 checkpoint 重测本身就有约 3pp 的波动，与 3.1 节中两次 step 0 相差 4.58pp 是同一量级。比较训练前后，至少要看 test
  上的配对差和区间，不能只看点估计。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261010/step0-base-t0-deepseek-{official,fixed}/`。对比脚本：`scripts/official-runner/variant_compare.py`。

### 13.2 本轮：fixed v2

配置与 13.1 相同：未训练的 Qwen3.5-4B，base 集 164 题，每题 4 次，policy 和 DeepSeek-Flash 用户都是 temperature
0。代码 `b31e8eb5`，在固定于该提交的 worktree 中运行，数据目录 `…-fixed-v2/data`。official 没有重跑：它的数据和评分都没变，沿用
13.1。

| 变体     | 领域 / 划分                   | 题数 | 排除 | mean@4 = pass^1 | pass^2 | pass^4 |
| -------- | ----------------------------- | ---- | ---- | --------------- | ------ | ------ |
| fixed v2 | 合计                          | 164  | 4    | 0.708           | 0.612  | 0.512  |
| fixed v2 | Airline                       | 50   | 2    | 0.388           | 0.283  | 0.229  |
| fixed v2 | Airline train                 | 30   | 2    | 0.356           | 0.233  | 0.179  |
| fixed v2 | Airline test                  | 20   | 0    | 0.438           | 0.358  | 0.300  |
| fixed v2 | Retail                        | 114  | 2    | 0.849           | 0.756  | 0.634  |
| fixed v2 | Retail train                  | 74   | 2    | 0.868           | 0.782  | 0.667  |
| fixed v2 | Retail 训练池（去重后 69 题） | 69   | 2    | 0.880           | 0.792  | 0.672  |
| fixed v2 | Retail test                   | 40   | 0    | 0.812           | 0.708  | 0.575  |

排除 4 条，按题计算的 pass^4 因此只覆盖 160 题：

- `airline:10:trial3`、`airline:34:trial2`：4 次尝试都超过 32K 上下文，和 v1 的 `airline:14` 同类。
- `retail:59:trial3`、`retail:99:trial3`：`MissingAttemptRecord`，见本节末尾的"新出现的挂起"。

**Airline 和预期一致：基本不变**。0.428 → 0.388（−4.0pp）。fixed v2 对 Airline 只改了反向转人工规则，这条规则只影响
airline:13， 这次没有清零任何 Airline episode。所以这 4pp 是运行间噪声，在 13.1 测得的约 3pp 的量级内。正向转人工规则在
Airline 清零了 75 条原本成功的 episode（v1 是 63 条）。

**Retail 没有降低，反而升高了**。0.805 → 0.849（+4.4pp）。按题分组归因（114 题，每组为 4 次平均分之和的变化）：

| 题组                                                               | 题数 | v1 → v2  | 折合 Retail pp |
| ------------------------------------------------------------------ | ---- | -------- | -------------- |
| 20 道打了补丁的题                                                  | 20   | +3.50 题 | +3.1           |
| v1 被正向规则清零、v2 按 Retail 例外恢复的题（1、32、43 等 10 题） | 10   | +2.25 题 | +2.0           |
| 其余未改动的题                                                     | 84   | −0.75 题 | −0.7（噪声）   |

- **补丁让分数上升，是补丁本身的性质**。补丁去掉的是"做对了也判错"的失败：

  - 38：v1 是 0/4，v2 是 4/4，补丁写明了取消原因；
  - 64：0/4 → 4/4，补丁加了"多款符合时选最便宜的"；
  - 59、60、63、66、110 也都升到了 4/4。

  这些失败原本就不是模型能力的问题，留着只会给训练加噪声，不会增加可学的空间。

- **本该降分的三题没有降**。57、65 改为必须查订单，但这个模型本来就会查：57 在 v1、v2 都是 4/4，65 从 3/4 升到 4/4。25 要求转人工，v2
  是 2/4，与 v1 相同。 "什么都不做也得分"的漏洞对这个模型的分数影响很小。

- **Retail 例外恢复了 v1 误清零的分数**（+2.0pp），这和 11.3 的判断一致。正向规则在 Retail 不再清零任何 episode。反向规则清零了 2
  条（retail train）。

**训练集上的组内对比**（temperature 0，每题 4 次，按 train 池统计）：

| 领域          | 题数 | v1：全对 / 全错 / 有对比 | v2：全对 / 全错 / 有对比     |
| ------------- | ---- | ------------------------ | ---------------------------- |
| Airline train | 30   | 4 / 11 / 15              | 5 / 12 / 13                  |
| Retail 训练池 | 69   | 45 / 0 / 22              | 47 / 0 / 22（另 2 题有排除） |

- Retail 训练池有 47/69 题在 temperature 0 下 4 次全对。可学的空间主要在 Airline，以及 Retail 剩下的约 22 题。
- 训练用 temperature 1、每组 8 条，组内对比会比这里多。上一轮 GRPO（含 Telecom、官方数据）训练期 Retail reward 是
  0.868，72.5% 的组没有组内优势（3.2）。所以"Retail 提升空间小"在训练里同样存在。
- 去掉不公平的失败后 Retail 分数上升，是修复的预期结果。如果想要更多 Retail 训练空间，办法只能是加更难的题或换更难的用户模拟器，不能靠让评分变得不公平。
  本轮没有做这类改动。

**两条 episode 挂起 1 小时：proxy 的 session 容量泄漏**（已复现，已修复：`cefb0772`）

- **现象**：controller 的 task 611、643（`retail:59:trial3`、`retail:99:trial3`）都分到 engine rank
  3。submit 之后，它们一直没有打出 `STARTING SIMULATION`，也没有写尝试记录，直到 `request_timeout`（3600
  s）后才被拒绝。评测因此多花了约 45 分钟（总耗时 77 分钟，v1 是 31 分钟）。这两条按 infra 排除，不影响其他 episode 的分数。之前 5
  次运行都没有出现过。

- **机制**（读代码确认，`areal/experimental/openai/proxy/`）：

  1. 每个 episode 先 `grant_capacity`（容量 +1），再 `start_session`（容量 −1，新建 session
     `<task>-<idx>`）。
  1. 客户端 `_start_session` 每次请求的超时是 10 s；遇到 HTTP 429、超时等错误时无限重试。
  1. 修复前，服务端的 `start_session` 不是幂等的。如果它在客户端超时之后才执行完，第一次请求照样建出 `<task>-0`、消耗 1 个容量，但这个
     session 没人使用；客户端的重试又建出 `<task>-1`，再消耗 1 个容量。
  1. 每发生一次，这个 proxy 就永久少 1 个容量。`_cleanup_stale_sessions` 在 3600 s 后会删掉孤儿
     session，但不会归还容量。这个 proxy 上最后到达的 episode 因此一直拿到 429，只能等 controller 的
     `request_timeout`。

- **与本次运行的日志吻合**：rank 3 的 task 267（`retail:1`）和 275（`retail:93`）都正常完成，但 proxy 在 11:11:23
  把 `267-0`、`275-0` 当作 stale session 删除。正常完成却留下 `-0`，说明它们实际用的是 `-1`，也就是各发生了一次泄漏。泄漏 2
  次，挂起也正好 2 条。11:11:23 往前推 3600 s 是 10:11 左右，那时 rank 3 正有 34–35 个 episode 并发。

- **修复**：

  1. **`start_session` 幂等**（根治）：客户端每次 start 生成一个 `request_id`，所有重试都带同一个 id。服务端见到已处理过的
     id，就返回当时建的 session，不再消耗容量；如果那个 session 已经结束，返回 409。记录的 id 随 stale 清理一起过期。
  1. **重试打日志**：start 的每次重试都记 WARNING，拿不到容量时不再无声地等到超时。
  1. **`export_trajectories` 移到线程里执行**：构造和序列化 token 张量是 CPU 计算，原来直接在 async handler
     里跑，会阻塞事件循环，让同时到达的 start 超过 10 s 的客户端超时。这是诱因的候选之一，移走后它不再能阻塞别的请求。

  - **没有采用**"删除 stale session 时归还容量"：episode 在 start 之后崩溃时，这样会多发容量。幂等修好之后，grant 与 start
    已经一一对应，不需要这个兜底。

- **验证**：

  - 复现脚本 `scripts/official-runner/repro_proxy_capacity_leak.py` 用真实的 proxy server
    和客户端代码，在 episode A 的 `start_session` 执行期间持有 session 锁 12 s。
    - 修复前的提交 `49dadd88`：A 用的是 `A-1`，`A-0` 成了孤儿，后到的 episode B 等 40 s 仍收到 429，输出
      REPRODUCED；
    - 修复后：客户端超时重试一次，拿回 `A-0`，B 立即启动，容量回到 0，输出 NOT REPRODUCED。
  - `tests/experimental/openai/test_proxy_rollout_server.py` 新增 6 个测试。分别去掉服务端去重、客户端
    `request_id`、`to_thread`，都有对应测试失败。
  - proxy server 与 gateway 测试 50 个、tau2 测试 176 个全部通过。`test_proxy_integration.py` 需要 GPU
    和 SGLang，没有运行。

- **没有确认的**：实际运行中是什么让 `start_session` 超过 10 s 才执行完，没有访问日志能证明。候选是同步 handler 在线程池里排队或等
  `_lock`，以及上面的 export 阻塞事件循环。幂等修复不依赖诱因是哪一个。

- **与 `2d7d342a` 无关**：涉及的文件在那次提交里没有改动，这条路径也不经过组内成员丢失的处理。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261010-fixed-v2/step0-base-t0-deepseek-fixed-v2/`。
对比脚本是 `scripts/official-runner/variant_compare.py`，按领域、划分和训练池输出 mean@4 /
pass^k，以及转人工规则清零的条数。

### 13.3 AReaL pack v2（2026-10-11）

口径与 12.9 第三步的 v1 评测相同：未训练的 Qwen3.5-4B，temperature 0，DeepSeek-Flash 做用户和 NL judge，每题 4
次，`benchmark_variant=fixed`，代码固定在 `1b17bc3f`。pack 是
`_cache/tau2-areal-pack-20261011-v2`（12.10，648 题）。

- 第一次运行（pueue 447）时 DeepSeek 余额耗尽，690 条 episode 报 `Insufficient Balance`，被当作 infra
  错误排除。充值后把涉及的 197 题 4 次全部重跑（pueue 448），按题替换合并，最终 2592 条 episode 全部有分。
- DeepSeek 用户侧：两次运行共 25.9M prompt token、3.6M completion token（含第一次运行里被替换掉的 98 条）。v1 的
  813 题用了 33M / 5.0M。
- 产物：`tau2-eval-20261011-areal-pack/step0-areal-pack-v2-merged/episodes.jsonl`（合并结果）。两次运行的原始目录
  `step0-areal-pack-v2-t0-deepseek`、`…-rerun` 和合并脚本 `merge_v2_rerun.py` 都在同一目录下。

| 范围                   | 题数 | mean@4 = pass^1 |    pass^2 |    pass^4 | 4 次全对 | 有对有错 | 4 次全错 |
| ---------------------- | ---: | --------------: | --------: | --------: | -------: | -------: | -------: |
| v1 全部                |  813 |           0.912 |     0.859 |     0.783 |      637 |      161 |       15 |
| v1 中 v2 保留的 648 题 |  648 |           0.934 |     0.887 |     0.815 |      528 |      114 |        6 |
| v1 中 v2 丢掉的 165 题 |  165 |           0.827 |     0.747 |     0.661 |      109 |       47 |        9 |
| **v2 合计**            |  648 |       **0.946** | **0.912** | **0.861** |      558 |       84 |        6 |
| v2 Airline             |  380 |           0.926 |     0.881 |     0.818 |      311 |       63 |        6 |
| v2 Airline keep        |  191 |           0.969 |     0.946 |     0.911 |      174 |       16 |        1 |
| v2 Airline fix         |  189 |           0.882 |     0.816 |     0.725 |      137 |       47 |        5 |
| v2 Retail              |  268 |           0.976 |     0.955 |     0.922 |      247 |       21 |        0 |
| v2 Retail keep         |   48 |           0.964 |     0.931 |     0.875 |       42 |        6 |        0 |
| v2 Retail fix          |  220 |           0.978 |     0.961 |     0.932 |      205 |       15 |        0 |

**结果**：

- **丢掉的题分数更低**（0.827 对保留题的 0.934）。v1 里 4 次全错的 15 题，有 9 题在 v2 被丢掉，说明大部分全错的题是题本身有问题。v2 剩下
  6 题 4 次全错，都是 Airline（airline_361、365、437、505、710、871），还没看轨迹。
- **同一批题，修过之后结果更稳**：有对有错从 114 降到 84，pass^4 从 0.815 升到 0.861，主要在 Retail（0.851 →
  0.922）。第三轮复核在指令里写明了"替谁办理"和地址写法，用户模拟器少了自由发挥的空间。
- **单题的变化多是噪声**：648 题里 505 题分数不变，83 题升、59 题降。temperature 0 下 DeepSeek 用户仍有随机性，单题对比没有意义。
- **最难的是修过的 Airline 题**（0.882，keep 是 0.969）：替换后的 gold 多是多步的改签、降舱。

**对后续实验设置的含义**：

1. **v2 对 Qwen3.5-4B 太容易，不适合拿它直接做 RL**。86% 的题 4 次全对，GRPO 只有 84 题（13%）的组内有分数差异；按训练时
   temperature 1 采样，差异会多一些，但这一点没测。
1. **AReaL 的题比官方 test 容易得多**。同一模型、同一口径下，官方 fixed v2 base 集 Airline 0.388、Retail
   0.849（13.2），AReaL v2 是 0.926、0.976。AReaL 的 Airline 题多是单一预订上的一两步操作，官方 Airline
   题要在多个预订之间推理 policy。所以在 AReaL 上练出的分数提升，不一定能迁移到官方 Airline。
1. **训练模型的候选是更弱的 Qwen3-4B-2507**（15.7）。它在官方 fixed 上 Airline / Retail 只有 0.225–0.245 /
   0.410–0.654，v2 的难度对它可能正合适。下一步先用本节口径在 v2 上评测 Instruct 和 Thinking
   两个版本，看有对有错的题有多少。DeepSeek 费用参照本节：Qwen3.5 跑 648 题约 26M prompt token，Thinking 的对话更长，会更多。
1. **Telecom**：Qwen3.5-4B 在官方 Telecom 上 456/456 全对（15.1），AReaL Telecom 也一样容易（0.993，见
   13.4）。按上一条，Telecom 只对更弱的模型有用。

### 13.4 AReaL Telecom pack（2026-10-11）

**为什么现在要 Telecom**（用户，2026-10-11）：12.1 没用 Telecom 是因为它对 Qwen3.5-4B 太简单；换更弱的 agent
模型后，它就有训练价值。筛选按 12.11 的流程走，pack 与 Airline / Retail 分开。

**机械过滤**（`tools/telecom_mech.py`，271 → 261）：

- AReaL 的 Telecom 数据（271 题）用的 DB 和官方 `db.toml` 逐字节相同。题目不与官方 test 集的初始状态重合；有 251 题与官方
  `telecom_full`（2285 题）同一个初始状态，只是 persona 文本不同。我们评测用的是 base 集，不用 `telecom_full`，所以保留，并在
  `out/telecom_mech.json` 里标出。
- **reward 设置有问题**：205 题没写 `reward_basis`，默认是 DB + COMMUNICATE。没有一题有
  `communicate_info`，所以 reward 实际只比较 agent 侧 DB 和用户设备状态的 hash。官方评测器重放 gold 后不调用
  `sync_tools`，而真实对话里每次工具调用都会同步，设备状态（比如 `mobile_data_usage_exceeded`）因此永远对不上：其中 136 题的
  gold 本身就得不了 1 分。官方 Telecom 题都用 ENV_ASSERTION。
- **处理**：195 题改用 ENV_ASSERTION，改完后 gold 1 分、空轨迹 0 分。剩下 10 题改了也区分不开 gold 和空轨迹，丢掉。为此 pack
  的 fix 加了 `reward_basis` 这一种改法（`6d477635`）。
- 代码：`check_task_pack.py` 按 tau2 的 `verify_task` 先应用 `initial_state`，再把每个 gold
  动作放在对应的一方（agent 或用户）重放（`0634cfe6`）。

**评测 v0**（Qwen3.5-4B，口径同 13.3，代码 `6d477635`，261 题 × 4）：mean@4 0.993，pass^2 0.987，pass^4
0.973；254 题 4 次全对，7 题有对有错，没有 4 次全错的题。DeepSeek 用户侧用了 157M prompt token，是 v2 的 6
倍：排障对话轮数多，每轮都带完整历史。

**Codex 审题**：

- **第一轮**（`TELECOM_REVIEW.md`）：keep 79、fix 85、drop 97。我抽查后发现 drop 多半不对，这些"题错"官方题里也普遍存在：

  | 第一轮判为题错的特征                                     |    第一轮引用它的条数 | 官方 2285 题中                             |
  | -------------------------------------------------------- | --------------------: | ------------------------------------------ |
  | ticket 写 "No Service"，手机其实有信号                   |            57 道 drop | 132 题同样如此（ticket 只有 5 个通用模板） |
  | persona 年龄与客户生日对不上                             | 49 道 drop、60 道 fix | 全部用同一个客户，760 题写 41 岁           |
  | 指令里写 `get_status_bar`（工具名是 `check_status_bar`） |             50 道 fix | 2285 题全都这么写                          |
  | 有一条断言初始就成立                                     |             4 道 drop | 15 题有                                    |

  另外 6 道 drop 是 rubric 照搬了 Airline 的转人工规则。97 道 drop 里 95 道在 v0 评测中 4 次全对。

- **第二轮**（`TELECOM_REVIEW2.md`，复核第一轮的 182 道 drop 和 fix）：rubric 写明上面这些不算题错。结果没有 drop，132
  题 keep，50 题 fix。50 道 fix 都是同一个改法：原指令说"愿意充 2 GB"，reward 却要求恰好 2
  GB，用户若少充也能解决问题，却拿不到分。改成"要恰好 2 GB"。

**v1 pack**：`_cache/tau2-areal-telecom-pack-20261011-v1`，261 题（50 题改了指令，195 题改了
reward_basis），全部 gold 1 分、空轨迹 0 分。v1 只改了 50 题的一句话，Qwen3.5 在 v0 上已经 0.993，所以没有用 Qwen3.5
重评。

**对后续实验的含义**：Telecom 对 Qwen3.5-4B 没有训练信号。Qwen3-4B-2507 在官方 Telecom 上约 0.37（15.1），所以
Telecom 是否有用取决于 Qwen3 在这份数据上的分数，见 13.5 表 B。DeepSeek 费用要按 Telecom 的对话长度估算：一个模型跑 261 题 × 4
约 150M prompt token。

### 13.5 训练前评测总表

所有未训练模型的 step 0 评测汇总在这两张表里，新模型按同一口径评测后加一行。两张表分开放：表 A 是官方 τ² 题，用来和官方 / 论文比；表 B 是筛选后的
AReaL 训练数据，用来判断这份数据对某个模型有没有训练信号。

**共同口径**：policy temperature 0；DeepSeek-Flash 做用户模拟器和 NL judge（temperature 0）；每题 4
次；Airline / Retail 用 `benchmark_variant=fixed`（fixed v2 数据，13.2）。"有对有错"的题，GRPO
的组内才有分数差异；训练时按 temperature 1 采样，这类题会更多，但还没测过。

**评测入口**：

- 表
  A：`scripts/qwen3-2507/launch_eval.sh NAME MODEL GPUS VARIANT DOMAINS THINKING PARSER`（Qwen3.5
  的配置见 13.2，Qwen3-2507 的配置见 15.2）。
- 表 B：同一个脚本，加环境变量 `TASK_PACK=_cache/tau2-areal-pack-latest`（或
  `tau2-areal-telecom-pack-latest`）和 `TAU2_SRC=<不早于 1b17bc3f 的固定 worktree>`。产物在
  `tau2-eval-20261012-qwen3-2507-packs/`。

**表 A：官方 base 集**（Airline 50、Retail 114、Telecom 114 题，train + test）

| 模型                   | 领域                       | 变体                | mean@4 | pass^4 | 4 次全对 | 有对有错 | 4 次全错 | 来源 |
| ---------------------- | -------------------------- | ------------------- | -----: | -----: | -------: | -------: | -------: | ---- |
| Qwen3.5-4B             | Airline                    | fixed v2            |  0.388 |  0.229 |       11 |       19 |       20 | 13.2 |
| Qwen3.5-4B             | Retail                     | fixed v2            |  0.849 |  0.634 |       73 |       39 |        2 | 13.2 |
| Qwen3.5-4B             | Telecom                    | official            |  1.000 |  1.000 |      114 |        0 |        0 | 15.3 |
| Qwen3-4B-Instruct-2507 | Airline                    | fixed v2            |  0.225 |  0.060 |        3 |       21 |       26 | 15.3 |
| Qwen3-4B-Instruct-2507 | Retail                     | fixed v2            |  0.410 |  0.096 |       11 |       72 |       31 | 15.3 |
| Qwen3-4B-Instruct-2507 | Telecom                    | official            |  0.377 |  0.167 |       19 |       60 |       35 | 15.3 |
| Qwen3-4B-Thinking-2507 | Airline                    | fixed v2            |  0.245 |  0.060 |        3 |       20 |       27 | 15.3 |
| Qwen3-4B-Thinking-2507 | Retail                     | fixed v2            |  0.654 |  0.351 |       40 |       65 |        9 | 15.3 |
| Qwen3-4B-Thinking-2507 | Telecom                    | official            |  0.373 |  0.246 |       28 |       25 |       61 | 15.3 |
| （下一个模型）         | Airline / Retail / Telecom | fixed v2 / official |      — |      — |        — |        — |        — |      |

Airline / Retail 在 official 变体下的分数见 15.3；official 与 fixed 的差别见 10.3、13.2。

**表 B：筛选后的 AReaL-tau2-data**（Airline / Retail 用 pack v2，648 题，12.10；Telecom 用 pack v0 /
v1，261 题，13.4）

| 模型                   | 领域                       | pack        | 题数 | mean@4 | pass^4 | 4 次全对 | 有对有错 | 4 次全错 | 来源      |
| ---------------------- | -------------------------- | ----------- | ---: | -----: | -----: | -------: | -------: | -------: | --------- |
| Qwen3.5-4B             | Airline                    | v2          |  380 |  0.926 |  0.818 |      311 |       63 |        6 | 13.3      |
| Qwen3.5-4B             | Retail                     | v2          |  268 |  0.976 |  0.922 |      247 |       21 |        0 | 13.3      |
| Qwen3.5-4B             | Telecom                    | telecom v0  |  261 |  0.993 |  0.973 |      254 |        7 |        0 | 13.4      |
| Qwen3-4B-Instruct-2507 | Airline                    | v2          |  380 |  0.593 |  0.289 |      110 |      211 |       59 | 本节      |
| Qwen3-4B-Instruct-2507 | Retail                     | v2          |  268 |  0.787 |  0.459 |      123 |      138 |        7 | 本节      |
| Qwen3-4B-Instruct-2507 | Telecom                    | telecom v1  |  261 |   待测 |        |          |          |          | pueue 460 |
| Qwen3-4B-Thinking-2507 | Airline                    | v2          |  380 | 运行中 |        |          |          |          | pueue 459 |
| Qwen3-4B-Thinking-2507 | Retail                     | v2          |  268 | 运行中 |        |          |          |          | pueue 459 |
| Qwen3-4B-Thinking-2507 | Telecom                    | telecom v1  |  261 |   待测 |        |          |          |          | pueue 460 |
| （下一个模型）         | Airline / Retail / Telecom | 当前 latest |      |        |        |          |          |          |           |

Qwen3.5 的 Telecom 用的是 v0。v1 只在 50 题里改了一句"要恰好 2 GB"，Qwen3.5 在 v0 上已经 0.993，所以没有重评。

**读表**（2026-10-12，Thinking 和 Telecom 跑完后补）：

- **Qwen3.5-4B 在 AReaL 数据上没有训练信号**：648 题里只有 84 题有对有错（13%）。
- **Qwen3-4B-Instruct-2507 的难度合适**：349 题有对有错（54%），4 次全错的只有 66 题。
- **Instruct 全错的题集中在修过的 Airline 题**：Airline fix 189 题 mean@4 0.463、全错 47 题；keep 191 题是
  0.721、全错 12 题。这些题的 gold 是多步改签、降舱，Qwen3.5 能做对（13.3），所以是模型能力不够，不是题错。
- **AReaL 数据比官方题容易**：Instruct 在官方 Airline / Retail 是 0.225 / 0.410，在 v2 上是 0.593 /
  0.787。在 v2 上练出的提升能不能迁移到官方题，要训练后用表 A 的口径评测才知道。

## 14. 能力缺口与训练期问题的 case study（2026-10-11）

目的：区分两件事——未训练的 Qwen3.5-4B 自身的能力缺口（每类该用 RL、加数据还是 harness 解决），以及在这个交互式场景做 RL
时训练信号本身的问题。只用已有记录，没有新采样或付费调用。

数据：fixed v2 的 step 0（13.2）为主，official / fixed v1 / Doubao 的 step 0 交叉核对；训练期信号用
2026-09-27 GRPO formal-v2 的 3,088 条 T=1 rollout（386 组 × 8，official 评分）。SAO 只落盘了 step 0 的
126 条、每题 1 条，做不了组内分析。

分工：Airline、Retail、训练信号各一个 agent，另一个做文献调研。我抽查了每份报告的关键证据 （airline:44 t0、airline:45
t0–3、retail:37 t0、retail:20 t0、GRPO 组内首句用户消息），都与报告一致；其余标注没有逐条复核。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-capability-gap-20261011/`
（`BRIEF.md` 是分类口径；`out/{airline_gap,retail_gap,train_signal,lit_scan}.md` 是各自报告，`.jsonl`
是逐条记录； `out/pass_table.json` 是每题在全部运行里的通过数）。

### 14.1 结论

1. **Airline 的头号缺口是"用转人工代替拒绝"（M2a）：69 条、25 题、34.8pp。**64/69 条 agent 先正确列出了不能取消的理由， 然后调用
   `transfer_to_human_agents` 而不是直接拒绝；9 条用户明确说过不要转。53/75 条引用了 policy 第 141 行 "already
   been flown … transfer is needed"，52/69 条说"API 不能取消"，把第 149 行 "The API does not
   check…" 读反了。 所以这是**行为选择**问题，不是资格推理问题。去掉 8 条"用户先发 `###TRANSFER###` 而侥幸得分"后，Airline 真实
   pass^1 约 0.35。
1. **Retail 的缺口不是长程或多实体**。gold 写操作数、订单数与通过率的 Spearman ρ 为 +0.17 / +0.15；只算模型失败时， 1
   个写操作的题失败率 11.2%，≥3 个的 6.0%。有区分度的是：按约束挑选（10.2% vs 4.0%），以及误解工具语义加报错后乱补救。
1. **新发现的一类：工具语义误解（M14）+ 报错后静默改参数重写（M15）**。例：airline:44 认为没有工具能查航班时刻
   （`search_direct_flight` 其实返回），3/4 trial 如此；retail:37 t0 以为一单只能调一次 modify，把不变的 item
   也填进去， 报错后不问用户就把背包换成更贵的款（\[30\]→\[32\]）。M15 在 T=1 失败中占 6.9%、成功中 1.4%。
1. **几乎所有失败过的题都在别的运行或 T=1 中成功过。**真正"从来做不对"的只有 airline:7（benchmark）、45（每次推理都对，然后转人工）、
   47（保险只覆盖健康/天气，0/8，T=1 1/18）。也就是说，按 4B 的现状，大多数缺口是**稳定性**问题（RL 能磨），不是能力缺失。
1. **训练信号里只有约 58% 的组内对比在教真本事**（95% CI 46–73%；Airline 88%，Retail 只有 42%）。其余：用户模拟器失败 29%、
   gold 错或歧义 10%、转人工骗分 3.5%。
1. **约 66% 的负梯度落在没有导致失败的 token 上**。人工定位的决定性错误轮中位数是第 9/15 轮，加上噪声失败的全部 token。

### 14.2 模型能力缺口与补救途径（fixed v2 step 0，按首要原因）

| 机制                                          | Airline pp | Retail pp | 补救                                        | 依据                                                                                                        |
| --------------------------------------------- | ---------- | --------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| M2a 用转人工代替拒绝                          | 34.8       | 0.9       | R-RL（用 fixed reward）+ R-HARNESS          | 推理已正确，只差最后一步的动作；T=1 下多数题有组内对比。接近 0 的题（27、39、43、47、49）用转人工门控更便宜 |
| M14/M15 工具语义误解、报错后静默重写          | 2.0        | 3.5       | R-HARNESS + R-RL                            | 工具描述写清返回字段、自动剔除 old==new 的项；成功样本会回头问用户，组内有对比                              |
| M1 政策资格规则（保险覆盖原因、同款换货）     | 4.5        | 0.7       | 43/44/18 R-RL；47 R-DATA 或资格检查器       | 47 是唯一稳定做不对的能力缺失                                                                               |
| M5 支付/退款路由                              | 3.0        | 1.8       | R-RL；airline:21 R-HARNESS                  | 成功样本都按原支付方式退                                                                                    |
| M4 多实体、M8 遗忘已读事实或子目标            | 3.0        | 1.3       | R-HARNESS（已读事实 + 待办写操作的 ledger） | airline:42 唯一的成功是用户替 agent 推理的                                                                  |
| M7 枚举/比较、M6 算术                         | 1.0        | 1.1       | R-HARNESS                                   | retail:20 列出 261.11 却选 258.71，3 个 trial 一样；`calculate` 每 200 条只用 4–11 次                       |
| M11 先写后报价                                | 2.0        | 0.2       | R-HARNESS                                   | 写前自动算差价并要求明确 yes                                                                                |
| M3/M12 写完就转人工、不回答追问               | 0.5        | 1.1       | R-RL                                        | T=1 成功率 > 90%                                                                                            |
| 非模型：benchmark、用户模拟器、转人工规则误伤 | 7.5        | 6.4       | R-REWARD                                    | 见 14.4                                                                                                     |

R-DATA 的需求很小：只有 airline:47 一类，以及 airline:44 这种训练集没覆盖的工具用法。加数据的主要理由仍是 12.7 的"训练题太简单"，
不是补能力。

### 14.3 训练期问题（交互式 RL 本身，与能力缺口分开）

按对训练的危害排序（P = 训练范式，B = benchmark/环境）：

| #   | 问题                                           | 证据                                                                                                       | 现有手段能否解决                                                                       |
| --- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1   | \[P+B\] 用户模拟器噪声占梯度大头，T=0 也消不掉 | 29% 的组内对比是"正常 agent vs 用户 STOP/偏离"，Retail 42%；同组 8 条的首句用户消息 386/386 组都不全相同   | 动态采样反而保留这些组（噪声恰好让组变成混合组）；fixed 只修 gold；用户 T=1 会更糟     |
| 2   | \[P\] 用户 STOP 惩罚"写前确认"                 | 5 条重复确认后被 STOP 的失败拿负优势，不确认的样本拿 1                                                     | 都不针对。reward 与 policy 直接冲突，训练久了会学会少确认                              |
| 3   | \[P\] 序列级优势的信用分配                     | 约 66% 负梯度落在非因果 token；决定性错误轮中位 9/15                                                       | 都不改变。需要 turn 级信用；SAO critic 的 EV 为负（3.4），目前没起作用                 |
| 4   | \[B\] official 评分奖励转人工                  | 应拒绝任务里转人工样本净优势 +14.9；占 Airline T=1 成功的 45%（198/510 条是骗分）                          | fixed 的转人工规则直接解决，但误伤 policy 要求转人工的已飞航段（airline:37、9，3.0pp） |
| 5   | \[B\] gold 错让组内排序反转                    | 10% 的对比；retail:98（G68）6 条遵从指令的样本全得 0                                                       | fixed v2 已修 98（11.2）；AReaL-tau2-data 引入新题时会再出现                           |
| 6   | \[P\] 二值 reward 抹掉部分进度                 | 225 条有 gold 写的失败中，44 条 DB 全做对（其中 16 条是 airline:7）、66 条做了一部分，与一个都没做的同得 0 | 按写操作给部分分能缓解，但会放大 gold 歧义                                             |
| 7   | \[P\] 有效样本少                               | 平均每步 8 组只有约 2.3 个混合组、约 1.3 个有效对比组                                                      | 动态采样提高混合组比例，但带着同样比例的噪声                                           |

4096 截断不是独立的失败原因：组内被截断与未截断的样本成功率无差异（+0.011 ± 0.095）。

### 14.4 可能的论文问题（结合文献调研，待核实）

文献调研只读了摘要，arXiv 编号未逐个打开确认，引用前要核实。

1. **用户模拟器噪声下的优势估计**（14.3 第 1、2 项）。文献里有人度量过换用户带来的分数差（UserProxyBench 2609.38043，15.2 分），
   没看到在训练中处理用户 STOP/偏离的方法。候选做法：用户在 agent 请求确认后 STOP 时把样本按 infra 失败剔除；同一前缀对用户重采样取平均。
1. **turn 级信用分配**（第 3 项）。约 66% 的负梯度落在非因果 token 上，需要 turn 级信用；SAO critic 想证明有用，至少要打赢"把结局
   advantage 均匀分给各轮"这个零信息基线。
1. **转人工捷径与"拒绝并留在对话里"**（14.2 M2a、14.3 第 4 项）。step 0 就有 34.8pp，official reward
   还在强化它；fixed 规则是 硬规则，会误伤合规的转人工。一个按 policy 判断"该不该转"的 reward 是空缺。
1. **部分进度 reward**（第 6 项）。按 DB 终态字段 diff 给分、并防止回滚刷分，文献调研里没找到做过的。

### 14.5 局限

- Airline 非转人工的 46 条失败读了约 40 条；Retail 69 条读了约 45 条，14 条按同题同路径规则核实。训练信号的 320 条标注只抽查了 2 组。
- "确认后立刻 STOP"的 pp 是可追回的上限。
- 训练期结论全部来自 GRPO、official 评分、旧数据；fixed v2 下训练信号的噪声比例需要新一轮训练后重算。
- 观测到的 Airline 转人工率从 epoch 1 到 2 下降 13.7pp（CI −22.5 到
  −5.0），但权重变化接近随机游走（3.3），原因未知，不能当作训练纠正了 hacking。
- Retail T=1 episode 计数在两个脚本里不同（1145 vs 994，后者按 `episode_id` 去重），没有查清。

## 15. Qwen3-4B-2507 的 step 0 评测与能力缺口（2026-10-11）

对象：未经训练的 Qwen3-4B-Instruct-2507（non-thinking）和 Qwen3-4B-Thinking-2507（thinking）。
目的：Qwen3.5-4B 在 fixed v2 的 Retail、Telecom 上已接近满分（13.2）。这里确认两个 Qwen3
模型的可提升空间，并弄清分数低的原因：是模型能力不足还是题目问题，如果是模型，具体缺哪种能力。另外核查我们为 Qwen3.5 做的加速配置有哪些对 Qwen3 仍然可用。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261010-qwen3-2507/`

- 每个运行目录下有 `summary.json`、`episodes.jsonl`、`trajectories/`
- `analysis/` 存放逐 episode 特征（`feat-*.jsonl`）和 actor probe 日志

脚本在本目录 `scripts/qwen3-2507/`：

- 启动：`launch_eval.sh`、`lane_a.sh`、`lane_b.sh`
- 特征与分类：`extract_episodes.py`、`aggregate.py`、`compare.py`、`write_diff.py`、`telecom.py`
- 查看轨迹：`view.py`
- Actor probe：`actor_probe.py`
- 模型下载与校验：`fetch_models.py`

### 15.1 结论

1. **可提升空间大得多，模型本身的能力也弱得多。**
   - fixed 下，Instruct 的 Airline / Retail mean@4 是 0.225 / 0.410，Thinking 是 0.245 /
     0.654，Qwen3.5 是 0.388 / 0.849。
   - Telecom 上两个 Qwen3 都约 0.37，Qwen3.5 是 456/456 全对。
   - temperature 0 下，Retail 有组内对比的题：两个 Qwen3 有 65–72 道，Qwen3.5 只有 39 道。
1. **分数低主要是模型问题。** 用户在确认句里结束对话、转人工漏洞这两项评测因素扣除后，与 Qwen3.5 的差距仍有 15–42pp（15.4）。
1. **缺口集中在多子任务、长程执行。** 这与 14.1 第 2 条 Qwen3.5 的情况相反：
   - 只需 1 个写操作的 Retail 题，Thinking 0.72，与 Qwen3.5 的 0.82 接近；
   - 需要 ≥3 个写操作时，Thinking 掉到 0.55、Instruct 0.19，而 Qwen3.5 是 0.92；
   - Telecom 注入 ≥6 个故障时，Thinking 为 0。
1. **两个模型短板不同。** Instruct 主要是不查证就行动、参数填错；Thinking 单步更准，但 policy 推理不完整，常以转人工收尾。
1. **对冷启动 / warm-up 的启示**（由失败分类推断，未经实验验证）：
   - 工具格式没有问题，缺的是把多子任务做完的行为；
   - Telecom 故障多的题两个模型几乎 4 次全错，单靠 RL 拿不到信号；
   - 示范数据应覆盖：遍历完所有实体、按手册走完排障流程、不把转人工当退路。
1. **加速配置：沿用现有 FSDP 配置，不再加其他优化**（2026-10-11 决定）。
   - FSDP + FA2 + gradient checkpointing + packing 能直接跑通，在测过的组合里最快。
   - Ulysses、tree training、archon 也能启动，但都不更快（15.6）。

### 15.2 配置

| 项                    | 设置                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 题集                  | base（train + test），每题 4 次，policy temperature 0。fixed：Airline 50 + Retail 114；official：再加 Telecom 114 |
| 用户模拟器 / NL judge | DeepSeek-Flash，temperature 0，同 13.2                                                                            |
| Instruct              | `enable_thinking=false`；concat 模板，与 Qwen3.5 评测相同；`tool_call_parser=qwen25`                              |
| Thinking              | `enable_thinking=true`；hf 模板 + individual 导出；`tool_call_parser=qwen25`                                      |
| 代码                  | `a6abe781`，在固定于该提交的 worktree 上运行                                                                      |
| 模型                  | Instruct `cdbee75f`，Thinking `768f209d`；从 ModelScope 下载，4 个 LFS 文件的 sha256 都与 HF 一致                 |

- **Telecom 不分变体。** fixed 只改了 Airline 和 Retail，Telecom 只在 official 里跑一次。Qwen3.5 的
  Telecom 对照是这次按相同配置新跑的。
- **与模型卡分数不可比。** 模型卡用 gpt-4.1 类用户模拟器、采样解码；这里是 DeepSeek 用户、temperature 0。

**为支持 Thinking 改的代码（`a6abe781`）。** Thinking-2507 的模板在生成前缀里就放了 `<think>`，所以模型输出里只有
`…</think>答案`。改之前有三个问题：

- proxy 不把这段识别为推理，整段思维链作为 `content` 返回；
- 用户模拟器会读到 agent 的思维链；
- 带工具调用的回复同时带文本，orchestrator 会判为错误。

改动内容：

- `ReasoningStrippedLLMAgent` 在 tau2 侧去掉 `</think>` 及之前的内容，完整回复保留在 `raw_data`；
- proxy 把只出现 `</think>` 的情况也识别为推理段；
- 每轮用 hf 模板完整渲染，历史里不保留之前各轮的思维链；
- 训练仍要求 `enable_thinking=false`。

三处都有单元测试，分别故意破坏后对应测试会失败。

已知限制：带工具调用的轮次，`raw_data` 里也没有思维链，所以 case study 只能看纯文本回复轮的推理。

### 15.3 分数

mean@4 = pass^1。

| 模型          | 变体     | 领域    | pass^1 | pass^2 | pass^4 | test  | train |
| ------------- | -------- | ------- | ------ | ------ | ------ | ----- | ----- |
| Qwen3.5-4B    | official | Airline | 0.780  | 0.710  | 0.620  | 0.787 | 0.775 |
| Qwen3.5-4B    | official | Retail  | 0.820  | 0.703  | 0.535  | 0.769 | 0.848 |
| Qwen3.5-4B    | fixed v2 | Airline | 0.388  | 0.283  | 0.229  | 0.438 | 0.356 |
| Qwen3.5-4B    | fixed v2 | Retail  | 0.849  | 0.756  | 0.634  | 0.812 | 0.867 |
| Qwen3.5-4B    | official | Telecom | 1.000  | 1.000  | 1.000  | 1.000 | 1.000 |
| Instruct-2507 | fixed    | Airline | 0.225  | 0.107  | 0.060  | 0.200 | 0.242 |
| Instruct-2507 | fixed    | Retail  | 0.410  | 0.235  | 0.096  | 0.438 | 0.395 |
| Instruct-2507 | official | Airline | 0.330  | 0.203  | 0.140  | 0.250 | 0.383 |
| Instruct-2507 | official | Retail  | 0.445  | 0.288  | 0.158  | 0.481 | 0.426 |
| Instruct-2507 | official | Telecom | 0.377  | 0.240  | 0.167  | 0.481 | 0.321 |
| Thinking-2507 | fixed    | Airline | 0.245  | 0.140  | 0.060  | 0.263 | 0.233 |
| Thinking-2507 | fixed    | Retail  | 0.654  | 0.494  | 0.351  | 0.619 | 0.672 |
| Thinking-2507 | official | Airline | 0.655  | 0.533  | 0.400  | 0.650 | 0.658 |
| Thinking-2507 | official | Retail  | 0.651  | 0.501  | 0.333  | 0.669 | 0.642 |
| Thinking-2507 | official | Telecom | 0.373  | 0.313  | 0.246  | 0.506 | 0.301 |

- **排除与超时。** 这次的 5 个运行都没有 infra 排除。Thinking official 有 7 条 1800 s 墙钟超时（Telecom 6
  条、Airline 1 条），按官方语义记 0 分。
- **成本。** 每轮生成的中位数：Thinking 956 token，Instruct 85，Qwen3.5 200。Thinking 单轮最长 7.5K
  token，没有截断，也没有超出上下文。

temperature 0 下各题 4 次的结果分布（全对 / 有对有错 / 全错）汇总在 13.5 表 A。

### 15.4 题目与评测因素

**用户在确认句里结束对话。** agent 请求确认后，DeepSeek 用户回 "Yes, please proceed. ###STOP###"，写操作来不及执行。这和
9.1 节 Doubao 的现象相同。

判定规则：

- 该 episode 失败；
- gold 有写操作，agent 没写；
- 最后一句用户消息带肯定词，并紧接 `###STOP###`。

| 运行              | Retail 条数 | Retail mean@4 | 去掉这些 episode 后 |
| ----------------- | ----------- | ------------- | ------------------- |
| Qwen3.5 fixed v2  | 12          | 0.848         | 0.871               |
| Instruct fixed    | 42          | 0.410         | 0.452               |
| Thinking fixed    | 41          | 0.654         | 0.718               |
| Instruct official | 48          | 0.445         | 0.498               |
| Thinking official | 34          | 0.651         | 0.704               |

- Qwen3 模型更常请求确认，所以受影响更多，但扣除后差距仍在。
- Airline 每个运行有 0–6 条，Telecom 为 0。
- 这些 episode 不该作为负样本进入训练，与 14.3 第 2 项是同一个问题。

**转人工漏洞（10.3）。** Thinking 在 Airline 大量转人工：fixed 规则清零 95 条，其中 57 条在 official 评分下是 1
分。official 0.655 与 fixed 0.245 之间的 41pp 差距就来自这里。Instruct 很少转人工，而是直接执行修改，所以两个变体只差 10pp。

**没有发现新的题目缺陷。** 精读过的失败轨迹都能对应到 agent 的具体错误。

### 15.5 能力缺口

**多子任务与长程执行（两个模型共同的主因）。** 按 gold 写操作数分组的成功率（fixed）：

| 模型          | Retail 1 | Retail 2 | Retail ≥3 | Airline 1 | Airline 2 | Airline ≥3 |
| ------------- | -------- | -------- | --------- | --------- | --------- | ---------- |
| Qwen3.5-4B    | 0.82     | 0.90     | 0.92      | 0.67      | 0.62      | 0.39       |
| Thinking-2507 | 0.72     | 0.52     | 0.55      | 0.62      | 0.12      | 0.11       |
| Instruct-2507 | 0.52     | 0.30     | 0.19      | 0.31      | 0.17      | 0.07       |

Telecom 按注入的故障数：

| 故障数   | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Instruct | 0.59 | 0.38 | 0.51 | 0.37 | 0.16 | 0.09 | 0.06 | 0.04 |
| Thinking | 0.60 | 0.48 | 0.51 | 0.31 | 0.02 | 0.00 | 0.00 | 0.00 |
| Qwen3.5  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

失败的具体形态：

- **列表没遍历完。**
  - Thinking airline:22 t1：用户要处理"纽约→芝加哥"的行程，模型只查了第一个预订（西雅图→达拉斯），思维链里写 "picked the first
    one"。
  - Thinking retail:54 t1：用户有 6 个订单，只查了 5 个，漏掉一单取消。
- **流程没走完就转人工。**
  - 转人工在 Telecom 失败中占 Thinking 264/286、Instruct 183/284。
  - 需要充流量的题，调用 `refuel_data` 的次数：Thinking 6/197，Instruct 48/200，Qwen3.5 200/200。
  - Thinking 的思维链写"基础检查都做完了，按 policy 应升级"，实际上流量问题还没处理。
- **只完成一部分子任务。** 部分 gold 写操作完成、其余缺失的失败 episode：Instruct 24，Thinking 17，Qwen3.5 4。

**参数 grounding：不查证就行动（Instruct 为主）。** Retail 写操作失败按类型分（fixed）：

| 模型     | 完全没写 | 参数错 | 只做了部分子任务 | 只调了非 gold 工具 | 有写操作报错 |
| -------- | -------- | ------ | ---------------- | ------------------ | ------------ |
| Instruct | 100      | 81     | 24               | 29                 | 55           |
| Thinking | 63       | 45     | 17               | 12                 | 20           |
| Qwen3.5  | 23       | 26     | 4                | 3                  | 10           |

- 出错最多的参数是 `new_item_ids`（选目标变体），其次是 `item_ids` 和 `order_id`。
- 例：Instruct retail:101 t1。它把 policy 里的格式示例 `#W0000000` 当订单号去查；没有调用
  `get_product_details`，就断言某个变体没货。

**Policy 遵守。**

- **该拒绝的 Airline 题**（0 个 gold 写操作），三个模型都在 0.11–0.24，与模型代际无关，同 14.2 的 M2a。
- **Instruct 会执行 policy 不允许的修改。** 例：airline:2 t0，它主动提出补偿并发了 \$50 证书，policy 规定不能主动提补偿。
- **Thinking 的 policy 推理不完整。** 例：airline:32 t0，policy 允许 basic economy
  先升舱再改签，它只记住了"basic economy 不能改"。它问了一句"要不要转人工"，没等用户回答就转了。

**不是缺口的部分。**

- 工具调用格式没问题：Instruct 没有空消息，Thinking 没有截断。
- 单步工具使用接近 Qwen3.5（Retail 1 个写操作的题，0.72 对 0.82）。
- Telecom 的环境和用户模拟器能支撑完整的排障流程：Qwen3.5 在同样配置下全对。

### 15.6 加速配置核查

Qwen3.5-4B 是 24 层 Gated DeltaNet 加 8 层 full attention；Qwen3-4B 是 36 层全部 full attention。

**推理（SGLang）：**

| 配置                                         | Qwen3 上                              | 说明                                               |
| -------------------------------------------- | ------------------------------------- | -------------------------------------------------- |
| `attention_backend: flashinfer` + CUDA graph | 可用                                  | 本次评测就在用；FA3 只支持 Hopper                  |
| FLA / causal-conv1d                          | 不使用                                | 只服务于线性注意力层，装着也不影响                 |
| KV cache 容量                                | 每卡 37.4 万 token，Qwen3.5 是 155 万 | `mem_fraction_static=0.75` 时实测，小 4.2 倍       |
| `disable_radix_cache: true`                  | 可以打开，收益未测                    | dense 模型的多轮对话每轮都要重算整段历史的 prefill |

- 评测时每个 SGLang 实例只有 4–10 条请求在跑，`num_retracted_reqs = 0`，瓶颈在 DeepSeek 用户的响应延迟。
- 训练 rollout 时每个实例 64 条并发、上下文也更长，要关注 `num_retracted_reqs`，必要时调高 `mem_fraction_static`。

**Actor update（4 卡 FSDP，用 `actor_probe.py` 跑 3 步）：** 合成数据每卡 8 条序列、长 6K–16K token，每 4
条共享一段 3K token 的前缀。

| 变体                                           | 能否启动                 | token/s | 峰值显存 | 说明                                                                            |
| ---------------------------------------------- | ------------------------ | ------- | -------- | ------------------------------------------------------------------------------- |
| **现有配置**：FSDP + FA2 + grad ckpt + packing | 是                       | 13,435  | 39.7 GB  | "每个 microbatch 只放一条序列"的规则只对 `qwen3_5` 生效，Qwen3 自动恢复 packing |
| Qwen3.5 现有配置（对照）                       | 是                       | 16,106  | 36.3 GB  |                                                                                 |
| Ulysses SP=2                                   | 是                       | 12,258  | 28.2 GB  | 32K 以内用不上                                                                  |
| Tree training                                  | 需 `pad_to_maximum=True` | 7,249   | 54.5 GB  | padding 的开销大于共享前缀省下的计算                                            |
| Archon，默认 selective AC                      | 否，OOM                  | —       | —        |                                                                                 |
| Archon，`ac_mode=full`                         | 是                       | 12,980  | 37.6 GB  |                                                                                 |
| Archon，`ac_mode=full` + compile               | 是                       | 13,368  | 39.5 GB  |                                                                                 |

- **结论：沿用现有 FSDP 配置，不再加其他优化。**
- 在这个长度区间，Qwen3 每 token 比 Qwen3.5 慢约 17%，原因是 36 层 full attention 的计算量。
- **没有测的：**
  - FP8：A100 不支持。
  - Qwen3 的 critic 路径：`Qwen35TokenCriticForCausalBackbone` 只适用于 Qwen3.5，Qwen3 会走通用路径。用
    SAO 之前要单独验证。

### 15.7 后续：在 AReaL-tau2-data 上训练 Qwen3

**依据。** 12.9 的 813 题对 Qwen3.5-4B 太容易：mean@4 0.912，temperature 0 下 78% 的题 4 次全对。两个 Qwen3
模型在 fixed 上比 Qwen3.5 低 15–44pp，这份数据的难度对它们可能正合适。这是一个候选方向，还没有测。

**已测（2026-10-12）。** 改为在筛选后的 v2（648 题）和 Telecom v1 上评测，结果在 13.5 表 B。Instruct 在 v2 上 54%
的题有对有错，难度合适。

**训练前要处理的两点：**

- 15.4 的"用户在确认句里结束对话"会让 Qwen3 产生更多假负样本；
- Qwen3 critic 路径还没验证（15.6）。

### 15.8 局限

- 失败分类靠规则抽取（`aggregate.py`、`write_diff.py`）。人工精读了约 10 条轨迹，主要类别各 1–2 条，没有逐条标注。
- temperature 0 同配置重测约有 3pp 波动（13.1）。本节比较的差距都在 15pp 以上，不受影响。
- Actor probe 只跑了 3 步、用的是合成数据，吞吐只能做相对比较。

## 16. 研究方向讨论：RL 训练中如何对待用户模拟器的质量（2026-10-11）

起因：与老师讨论的方向是"现在的 RL 没有考虑用户"。设想是：给每个用户的反馈维护一个置信度；置信度高的反馈多学（大优势、小裁剪）， 置信度低的少学；PPO 中 critic
的 value 估计和 GAE 也要考虑用户。实验设计是：找一个弱用户、一个强用户，各训一轮 baseline， 再各训一轮新方法，预期新方法都更高。

本节最初回答的问题是"这个问题是否存在、是否是缺口"。讨论中方向几经调整，16.0 给出**当前**的问题定义和决策轨迹；16.1–16.8 是推导过程的记录，
其中被后来的决策推翻的地方在 16.0 的表里注明。还没有做实验，判断来自第 9、10、14 节已有的数据和文献检索。

### 16.0 当前问题定义与决策轨迹

**问题陈述**（2026-10-11；方法尚未确定）

用 LLM 用户模拟器训练多轮客服 agent 时，结局 reward 由 agent 和用户共同决定。用户可能在确认的同一句里结束对话、不提供必要的信息，或偏离自己的目标；
这时 agent 即使每一步都做对，也拿 0 分，这次失败还会作为负优势落在 agent 此前的正确行为上。在 τ² 上这个影响不小：我们的 GRPO 训练中，约 29%
的组内对比是 "agent 做对、用户出错"（Retail 为 42%，14.3）；同一个模型只把用户从 DeepSeek 换成 Doubao，Retail 成功率就从
0.851 降到 0.651（9.1）。 现有做法都假设可以重复或检查用户：丢弃 judge 判为用户出错的轨迹（AutoForge，judge 要看用户剧本），对用户模型做
SFT 或换更强的用户， 或对同一题采样多条、丢掉全对或全错的组。真实用户的在线训练不满足这些前提：每次对话只发生一次，不能重采样，也没有剧本可查；同一题只有一条
rollout， GRPO 的组内基准也就无从计算。我们研究这一设定下的信用分配：agent 应当为它能影响的结果负责，而不为用户单方面造成的成败负责。初步方向是在单条
rollout、 带 critic 的 PPO 类方法（SAO）上，利用用户会反复出现这一点，从同一用户的历史交互中在线估计其可靠度，并据此缩放该用户的对话产生的更新。
**核心困难是可辨识性**：一个用户的结果差，可能是因为他的反馈不可靠，也可能是因为 agent 还不会应对这类用户；前者应当少学，后者恰恰最该学。
因此，这个方向成立的前提是：找到一个只用线上可观测信息、能区分这两种情况的可靠度估计。做不到的话，降权会把训练推向容易的用户，方法不成立。

**设定要点：**

- 可用的信息：对话文本、结局 reward、同一用户的历史交互；不用用户的隐藏目标，不用带剧本的 judge。
- 测试平台：τ² Airline/Retail，强弱用户混合的用户池，由模拟器强制执行"每个（用户、题目）只交互一次、不重采样"的协议（16.9）。
- 方法草案见 16.11：按用户给 advantage 乘一个权重，权重取"agent 最后一次行动之后、结果还剩多少不确定性"的倒数。还没有实验。
- 必须区分的已有工作：P-GRPO（按用户平移基准）、RGPO（离线标注者可靠度）、AutoForge（带特权信息的整条 mask），见 16.10。

**决策轨迹：**

| #   | 决策                                                                                                             | 原因                                                                                                                                                                                                                       | 位置       |
| --- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 1   | 起点：老师提出按用户反馈的置信度调整 advantage、clip，critic 和 GAE 也考虑用户；弱、强用户各训 baseline 和新方法 | 观察到好的用户模拟器下分数高、差的用户下学不会                                                                                                                                                                             | 本节开头   |
| 2   | 离线真实数据按用户专业程度加权：不做                                                                             | 离线数据可以先清洗、改写、合成，PPO 的 advantage/clip 也用不上；τ² 没有专业程度这个维度                                                                                                                                    | 16.2       |
| 3   | 把"用户不好"拆成 A 随机性（方差）和 B 系统性偏离（偏差）                                                         | 两者解法不同：A 有经典降方差方法；B 是优化目标被带偏，样本再多也学错                                                                                                                                                       | 16.3       |
| 4   | "用户历史成功率"不单独作为置信度，列出 6 种候选信号                                                              | 成功率混入 agent 能力，粒度只到用户；按成功率给 advantage 加权已有人做（ProxMO 2602.19225）                                                                                                                                | 16.4       |
| 5   | 问题 A 不作为贡献                                                                                                | 自托管用户时可固定用户噪声，经典方法无偏可用；API 用户拿不到随机数，只能统计处理                                                                                                                                           | 16.3、16.8 |
| 6   | AutoForge 式整条轨迹 mask（E1）只作 baseline，且不是上界                                                         | AutoForge（2512.22857）已做；整条丢掉会丢掉出错前 agent 做对的部分。可重采样场景的真正上界是重采样到干净数据，同行已做很多                                                                                                 | 16.5       |
| 7   | 删去 E0（用户反应探针）和 E0′（运气占比）                                                                        | E0 是 benchmark 诊断，不检验方法；E0′ 在 API 用户下分不开运气和反应                                                                                                                                                        | 16.6       |
| 8   | **转向不能重采样的在线场景**                                                                                     | 可重采样时评审会问"为什么不直接拿更好的数据"；不能重采样时 E1/E3 不可用，同行工作少。GRPO 不可用，critic 成为必需，与 SAO 工作衔接；用户历史残差成为合理信号                                                               | 16.9       |
| 9   | Coverage, Not Targeting（2609.02417）不再引用                                                                    | 证据太弱：τ² 结论只来自一次 LoRA 更新，效应与评测噪声同级                                                                                                                                                                  | 16.8       |
| 10  | 不与 GiGPO 比较                                                                                                  | 它按完全相同的中间状态分组算 step 级优势，需要同一题多条 rollout；τ² 用户回复各不相同，状态几乎不重合；在线场景每题只有一条                                                                                                | 16.8       |
| 11  | 在线场景不撞车，继续；要正面回应三篇中度相关工作，并修改可靠度的定义                                             | 检索（16.10）没找到同时做到"单条 rollout + critic、用户反复出现、在线估计用户可靠度并缩放更新"的工作，但各部件都有人做过。$\bar r_u$ 有两个问题：critic 认出用户后残差均值趋于 0；降权"做不好的用户"会把训练推向容易的用户 | 16.10      |
| 12  | 方法草案：按用户给 advantage 乘权重，权重取 agent 最后一次行动后残差平方的倒数（带收缩、用历史对话算）           | 能改的只有 critic（只降方差）和乘系数（改目标）；残差均值会被 critic 吸收，残差平方不会；在"用户只加噪声"的模型下，乘正权重不改方向，逆方差权重方差最小                                                                    | 16.11      |

**下一步**：先把 16.10 的"不可靠和难怎么区分"想清楚，定下可靠度的定义；同时按 16.9 做"训练本身有效"的单条 rollout SAO
baseline（需要批准和 DeepSeek 费用）。

### 16.1 结论

- **问题存在**。训练里约 29% 的组内对比是"agent 做得对，但用户出了错"（Retail 42%，14.3），换一个用户模型 Retail 就差
  20pp（9.1）。
- **"用户不好"其实包含两个性质不同的问题**（16.3）：
  1. 用户的随机性让 reward 变得嘈杂。这是降方差的问题，经典 RL 有成熟方法，不算缺口。
  1. 用户系统性地不按应有的方式表现，导致 agent 学到错误的行为。这是优化目标被带偏的问题，我们有数据支撑，检索中没看到有人直接做。
- **最关键的未知数**：用户噪声到底有没有伤到训练结果。上一轮训练权重几乎没动（3.3），回答不了。要先用一个便宜的上界实验（16.6）确认，再设计方法。

### 16.2 "坏数据为什么不直接丢掉"

这个质疑要分场景回答：

| 场景                                       | 能不能直接丢 / 先洗干净                                                                                                                                       | 对本方向的意义                   |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 离线的真实用户数据（例如杨老师收集的数据） | 能。筛选、改写、合成更直接；而且这是离线数据，PPO 的 advantage 和 clip 用不上                                                                                 | 质疑成立，不宜作为 motivation    |
| 在线 RL + 用户模拟器（我们现在的场景）     | 用户好不好，要等这条对话跑完才知道；而且坏往往只坏在一轮：agent 前面 10 轮都对，用户最后一句提前结束，整条得 0。整条丢掉就是现在处理 infra 失败的做法（12.3） | 问题是怎么只扣掉用户那部分       |
| 在线 RL + 真实用户（将来的 online RL）     | 不能选用户，用户质量只能在线估计                                                                                                                              | 动机最强，但目前只能用模拟器近似 |

另外要注意：**弱用户造成的失败不全是噪声。**

- 用户不主动给信息、说法含糊时，agent 应该追问。这是该学的能力，真实客服的用户就是这样。如果对这类对话降权，agent 遇到弱用户只会更差。
- 只有 agent 无论怎么做都挽回不了的部分，才应该从 agent 的功劳或过错里扣掉。tau2 里的例子：用户确认的同一句就结束对话（9.1）、
  用户不按剧本走、用户抢先发 `###TRANSFER###`（14.2）。

所以方法必须区分"agent 能挽回的用户缺陷"和"挽回不了的用户缺陷"。

### 16.3 两个问题的拆分

#### 先说清楚几个概念

强化学习（RL）把问题建模成马尔可夫决策过程（MDP）。这是 RL 的标准建模方式，不是某个特定理论：

- **agent**：我们训练的模型，只控制自己说什么、调用什么工具。
- **环境**：agent 之外、会对 agent 的动作做出反应的一切。agent 每做一个动作，环境就返回下一个状态。
- **状态** $s_t$：到第 $t$ 步为止的全部对话历史，加上数据库状态。
- **动作** $a_t$：agent 这一轮的回复或工具调用。
- **转移**：动作之后状态怎么变。在 tau2 里，工具调用的返回是确定的（查数据库）；用户的回复是用户模拟器采样出来的，是随机的。

所以"用户是环境的一部分"的意思是：从 agent 的角度看，用户的回复和工具的返回一样，都是"我做了动作之后，外界给我的反应"。 agent 控制不了用户说什么，只能影响它。

训练的目标是：

$$
J(\theta) = \mathbb{E}_{\text{题目}}\ \mathbb{E}_{\text{agent 采样},\ \text{用户采样}}\big[R\big]
$$

即在很多道题上，agent 和用户都按各自的概率说话时，平均能拿到多少 reward $R$。这就是"对用户分布取期望"：我们关心的是 agent
面对这类用户的**平均**表现，而不是某一次碰巧遇到的那句用户回复。

#### 问题 A：用户的随机性让 reward 变嘈杂（方差问题）

**例子**。Retail 的一道换货题，agent 列出了换货的全部细节，请用户确认。同一个前缀，用户有两种可能的回复：

- 一半概率回 "yes"，agent 接着执行换货，reward = 1；
- 一半概率回 "Yes, I confirm. ###STOP###"，对话结束，没执行写操作，reward = 0（9.1 中 Doubao 用户的典型行为）。

agent 在这两条轨迹里说的话完全一样，做得也都对。GRPO 用组内平均做基准：组内平均 0.5，第一条的优势是 $+0.5$，第二条是 $-0.5$。 同样的 agent
行为，一次被鼓励，一次被惩罚。

**为什么说这是方差问题，而不是方向错了。**

- 策略梯度是很多样本的平均。把足够多的样本平均起来，$+0.5$ 和 $-0.5$ 会互相抵消，这道题上"确认"这个动作的净信号是 0。 它在这个用户下确实只有 50%
  的成功率，所以净信号为 0 在数学上是对的（无偏）。
- 问题是每一条样本的信号都很大、符号还相反，要很多样本才能平均出可靠的方向。样本不够时，梯度主要在追噪声，这就是"方差大"。 我们每步只有约 1.3 个有效对比组（14.3
  第 7 项），样本远远不够。

**经典的降方差办法：基准（baseline）。**

- 优势 = 实际 reward − 基准。基准越接近"这条轨迹本来能拿多少"，优势里剩下的噪声越少。
- PPO 的 critic 就是一种基准：$V(s_t)$ 预测"从状态 $s_t$ 开始平均能拿多少"。
- 但在上面的例子里，$V(s_t)$ 是在 agent 发出确认请求时计算的，它看不到用户下一句会不会提前结束，所以两条轨迹的基准相同，噪声一点没减。

**经典 RL 怎么处理这种外部随机性。**[Mao et al., ICLR 2019](https://arxiv.org/abs/1807.02264)
研究的是"环境里有一个外部随机输入" 的问题（例如排队系统里随机到达的请求）。他们让基准额外看到**之后实际发生的外部输入**：如果基准知道"这次用户会提前结束"， 它就会预测
reward ≈ 0，第二条轨迹的优势 ≈ 0，agent 不再被冤枉。论文证明，**只要外部输入不受 agent 动作影响**，这样做不会把梯度带偏。

**难点就在这个前提**。用户的回复由两部分决定：用户模拟器自身的采样随机性，以及它对 agent 说的话的反应。

- **这两部分在定义上是可分的**。用户模拟器每一轮的回复，可以写成 $u_t = f(\text{历史}, \epsilon_t)$：$\epsilon_t$
  是采样用的随机数（随机种子）， 与 agent 无关；"历史"里包含 agent 说过的话。Mao et al. 要求的"外部输入与动作无关"，对 $\epsilon_t$
  成立，对 $u_t$ 本身不成立。
- **在我们的训练数据里它们是混在一起的**。我们只记录了用户说出的文字 $u_t$，没有记录 $\epsilon_t$；用户模拟器是外部 API，也拿不到种子。
  所以不能直接套用 Mao et al.：把"用户这次结束了"当作外部输入去扣，就默认了"结束"与 agent 的说法无关。
- **"结束与 agent
  说法无关"这个假设，现有数据既不能证实也不能否定**。粗略的相关性统计（`scripts/lead_confirm_stop.py`，Retail， agent
  请求确认后用户回复肯定的轮次）：Doubao 用户在同一句里结束的比例是 90/553（16.3%），DeepSeek 是 20/627（3.2%）。 按 agent
  的说法拆分：确认里列出 3 条以上操作时，Doubao 17.6%（78/444），否则 11.0%（12/109）；agent 写明"你确认后我再执行"的只有 8
  条，其中 1 条结束。这说明用户是否结束可能和 agent 的说法有关，但样本小、而且混着题目难度，不能当作因果证据。 要回答这个问题，需要对同一个前缀只改 agent
  的说法、各采样多次用户回复，看结束率是否变化。这是 benchmark 诊断，不是方法实验（16.6 已从实验表中删去）。

所以，"可分不成立"的准确说法是：**理论上可分，但我们观测不到用户的随机种子，只能从文字回复里去估计哪部分是运气、哪部分是反应**。这正是要解决的问题。 它有两条路：

1. **工程上直接把两部分分开**：只在自己部署的用户模拟器上可行。由我们控制它的随机数，让用户每个 token 的采样噪声只由（episode、用户第几轮、第几个
   token） 决定，与 agent 说了什么无关（Gumbel-max 采样加按事件编号的随机数，见 16.8）。这样噪声成了与动作独立的外部输入，Mao et al.
   的结论直接适用： 同一组 rollout 共用同一份用户噪声，组内对比就去掉了用户运气。
   - **调用 API 的用户（DeepSeek、Doubao）做不到**：拿不到它的随机数，T=0 下剩下的随机性来自服务端，只能事后做统计，不能控制。
   - 对话一旦分叉，同一份噪声对应的用户回复就不再可比，长对话里效果会变弱。
1. **从数据里学**：让 critic 在用户回合前后各估一次 value（16.4 的 $\Delta_u$），把用户回合带来的变化单独拿出来。难点是 critic
   本身要估得准，而且它同样分不清"用户运气差"和"agent 上一句话引出了这个反应"。经典结论是：用用户回复的文字做基准的条件是有偏的， 因为文字依赖 agent
   的动作（16.8）。

结论：在我们用 API 用户的设置下，问题 A 分不开，只能在统计意义上处理；在自托管用户下有现成的无偏做法，不算贡献。

#### 问题 B：用户系统性偏离应有的行为（目标被带偏）

- tau2 的用户指令要求用户在任务完成后才结束对话（9.1），所以"确认的同一句就结束"是用户模拟器违反了指令。 Doubao 这样做的频率是 DeepSeek 的 4.5
  倍。
- 如果训练时用的就是这样的用户，那么在这个用户下平均 reward 最高的策略是"少确认、直接写"，而 retail policy 要求写前必须确认。
- 这就不是噪声能平均掉的问题了：样本再多，学到的方向也是错的。原因是我们想要的是"面对按指令行事的用户时表现好"，采样却来自一个不按指令行事的用户。
- 这里给用户回复打置信度有明确的含义：衡量"这一轮用户回复有多符合它应有的行为"，并据此修正 agent 学到的信号。

问题 B 必须打赢一个简单的 baseline：在 rollout 中检测到不合格的用户回复（例如确认的同一句里结束），就丢掉这句、让用户重新回复一次。
这是对用户回合做拒绝采样，发生在训练采样阶段，不需要改 advantage 或 critic，相当于把"数据清洗"放进 rollout 过程里。 它与 16.2
的质疑是同一个问题：如果重采样就够用，加权方法就没有必要。它的局限是只能处理检测得出来的失败类型，而且改变了用户分布（训练时的用户比评测时的更守规矩）。

### 16.4 用户置信度可以从哪里来

置信度不止一种算法。下表列出候选信号，以及它们在模拟器和真实用户场景下是否可用：

| 信号                                        | 衡量什么                                                           | 模拟器     | 真实用户 | 问题                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------ | ---------- | -------- | --------------------------------------------------------------------------------------------------- |
| 用户过去任务的成功率                        | 这个用户整体上好不好合作                                           | 可用       | 可用     | 混入了 agent 的能力（agent 变强，所有用户的成功率都上升）；只有用户级粒度，分不出是哪一轮出的问题   |
| 规则标签                                    | 已知的失败模式：确认后立刻结束、首句就结束、抢先转人工、中途换语言 | 可用       | 部分可用 | 只能覆盖已知类型                                                                                    |
| 对照用户的隐藏指令判断忠实度（LLM judge）   | 这一轮用户回复是否符合剧本                                         | 可用       | 不可用   | 真实用户没有剧本；judge 本身有噪声                                                                  |
| critic 在用户回合前后的 value 差 $\Delta_u$ | 这一轮用户回复让成功的希望升高还是降低                             | 可用       | 可用     | 依赖 critic 质量（现在 SAO critic 的 EV 为负，3.4）；分不出"用户的错"和"agent 上一轮引出了这个反应" |
| 同一前缀重采样多次用户回复                  | 这一步的结果有多大程度取决于用户运气                               | 可用       | 不可用   | 费用随重采样次数线性增加                                                                            |
| 反馈的专业程度                              | 用户是不是领域专家                                                 | 基本不可用 | 可用     | tau2 的用户只照剧本陈述事实，没有专业程度这个维度；要做就得构造 persona 或换 benchmark              |

关于第一行：[Proximity-Based Multi-Turn Optimization（2602.19225）](https://arxiv.org/abs/2602.19225)
已经按成功率给 advantage 重新加权。
我只读了摘要，没有核对它的成功率是按题、按步还是按用户算的。不管哪种，"用成功率做权重"本身已经不新，新方法的置信度需要来自其他信号， 或者把多种信号结合起来。

$\Delta_u$ 这一行需要单独说明。

**先说 critic 在哪里输出 value**。critic 和 actor 一样是 causal transformer，只是最后一层换成输出一个数的 value
head。它对整条对话 （system、user、assistant、tool 的全部 token）做一次前向，每个位置都输出一个
value，表示"看到这里为止的内容，预计最终能拿多少 reward"。 现在训练时只用 assistant token 位置上的
value（`areal/trainer/ppo/gae.py` 中 `loss_mask` 为 0 的位置被跳过），其余位置的输出被丢弃。

**$\Delta_u$ 不需要让 agent 在用户回答前多输出任何东西，也不需要 mask 掉用户 token**。做法是：

$$
\Delta_u = V(\text{用户这一轮最后一个 token}) - V(\text{agent 上一轮最后一个 token})
$$

- 两个 value 来自同一次前向，只是取不同位置的输出，不增加计算量。
- "agent 上一轮最后一个 token"处，critic 还没看到用户的回复；"用户这一轮最后一个 token"处，critic 已经读完了用户的回复。
  两者之差就是这句用户回复让预期 reward 变了多少。
- 用户 token 的概率高不高不影响这件事。value 不是 token 的概率，而是 critic 读完这一段后的判断；用户 token 不参与 actor 的
  loss， 但照样作为输入被 critic 读到，causal attention 保证每个位置只用到它之前的内容，因果关系是对的。
- 用户结束对话时，最后一句（含 `###STOP###`）之后没有 assistant token，但这句用户回复本身有 token，可以在它的最后一个 token 上取
  value。

**会不会算不准**。会，而且这是这条路线的主要风险：

- 现在 critic 从来没有在用户 token 位置上被训练过，这些位置的输出没有意义。要让 $\Delta_u$ 可用，critic 的训练目标必须覆盖用户回合结束的位置
  （用同样的最终 reward 做回归目标）。这是一个小改动，但没有验证过。
- 现在的 SAO critic 在 assistant 位置上的 explained variance
  都是负的（3.4），也就是还不如常数预测。在它学会之前，$\Delta_u$ 只会是噪声。
- 即使 critic 准确，$\Delta_u$ 衡量的也是"这句用户回复让结果变好还是变坏"，不区分是用户自己的运气，还是 agent 上一句话引出的反应。 后者应该算
  agent 的功劳或过错。所以 $\Delta_u$ 只能直接用于明显与 agent 无关的回合（例如首句），其余回合目前没有可靠的区分办法。

最接近的前人工作是
[SWEET-RL（Meta，2503.15478）](https://arxiv.org/abs/2503.15478)：它在与人类模拟器协作的多轮任务上，只给 critic
提供训练时才有的额外信息（参考答案），用来做逐轮的信用分配。它假设用户是可靠的；我们的区别是把"用户不可靠"作为建模对象。写论文时必须和它正面比较。

### 16.5 实验设计：simple but work

原设计（弱用户、强用户各训一轮 baseline，再各训一轮新方法）有两个问题：

1. 如果置信度按用户模型来算，一次训练里只有一个用户，所有样本的置信度都相同，方法就退化成调学习率或 clip 系数。置信度必须按每条对话或每一轮来估计。
1. 评测用哪个用户要事先固定。弱用户训练、弱用户评测，提升可能只是适应了这个用户的毛病。至少要在强用户上评测，最好再加一个训练中没见过的用户。

在这两点之内，最简单的设置是：

| 项       | 设置                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 模型     | Qwen3-4B-Instruct-2507（fixed 下 Airline 0.225 / Retail 0.410，15.1；有对有错的题多）                                          |
| 题集     | fixed v2 的 train pool 99 题，或 12.9 的 AReaL task pack 中按 Qwen3 筛过的题                                                   |
| 训练用户 | 强：DeepSeek-Flash；弱：Doubao 2.1 Turbo（同代码下 Retail 低 20pp，主要原因是确认后结束，9.1）                                 |
| 评测用户 | 固定 DeepSeek-Flash；另加一个训练中没用过的用户                                                                                |
| 算法     | GRPO（无 critic，最简单）。只有 GRPO 上的版本有效后，再上 SAO                                                                  |
| 方法     | 最简单的版本只改一处：被判为"用户造成的失败"的样本优势置 0。判定先用规则（确认后结束、首句结束、抢先转人工），不训练任何新模块 |

**这个最简单的方法已经有人做过**（16.8）：AutoForge（2512.22857）用 LLM judge 判断用户是否给出幻觉或遗漏的信息，把判错的整条轨迹 从
advantage 和 loss 中去掉，在 τ² 上训练时更稳定。所以 E1 在论文里只能当 baseline，不是贡献。

它**不是**这一类方法的上界（此前写成"上界"是错的）：

- 整条丢掉会连出错之前 agent 做对的部分一起丢掉。"确认后结束"这类失败里，agent 在用户出错之前已经做对了，正确的处理是给正 credit，而丢掉只给 0。
- judge 本身有误判，丢掉样本后组变小，advantage 的方差变大。

可以重采样的场景里，真正的上界是"用完美的检测器把出错的用户回合重采样到合格为止"，也就是拿到干净数据。这条路同行已经做了很多 （AutoForge、修用户的
AReaL-SEA、多样化用户的 2608.12253），评审会问"为什么不直接拿更好的数据"。这是 16.9 转向不能重采样场景的原因。

### 16.6 实验顺序

**第 0 步：先证明训练本身有效**。上一轮在小批量数据上训练，效果是退化或看不出来（第 3 节），原因是评测噪声、有效信号少、学习率太小权重几乎没动。
在这个前提下比较"有没有用户处理"，差异会被噪声淹没。所以第一件事是在修复后的数据上，用 Qwen3-4B、GRPO、DeepSeek 用户训一个能看出提升的
baseline：test 上 mean@4 的配对提升要超出 13.1 测得的约 3pp 运行间波动。做不到，就先解决训练本身的问题，用户方向的实验都往后放。

**第 1 步：三个实验并行**。baseline 有效之后，在同一套配置上同时跑：

| 实验                             | 做法                                                                    | 判断                                                            |
| -------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------- |
| E1 整条轨迹 mask（AutoForge 式） | 16.5 的最简单方法，DeepSeek 与 Doubao 用户各训一轮                      | 都没有提升：方向不成立，换 idea；有提升：它是要打赢的 baseline  |
| E3 用户回合拒绝采样              | 检测到不合格的用户回复就让用户重说一次，DeepSeek 与 Doubao 用户各训一轮 | 新方法必须超过它；如果它已经接近 E1，说明问题用工程手段就能解决 |
| E2 用户质量的损失                | 不加处理的 Doubao 训练对比 DeepSeek 训练（与第 0 步的 baseline 共用）   | 两者的差就是用户质量造成的训练损失上界                          |

此前列的 E0（只改 agent 说法、看用户结束率是否变化）和 E0′（运气占比）已删去：

- E0 是评测性质的实验，回答的是"τ² 用户模拟器在确认后结束是不是由 agent 的说法引起"，属于指出 benchmark 问题，不检验方法。需要时放进
  benchmark 分析部分。
- E0′ 在 API 用户下分不开用户的运气和反应，再怎么控制都有很大的随机性，测出来的数字解释不了什么。

**第 2 步：根据结果决定。**

- E1 和 E3 都没有提升：用户噪声不是当前训练的瓶颈，换问题。
- E3 与 E1 差不多：拒绝采样已经够用，在可重采样的场景里研究价值有限（见 16.9）。
- E1 明显好于 E3：用户造成的失败只能事后识别、不能靠重采样避开，这时才值得设计 $\Delta_u$ 或置信度方法去逼近 E1。

前提：

- 学习率要大到权重确实会变化。上一轮 1e-6 时权重的变化接近随机游走（3.3）。
- 用 Qwen3-4B 而不是 Qwen3.5。Qwen3.5 在 Retail 上已接近满分（0.849），Qwen3 有对有错的题更多（15.1）。
- 代价是第 0 步 1 个训练、第 1 步约 4 个训练，都需要 DeepSeek 和 Doubao 费用，需要批准后启动。

如果按 16.9 转向不能重采样的场景，E1、E3 不再是要比较的方法，第 1 步改用 16.9 的实验。

### 16.7 判断

| 版本                                                                                                  | 是否成立   | 是否新                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 用户噪声大，需要降方差                                                                                | 成立       | 不新：问题已被测量（τ² 原文统计过用户模拟器出错率；UserProxyBench 只换用户 reward 变 15.2 分），修法有加大 batch、修用户（AReaL-SEA）、多样化用户（2608.12253） |
| 检测出用户出错的轨迹，整条去掉                                                                        | 成立       | 不新：AutoForge MEU                                                                                                                                             |
| 按全局噪声率修正 advantage                                                                            | 成立       | 不新：Noisy RLVR 的 forward/backward correction（2510.00915），但它的噪声来自 verifier，且与策略无关                                                            |
| 按轮次定位用户出错，只扣掉 agent 挽回不了的部分，出错之前和 agent 能影响的部分照常学；用于 critic/GAE | 有数据支撑 | 检索中没找到（16.8）。建议用这个版本                                                                                                                            |
| 离线真实用户数据按专业程度加权                                                                        | 质疑成立   | 数据清洗是更直接的解法；tau2 也没有专业程度这个维度                                                                                                             |

### 16.8 相关工作（2026-10-11 检索）

三路检索各一个 agent，结果在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-capability-gap-20261011/out/`：
`related_user_rl.md`（带用户模拟器的 agent RL、反馈可靠度加权）、`related_credit.md`（多轮 credit
assignment、critic、自适应 clip）、 `related_classical_rl.md`（经典 RL 中的外部噪声与反事实 baseline）。下表中我打开
arXiv 页面核对过的标"已核对"，其余以 agent 报告为准。

**最接近的工作：**

| 工作                                                                                         | 做了什么                                                                                                                                                                                                                | 与本方向的关系                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AutoForge（2512.22857）](https://arxiv.org/abs/2512.22857)，已核对 HTML 正文                | LLM judge 判断用户是否给出幻觉或遗漏信息，判错的整条轨迹不进 advantage 和 loss，不重采样；std 在整个环境上算。Qwen3-30B-A3B τ² Retail/Airline 58.8/58.0 → 74.8/62.0；去掉 mask 后训练曲线后期下降（只有曲线，没给数字） | 覆盖了"轨迹级、二值、把用户出错从 advantage 去掉"。未覆盖：按轮次定位（确认后结束这类失败，出错前 agent 已经做对）、区分 agent 能否影响、critic/GAE。它的 judge 只查幻觉和遗漏，不查确认后结束 |
| [FACA（2608.17499）](https://arxiv.org/abs/2608.17499)，已核对摘要                           | 把用户下一句回复当作对前一段 agent 行为的局部评价，算出的 advantage 加到终局 advantage 上，记在 agent 名下。8B/14B 在 τ 系列 9 个 domain 平均 +5.9/+10.2pp，主要来自 Telecom                                            | 方向相反：它把用户反应当作 reward 来源，本方向把其中与 agent 无关的部分当作噪声。合起来可以表述为"用户反应 = 对 agent 的评价 + 用户自身噪声"                                                   |
| [Noisy RLVR（2510.00915）](https://arxiv.org/abs/2510.00915)，已核对摘要；UARM（2606.19818） | 把 verifier 错误建模为有固定假阳性/假阴性率的信道，给出 GRPO 中的无偏修正；UARM 按单样本 RM 不确定度加权 advantage                                                                                                      | 提供了修正的数学形式。"用户出错让正确的 agent 得 0"对应假阴性；但它们假设噪声与策略无关，而用户是否出错部分取决于 agent                                                                        |
| Off-policy DAE（ICLR 2024）                                                                  | 把回报分解为 agent 的 skill 和环境的 luck，luck 项 $V(s') - \mathbb{E}[V(s')]$ 用 CVAE 估计                                                                                                                             | $\Delta_u$ 在理论上就是这里的 luck 项。LLM 对话中用户回合的位置已知，不需要 CVAE，这是可以主张的简化                                                                                           |
| [SWEET-RL（2503.15478）](https://arxiv.org/abs/2503.15478)                                   | 非对称 critic，训练时给 critic 参考答案，逐轮 credit                                                                                                                                                                    | 可借用结构：给 critic 用户的隐藏剧本和可靠度。它假设用户可靠                                                                                                                                   |

tau2 上目前报告的 4B 结果：IAPO（2608.24588）Qwen3-4B 38.4，GRPO 34.8（agent 报告，未核对）。

**经典 RL 中的对应理论**（`related_classical_rl.md`，14 篇读了方法和定理）：

| 工作                                                                                                                                                               | 结论                                                                                                                                                                                                   | 对本方向的含义                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| [Mao et al., ICLR 2019](https://arxiv.org/abs/1807.02264)；Counterfactual Credit Assignment（Mesnard et al. 2021，[2011.09464](https://arxiv.org/abs/2011.09464)） | 基准可以以未来信息为条件而不引入偏差，前提是这份信息与 agent 的动作独立                                                                                                                                | 以用户的**采样噪声**为条件：无偏；以用户**说出的文字**为条件：有偏，因为文字依赖 agent 说了什么                         |
| COMA（反事实多智能体 baseline）                                                                                                                                    | 固定另一个智能体的动作、对自己的动作求平均                                                                                                                                                             | 假设双方同时行动。用户是在 agent 之后回复的，直接套用有偏                                                               |
| Gumbel-max 结构因果模型（Oberst & Sontag 2019、Chatzi et al. 2025）；按事件编号的随机数（2603.11084）                                                              | LLM 采样可以写成 $\arg\max(\log p + g)$，噪声 $g$ 是外部变量；随机数要按事件编号生成，不能按"第几次调用"生成                                                                                           | 自托管用户时，可以让用户噪声在构造上与 agent 动作独立，同时用户的行为分布不变                                           |
| [Luck Is Not Skill（2609.24144）](https://arxiv.org/abs/2609.24144)，已核对摘要                                                                                    | GRPO 组内共用同一份环境噪声（paired rollouts），用 leave-one-out 基准时无偏，除以组内 std 会破坏这个性质。2B 工具 agent：工具故障下最终成功率 +5.1pp，但没达到预注册的学习曲线判据；评分翻转下基本无效 | 最接近的方法论先例，噪声来自工具和评分，不是用户。它也说明：reward 对比的方差下降，不等于梯度方差下降，更不等于学得更快 |

**经典方法在我们场景里没有解决的问题：**

1. **我们拿不到用户的噪声**。用户是 DeepSeek API、temperature 0，剩下的随机性来自服务端的非确定性（14.3：同组首句用户消息 386/386
   组都不全相同），我们控制不了。
1. **"运气"的分解不唯一**。同一个用户分布可以用不同方式写成"确定函数 + 噪声"，每种都无偏，但给出不同的运气/能力划分。
1. **对话分叉后共用噪声失效**。agent 第一轮说法不同，用户后面每一轮的上下文就不同，同一份噪声对应的回复不再可比。
1. **只去噪声，不去偏差**。用户模拟器系统性的毛病（过度配合、确认后结束）仍然保留，这是问题 B，经典方法不处理。

所以问题 A 在自托管用户下有现成的无偏做法；我们能贡献的是问题 B，以及在拿不到用户噪声时（API 用户、真实用户）如何估计。

**检索中没找到的**（检索次数有限，"没找到"不等于"没有"）：

1. 把用户出错的轮次当作截断点：之后的结果不传回给 agent，用 $V(s_k)$ bootstrap 代替 0 reward，出错之前的 agent 行为得到
   critic 估计的 credit。 现有做法只有"整条丢弃"（AutoForge）和"整条计 0"（其余所有）。
1. 区分"agent 能影响"和"外生"的用户错误：固定 agent 前缀、重采样 K 次用户回合估计 $P(\text{用户出错} \mid \text{前缀})$，
   这同时也是一个"只重采样用户回合"的 baseline，也没有公开工作。它要求能重采样，只适用于模拟器场景。
1. 按用户或环境可靠度设置 clip。现有自适应 clip 用的是 token 概率、entropy、information gain。
1. "弱用户加修正能否追平直接换强用户"这个对照。多篇工作报告强用户训练更好（AutoForge 换 GPT-5-thinking 用户 Telecom
   +14，AReaL-SEA 用 SFT 后的用户 +20），但没人回答损失有多少能靠修正 credit 挽回。

**对实验设计的修正：**

- E1 改称"AutoForge 式整条 mask"，作为 baseline（已改入 16.5、16.6）。
- 不再引用 Coverage, Not Targeting（2609.02417）：τ² 结论只来自一次 policy 更新（LoRA），主结果
  +3.6pp，作者报告的评测噪声与效应同级（一个 seed 重评测变化 +0.159），4B 上无差异。
- 不与 GiGPO（[2505.10978](https://arxiv.org/abs/2505.10978)，2025-05，"Group-in-Group Policy
  Optimization"）比较。它不训练 critic， 在同一组 rollout 中找完全相同的中间状态（anchor state），在这些状态上比较后续回报，得到
  step 级优势；在 ALFWorld、WebShop 上有效。 τ² 里用户回复各不相同，中间状态几乎不会完全重合，它退化为 GRPO；在线场景每题只有一条
  rollout，更用不了。IAPO 论文里 Qwen3-4B 上 GiGPO 37.2、 GRPO 34.8（agent 报告），差距在噪声量级。
- 做 $\Delta_u$ 之前，先按用户回合边界分桶报告 SAO critic 的 EV；critic 在中间状态预测偏平时，$\Delta_u$ 会被压向 0。
- UserProxyBench 报告成功 episode 中 24.4% 含用户违规，与 14.3 的 29% 可以互相印证，可写入 motivation（agent
  报告，未核对）。

文献部分除标注"已核对"的以外，都以检索 agent 的报告为准，引用前要核实全文。

### 16.9 转向不能重采样的场景（候选，2026-10-11 讨论）

**为什么转**。可以重采样用户的场景（离线模拟器训练）里，"用户不好"有直接的工程解法：检测、丢弃、重采样、换更强的用户、修用户。
同行已经做了很多（16.8），评审会问"为什么不直接拿更好的数据"。在线场景里这些都做不到：真实用户每次对话只发生一次，不能重来，也没有剧本可以对照。
这里做的工作更少，差异也更容易做出来。

**场景定义**（实验时用模拟器强制执行这些约束）：

| 约束                         | 含义                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------- |
| 每个（用户、题目）只交互一次 | 不能重采样用户回合，不能对同一用户、同一题跑一组 8 条                           |
| 方法只能用线上可观测的信息   | 对话文本、结局 reward、同一用户的历史交互；不能用用户的隐藏剧本或带剧本的 judge |
| 用户反复出现、有 ID          | 同一用户的历史表现可以积累                                                      |
| 用户质量有高有低             | 用户池混合强、弱用户                                                            |

**这个定义带来的几个直接后果：**

1. **GRPO 用不了**。它的基准是同一题的组内平均，要求同一题采样多次。不能重采样时只能用 critic 当基准，单条 rollout 的 PPO/SAO 是自然的选择
   （SAO 单条 rollout 的分析见
   `docs/research/2026-09-23-sao-single-rollout-async-is.md`）。critic
   就必须吸收用户带来的变化，这让"critic 怎么考虑用户"从可选项变成必需。
1. **E1（AutoForge 式 mask）、E3（拒绝采样）不再是要比较的方法**：前者要带剧本的
   judge，后者要重采样，线上都没有。它们只能作为"有特权信息时能做到多好"的参照。
1. **老师提的"用户历史成功率"在这里成了合理的信号**，因为它是线上能观测到的少数信息之一。它的问题（16.4）是混入了 agent 的能力。一个具体的修正： 用
   critic 的预测扣掉 agent 的部分，用户可靠度取这个用户过去对话的残差平均 $\bar{r}_u = \text{mean}(R - V(s_0))$。
   $V(s_0)$ 随 agent 变强而上升，残差只剩用户相对平均用户的好坏。
1. **要比较的方法变成**：普通 critic 的 PPO/SAO；按估计噪声率修正 advantage（Noisy RLVR
   2510.00915，噪声率在线估计）；把用户反应当 reward 的 FACA（2608.17499）；按用户分组归一化 advantage。

**最简单的实验**：

| 项       | 设置                                                                                                   |
| -------- | ------------------------------------------------------------------------------------------------------ |
| 用户池   | DeepSeek（强）、Doubao（弱），可再加改了 persona 的变体（例如更容易提前结束、不主动给信息），每个带 ID |
| 训练     | 每个（用户、题目）一条 rollout，单条 rollout 的 SAO                                                    |
| baseline | 同样设置的普通 SAO                                                                                     |
| 方法     | 在 advantage 或 clip 上乘一个由 $\bar{r}_u$ 得到的用户可靠度；先不改 critic 结构                       |
| 评测     | 固定 DeepSeek 用户，另加一个训练中没见过的用户；同时报用户池平均                                       |

前提仍是 16.6 第 0 步：先证明训练本身有效。这里换成单条 rollout 的 SAO，所以第 0 步的 baseline 也要用 SAO 做。

**风险：**

- **仍然是模拟**。评审可能说"在模拟器里明明可以重采样"。回答只能是：约束来自目标部署场景，论文里必须把协议写清楚，并说明为什么不用特权信息。
- **reward 从哪来**。真实在线场景没有 DB 终态检查。先用 DB
  reward，当作"延迟到达的真实结局"，保持简单；之后可以换成用户自己报告的满意度，噪声更大，也更贴近在线。
- **相关工作还没查**。这个场景下最相关的是从用户反馈做在线 RL（如 RLUF）、带不同水平标注者的 RLHF、continual / online RL，16.8
  的检索没有覆盖，转向前要补。
- **critic 现状**。现在 SAO critic 的 EV 为负（3.4），$\bar{r}_u$ 依赖 $V(s_0)$，critic 学不好时这个信号也不可靠。

### 16.10 在线场景的撞车检查（2026-10-11 检索）

一个检索 agent，38 次检索，结果在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-capability-gap-20261011/out/related_online_user_rl.md`
（26 篇，按撞车程度分级，高、中度的读了方法部分）。P-GRPO、RGPO 的编号和标题我用 arXiv 页面核对过，其余以 agent 报告为准。

**结论：没有高度撞车的工作**。没有一篇同时做到：每个（用户、题目）只有一条 rollout 并用 critic；用户反复出现、带 ID；在线估计每个用户的可靠度，并用它缩放
advantage、clip 或 critic 目标。但每个部件都有人做过，评审可以把本方向拆成"AutoForge 的软化在线版 + P-GRPO 的 critic 版 +
RGPO 的 RL 版"，必须正面回答这个拆分。

| 工作                                                                                   | 已覆盖的部分                                                                                                                        | 没覆盖的部分                                                                                  |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| [P-GRPO（2603.10009，Apple，2026）](https://arxiv.org/abs/2603.10009)，已核对标题      | 按用户或用户组 ID 在线维护 reward 的均值和方差，advantage 取 $(R-\mu_u)/\sigma_u$。Qwen3-8B MovieLens 0.719 → 0.757；随机分组无增益 | 目的是偏好不同，不是用户可靠；是平移和归一化，不是降权；仍要同一 prompt 采样多条；没有 critic |
| [RGPO（2607.20515，2026）](https://arxiv.org/abs/2607.20515)，已核对标题；Crowd-PrefRL | 估计每个标注者的可靠度，按它给样本加权                                                                                              | 离线；靠多人标注同一样本来估计；不是 RL 交互                                                  |
| AutoForge MEU（2512.22857）                                                            | 在 τ/τ² 上去掉用户造成失败的轨迹                                                                                                    | judge 看用户意图（特权信息）；按 episode 0/1；要组采样；不跨 episode 积累用户信息             |
| FACA（2608.17499）                                                                     | 每轮用户反应进入 advantage，τ 系列                                                                                                  | 读模拟器私有字段；没有可靠度；GRPO                                                            |
| SAO（2607.07508）、CLaaS                                                               | 单条 rollout 加 critic，一次性场景                                                                                                  | 没有用户维度                                                                                  |
| RLHI（2509.25137）                                                                     | 在真实用户日志上按用户质量过滤很关键（+23.4，不过滤 +2.5）                                                                          | 离线、LLM 判断、硬过滤                                                                        |

**仍然空着的：**

1. 单条 rollout 下，只用用户 ID、可观测对话和结局，跨 episode 在线估计用户可靠度。
1. 用可靠度连续缩放更新，并在某个用户样本还很少时向总体均值收缩（empirical Bayes shrinkage）。
1. 把用户池的构成（提前结束、隐瞒信息、偏离目标的比例）作为受控变量。
1. **区分"用户不可靠"和"用户难但有信息量"**。这一点决定方法是否成立，是这个方向真正的研究问题：agent 在难用户上做得差，恰恰最该学。

**对 16.9 方案的两处修正**（推导见 16.11）：

- 最初设想的可靠度 $\bar r_u = \text{mean}(R - V(s_0))$ 不可用：critic 训练好之后，它对每个用户都趋于 0（16.11 第 2
  步）。
- 降低某些用户的权重，等于改变训练目标中各用户的比重，可能把训练推向容易的用户（16.11 第 1 步）。评测必须按用户类型分别报告，并报告最差用户组的成功率。

**必须比较的 baseline**（都满足"单条 rollout、不用特权信息"）：

| #   | baseline                                      | 回答的问题                         |
| --- | --------------------------------------------- | ---------------------------------- |
| 1   | 不做用户加权的单条 rollout SAO                | 主 baseline                        |
| 2   | REINFORCE++，batch 全局归一化，无 critic      | 没有 critic 的下界                 |
| 3   | 按用户的 running baseline（P-GRPO 单条版）    | 缩放是否优于平移                   |
| 4   | AutoForge 式 judge mask：看剧本版、不看剧本版 | 有特权信息时的参照；现实可行的版本 |
| 5   | 逆方差加权，不用用户 ID                       | 是否需要用户 ID                    |
| 6   | 以用户 ID 为条件的 critic，不缩放             | 把可靠度吸收进 $V$ 是否就够了      |
| 7   | 按用户可靠度的样本权重（RGPO 式，带收缩）     | 与已有可靠度加权的区别             |

baseline 数量多，第一版只需要 1、3、6 和我们的方法，分别回答"有没有用""缩放还是平移""critic 自己能不能吸收"。

### 16.11 方法设计（草案，2026-10-11）

本节从需求出发，推出一个最简单、数学上说得通的做法，并写清它在什么条件下成立、什么条件下不成立。不追求严格无偏或最小方差，剩下的偏差和方差留给工程手段处理。

全节的顺序是：第 1 步找出能改的地方；第 2 步用一个例子和一个简单模型，回答"降低哪些用户的权重是无害的、权重该多大"；第 3–5
步回答"这个权重怎么从线上数据估出来"；第 6 步列出模型不成立的情况。

#### 第 1 步：能改的地方只有两处

单条 rollout 的策略梯度（PPO 去掉 clip 后的形式）是

$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A_t\Big],\qquad A_t = R - V(s_t)
$$

GAE 只是把 $R - V(s_t)$ 换成多步 TD 误差的加权和，结论不变。$R$ 是环境给的，不能改；能改的只有：

1. **$V$（critic）**：它以什么为输入、用什么目标训练。
1. **$A_t$ 外面再乘一个系数
   $w$**：$\nabla_\theta J_w = \mathbb{E}[\sum_t \nabla \log \pi \cdot w \cdot A_t]$。

per-user 的 clip 宽度也能改，但它的作用是限制单步更新幅度，效果接近给样本的步长打折，数学上不如乘系数清楚，先不考虑。

这两处改动在数学上的作用不同：

- **改 $V$ 只影响方差，不影响方向**。只要 $V(s_t)$ 只依赖 $s_t$、不依赖 $a_t$，就有
  $\mathbb{E}[\nabla \log \pi(a_t \mid s_t)\, V(s_t)] = 0$（策略梯度的 baseline 定理）。用户
  ID、同一用户的历史都在 $s_t$ 里，所以让 critic 看到用户信息是合法的，它只会让 $A_t$ 更小、更稳，不改变 agent
  往哪个方向学。这就是"平移"，P-GRPO 做的就是这件事。

- **乘系数会改变目标**。如果 $w$ 只依赖用户 $u$（在这条对话开始前就定了），那么

  $$
  \nabla_\theta J_w = \sum_u p(u)\, w_u\, \nabla_\theta J_u
  $$

  其中 $J_u$ 是 agent 面对用户 $u$ 时的成功率。也就是说，乘 $w_u$ 等于把训练目标从"所有用户上的平均成功率"换成"按 $w_u$
  加权的平均成功率"。降低某个用户的权重，就是告诉模型"这个用户上的表现不重要"。

所以"让 agent 少从不可靠的用户身上学"只能通过第 2 处实现，而它的代价是改变了目标。下一步回答：什么情况下这个改变是无害的。

#### 第 2 步：什么时候降权无害——把"不可靠"写成一个模型

**这一步在做什么**。第 1 步说乘 $w_u$ 会换掉训练目标。换得好不好，取决于不可靠的用户到底对 reward
做了什么；只说"用户不可靠"推不出任何结论。所以先写一个最简单的模型，把"不可靠"翻译成公式，再在模型里算两件事：降权会不会把 agent
带偏，以及权重该取多大。这是统计里处理测量误差的常规思路：先写出误差怎么产生，再决定怎么加权。

**例子**。同一道题、同一个 agent，面对三类用户：

| 用户           | 行为                                                                      | 成功率 |
| -------------- | ------------------------------------------------------------------------- | ------ |
| 普通用户       | 按剧本配合，偶尔出错                                                      | 0.9    |
| 用户 A：不可靠 | agent 每一步都做对，但 A 有一半概率在确认的同一句里结束对话，写操作没执行 | 0.5    |
| 用户 B：难应付 | B 不主动说订单号，agent 有一半概率没有追问，于是改错了订单                | 0.5    |

两类用户都是 τ² 里实际出现的情形，不是极端构造。A 就是 9.1 的 Doubao 提前结束（例如 `retail:6:trial1`），Doubao
用户在确认轮次里同一句结束的比例是 16.3%，DeepSeek 是 3.2%（16.3 问题 A）。B 是 16.2 说的"用户不主动给信息，agent
应该追问"；这一类在现有数据里还没有单独计数。唯一刻意的地方是让 A、B 的成功率都等于 0.5：两者平均结果一样，才能看清"只看平均结果"会漏掉什么（第 4
步）。比例不相等时结论不变，见第 4 步。

我们希望的结果是：少从 A 的对话里学（A 的成败与 agent 无关），照常从 B 的对话里学（agent 在 B 身上失败，是因为缺"追问"这项能力，正该学）。

**模型：用户只给结果加噪声**

**为什么要写公式**。我们要回答的问题是：agent 变强一点，它在某个用户身上的成功率涨多少？RL 训练从用户 $u$ 的对话里得到的梯度，估计的正是"成功率对 agent
参数的导数"。所以只要写出"成功率 = 某个关于 agent 能力的函数"，求导就能看出这个用户的对话把 agent 往哪个方向推、推多大力。

**符号**：

| 符号                        | 含义                                                                                  | 例子里的取值                    |
| --------------------------- | ------------------------------------------------------------------------------------- | ------------------------------- |
| $\theta$                    | agent（policy 模型）的参数，训练改的就是它                                            |                                 |
| $q(\theta)$                 | agent 把该做的事全部做对的概率（查对订单、调对工具、提交对的写操作），只由 agent 决定 | 例子中设 agent 每步都做对，即 1 |
| $\rho_u$                    | agent 做对了，用户 $u$ 仍把它搞砸的概率（例如确认的同一句就结束，写操作没执行）       | 普通 0.1，A 为 0.5              |
| $\eta_u$                    | agent 做错了，用户 $u$ 仍让它成功的概率（例如用户自己纠正了 agent）                   | 都取 0                          |
| $J_u(\theta)$               | agent 面对用户 $u$ 的成功率，也就是 reward 的期望                                     | 普通 0.9，A 0.5                 |
| $c_u = 1 - \rho_u - \eta_u$ | agent 做对与做错时成功率的差，即 agent 的能力对结果有多大影响                         | 普通 0.9，A 0.5                 |
| $\nabla$                    | 对 $\theta$ 求梯度。$\nabla q$ 指"让 agent 更常做对"的参数方向                        |                                 |

**公式怎么来**。按 agent 做对还是做错分两种情况：

- agent 做对（概率 $q$），用户不搞砸（概率 $1 - \rho_u$）才成功；
- agent 做错（概率 $1 - q$），用户救回来（概率 $\eta_u$）才成功。

两项相加：

$$
J_u = q\,(1 - \rho_u) + (1 - q)\,\eta_u = \eta_u + (1 - \rho_u - \eta_u)\, q = \eta_u + c_u\, q
$$

**数字例子**。设 agent 从 $q = 0.8$ 提高到 $q = 0.9$：

- 普通用户（$\rho = 0.1$）：成功率从 $0.72$ 涨到 $0.81$，涨 $0.09$；
- A（$\rho_A = 0.5$）：成功率从 $0.40$ 涨到 $0.45$，只涨 $0.05$。

两者都在 agent 做对更多时上涨，只是 A 涨得少。求导就是 $\nabla J = 0.9\, \nabla q$ 和
$\nabla J_A = 0.5\, \nabla q$：同一个方向 $\nabla q$，长度不同。

**B 不能写成这个形式**。B 不说订单号，agent 要先追问（概率记为 $r(\theta)$，也由 agent 决定），再把其余事做对（概率
$q$），两件都做到才成功：

$$
J_B = q(\theta)\, r(\theta),\qquad \nabla J_B = r\, \nabla q + q\, \nabla r
$$

第二个式子是乘积求导。第一项教"把事做对"，与普通用户、A 一样；第二项 $q\, \nabla r$ 教"追问"。普通用户和 A 会主动给订单号，agent
问不问都不影响结果，他们的对话里没有这一项，只有 B 的对话能教追问。

由此得到三个结论：

1. **对 A 这类用户，降权不会带偏方向**。$\nabla J_u = c_u \nabla q$：普通用户给 $0.9\, \nabla q$，A 给
   $0.5\, \nabla q$，方向相同，A 的只是更短、更嘈杂。所以对符合模型的用户，任何正的 $w_u$ 都不会把 agent 带偏，第 1
   步说的"改变目标"在这里无害。对 B 则相反：降低 B 的权重会削弱 $q\, \nabla r$，agent 少学追问。所以方法必须能把 A 和 B 分开，这就是
   16.0 说的可辨识性问题，第 4 步专门回答。

1. **权重该多大：逆方差加权**。直觉是两支温度计测同一个温度，一支准、一支不准，平均时应该让准的那支说了算，各自的权重与读数的方差成反比。这里每个用户的样本是对
   $\nabla q$ 的一次带噪估计 $\hat g_u$：均值 $c_u \nabla q$，方差记为 $\sigma_u^2$。先除以 $c_u$ 得到无偏估计
   $\hat g_u / c_u$，方差 $\sigma_u^2 / c_u^2$；对它做逆方差加权、再换回 $\hat g_u$，得到

   $$
   w_u \propto \frac{c_u}{\sigma_u^2}
   $$

   信号越弱（$c_u$ 小）、噪声越大（$\sigma_u^2$ 大），权重越低。这就是"可靠度"在这个模型里的准确含义。在例子里，A 的 $c_A = 0.5$
   比普通用户小；第 3 步会算出，A 的对话里同样的行为一次得 $+0.5$、一次得 $-0.5$，$\sigma_A^2$ 也大。两项都让 A 的权重变低。

1. **要估什么**。需要一个能在线算出来的量，近似 $\sigma_u^2$（最好还有 $c_u$），而且必须对 A 大、对 B 小。线上能用的只有 reward $R$
   和 critic 的 $V$，所以下面先弄清 critic。

#### 第 3 步：critic 学到的是什么，能怎么改

**为什么现在讲 critic**。第 2 步要的估计量只能从 $R$ 和 $V$ 算出来；16.9 最初想用的可靠度，残差均值
$\bar r_u = \text{mean}(R - V(s_0))$，也是这样算的。这些量的含义完全取决于 critic 学到了什么，所以要先弄清 critic
怎么训练、训练出来的 $V$ 代表什么。顺带也检查：只改 critic 能不能代替乘系数。

**3a. 现在 SAO 的 critic 怎么训练。**

- **在哪些位置预测**。critic 对整条对话做一次前向，只在 agent 生成的 token 位置（`loss_mask` 为 1 的位置）计算 loss。位置 $t$
  的输入是它之前的全部内容 $s_t$：system prompt、policy、工具定义、此前的用户消息、工具返回和 agent 自己的回复。

- **回归什么**。`examples/tau2/config_sao.yaml` 中
  `critic_gae_lambda: 1.0`、`discount: 1.0`，reward 只在对话结束时给一次，所以每个位置的回归目标都是这条对话的最终 reward
  $R$（0 或 1）。代码见 `areal/trainer/ppo/actor.py` 中 `critic_gae_lambda` 一段，GAE 在
  $\lambda = 1$ 时退化为蒙特卡洛回报。

- **loss**。`areal/trainer/ppo/critic.py` 的 `ppo_loss_fn`，SAO 配置 `eps_clip: null`，即不做
  value clipping 的均方误差：

  $$
  L_V = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \big(V(s_t) - R\big)^2
  $$

  $\mathcal{T}$ 是 agent token 的位置集合。

**3b. 这个 loss 训练出来的 $V$ 是什么**。固定一个状态 $s$，在所有经过 $s$ 的对话上，常数 $c$ 让
$\mathbb{E}[(R - c)^2 \mid s]$ 最小的取值是 $c = \mathbb{E}[R \mid s]$（对 $c$
求导：$2(c - \mathbb{E}[R \mid s]) = 0$）。所以 critic 足够强、数据足够多时：

$$
V^*(s_t) = \mathbb{E}[R \mid s_t] = \Pr(\text{成功} \mid \text{对话进行到 } t \text{ 为止的全部内容})
$$

也就是"看到目前为止的对话，这条对话最终成功的概率"。这里的"全部内容"包括用户已经说过的话，所以 critic 天然会根据用户的表现调整预测。

**3c. 为什么只改 critic 做不到"少从 A 身上学"**

看 A 的一条对话。agent 做完所有事、发出确认请求时，critic 能给出的最好预测是 $V = 0.5$：之后成不成，取决于 A 会不会提前结束，critic
再准也预测不了这一次的结果。于是 agent 后面的 token 拿到的 advantage 是

$$
A_t = R - V(s_t) = \begin{cases} 1 - 0.5 = +0.5 & \text{A 这次没提前结束} \\ 0 - 0.5 = -0.5 & \text{A 这次提前结束} \end{cases}
$$

agent 的行为两次完全相同，却一次被推向"多这样做"，一次被推向"少这样做"，幅度都是 0.5。critic 在这里已经做到最好，这 $\pm 0.5$ 仍然在。原因是
critic 只出现在减号后面：它能把 advantage 的中心移到 0，但不能把 $\pm 0.5$ 缩小。要缩小，只能在外面乘一个系数（第 1 步），比如对 A 乘
0.2，变成 $\pm 0.1$。

critic 上可以改的地方，以及它们为什么都不能代替乘系数：

| 改什么     | 怎么改                                          | 结果                                                                                                  |
| ---------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 输入       | 把用户 ID 或这个用户的历史放进 critic 的输入    | critic 更早知道"这是 A"，预测更准，advantage 更小更稳；但上面的 $\pm 0.5$ 依然存在。P-GRPO 属于这一类 |
| 回归目标   | 用户出错之后的结果不算，改用出错前的 $V$ 当目标 | 能消掉 A 的 $\pm 0.5$，但要知道用户在哪一轮出错。线上没有这个标签，所以第一版不用                     |
| 预测位置   | 在用户回合结束处也训练 critic                   | 不改变 agent 的 advantage，只多得到"这句用户回复让成功率变了多少"的读数                               |
| loss 权重  | A 的对话在 critic loss 里权重低                 | 基本无效：权重只影响 critic 拟合得多准，不改变它要逼近的"成功概率"                                    |
| 多一个输出 | 另加一个 head，预测"结果还有多不确定"           | 不改变 agent 的 advantage，但能直接给出第 5 步要用的不确定性，是备选做法                              |

另有两条不能改：$R$ 是环境给的；$V$ 不能依赖当前这一步的动作，否则 advantage 会有偏差。

所以 critic 在本方法里只做原来的事：提供 $V$。第一版不改 critic 的 loss，只在 advantage 外面乘按用户的权重。

#### 第 4 步：用什么量区分 A 和 B——残差均值不行，残差平方可以

**残差均值原来是干什么的**。16.9 把 $\bar r_u$ 当作每个用户的"可靠度"：结果总比 critic 预期差的用户，就认为不可靠、该降权。第 2
步说明，可靠度必须把 A（该降权）和 B（不该降权）分开，所以这里拿 A、B 检验它。

$\bar r_u$ 是"这个用户的对话，结果比 critic 在开头的预测 $V(s_0)$ 平均差多少"。critic
在开头能看到用户的第一句话，可能认得出用户类型，也可能认不出，分两种情况：

- **critic 认得出 A、B**：它在开头就预测 A 是 0.5、B 是 0.5，与实际一致，$\bar r_A = \bar r_B = 0$，什么都测不出来。
- **critic 认不出**：它只能按"所有用户在这道题上的平均"来预测，假设是 0.8。那么
  $\bar r_A = 0.5 - 0.8 = -0.3$，$\bar r_B = 0.5 - 0.8 = -0.3$，两者一样。

**比例不相等时也一样分不开**。若 A 的成功率是 0.6、B 是 0.4，critic 认不出用户时
$\bar r_A = -0.2$，$\bar r_B = -0.4$：B 看起来更"不可靠"，反而被降得更多，正好降错。$\bar r_u$
只量出"这个用户让成功率低了多少"，不管低的原因是用户捣乱还是 agent 缺能力。

不同的是成功率低是怎么来的：

- A：agent 做完事之后，结果是随机的。看到 agent 已经做完、正在等确认，critic 也只能说 0.5。
- B：结果由 agent 有没有追问决定。agent 做完时，critic 已经看得出它有没有问到订单号：问到了就接近 1，没问到就接近 0，剩下的不确定性很小。

这就是"成功率"和"噪声"的区别：A、B 都把成功率拉低，但只有 A 让"agent 做完之后的结果"变得不确定。

**所以看 agent 最后一次行动之后的残差平方**。一条对话里，agent 最后一次发言或调用工具之后，剩下的事全由用户决定（确认、结束或偏离）。取

$$
e = R - V(s_{\text{last}})
$$

$s_{\text{last}}$ 是 agent 最后一个 token 的位置，$V(s_{\text{last}})$ 是"agent 已经做完所有事时，critic
预计能成功的概率"，$e$ 是用户最后这一段带来的意外，也就是 16.4 里 $\Delta_u$ 在最后一段的特例。对 0/1 的 reward，若 critic
校准良好、$V(s_{\text{last}}) = p$，则

$$
\mathbb{E}[e \mid s_{\text{last}}] = 0,\qquad \mathbb{E}[e^2 \mid s_{\text{last}}] = p(1 - p)
$$

均值会被 critic 吸收为 0，平方不会：critic 再准，也消除不了"agent 做完事以后结果还不确定"这部分。代入例子：

- 普通用户：$p \approx 0.9$，$e^2 \approx 0.09$；
- A：agent 做对了，结果仍然一半一半，$p \approx 0.5$，$e^2 \approx 0.25$；
- B：agent 做完时成败已基本确定，$p \approx 1$ 或 $\approx 0$，$e^2 \approx 0$。

A 最大、B 最小，正是第 2 步要的方向。它也是 $\sigma_u^2$ 的一个代理：3c 里 A 的 $\pm 0.5$ 就来自这部分不确定性。critic
认出用户也没关系，它只会把 $p$ 估准，$e^2$ 不会变成 0。

#### 第 5 步：估计和实现

$$
\hat\sigma_u^2 = \frac{n_u\, \overline{e^2}_u + k\, \overline{e^2}_{\text{all}}}{n_u + k},\qquad
w_u = \frac{1/(\hat\sigma_u^2 + \tau)}{\text{batch 内的平均值}}
$$

- $\overline{e^2}_u$ 是用户 $u$ **过去**对话的平均值，$n_u$ 是他的对话数。
- $k$ 控制收缩：对话少的用户先向总体平均靠拢，避免几条对话就给出极端权重（empirical Bayes 的标准做法）。
- $\tau$ 防止除以接近 0 的数。
- 除以 batch 平均，使平均权重为 1，总学习率不变。
- $w_u$ 只用过去的对话算，这条对话开始前就固定了，所以它只依赖用户，满足第 1 步"只依赖 $u$"的条件。如果用当前对话的结果算权重，agent
  会因为"失败看起来像用户的错"而少受惩罚，可能学会把失败推给用户。

这一版只用了逆方差 $1/\sigma_u^2$，没有估计 $c_u$。$c_u$ 需要知道 agent 的真实水平，单条 rollout 下估不出来；$\rho_u$
大的用户 $\sigma_u^2$ 也大，所以逆方差已经部分体现了 $c_u$ 的作用。

**实现上的改动很小：**

- critic 不改。$V(s_{\text{last}})$ 就是最后一个 assistant token 位置的 value，现有 SAO 已经在算。
- 在 GAE 算出 $A_t$ 之后、进 PPO loss 之前，乘上这条对话所属用户的 $w_u$。
- 维护一张"用户 → $(n_u, \overline{e^2}_u)$"的表，每个 batch 结束后更新。

#### 第 6 步：这个模型什么时候不成立

第 2 步的结论依赖两个假设，都可能不成立：

1. **"用户只加噪声，所有用户考的是同一种能力"**。B 这类用户就违反它：$J_B$ 依赖"追问"这项另外的能力，不是 $c_u q$ 的形式，降权等于少学这项能力。第
   4 步的 $e^2$ 对此有一点区分能力：用户不给信息时，agent 往往在最后一步之前就已经出错，失败在 $V(s_{\text{last}})$
   里已经体现，$e^2$ 不大；用户在 agent 做完之后才坏事，失败才落在 $e^2$ 里。但这只是倾向，不是保证。例如 critic 看不出 agent
   改错了订单时，B 的失败也会落进 $e^2$。
1. **"用户出错的概率与 agent 无关"**。如果 agent 换一种确认方式就能让用户不提前结束，$\rho_u$ 就依赖 $\theta$，$c_u$ 也依赖
   $\theta$。这时用户出错部分是 agent 的责任，降权会让 agent 少学这一点。16.3 问题 A 的统计显示，Doubao 是否结束可能与 agent
   列出的操作数有关，但证据很弱。

**缓解和检查：**

- 给 $w_u$ 设下限（例如不低于 0.3），保证任何用户都还在学；
- 评测按用户类型分别报告，并报告最差用户组的成功率；如果它比 baseline 低，就是第 1 条假设被违反；
- 在训练前先做一个能失败的检查：用 persona 构造已知提前结束率不同的用户（例如 0%、20%、50%），用现有 critic 跑一批对话，看
  $\hat\sigma_u^2$ 能否把它们按提前结束率排对序。排不对，说明这个估计量在 τ² 上不可用，方法要改；这一步需要 critic 先有正的 EV（现在 SAO
  critic 的 EV 为负，3.4）。再加一组"不主动给信息"的 persona，检查它的 $\hat\sigma_u^2$ 是否明显低于提前结束组，用来检验第 4
  步说的 A/B 区分。

#### 与第 16.10 节 baseline 的对应

| 改动                                | 对应的 baseline              | 区别                                                   |
| ----------------------------------- | ---------------------------- | ------------------------------------------------------ |
| 只改 $V$：critic 看用户 ID          | 第 6 项（用户条件 critic）   | 只降方差，不降低不可靠用户的权重                       |
| 只改 $V$：按用户的 running 均值     | 第 3 项（P-GRPO 单条版）     | 同上，而且用的是均值，第 4 步说明它测不出可靠度        |
| 乘 $w_u$，$w_u$ 由 $e^2$ 得到       | 本方法                       | 降低"agent 做完事后结果仍不确定"的用户的权重           |
| 乘 $w$，$w$ 由当前对话的 judge 得到 | 第 4 项（AutoForge 式 mask） | 依赖特权信息；按当前对话算权重，有把失败推给用户的风险 |

这一节是推导和设计，没有任何实验支持。第 2 步的最优权重是标准结论；"$e^2$ 能反映用户噪声、并把 A 和 B 分开"是基于模型的推断，要靠第 6 步的检查验证。
