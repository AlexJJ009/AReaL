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
- 第 15 节：Qwen3-4B-2507 两个模型的 step 0 评测、能力缺口与加速配置核查（2026-10-11）。

第 11–12 节只保留最新的决策和执行步骤。τ² 的调研、对比、找 gap 都记在本文，不另开文档。

原始数据根目录：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking`。
核查脚本在本目录 `scripts/`，大体积中间结果保留在
`/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/_local/scratch/claude-verify-20261009/`（不入库）。
第 7.1 节的 Doubao 首轮探测、第 9.5 节的高分核查证据归档在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-scratch-archive-20261010/`。

## 1. 结论

**初始分数高（Retail/Airline 约 80%，Telecom 100%）的主因是模型本身能完成多数任务，加上 benchmark
自身的评分盲区；不是数据或评测代码被改过，也不是用户模拟器代劳。**

\*\*2026-10-10 更正：\*\*第 2 节"用户模拟器过度配合：否"只说明用户没有代劳；用户模拟器的选择本身决定 Retail 分数。
同一份代码下，DeepSeek 用户 Retail 0.851、Doubao 用户 0.651，差距主要来自 Doubao 在确认的同一句话里结束对话（第 9 节）。

\*\*训练看不出效果，原因是评测噪声大于训练带来的变化、有效学习信号少、权重几乎没有朝一致方向移动。\*\*现有证据只能说"无法判断训练是否有效"，不能说"训练变差"。

\*\*2026-10-10 补充：航空的高分里约 30pp 是转人工得来的。\*\*航空任务按 DB + COMMUNICATE 打分，在"应该拒绝"的任务上，agent
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

另有两处标注问题：Retail 105 的 gold 换货操作本身报"礼品卡余额不足"（$17 \< $21.10），NL 却要求完成换货；Retail 38 的取消理由
gold 固定为 "no longer needed"，用户多数时候选 "ordered by mistake"。

官方 NL 实现有一个隐患：judge 返回空 `results` 时 `all([])`
为真，直接判通过（`tau2/evaluator/evaluator_nl_assertions.py:51,134`）。历史记录未保存 judge
原始返回，无法确认是否发生过。

### 与 Recuris 论文（Qwen3.5-4B：Retail 68.0 / Airline 75.3）的差距

已在第 9 节查清：两边都用 Doubao 用户时，差距在抽样误差内；Recuris 的 harness 对这一行没有影响。下面是 2026-10-09
的原始判断，保留作记录。

- \*\*不是题集差异。\*\*未训练模型在 train split（epoch 1 覆盖全部任务，权重相对变化约 1e-4）上为 Retail 84.8%、Airline
  78.3%；与 test 合并估算全量 114/50 题约 82%/79%。
- \*\*论文 Table 1 的用户模拟器不是 4B 自己。\*\*Recuris 源码 `src/recuris/adapters/tau2/treatment.py`
  的 transfer 配置固定 Doubao Seed 2.0 Pro 作为用户模拟器和 NL
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
- **pass^k**：官方 $\\binom{c}{k}/\\binom{n}{k}$，即同一题 k 次全部成功的概率，再在题间平均。

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

- \*\*是什么。\*\*Mode A 指"用户的确认里带着 `###STOP###`，之后没有 agent 回合"，也就是 9.1 的模式（RT2
  `orchestrator.py:303`）。代码只在这一处定义它。
- **怎么处理。**`TAU2_GATE_TERM=1` 时，在 `USER_STOP` 之后、`_finalize()`
  之前检查：全程没执行过写工具，且任意一条用户消息命中 确认词正则（yes / sure / go ahead / confirm / ok / perfect
  等）。命中就往轨迹末尾追加一条 `role="user"` 的
  `[SYSTEM CHECK - not from the customer] ... execute the confirmed change now` 消息，让
  agent 再跑最多 4 轮并真实执行工具。 用户的 STOP 不删不改，`termination_reason` 仍是
  `USER_STOP`；评分重放整条轨迹，所以补执行的写操作会计入 DB 分。 变体 `_V5` 处理部分执行，`_A2` 看 agent 是否在请求确认，`_WM` 看
  Recuris agent 的待办账本。开关默认全关。
- \*\*解决了没有。\*\*仓库里没有可复核的效果数据：唯一的数字是代码注释里的自述（在 977 条 retail 仿真上调参，对通过的仿真触发 2%、失败的
  47%），没有对应的结果文件或日志。从正则看有明显的误报（开场 "Yes, I need help"、告别 "Perfect, thanks"、 否定句 "that's
  not correct" 都会命中，可能把应当不写的任务强行写坏）和漏报（确认措辞不在词表里；已有任意写操作时默认路径
  直接跳过）。它改变了评分对象（多了一条伪用户消息和用户看不到的 agent 回合），结果和官方 `tau2 run` 不可比。
- \*\*发表的 68.0 / 75.3 开了没有。\*\*base config 和 README 命令三个开关都是开的，但 `TAU2_GATE_TERM_WM=1` 时
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

| 类别       | 子类型                           | DeepSeek | Doubao | 口径 | 例子                                                                                                 | 改进途径                          |
| ---------- | -------------------------------- | -------- | ------ | ---- | ---------------------------------------------------------------------------------------------------- | --------------------------------- |
| benchmark  | gold 不可行                      | 4        | 4      | 精确 | retail:105：礼品卡 $17 付不了 $21.10，agent 正确拒绝，NL 断言却要求完成换货                          | 无，记为上限损失                  |
| benchmark  | gold 写死取消原因                | 3        | 1      | 精确 | retail:38：用户说 "ordered by mistake"，gold 是 "no longer needed"                                   | 同上                              |
| benchmark  | gold 动作顺序错                  | 2        | 3      | 精确 | retail:64                                                                                            | 同上                              |
| benchmark  | 任务有歧义                       | 3        | 4      | 精确 | airline:7：问的是 "other upcoming flights"，gold 把要取消的两单也算进金额（$1628 vs $708），8 条全挂 | 同上                              |
| benchmark  | 未指定支付方式                   | 5        | 3      | 估计 | retail:60：任务没写支付方式，用户自己选了礼品卡，gold 是 paypal                                      | 同上；归 benchmark 还是用户有争议 |
| benchmark  | 环境实现 bug                     | 2        | 0      | 精确 | retail:100：`modify_pending_order_items` 把最后一个 variant 的价格写给所有 item，DB hash 对顺序敏感  | 同上                              |
| benchmark  | NL 断言要求空操作                | 0        | 2      | 精确 | retail:110：默认地址已经是新地址，断言仍要求 "modify user address"                                   | 同上                              |
| 用户模拟器 | 确认后立刻 STOP                  | 21       | 88     | 估计 | retail:17："Yes, I confirm… ###STOP###"，agent 还没执行写操作                                        | 换用户模型，或在 prompt 层修      |
| 用户模拟器 | 首轮即 OUT-OF-SCOPE              | 0        | 3      | 精确 | retail:109：第一条用户消息就是 `###OUT-OF-SCOPE###`                                                  | 换用户模型                        |
| 用户模拟器 | 中途 OUT-OF-SCOPE                | 0        | 5      | 估计 | airline:29                                                                                           | 换用户模型                        |
| 用户模拟器 | 编造或偏离指令                   | 3        | 4      | 估计 | retail:79（Doubao）编造"另一只瓶子是黑色"；retail:5（DeepSeek）known_info 里有姓名却拒绝提供         | 换用户模型                        |
| NL judge   | 判错                             | 0        | 0      | 精确 | DeepSeek 的 4 条 NL-only 失败都是 retail:105；Doubao 的 9 条 judge 都按字面判对                      | 当前不是瓶颈                      |
| 模型       | 参数、物品或支付方式错           | 25       | 46     | 估计 | airline:18：5 张票的退款全打到一张卡，应原路退回；retail:49 选错 variant                             | 训练                              |
| 模型       | 违反政策的写操作，或该转人工没转 | 13       | 15     | 估计 | airline:47：不符合条件的 basic economy 照样取消                                                      | 训练                              |
| 模型       | 不该转人工却转了                 | 21       | 17     | 估计 | airline:44：本该升舱，却转了人工                                                                     | 训练，但见 10.3                   |
| 模型       | 漏步骤、漏传达、错误拒绝、幻觉   | 4        | 7      | 估计 | retail:110：错称 pending 订单不能修改；retail:39 编造地址                                            | 训练                              |
| infra      | —                                | 0        | 0      | 精确 | 见 10.1                                                                                              | 充值 Ark；32K 是有意设定          |

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
- \*\*#580 没有作为代码合入。\*\*它的作用是：gold 写操作执行抛错时，DB 判定不算通过。fixed 下，`check_gold_replay.py`
  没有查出抛错的 gold 写操作，合入它不会改变任何分数。
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

| 步骤            | 配置                                            | 步数与 episode                                                           | 存档与评测                                  | 评测内容                                                |
| --------------- | ----------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------- |
| 1. critic       | `scripts/tau2/train_critic.sh`，formal          | 2 epoch：$\\lceil 99/16 \\rceil \\times 2 = 14$ 步，198 条训练 episode   | 第 5、10、14 步存档；第 0、5、10、14 步验证 | critic 在 60 条固定 test 轨迹上的 MSE、EV（按领域分开） |
| 2. SAO          | `run.sh policy mixed sao`，加载第 1 步的 critic | 2 epoch，26 步；每 epoch Airline 240 + Retail 552 条，共 1584 条         | 每 5 步存档并评测，训练前先评一次           | test 60 题 × 4 次，temperature 0：mean@4、pass^1/2/4    |
| 3. GRPO（对照） | `run.sh policy mixed grpo`                      | 2 epoch：$\\lceil 99/8 \\rceil \\times 2 = 26$ 步，每题 8 条，共 1584 条 | 同上                                        | 同上                                                    |

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
  - 存活成员 ≥ $\\lceil n/2 \\rceil = 4$ 时，在存活成员上归一化。不足 4 个时，这道题放到本 epoch 末尾重跑一次；再被拒就记为
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

### 12.9 AReaL-tau2-data 的筛选与修复（进行中）

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

- **进入评测的候选集**（`out/candidates_after_review.json`）：

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

1. **失败轨迹复核**（pueue 任务 438，Codex gpt-6.1-sol，规则见 `EVAL_REVIEW.md`）：

   - 范围：4 次全错的 15 题，加上从"有对有错"的 161 题里随机抽的 40 题。
   - 对每条失败轨迹，找出它和 gold 第一次在 DB 写操作上分开的地方，判断是 agent 的错还是题的错。题的错就丢掉，或者再修一次。

1. **训练集**（待复核后决定）：只留 temperature 1、每组 8 条时有组内对比的题。按 temperature 0 的结果估计，可用的大约是"有对有错"的
   161 题，加上一部分 temperature 1 下会出错的全对题。要在 temperature 1 下按训练口径再筛一次，才能定下来。

工作目录：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/areal-tau2-data-screen-20261010/`。其中：

- `PROMPT.md` 是给 Codex 的审题规则；
- `tools/` 是 `dossier.py`（打印题目和相关 DB 记录）和 `score.py`（重放打分）；
- `shards/` 是分片题号；
- `out/` 是每片的逐题结论（JSONL）和总结。

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

\*\*Airline 和预期一致：基本不变。\*\*0.428 → 0.388（−4.0pp）。fixed v2 对 Airline 只改了反向转人工规则，这条规则只影响
airline:13， 这次没有清零任何 Airline episode。所以这 4pp 是运行间噪声，在 13.1 测得的约 3pp 的量级内。正向转人工规则在
Airline 清零了 75 条原本成功的 episode（v1 是 63 条）。

\*\*Retail 没有降低，反而升高了。\*\*0.805 → 0.849（+4.4pp）。按题分组归因（114 题，每组为 4 次平均分之和的变化）：

| 题组                                                               | 题数 | v1 → v2  | 折合 Retail pp |
| ------------------------------------------------------------------ | ---- | -------- | -------------- |
| 20 道打了补丁的题                                                  | 20   | +3.50 题 | +3.1           |
| v1 被正向规则清零、v2 按 Retail 例外恢复的题（1、32、43 等 10 题） | 10   | +2.25 题 | +2.0           |
| 其余未改动的题                                                     | 84   | −0.75 题 | −0.7（噪声）   |

- \*\*补丁让分数上升，是补丁本身的性质。\*\*补丁去掉的是"做对了也判错"的失败：

  - 38：v1 是 0/4，v2 是 4/4，补丁写明了取消原因；
  - 64：0/4 → 4/4，补丁加了"多款符合时选最便宜的"；
  - 59、60、63、66、110 也都升到了 4/4。

  这些失败原本就不是模型能力的问题，留着只会给训练加噪声，不会增加可学的空间。

- \*\*本该降分的三题没有降。\*\*57、65 改为必须查订单，但这个模型本来就会查：57 在 v1、v2 都是 4/4，65 从 3/4 升到 4/4。25
  要求转人工，v2 是 2/4，与 v1 相同。 "什么都不做也得分"的漏洞对这个模型的分数影响很小。

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

temperature 0 下各题 4 次的结果分布（全对 / 全错 / 有对有错）：

| 模型          | Airline（fixed） | Retail（fixed） | Telecom      |
| ------------- | ---------------- | --------------- | ------------ |
| Qwen3.5-4B    | 11 / 20 / 19     | 73 / 2 / 39     | 114 / 0 / 0  |
| Instruct-2507 | 3 / 26 / 21      | 11 / 31 / 72    | 19 / 35 / 60 |
| Thinking-2507 | 3 / 27 / 20      | 40 / 9 / 65     | 28 / 61 / 25 |

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
- **Instruct 会执行 policy 不允许的修改。** 例：airline:2 t0，它主动提出补偿并发了 $50 证书，policy 规定不能主动提补偿。
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

**要先测的。** 用 13.2 的口径，在 813 题上评测两个 Qwen3 模型，看全对、有对有错、全错各多少。这一步需要 DeepSeek 费用：Qwen3.5
跑一轮用户侧用了 33M prompt token；Thinking 对话更长，会更多。

**训练前要处理的两点：**

- 15.4 的"用户在确认句里结束对话"会让 Qwen3 产生更多假负样本；
- Qwen3 critic 路径还没验证（15.6）。

### 15.8 局限

- 失败分类靠规则抽取（`aggregate.py`、`write_diff.py`）。人工精读了约 10 条轨迹，主要类别各 1–2 条，没有逐条标注。
- temperature 0 同配置重测约有 3pp 波动（13.1）。本节比较的差距都在 15pp 以上，不受影响。
- Actor probe 只跑了 3 步、用的是合成数据，吞吐只能做相对比较。
