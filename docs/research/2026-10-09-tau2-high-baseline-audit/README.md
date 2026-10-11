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
- 第 12 节：训练计划。
- 第 13 节：两轮 step 0 评测。

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
1. 修复 13.2 的 proxy 容量泄漏（修法候选见 13.2，需要批准）。不修也能训练，代价是偶尔有一步多等 1 小时。
1. Retail 训练信号不足的处理（12.7 的两条建议）。

### 12.7 Retail 训练信号不足：社区的做法（2026-10-10 调研）

**问题**：fixed v2 下，Retail 训练池 69 题里有 47 题在 temperature 0 下 4 次全对（13.2）。

- 对 GRPO 来说，全对的组 advantage 为 0，没有梯度。
- 对 PPO / SAO 来说，critic 能给全对轨迹的中间步骤非零的 advantage。但这些题上从来没有失败样本，critic 学不到"哪里会失败"，信号很弱。所以
  PPO 也只是比 GRPO 好一些，题太简单的问题仍然存在。

下面的结论来自调研 agent，标注说明来源：\[代码/论文\] 表示读过原文，\[摘要\] 表示只读了摘要或 README。两个 HF 数据集的存在、license 和 DB
文件，我用 HF API 独立核对过；其余没有独立核对。

**现成的、能直接当 RL 任务用的数据**：

| 数据                                                                                                         | 规模                                           | 内容                                                               | License     | 用于我们的代价                                                                                          |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------------------- |
| [inclusionAI/AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)（AReaL-SEA 论文） | RL 任务：Airline 1148、Retail 563、Telecom 271 | 每题带 gold actions 和 `db_path`。Retail 用 3–4 个新合成的 DB      | Apache-2.0  | 要能按任务加载各自的 DB；现在 `agent.py` 只用默认 DB。也需要核对它和官方 test 有没有重复（card 里没写） |
| [wwwenlin/skillscaffold-tau-rl](https://huggingface.co/datasets/wwwenlin/skillscaffold-tau-rl)               | 每个领域 300 题                                | 上面那份数据的子集：用 Qwen3-8B 每题跑 8 次，只保留通过 1–6 次的题 | Apache-2.0  | 格式与上面相同。它是按 8B 筛的，要用我们的 4B 重新筛                                                    |
| tau-bench v1 Retail train                                                                                    | 500 题                                         | 与 tau2 Retail 共用同一个 DB                                       | MIT         | **不推荐**：调研 agent 实测，tau2 Retail test 的 41 个写操作 order_id 中有 26 个出现在 v1 train 里      |
| APIGen-MT-5k、CoVe-12k                                                                                       | 5k / 12k 条轨迹                                | 只有轨迹，没有 gold 或 DB                                          | NC / Apache | 只能用于 SFT                                                                                            |

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
   - 工程量估计 1–2 天：按任务加载 DB、核对与官方 test 是否重复、fixed 变体的补丁对新 DB 是否适用。
   - API 费用：每道候选题要用 DeepSeek 用户跑 8 次。

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

**两条 episode 挂起 1 小时：proxy 的 session 容量泄漏**（已复现，未修）

- **现象**：controller 的 task 611、643（`retail:59:trial3`、`retail:99:trial3`）都分到 engine rank
  3。submit 之后，它们一直没有打出 `STARTING SIMULATION`，也没有写尝试记录，直到 `request_timeout`（3600
  s）后才被拒绝。评测因此多花了约 45 分钟（总耗时 77 分钟，v1 是 31 分钟）。这两条按 infra 排除，不影响其他 episode 的分数。之前 5
  次运行都没有出现过。

- **机制**（读代码确认，`areal/experimental/openai/proxy/`）：

  1. 每个 episode 先 `grant_capacity`（容量 +1），再 `start_session`（容量 −1，新建 session
     `<task>-<idx>`）。
  1. 客户端 `_start_session` 每次请求的超时是 10 s；遇到 HTTP 429 等错误时，用 `stop_never` 无限重试。
  1. 服务端的 `start_session` 不是幂等的。如果它在客户端超时之后才执行完，第一次请求照样建出 `<task>-0`、消耗 1 个容量，但这个
     session 没人使用；客户端的重试又建出 `<task>-1`，再消耗 1 个容量。
  1. 每发生一次，这个 proxy 就永久少 1 个容量。`_cleanup_stale_sessions` 在 3600 s 后会删掉孤儿
     session，但不会归还容量。
  1. 之后每个新 episode 的 grant 和 start 仍是一一对应，但总数差 1。这个 proxy 上最后到达的 episode 永远拿到 429，只能等
     controller 的 `request_timeout`。

- **与本次运行的日志吻合**：rank 3 的 task 267（`retail:1`）和 275（`retail:93`）都正常完成，但 proxy 在 11:11:23
  把 `267-0`、`275-0` 当作 stale session 删除。正常完成却留下 `-0`，说明它们实际用的是 `-1`，也就是各发生了一次泄漏。泄漏 2
  次，挂起也正好 2 条。11:11:23 往前推 3600 s 是 10:11 左右，那时 rank 3 正有 34–35 个 episode 并发。

- **复现**：`scripts/official-runner/repro_proxy_capacity_leak.py`。用真实的 proxy server
  和客户端代码，在 episode A 的 `start_session` 执行期间持有 session 锁 12 s，结果是：A 用的是 `A-1`，`A-0`
  成了孤儿，后到的 episode B 持续收到 429（等 40 s 仍未启动），脚本退出码为 0。 把锁的时间改成 5 s（小于 10 s 超时）就不会泄漏，退出码为
  1。

- **没有确认的**：实际运行中是什么让 `start_session` 超过 10 s 才执行完，没有访问日志能证明。候选有两个：

  - 同步 handler 在线程池里排队，或在等 `_lock`；
  - `export_trajectories` 在 async handler 里同步执行 `export_interactions`，阻塞了事件循环，导致响应发不出去。

  复现脚本只注入了"执行变慢"这一个前提，没有复现这个前提是怎么产生的。

- **与 `2d7d342a` 无关**：涉及的 `client_session.py` 和 `proxy_rollout_server.py`
  在那次提交里没有改动。这条路径也不经过组内成员丢失的处理。

- **对训练的影响**：同样的泄漏会让训练里的一个成员等满 3600 s。按 ADR 0011，这个成员会被当作丢失成员，训练不会中止，但那一步要多等 1 小时。
  泄漏是累积的，一次训练运行越长，最后挂起的 episode 越多。

- **修法候选**（尚未实现；改的是 proxy 的公共代码，需要批准）：

  1. 让 `start_session` 幂等：客户端为每次 episode 生成一个请求 id，重试时带上同一个 id，服务端遇到已有的 id 就返回原来的
     session，不再消耗容量；
  1. 删除 stale session 时归还它占用的容量；
  1. 把 `export_interactions` 移到线程里执行，不阻塞事件循环。

  1 是根治，2 是兜底，3 处理的是候选诱因之一。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261010-fixed-v2/step0-base-t0-deepseek-fixed-v2/`。
对比脚本是 `scripts/official-runner/variant_compare.py`，按领域、划分和训练池输出 mean@4 /
pass^k，以及转人工规则清零的条数。
