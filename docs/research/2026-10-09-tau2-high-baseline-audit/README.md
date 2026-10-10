# τ² 初始分数偏高与训练无效：核查记录（2026-10-09）

短期核对用。对象：GRPO `formal-v2` 与 SAO
`mixed-v1`/`mixed-v4-disk`（Qwen3.5-4B，non-thinking，DeepSeek Flash 用户模拟器与 NL judge，评测
temperature=1、每题 8 次）。全部结论来自已有记录的重算，没有新增采样、训练或付费模型调用。

第 1–6 节是对 2026-09-27 训练记录的核查；第 7–8 节是同日评测调度对齐官方与新口径 step 0 评测；第 9 节是与 Recuris、 官方榜单的对比和
Doubao 用户失败精读（2026-10-10）。τ² 的调研、对比、找 gap 都记在本文，不另开文档。

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

1. 不单独跑 DeepSeek 用户下的 temperature=0 step 0 评测：GRPO/SAO 配置
   `eval_before_train: true`，训练启动时会在新口径（test、temperature 0、每题 4 次）下自动评一次 step 0。
1. 复现 Recuris 的小规模实验（待购买 API）：沿用我们的 harness，只把用户模拟器与 NL judge 换成 Recuris 源码固定的
   `doubao-seed-2-0-pro-260215`（temperature 0，reasoning_effort medium），agent temperature
   0，Airline+Retail base 全量，每题 1 次（164 条）。同一配置下另跑一组 DeepSeek 作为对照。若 Doubao 组接近
   68%/75%，差距来自用户模拟器；若仍约 80%，差距来自 Recuris fork harness（如 `TAU2_GATE_TERM`
   等开关）或其他协议差异。注意我们的 `user_llm_args.max_completion_tokens: 512` 对推理模型可能截断，换 Doubao
   时需去掉或调大。
1. 重训 airline+retail critic，再跑 SAO；GRPO 可直接跑。
1. 若仍需提升学习量，考虑调大 actor lr 或增加步数（本次未改）。

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

| #   | 维度                 | 官方                                                | 我们（评测）                                                        | 我们（训练）                       | 状态               | 影响                                                                       |
| --- | -------------------- | --------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------- | ------------------ | -------------------------------------------------------------------------- |
| 1   | 用户模拟器模型       | gpt-4.1，temperature 0；榜单推荐 gpt-5.2            | DeepSeek-Flash（默认）或 Doubao，temperature 0，top_p 1，超时 120 s | 同评测                             | 未对齐             | 最大的一项：只换用户，Retail 差约 20pp（9.1）；与榜单数字不可直接比        |
| 2   | NL judge             | gpt-4.1，temperature 0                              | 与用户模拟器同一模型和参数                                          | 同评测                             | 未对齐（暂不拆）   | 只影响含 NL_ASSERTION 的任务，约 1pp（9.1）                                |
| 3   | seed                 | `random.seed(300)` 为每个 trial 生成 seed           | 不传                                                                | 不传                               | 未对齐（决定不改） | 只影响可复现性；provider 是否支持 seed 未核实                              |
| 4   | policy `num_retries` | 3                                                   | 0，重试放在 episode 层                                              | 同评测                             | 未对齐（决定不改） | 一次传输错误会让整段 episode 重跑                                          |
| 5   | 上下文               | 不设窗口                                            | 32K；溢出抛异常，重试后排除                                         | 溢出记 `context_limit`，0 分并计入 | 有意不同           | 只有极长 episode 会被排除；两组 step 0 评测都没有发生                      |
| 6   | 墙钟超时             | 默认不设                                            | 1800 s，超时按官方 TIMEOUT 记 0 分                                  | 1200 s，超时当 infra 重试          | 有意不同           | 实测最长 651 s，评测基本不触发                                             |
| 7   | policy 生成设置      | temperature 0                                       | temperature 0，thinking 关闭，concat 模板加 qwen3_coder 解析        | temperature 1，单次回复 4096 上限  | 有意不同           | concat 拼接与服务端完整渲染的 token 差异未量化                             |
| 8   | 领域                 | 榜单覆盖全部领域                                    | 只跑 Airline + Retail                                               | 同评测                             | 有意不同           | 不能和榜单总分比，只能按领域比                                             |
| 9   | `pass^k` 的 k        | k 裁到每题最少有效 trial 数，排除 1 条就没有 pass^4 | 只报 k=1/2/4，每个 k 在有效 trial 数 ≥ k 的题上平均，并报题数       | 同评测                             | 有意不同           | 无排除时两种算法结果相同；有排除时多报出一个 pass^4                        |
| 10  | 失败 episode 处理    | —                                                   | 同官方                                                              | 被拒结果会让整组 GRPO 中止         | 训练有意不同       | 训练一旦有排除或用户首句就 stop，整轮训练中止（DeepSeek 用户下还没发生过） |
| 11  | 用户工具             | `get_user_tools(include=task.user_tools)`           | `get_user_tools()`，不按任务过滤                                    | 同评测                             | 未对齐（新发现）   | Airline/Retail 没有用户工具，无影响；重新加 Telecom 前要改                 |
| 12  | `evaluate.py` 题集   | 榜单用 base 全量，≥4 trials                         | `--task-set` 默认 test，需显式传 `base`                             | 训练中评测用 test（formal）或 dev  | 默认值不同         | 忘传 `--task-set base` 的结果和榜单不可比                                  |

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

1. 用 gpt-5.2（reasoning low）做用户跑一次 base 集，和榜单定标（需要 OpenAI 渠道和费用，需批准）。
1. 选训练用的用户模拟器前先定：Doubao 的提前结束会把正确行为记 0 分；DeepSeek 也有，频率约 1/4.5。
1. 把 NL judge 和用户模拟器分开配置（`econfig` 新增 judge 字段，属于配置结构改动，需批准），judge 固定为官方 gpt-4.1 或同一模型。

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

| 方向                           | 涉及的失败                      | 量级（pp，DeepSeek / Doubao）     | 做法                                                                                            | 是否改变与官方的可比性        |
| ------------------------------ | ------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------- |
| 训练用的用户模拟器             | 确认后 STOP、OUT-OF-SCOPE、编造 | 零售 5 / 21                       | 训练用 DeepSeek（已定），不用 Doubao                                                            | 不改评测                      |
| 训练 reward 罚转人工           | 转人工假阳性，以及不该转却转    | 航空约 30 的假阳性 + 约 10 的失败 | 航空任务把 NL 断言纳入训练 reward，或对无 gold 转人工的转人工罚分。属于 reward 设计改动，需批准 | 只改训练 reward，评测仍用官方 |
| 训练（模型能力）               | 参数、物品、支付、政策违规      | 航空约 17，零售 6–11              | GRPO / SAO（第 11 节）                                                                          | —                             |
| 评测时附报一个去掉转人工的视图 | 转人工假阳性                    | 航空约 30                         | 在官方指标之外，报"不含转人工成功的 pass^1"，用于判断训练是否只是学会了转人工                   | 不改官方指标                  |
| benchmark 本身                 | gold 不可行、歧义、环境 bug     | 1.5–3.5                           | 不修，作为固定的上限损失                                                                        | —                             |
| infra                          | Ark 欠费；32K 边缘超限          | 排除 2 条                         | Ark 充值并换 key；32K 不改                                                                      | —                             |

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

**没有任何来源修过的**：

- 评测端针对转人工的直接规则；
- airline:7 的金额歧义（只有 issue
  [#495](https://github.com/sierra-research/tau2-bench/issues/495)）；
- retail:38 的取消原因、retail:60 的支付方式、retail:110 的空操作断言；
- Airline `payment_history` 的列表顺序
  hash（[#514](https://github.com/sierra-research/tau2-bench/issues/514)）。

另外 NL judge 本身有已报告的缺陷，例如
[#554](https://github.com/sierra-research/tau2-bench/issues/554) 中空回复会被判通过。把
NL_ASSERTION 放进训练 reward，这些缺陷会直接变成 reward 噪声。

**转人工规则的边界很清楚**：gold 动作含 `transfer_to_human_agents` 的只有 Airline 1 题（13）和 Retail 4
题（10、12、26、50）。Airline 有 23 题、Retail 有 7 题没有 gold 写操作。

分类脚本和逐条结果：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-failure-taxonomy-20261010/`。其中
`auto.json` 的 `final` 字段是逐条分类，`table.py` 的 `MAN` 字典记录精读时的人工改判。

## 11. 下一轮训练计划（2026-10-10，待批准启动）

### 11.1 题目划分

不用 Telecom，其余划分与之前相同（[ADR 0003](../../decisions/0003-tau2-data-and-evaluation.md)）。

| 模式           | 训练集                                         | 训练中评测                                        |
| -------------- | ---------------------------------------------- | ------------------------------------------------- |
| formal（本轮） | 官方 train 104 题：Airline 30、Retail 74       | 官方 test 60 题：Airline 20、Retail 40            |
| tune           | 从官方 train 切出 83 题：Airline 24、Retail 59 | dev 21 题：Airline 6、Retail 15，也来自官方 train |

- tune 的 dev 切法：每个领域先排序，再用 `Random("42:<domain>")` 打乱，取 `round(n × 0.2)` 题。代码会检查 dev
  与训练集不重叠。
- 每个领域的题数和 sha256 必须与 `examples/tau2/contracts.py` 的 `OFFICIAL_SPLIT_SNAPSHOTS`
  一致，否则拒绝启动。
- **之前的划分**：最初是 Airline + Retail + Telecom 三域。上一次 formal critic 用了 178 题，共 24 步。Telecom
  在训练前就是 320/320 全对，166 个 GRPO 组里 160 个全对，没有组内对比，却占了约 42% 的 rollout，所以 ADR 0003 把 policy
  训练改成两域。这次 critic 也去掉 Telecom。
- 训练只在 train 上做，test 只记录、不用来选 checkpoint。base 集（164 题）里包含 train，所以训练后的 base 分数不是
  held-out 结果，test 60 题才是。

### 11.2 运行顺序与配置

都用 DeepSeek-Flash 做用户模拟器和 NL judge：temperature 0，无回复上限，key 从 `TAU2_DEEPSEEK_ENV_FILE`
读取。

| 步骤            | 配置                                            | 步数                                                                    | 存档与评测                                  | 评测内容                                                |
| --------------- | ----------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------- |
| 1. critic       | `scripts/tau2/train_critic.sh`，formal          | 2 epoch：$\\lceil 104/16 \\rceil \\times 2 = 14$ 步，208 条训练 episode | 第 5、10、14 步存档；第 0、5、10、14 步验证 | critic 在 60 条固定 test 轨迹上的 MSE、EV（按领域分开） |
| 2. SAO          | `run.sh policy mixed sao`，加载第 1 步的 critic | 2 epoch，26 步                                                          | 每 5 步存档并评测，训练前先评一次           | test 60 题 × 4 次，temperature 0：mean@4、pass^1/2/4    |
| 3. GRPO（对照） | `run.sh policy mixed grpo`                      | 2 epoch：$\\lceil 104/8 \\rceil \\times 2 = 26$ 步，每题 8 条           | 同上                                        | 同上                                                    |

- 已用 `--check-config` 核对过三份配置，都退出 0，没有调用 API 或 GPU：
  - critic：train 104、validation 60、步数 14；
  - GRPO / SAO：领域只有 Airline + Retail，步数都是 26，每 5 步存档和评测；
  - 用户只设 temperature 0。
- **critic 的 epoch**（ADR 0005）：actor 固定（lr 0），每个 epoch 给每道训练题现采一条 temperature 1 的
  episode，只更新 critic。每个 epoch 的最后一个 batch 只有 8 题（`drop_last: false`）。
- **critic 的验证**：开头用训练的 `gconfig`（temperature 1，每题 1 次）采一次 60 条固定轨迹，之后每次验证都复用。验证测的是
  critic，不是 policy，所以没有 mean@4 / pass^k。这一点写死在 `train_critic.py`，要改成 temperature 0 × 4
  必须改代码。
- **资源**：8 张 A100-80GB，当前全部空闲。
  - critic：rollout 4 卡，actor 和 critic 共用另外 4 卡，阶段之间 offload。
  - GRPO / SAO：rollout 3 卡，独立评测 1 卡，actor 4 卡。
- **critic 耗时与用量（估计）**：上一次 critic 跑 24 步用了 1h44m，按比例本次约 1–1.5 h。DeepSeek 用户侧约 2.2M
  prompt token、0.55M completion token，按 temperature 1 和重试留 1.3 倍余量，约 3M / 0.7M。这个用量里是否包含
  NL judge 调用，没有核实。
- 每次运行都要用新的 `TAU2_RUN_ROOT`。`recover.mode=auto` 会续跑同一目录，而旧 `critic-formal-v1` 是含 Telecom
  的运行，固定验证集的 digest 对不上。

### 11.3 这轮能验证什么

- **critic 能否在两域上学到价值**：60 条 test 轨迹上 MSE 下降、EV 上升，按领域分开看。
- **SAO / GRPO 是否提高 test 成功率**：每 5 步一次 mean@4 和 pass^1/2/4。step 0 的 temperature 0
  评测由训练启动时自动产出，作为同口径基线。
- **SAO 与 GRPO 的对比**：两者 rollout 预算相同，题集相同，评测相同。
- **稳定性（pass^4 − pass^1 的差）**：训练是否让模型在同一题上更一致。
- **链路**：去掉用户回复上限后，在线采样、训练、存档和 recover 都能跑完。

### 11.4 这轮验证不了什么

- **和 Recuris 或榜单的绝对分数对比**：用户模拟器不同（7.2 第 1 项），且没有 Telecom。
- **成功是真解决还是转人工**：官方指标分不出来（10.3）。训练 reward 不改的话，航空的提升可能来自更多转人工。要分辨，需要第 10.4
  节的附加视图，这个视图还没实现。
- **小于噪声的提升**：test 只有 60 题。此前同一初始模型的两次 step 0 在 Retail+Airline 上差 4.58pp，仅采样噪声的 95%
  半宽合并约 ±4.1pp、Airline ±6.4pp（3.1）。164 题 base 集上的 bootstrap 区间宽 12–18pp（9.2），60
  题只会更宽。几个点的变化不能当成训练有效。
- **对 Doubao 或 gpt-5.2 用户的泛化**：训练和评测都用 DeepSeek 用户，换用户后的效果需要另跑评测。
- **critic 对 SAO 的具体贡献**：critic 的 MSE / EV 只说明它拟合了 temperature 1 下的回报；它对 SAO 有没有帮助，只能从
  SAO 与 GRPO 的对比间接看出，而这个对比还混着两种算法的其他差异。
- **训练中途的中止风险**：GRPO 组里一旦有 episode 被排除，或用户首句就 stop，训练会中止（7.2 第 10 项）。DeepSeek
  用户下还没发生过，但不能排除。

启动需要批准。启动前先把当前改动 commit，因为相关文件的哈希会写进 critic 固定验证集的 manifest。

## 12. 合入上游修复与训练侧改动（2026-10-10）

### 12.1 两个 benchmark 变体

可用的上游 PR 已合入，原版也保留，按 `econfig.benchmark_variant`
选择（[ADR 0010](../../decisions/0010-tau2-official-and-fixed-benchmark-variants.md)）。

| 变体       | 内容                                                                                                                     | 数据目录                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `official` | tau2-bench `b7ea907` 原样，可与榜单比                                                                                    | `_cache/tau2-bench-b7ea9074…/data`                                                               |
| `fixed`    | #411 的 reward_basis、#571 替换 retail 64/105、#322 修改价环境 bug、转人工规则（gold 不含转人工时，agent 转人工记 0 分） | `_cache/tau2-bench-b7ea9074…-fixed-pr411-pr571/data`，由 `scripts/tau2/build_fixed_data.py` 生成 |

- #580 没有作为代码合入。它的作用是：gold 写操作执行抛错时，DB 判定不能算通过。合入 #571
  后，`scripts/tau2/check_gold_replay.py fixed` 显示 Airline/Retail 没有执行抛错的 gold
  写操作（official 有 retail 64 和 105 两条），#580 不会改变任何分数。这个脚本保留下来，用于持续检查。
- 生成的 fixed 数据与独立做 git merge（#411 + #571）得到的结果逐题一致。两个变体都固定了 `tasks.json` 的
  sha256，数据目录和变体不匹配时，评测、训练和 critic 启动都会直接报错（已实测）。
- #322 改变了 5 道 Retail 题的 gold DB hash：train 20、21、37，test 36、100。

### 12.2 训练集也被修复了

这些修复按题生效，不区分 train/test，所以官方 train 104 题也在修复范围内：

| 领域    | 划分        | reward_basis 改动 | 新增 NL 断言计分 | 无 gold 写操作 | gold 含转人工 | 题目替换 |
| ------- | ----------- | ----------------- | ---------------- | -------------- | ------------- | -------- |
| Airline | train（30） | 25                | 17               | 17             | 无            | 无       |
| Airline | test（20）  | 19                | 9                | 7              | 13            | 无       |
| Retail  | train（74） | 67                | 0                | 6              | 10、50        | 105      |
| Retail  | test（40）  | 38                | 0                | 4              | 12、26        | 64       |

- Retail 的 basis 改动大多只是删掉空的 NL_ASSERTION（没有断言却列在 basis 里），或补上已有的
  COMMUNICATE/ACTION，不改变实际评分对象。Airline 的改动才是实质性的：17 道 train 题开始对 NL 断言计分。
- 社区没有专门针对 train 的修复。10.2 里 train 上的问题，有修复的是 retail:105（#571）、retail:20/21/37（#322 价格
  bug），以及 Airline 拒绝题的转人工（#411 + 转人工规则）。airline:7（train）和
  retail:38（test）、60（test）、110（train）仍然没有任何来源修过。
- 代价：Airline 17 道 train 题的 reward 现在依赖 NL judge，而 judge 就是 DeepSeek 用户模型（7.2 第 2
  项），judge 的噪声会进入训练 reward。

### 12.3 训练侧改动

- **单次回复上限**：保留 4096，以节省资源（用户决定）。12% 到 17% 的 Airline 训练 episode
  至少有一次回复被截断（9.5），这一代价已知并接受。
- **训练墙钟超时**：已改为保留 episode，标记为 truncated，reward 0，不再重跑（ADR 0009 修订）。原来的重跑做法只保留按时跑完的
  episode，会系统性地去掉慢的长轨迹。
- **组内成员丢失**：12.4
  的方案已批准并实现（[ADR 0011](../../decisions/0011-tau2-lost-episodes-in-training.md)），细节见 12.4
  末尾。

### 12.4 组内成员丢失：同类框架的做法

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

**建议**：

1. **我们已经做了分类**：上下文超限和墙钟超时记 0 分训练（模型或预算导致的失败），其余异常算 infra。
1. **infra 失败先对单个成员从头重跑**，上限 3 次，这是现有行为。
1. **重试用尽后**：
   - 存活成员 ≥ $\\lceil n/2 \\rceil = 4$ 时，用 `traj_group_sizes` 在存活成员上算 mean/std。
   - 不足 4 个时，把这道题推到同一个 epoch 的末尾再跑一次，保持"每题一次"；还失败就记为 skipped。
   - 每个 epoch 允许的 skipped 题数要有上限（建议 ≤ 2 题，约 2%），超过就中止训练。
   - 阈值 4 是推断出来的，没有框架给出过这个具体数字。
1. **用户第一句就结束、policy 一个 token 都没产出**：这种成员的 reward 与策略无关，不计入
   baseline，也不训练。同一题反复出现时单独报出来。DeepSeek 用户下 656 条里没有出现过。
1. **训练指标**：报每道题的 infra 失败率，以及失败与轮数的相关性。这是判断第 2、3 步是否无偏的唯一依据。

本地 fork 与 upstream 的差别：本地的 `_normalize_group_rewards` 只要组里有一个 None 就丢整组，再加上
`fail_on_rejection=True`，结果就是训练中止。upstream 已经改为在存活成员上归一化。

**实现**（已获批准，2026-10-10）：

- 训练中，episode 重试用尽后从组里丢掉，不再让整组失败（`reject_exhausted_episode`）。
- 不足 4 个成员的组被拒绝（`min_usable_group_size`）。
- 被拒的题在本 epoch 末尾重跑一次；重跑再被拒就记为 skipped。每个 epoch 超过 2 次
  skipped，训练中止（`max_skipped_tasks_per_epoch`）。
- critic 每题只有一条 episode，所以在同一步内重跑一次，预算相同。
- recover 时，等待重跑的题会写进恢复缓冲区。
- 训练指标新增 `tau2_batch/epoch_deferred_tasks` 和 `tau2_batch/epoch_skipped_tasks`。

还没实现的是第 5 条：按题统计失败率，以及失败与轮数的相关性。

### 12.5 下一步

1. 12.4 已实现。
1. 训练使用 fixed 变体（用户决定）。两个变体的评测都要报。
1. 在两个变体上各跑一次 step 0 评测，结果见 12.6。
1. 按第 11 节启动 critic、SAO 和 GRPO，需要另行批准。

### 12.6 两个变体的 step 0 评测（2026-10-10）

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
| fixed    | 合计          | 164  | 1    | 0.690           | 0.571  | 0.460  |
| fixed    | Airline       | 50   | 1    | 0.428           | 0.297  | 0.224  |
| fixed    | Airline train | 30   | 1    | 0.364           | 0.228  | 0.138  |
| fixed    | Airline test  | 20   | 0    | 0.525           | 0.400  | 0.350  |
| fixed    | Retail        | 114  | 0    | 0.805           | 0.692  | 0.561  |
| fixed    | Retail train  | 74   | 0    | 0.845           | 0.750  | 0.635  |
| fixed    | Retail test   | 40   | 0    | 0.731           | 0.583  | 0.425  |

fixed 排除了 1 条 episode，`airline:14:trial1`：4 次尝试都超过了 32K 上下文。它所在题的 pass^4 只按 3 次有效 trial
无法计算，所以 Airline 的 pass^4 只在 49 题上算。

**fixed 的下降几乎全在 Airline，主要来自转人工规则**：

- Airline 从 0.780 降到 0.428。fixed 运行中，转人工规则之前的分数是 0.744，规则把 199 条 episode 里 63 条原本成功的记为
  0，扣了 31.7pp。剩下约 3.6pp 是 #411 新增的 NL 断言计分，这个量与下面的运行间噪声同级，不能单独确认。
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

### 12.7 Retail 的题目问题与社区修复（2026-10-10 调研）

**结论**：

- 官方 main 在 `b7ea907` 之后没有改过 Retail 的数据、工具或评估器。
- 社区的 Retail 修复都还是 open 或已关闭未合并的 PR。我们发现的 38、60、110，没有任何 issue、PR 或社区修订版处理过。
- 与 Airline 不同，Retail 的问题不在评分规则，而在题目本身：
  - 指令没写清楚关键参数；
  - 有些题什么都不做也能得分；
  - train 和 test 之间有重复题。

下表中，"已核实"表示我或调研 agent 读过代码或数据，或在本地重放过；其余来自 issue 或论文文字。

| 问题                                                                 | 题号（划分）                                                                                    | 来源                                        | 现状                    | 影响                                                                         |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| 指令没写支付或退款方式，gold 却固定了一种                            | 41、63、16、98（train）；60、61（test）                                                         | 我们核查；#319                              | 无修复                  | 用户模拟器选另一种合法方式时，正确操作也记 0 分；训练 reward 有噪声          |
| 指令没写取消原因，gold 固定为 "no longer needed"                     | 66（train）；38（test）；30、31、54、59、69（train）和 32、55（test）只是暗示了原因             | 我们核查；#392 只修了 66                    | #392 open               | 同上                                                                         |
| NL 断言要求一次空操作写入                                            | 110（train）                                                                                    | 我们核查，已复核：gold 地址本来就是目标地址 | 无修复                  | 按断言做是空写入，不做又不满足断言                                           |
| NL 断言写的是"应当执行某操作"，但 judge 只看对话文本，看不到工具调用 | 105–107、109、110、113（train）；111（test）                                                    | 读评估器代码                                | 无修复                  | 与 DB 检查重复，只增加 judge 噪声                                            |
| gold 不改 DB，什么都不做也能得 1 分                                  | 10、25、50、57（train）；12、65（test）                                                         | 本地重放；#384                              | #411 只覆盖了其中一部分 | 训练时会奖励不作为                                                           |
| gold 要求转人工，agent 不转也得分                                    | 50（train）；26（test）                                                                         | #320；arXiv 2605.10448                      | 无修复                  | 我们的转人工规则只处理反方向（不该转却转了）                                 |
| 指令与 gold 不一致                                                   | 10（train）：指令说退两单，gold 只读取并转人工；82（train）：指令说收到两台平板，订单里只有一台 | #316；82 是我本次在轨迹里发现的             | 无修复                  | 三次 DeepSeek 运行 82 都只有 0.25，41 是 0–0.25；41 的失败来自支付方式未指定 |
| train 和 test 重复题：同一用户、gold 写操作完全相同                  | 72↔71、96↔97、37↔36（38 是同一订单）、41↔42、67↔68                                              | 已复核                                      | —                       | test 被训练集污染                                                            |
| 近似重复：指令相似度 ≥ 0.8，同一用户                                 | 31↔32、93↔94、6/7/8↔5/9、63↔62、13/14↔12、34↔33、25↔26                                          | 我们核查                                    | —                       | 轻度污染                                                                     |
| 已修复                                                               | 105、64（#571）；20、21、36、37、100（#322）                                                    | —                                           | 已在 fixed 中           | —                                                                            |

修正一个之前的说法：Retail 的"DB hash 对列表顺序敏感"其实是 #322 那个价格 bug 造成的（issue #325）。打上 #322 后，没有任何 gold
写操作对顺序敏感。

**不建议采用的**：

- Amazon verified 对 49、18、91、21 的改动。49 的新 gold 不在订单里，与指令矛盾；18、91 的目标商品是
  `available: false`；21 把 product id 当作 item id。我们 pin 的版本在这几题上是对的。
- #561：它会替换 `pyproject.toml`。

**可以考虑的**：

- #392 的数据部分（66 的取消原因、93）。不要采用它把用户模型默认换成 gpt-5.5，也不要采用它对转人工停止条件的改动。
- #356（已关闭）对 5–9 的 `known_info` 修改。
- Amazon verified 只改指令提示的那部分（8、35、42、59、71、72、74、93、109）。

**Retail 的处理建议**，作为下一版 fixed 的候选，尚未实现：

- **train（74 题）**，目标是降低训练 reward 的噪声：
  - 在指令里补上支付或退款方式（41、63、16、98），补上取消原因（66，取自 #392）。
  - 删掉 110 的空操作断言，以及 105–107、109、113 中描述动作的 NL 断言，只保留 DB 检查。
  - 修正 82 的指令，或把 82 排除。
  - 把重复题 72、96、37、41、67 移出训练，否则 test 上对应的 71、97、36、38、42、68 等于见过的题。另一种做法是保留它们，但 test
    单独报一个"去重后"的子集。
  - 什么都不做也能得分的 10、25、50、57：要么排除，要么要求出现 gold 中的转人工或必须传达的信息。
  - 增加反向的转人工规则：gold 中有转人工而 agent 没转，记 0 分。
- **test（40 题）**：
  - official 保持原样不动。
  - fixed 中，38 接受任一取消原因，60、61 接受任一支付方式；12、65 单独标出。
  - 另报一个"排除重复题"的 test 子集（去掉 71、97、36、38、42、68）。

**需要决定**：

1. 是否把上面的 Retail 修订做进 fixed。这会改变 fixed 的定义，step 0 要重跑，结果与 12.6 不再可比。
1. 重复题怎么处理：从训练中移除，还是保留并在 test 上单独报去重子集。

调研脚本和中间结果：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-retail-audit-20261010/`。
