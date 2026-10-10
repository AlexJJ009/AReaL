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

## 7. Doubao 2.1 Turbo 评测中止与官方行为对齐（同日补充）

### 7.1 中止原因

`step0-base-t0-doubao21turbo` 跑到 74/656 个 episode 时中止。其中 6 个 episode 都是 Doubao
第一句用户回复就触发了 `UserSimulator.is_stop()`，官方 orchestrator 以 `user_stop` 结束，policy
一次都没被调用。结论来自排除法：tau2 在第一句用户回复后 到调用 agent 之前，只有 stop token 这一条路会正常打分结束；异常、超时、空回复都会 变成
infra error，不会打印 FINISHED。用量行（每个只有 1 次 user 调用，prompt tokens 与第一轮完全一致）也与此相符。

用同一配置探测 236 次（其中 192 次并发），没有一次复现出 stop token。所以根因 （Doubao 为什么偶尔在第一轮就输出 stop
token）仍未确认；当时的回复原文没有保存。

下游链路：proxy session 里没有 interaction，`set_reward` 返回 400，concat 导出 0 条轨迹，触发
`Tau2ExportCardinalityError` 并被当作暂时性故障重试。retail/69 t0 连续两次都这样，整轮评测中止。官方 tau2 会把这个
episode 正常记分。

### 7.2 修改

| 位置                                          | 修改                                                                                                                                                                                                                                                         |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `examples/tau2/agent.py`                      | 设置 `TAU2_TRAJECTORY_DIR` 时，每次 episode 尝试都写一份 JSON（成功、零 policy 回合、异常或取消都写），包含完整官方消息、`stop_reason`、`policy_turns`、`reward_info`、错误信息；失败时保留 orchestrator 已产生的部分轨迹                                    |
| `examples/tau2/agent.py`                      | concat 导出 0 条轨迹不再报 cardinality 错误，因此不再被重试                                                                                                                                                                                                  |
| `areal/experimental/openai/proxy/workflow.py` | 正常模式下导出为空时返回 None（拒绝该轨迹），与 online 模式一致                                                                                                                                                                                              |
| `scripts/tau2/evaluate.py`                    | 写 `trajectories/<source_id>/<attempt>.json`。被拒绝的结果没有 task_id，用 episode 自己的记录（completed 且 `policy_turns == 0`）把它绑定回对应 episode，按官方分数记为 completed；其余没有 task_id 的失败仍然中止。summary 新增 `eval/policy_free_episodes` |

训练行为：零 policy 回合的 episode 被拒绝（以前是重试一次后再拒绝），GRPO 组按 `drop_incomplete_group` 处理。（更正：AReaL
的 `dump_to_file` 本来就会写 policy token 视角的轨迹；缺的是官方消息视角的逐次尝试记录，见 ADR 0009。）

### 7.3 与官方 `tau2 run` 仍不一致的行为

逐项对照官方源码（pinned b7ea907）。前三项已在源码中复核：`runner/progress.py`、
`metrics/agent_metrics.py`、`runner/batch.py`。

| #   | 行为                               | 官方                                                                      | 我们                                                                                                 | 影响                                         |
| --- | ---------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 1   | episode 失败重试                   | 任何异常都整段重跑，最多再试 3 次                                         | 只重试 timeout、连接、限流、5xx、导出形状这几类错误，只重试 1 次；其他错误让评测中止                 | 官方能恢复的失败，我们会中止整轮             |
| 2   | 最终失败的 episode                 | 记为 `INFRASTRUCTURE_ERROR`，从 avg_reward 和 pass^k 中排除，评测继续     | 整轮中止，summary 标为 partial                                                                       | 计数口径不同                                 |
| 3   | 随机种子                           | `random.seed(300)` 为每个 trial 生成 seed，传给 agent 和 user 的 LLM 请求 | orchestrator 不传 seed，`eval_gconfig.seed` 必须为空                                                 | 可复现性；provider 是否支持 seed 未核实      |
| 4   | policy 上下文溢出                  | 无专门处理，按异常重试，最终排除（推断）                                  | `context_limit`，记 0 分并计入                                                                       | 我们的分数偏保守                             |
| 5   | 墙钟超时                           | 默认不设                                                                  | 1200 s，超时或总耗时 ≥ 1200 s 的已完成 episode 都当 infra error 重试                                 | 慢但有效的 episode 被丢弃重采                |
| 6   | 用户空回复、用户工具 JSON 格式错误 | 抛异常后整段重跑                                                          | 原地重试一次请求，再失败就中止                                                                       | 计数口径不同                                 |
| 7   | NL judge                           | `gpt-4.1-2025-04-14`，temperature 0                                       | 用户模拟器同款模型和参数                                                                             | 只影响 reward_basis 含 NL_ASSERTION 的任务   |
| 8   | 用户 LLM 参数                      | 只有 temperature 0                                                        | 训练配置还有 `max_completion_tokens: 512`、`top_p` 1.0、120 s 超时；step 0 评测启动时已去掉 512 上限 | 512 上限可能截断用户回复                     |
| 9   | policy LLM 参数                    | temperature 0，num_retries 3                                              | 另有 max_completion_tokens 4096、上下文 32K、num_retries 0                                           | 回复可能被截断；单次传输错误会升级为整段重跑 |
| 10  | 汇总                               | avg_reward 加 pass^1..k                                                   | 只报 pass1_mean                                                                                      | 只是报告形式不同                             |

另有几项一致或只影响展示：max_steps 200、max_errors 10、不开 communication 校验、 LLMAgent、无 think
工具、airline/retail 无用户工具、cost/日志相关 monkey patch。

2026-10-09 决定：1、2、4（仅评测）、5（仅评测，墙钟 1800 s）、6、8（去掉 512 上限）、10 按官方对齐；3 不改；7 待定；9 评测去掉 4096
回复上限、训练保留。见 [ADR 0009](../../decisions/0009-official-tau2-runner-semantics.md) 和
[PRD](../../prd/2026-10-09-tau2-official-runner.md)。

## 8. 官方 runner 口径下的 step 0 评测（同日补充）

两组都用 commit `31634dee`：未训练的 Qwen3.5-4B（non-thinking），base 集（Airline 50 + Retail 114 题，不含
Telecom），policy temperature 0，每题 4 次，用户模拟器 temperature 0、无回复上限，32K context，评测不设单次回复上限。

| 用户模拟器       | 领域    | episode（排除） | mean@4 | pass@2 | pass@4  | pass^2 | pass^4  |
| ---------------- | ------- | --------------- | ------ | ------ | ------- | ------ | ------- |
| DeepSeek-Flash   | 合计    | 656（0）        | 0.838  | 0.925  | 0.970   | 0.752  | 0.646   |
| DeepSeek-Flash   | Airline | 200（0）        | 0.810  | 0.887  | 0.940   | 0.733  | 0.660   |
| DeepSeek-Flash   | Retail  | 456（0）        | 0.851  | 0.942  | 0.982   | 0.760  | 0.640   |
| Doubao 2.1 Turbo | 合计    | 656（2）        | 0.691  | 0.834  | 0.920\* | 0.548  | 0.426\* |
| Doubao 2.1 Turbo | Airline | 200（1）        | 0.783  | 0.863  | 0.898\* | 0.703  | 0.633\* |
| Doubao 2.1 Turbo | Retail  | 456（1）        | 0.651  | 0.822  | 0.929\* | 0.480  | 0.336\* |

\* Doubao 有 2 个 episode 在重试时遇到 Ark 账户欠费，被排除；pass@4 / pass^4 只在其余 162 题上算。

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
- 每个训练 run 只有 4 条预算截断，截断不是失败主因；SAO 长度自适应 GAE 让终局信号衰减约 0.513，不会指数消失。
- 训练过 train 部分后，base 集分数不是完全 held-out 的泛化结果。MUA-RL、RWML、AReaL-SEA
  等同类工作的题集和用户模拟器都不同，分数不能直接比。
- 逐 step 的训练中 validation 表（mean@4 / mean@8）以飞书文档为准。

### 9.6 下一步

1. 用 gpt-5.2（reasoning low）做用户跑一次 base 集，和榜单定标（需要 OpenAI 渠道和费用，需批准）。
1. 选训练用的用户模拟器前先定：Doubao 的提前结束会把正确行为记 0 分；DeepSeek 也有，频率约 1/4.5。
1. 把 NL judge 和用户模拟器分开配置（`econfig` 新增 judge 字段，属于配置结构改动，需批准），judge 固定为官方 gpt-4.1 或同一模型。

复现脚本：`scripts/official-runner/fault_user_proxy.py`（PRD 第 6 节的故障注入
smoke）、`scripts/official-runner/metrics_table.py` （从 `episodes.jsonl` 重算第 8 节的 mean@4 /
pass@k / pass^k，在仓库根目录运行）。
