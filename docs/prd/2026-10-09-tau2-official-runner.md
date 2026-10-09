# PRD：τ² 评测调度层按官方 runner 语义重写

日期：2026-10-09 状态：已实施（afb42082）

相关记录：[ADR 0009](../decisions/0009-official-tau2-runner-semantics.md)、
[2026-10-09 审计第 7 节](../research/2026-10-09-tau2-high-baseline-audit/README.md)

## 1. 背景

我们的 τ² 接入分两层：

| 层                     | 官方 `tau2 run`（b7ea907）                                                                        | 我们                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 单个 episode           | `Orchestrator`、`LLMAgent`、`UserSimulator`、`evaluate_simulation`                                | 同一份官方代码                                                          |
| episode 之上（调度层） | `tau2/runner/progress.py` 重试，`tau2/runner/batch.py` 分发，`tau2/metrics/agent_metrics.py` 汇总 | AReaL rollout controller、`OpenAIProxyWorkflow`，加上我们自己的失败处理 |

单个 episode 已经和官方一致；调度层被我们自己的规则替换掉了。ADR 0001 当时定了"infra 失败不能变成 0
分；与其悄悄缩小分母，不如中止"，于是任何没有归类的异常都会让整轮评测中止。官方的做法是：重试，仍失败就记为
`INFRASTRUCTURE_ERROR`，从指标中排除并报告数量，评测继续。2026-10-09 的 Doubao 2.1 Turbo 评测就是这样中止的：
用户在第一轮输出 stop，导致零 policy 回合，被当成导出错误。

AReaL 并不妨碍完全照官方做。唯一的结构性约束是：AReaL 里每个结果都要导出 token 轨迹，所以零 policy 回合或最终失败的 episode
没有轨迹可交，dispatcher 交回的是一个不带 task ID 的 `None`。

## 2. 官方语义（源码出处，均在 `tau2` 包内）

- **重试**：任何 `Exception` 都整段重跑，最多再试 `DEFAULT_RETRY_ATTEMPTS = 3` 次，每次间隔
  `DEFAULT_RETRY_MIN_WAIT = 1.0` 秒（`config.py:41-42`、`runner/progress.py:52-103`）。
- **最终失败**：生成一个占位 `SimulationRun`，`termination_reason = INFRASTRUCTURE_ERROR`、没有 reward，
  不抛出异常（`runner/progress.py:113-131`）。
- **汇总**：先排除 `INFRASTRUCTURE_ERROR`，在剩下的 episode 上算 `avg_reward`，并按任务算
  `pass^k = C(c,k)/C(n,k)`；k 的上限是各任务中最少的 trial
  数（`metrics/agent_metrics.py:136-181, 202-250`）。
- **终止原因**：`AGENT_STOP` 和 `USER_STOP`
  正常打分；其他终止原因（`MAX_STEPS`、`TIMEOUT`、`TOO_MANY_ERRORS`） 直接得 0
  分，但计入分母（`evaluator/evaluator.py:119-129`）。用户第一轮就 stop 属于 `USER_STOP`，正常打分。
- **上下文溢出**：这个 revision 里没有任何代码设置 `CONTEXT_WINDOW_EXCEEDED`。溢出时 provider 报错，走上面的重试，
  最终被排除。
- **用户、judge 和 agent 的 LLM 参数**：用户只有 `temperature 0`；judge 是
  `gpt-4.1-2025-04-14`，`temperature 0`； LLM 请求的 `num_retries` 默认
  3（`config.py:21-26, 40`、`utils/llm_utils.py:379-380`）。
- **墙钟**：`--timeout` 默认不设。设了也只在每一步之间检查（协作式），超时记为 `TIMEOUT`，得 0 分并计入。

## 3. 缺口清单与处理

| #   | 缺口                                                                                   | 处理                                                                                                   | 理由                                                        |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| G1  | 只有白名单内的异常才重试，且只重试 1 次                                                | **对齐**：任何异常最多再试 3 次，间隔 1 s，每次重开 proxy session                                      | 官方语义                                                    |
| G2  | 最终失败让整轮中止                                                                     | **对齐**：记为 `infrastructure_error`，排除出指标，报告数量和原因，评测继续                            | 官方语义                                                    |
| G3  | 不传 seed                                                                              | **不改**                                                                                               | 用户决定（2026-10-09）                                      |
| G4  | 上下文溢出记 0 分并计入                                                                | **评测对齐**：抛异常，重试后排除；上下文仍是 32K。**训练不变**（ADR 0002：记 0 分，作为学习信号）      | 用户要求保留 32K；官方没有训练模式                          |
| G5  | 评测超时当作 infra 错误；已完成但耗时超限的 episode 也被丢弃                           | **评测对齐**：`TIMEOUT` 按官方打分（得 0、计入）；已完成的 episode 不再按耗时丢弃。评测墙钟设为 1800 s | 用户要求设一个偏宽的硬上限；实测 eval p99 538 s、最长 651 s |
| G6  | 用户回复为空或工具 JSON 格式错误时原地重试一次；evaluator 的 JSON 错误原地重新打分一次 | **对齐**：去掉原地重试，抛异常，按 G1 整段重跑；保留 incident 记录                                     | 官方语义                                                    |
| G7  | NL judge 用的是用户模拟器的模型                                                        | **待定**：没有 OpenAI key                                                                              | 只影响 reward_basis 含 NL_ASSERTION 的任务                  |
| G8  | 用户回复上限 `max_completion_tokens: 512`                                              | **对齐**：所有配置都去掉                                                                               | 官方只设了 temperature                                      |
| G9  | policy 有 4096 输出上限，`num_retries 0`                                               | **不改**：4096 和 32K 属于生成协议（ADR 0002）；policy 走本地 proxy，重试放在 episode 层               | 有限条件                                                    |
| G10 | 只报 pass1_mean                                                                        | **对齐**：加上 `avg_reward`、`pass^k` 和排除数                                                         | 官方语义                                                    |
| G11 | 零 policy 回合的 episode 在 AReaL 里拿不回 task ID                                     | **修**：dispatcher 可返回带 task ID 的结果；episode 记录说明它是怎么结束的                             | G2/G5 的前提                                                |
| G12 | 训练中的 eval 有自己一套汇总和失败处理                                                 | **对齐**：和 `evaluate.py` 共用同一份调度模块                                                          | 两条 eval 路径必须同口径                                    |

不在本次范围、需要后续决定的：

- G7 的 judge 渠道。
- 训练里被排除或零 policy 回合的 episode 怎么处理。GRPO 需要完整的组，现在一组不完整训练就会中止
  （`fail_on_rejection=True`）。这会改变组的构成，属于算法层面的决定。DeepSeek 用户下至今没出现过。

## 4. 需求

R1 **共享调度模块** `examples/tau2/official_runner.py`。evaluate.py 和训练中的 eval 都用它，它是唯一实现官方
runner 语义的地方：

- 把每个任务展开成 trial 行，带 `source_id`。
- 滚动提交、按 task ID 收结果。
- 判定每个 episode 的结局：`completed` 或 `infrastructure_error`。
- 按官方公式汇总。

这个模块不依赖 AReaL 的训练语义，可以作为以后接入 PHAI Relay 的样板。

R2 **重试**：`Tau2AgentWorkflow.should_retry_episode` 对任何 `Exception` 返回
True，`CancelledError` 不重试，和官方 shutdown 时直接放弃一致。重试次数默认 3、间隔 1.0 s，都从 `tau2.config`
读取。所有配置的 `infra_retries: 3`。

R3 **结局记录**：每次 episode 尝试都写一份 JSON 到
`trajectory_dir`，包含官方消息、`stop_reason`、`policy_turns`、`reward_info` 和错误信息。目录由 workflow
kwargs 显式传入：评测每次运行一个新目录；训练写在本次 run 的日志目录下。

R4 **结局判定**：

| 结果                                                            | 判定                               |
| --------------------------------------------------------------- | ---------------------------------- |
| 有 token 轨迹                                                   | `completed`，分数读张量            |
| 无轨迹，且最后一次尝试的记录是 `completed`、`policy_turns == 0` | `completed`，分数读记录            |
| 其他无轨迹的情况，包括没有记录                                  | `infrastructure_error`，排除出指标 |

R5 **AReaL 最小改动**：

- `BatchTaskDispatcher.wait_results(..., with_task_ids=True)` 返回
  `(task_id, result)`，默认行为不变。
- proxy workflow 在导出为空时拒绝该轨迹；重试间隔读 agent 的 `retry_delay`。

R6 **评测结局规则**（workflow kwarg `evaluation=True`）：

- 上下文溢出：抛异常。
- 超时：按官方打分。
- 已完成的 episode：不按耗时丢弃。

训练（`evaluation=False`）保持 ADR 0002/0007 的规则不变。

R7 **评测墙钟**：新增 `eval_episode_timeout_seconds`，默认 1800；`evaluate.py` 和训练中的 eval 都使用它。

R8 **summary 字段**：

- `eval/avg_reward` 和 `eval/pass^k`，总体和分领域各一份。
- `eval/infra_excluded_episodes`，以及按原因的明细。
- `eval/policy_free_episodes`。
- 保留现有的 `pass1_mean`、`by_domain`、`by_domain_split`。
- 有 episode 被排除时，评测仍然正常结束、返回码为 0；`status` 写为 `complete`。

## 5. 验收

A1 单元测试覆盖：

- 任何异常都重试，取消不重试；最多 4 次尝试。
- 结局判定：三种情况各一条。
- 官方 pass^k 与排除规则：和官方 `pass_hat_k` 对拍。
- 带 task ID 的 dispatcher 结果。
- 评测结局规则：溢出和超时。
- 训练中的 eval 走共享模块。

A2 smoke test：真实 SGLang 加 Qwen3.5-4B，DeepSeek 用户前面挂一个本地故障注入代理。每个场景跑一条 episode：

| 场景 | 注入                               | 预期                                                                          |
| ---- | ---------------------------------- | ----------------------------------------------------------------------------- |
| S1   | 第一次尝试时，用户第一句返回空内容 | 尝试 1 抛 `ValueError`，尝试 2 正常完成并打分；记录里有 2 次尝试；不排除      |
| S2   | 用户第一句返回 `###STOP###`        | `user_stop`，`policy_turns = 0`，按官方打分并计入；`policy_free_episodes = 1` |
| S3   | 每次尝试用户第一句都返回空内容     | 4 次尝试后记为 `infrastructure_error`，被排除；评测正常结束，返回码 0         |

A3 pre-commit 全部通过，tau2 测试全部通过。

A4 ADR 0009 写成，并更新 0001、0002、0003 的相关部分；README 的 ADR 索引同步更新。

## 6. 验收结果（2026-10-09）

- A1/A3：tau2 相关测试与 dispatcher 测试共 189 个全部通过；新增 `tests/test_tau2_official_runner.py`。
  故意改坏 `max_k` 截断和 `policy_turns == 0` 判定后，对应测试变红，恢复后变绿。改动文件的 pre-commit 全部通过。

- A2：一次运行覆盖三个场景（`--task-set base --repeats 1 --tasks retail:51,retail:38,retail:69`，
  Qwen3.5-4B 加 SGLang d1，DeepSeek 用户前挂故障注入代理），返回码 0，`status: complete`。

  | 场景 | 任务      | 尝试记录                                                                 | 结局                                     |
  | ---- | --------- | ------------------------------------------------------------------------ | ---------------------------------------- |
  | S1   | retail:51 | 第 1 次 `EmptyUserResponseError`，第 2 次 `user_stop`、10 个 policy 回合 | 得 1.0，计入                             |
  | S2   | retail:38 | 1 次，`user_stop`，`policy_turns = 0`                                    | 得 0.0，计入，`policy_free_episodes = 1` |
  | S3   | retail:69 | 4 次，均为 `EmptyUserResponseError`                                      | `infrastructure_error`，排除             |

  汇总：`avg_reward = pass^1 = 0.5`（2 个计入 episode），`infra_excluded_episodes = 1`。
  逐次尝试记录大小：10 个 policy 回合的 episode 共 62 KB，零回合的约 5 KB。 排除记录的 `error_type` 是外层的
  `Tau2InfrastructureError`，原始异常类型在 `error` 文本里。

- A4：ADR 0009 已写，0001、0002、0003 和索引已更新。
