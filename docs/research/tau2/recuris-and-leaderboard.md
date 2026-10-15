# 与 Recuris 论文和官方榜单的对比（2026-10-10）

> 来源：原 README 第 9.2–9.6 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 与 Recuris 的差距

Recuris Table 1 里 Qwen3.5-4B 的 68.0 / 75.3 是不加 harness 的基线（stock
`llm_agent`，`README.md:71-73`）；+Recuris 是 68.3 / 79.0。他们的 tau2 是上游 `8ebb749` 加
`third_party/tau2/recuris.patch`：`llm_agent.py`（含 prompt）、 `user_simulator.py`、NL/action
evaluator、retail/airline environment 和我们 pinned 的 b7ea907 逐字节相同，用户 guideline 和数据 `diff`
为空；口径同为 base 全量、4 trials、max_steps 200、max_errors 10、失败 episode 排除。

| 我们的用户（口径）                       | 领域    | 我们 | 按题 bootstrap 95% | Recuris | 差    |
| ---------------------------------------- | ------- | ---- | ------------------ | ------- | ----- |
| DeepSeek-Flash（official，base 集）      | Retail  | 84.0 | \[79.4, 88.2\]     | 68.0    | +16.0 |
| DeepSeek-Flash（official，base 集）      | Airline | 76.5 | \[67.0, 85.0\]     | 75.3    | +1.2  |
| Doubao 2.1 Turbo（fixed v3，test 60 题） | Retail  | 65.0 | \[54.4, 75.0\]     | 68.0    | −3.0  |

Turbo 一行只有 Retail 可比：Airline 在 fixed 下有转人工规则，与 Recuris 的 official 口径不同，也不是 base 全量。

结论：Airline 的 official 分数与 Recuris 接近。Retail 随用户模拟器变化：Doubao 类用户下我们的 65.0 与 Recuris 的
68.0 在误差内，看不出谁高；我们"偏高"只发生在 DeepSeek 用户的 Retail，原因是
[user-simulator.md §1](user-simulator.md) 的用户行为差异。剩余不确定：Doubao 2.0 Pro（Recuris）与 2.1
Turbo（我们）的差别无法复现；Recuris 没公开 4B 的部署参数（推理框架、tool parser、thinking）。

**早期核查中仍成立的两点（2026-10-09）**：

- **不是题集差异**。未训练模型在 train split（epoch 1 覆盖全部任务，权重相对变化约 1e-4）上为 Retail 84.8%、Airline
  78.3%；与 test 合并估算全量 114/50 题约 82%/79%。
- **论文 Table 1 的用户模拟器不是 4B 自己**。Recuris 源码 `src/recuris/adapters/tau2/treatment.py` 的
  transfer 配置固定 Doubao Seed 2.0 Pro 作为用户模拟器和 NL
  judge（`reasoning_effort=medium`，temperature=0），被测 agent temperature 强制为 0。

## 2 Recuris 的 Mode A 与 terminal gate

- **是什么**。Mode A 指"用户的确认里带着 `###STOP###`，之后没有 agent 回合"，也就是
  [user-simulator.md §1](user-simulator.md) 的模式（RT2 `orchestrator.py:303`）。代码只在这一处定义它。
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

## 3 官方榜单

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
- 所以 Airline > Retail 本身不算倒挂。我们的 Retail 分数随用户模拟器变化：Doubao 2.1 Turbo 用户比 DeepSeek 低
  15.6pp（fixed-v3 test， [user-simulator.md §2](user-simulator.md)）；DeepSeek
  用户下的排序（Retail 更高） 和 GPT-5.2 (none)、Qwen3.5-397B 一致。
- 榜上没有 ≤8B 的模型。4B non-thinking 在 official Airline 上 0.765，比 Opus 4.5 的 84.0 低 7pp、远高于
  GPT-5.2 (none) 的 52.5，说明我们的 用户模拟器可能比 gpt-5.2 更好说话。要和榜单对齐，需要用 gpt-5.2（reasoning
  low）做一次用户。

## 4 仍有效的早期核查

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
