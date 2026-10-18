# 用户模拟器重跑与 U1 prompt patch：主要缺陷可复现，修补效果依赖模拟器

2026-10-18 实验批次。单一标注者独立完成；未启动 sub-agent，未重采样、训练或调用付费 API。仓库只读，分析和报告保存在本目录。研究对象是同一 actor 在已有九组运行中的行为与评分来源。

## 1. 结论

**Turbo/Lite 的授权时同句 STOP、Qwen 的任务约束偏离，以及 agent 用转人工代替政策拒绝，都在重跑中复现。** 但次数和具体 episode 不稳定：未打 patch 的五组中，用户致败从 v3 的 168 降为 147，agent 致败从 204 增为 223，benchmark 致败从 13 变为 12；总成功只从 815 变为 818。相近总分掩盖了失败来源和具体题目的变化。DeepSeek 后端、NL judge 版本以及 Turbo 接口发生变化，这些比较不能解释为同一系统的纯随机复验。

**U1 patch 明显减少授权时结束，但没有消除用户问题。** DeepSeek、Turbo、Lite 的 U1 分别为 **3→0、42→16、20→0**，用户致败总数为 **10→1、52→27、32→16**。原 U1 消失后分别有 3、32、19 条成功；另有 1 条转为 agent 错误、3 条转为其他用户错误。Lite 仍会撤销必要地址目标、编造已送达事实；Turbo 仍会确认与 STOP 同句。不能把“U1 减少”写成“所有用户缺陷减少”。

**DeepSeek 关 thinking 后用户致败减少，原始成功没有随之上升。** U1 3→0、失败侧 primary U7a 5→0，用户致败 10→3；原始成功 180→177。四条认证失败暴露了 retail:9 未补姓名的 B1，不能算用户任务偏离。用户正文稍短，交互次数略增。

九组共 **2,160 个 completed episode，635 个失败：agent 406、用户 194、benchmark 35**。另有 20 个 U6 成功、45 个转接工具前收到 TRANSFER 的成功。这里的**致败原因**指实际轨迹中，使最终 reward 无法再为 1 的决定性一步由谁造成；不声称穷尽或执行了所有可能的反事实修复。每个失败只计一个原因，成功异常单列。本轮**未新增代码**。

## 2. 数据、口径和总表

### 2.1 九组运行均通过一致性核对

每组 test 60 题（Airline 20、Retail 40）×4 trial，恰有 240 个唯一 completed episode，Airline 80、Retail 160。实际数据目录从 run.log 读回，未使用启动脚本默认值替代；actor 为 Qwen3.5-4B step 0，snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，non-thinking，policy `eval_gconfig.temperature=0`，代码 `e0fdf4a318453e6915d33f2cc30cd864666b8b50`。任务文件哈希与 fixed-v3 cache 一致；patch 组任务和 gold 不变。NL judge 配置及逐轨迹元数据均为官方 `openai/deepseek-flash`、host `api.deepseek.com`。详见 [preflight.json](preflight.json)，各完整运行路径见 [matrix.md](matrix.md)。

用户 temperature 参数均为 0，但不能据此保证确定性：DeepSeek thinking 不按普通 temperature 采样约束解释，API 后端和 actor 并发批次也可能改变路径。Turbo/Lite 使用订阅接口、`reasoning_effort=medium`，GLM 并发 32；五组对照沿用 brief 指定的 v3 运行。旧实验记录把 DeepSeek 写为 0.713，而指定 v3 轨迹实际是 **173/240=72.08%**；本报告使用已核验的原轨迹和冻结标签，不沿用该笔误。

### 2.2 归因与证据

遵照 [v3 REPORT](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/research/tau2/user-sim-case-study-v3/REPORT.md)：用户违反 task scenario 或模拟协议归 U；agent 违反 policy、漏执行或用错工具归 M/T；双方按说明行事仍被任务、gold 或 fixed 规则拒绝归 B。已经恢复的错误不作致败原因，后发生的问题记 secondary、不叠加。用户被任务要求坚持、报错或改主意时，不因行为麻烦就归用户。实际工具报错与实际数据库写入分开读取。

U1 是尚待执行的正确授权与 STOP 同句；U3 是目标、约束或事实偏离；U4 是无依据额外要求，U4b 是未完成即结束或缩小目标；U5 是用户与客服角色混淆；U7a 是实际转接前发 TRANSFER。U6 只标成功中的可见帮助，不自动判作违规。其余 M/T/B 定义沿用 v3 和既有 capability codebook，见 [matrix.md 的代码表](matrix.md#%E4%BB%A3%E7%A0%81)。未出现的类别保持 0，不因本轮未发生而删去旧定义。

消息为 JSON `messages` 数组 **0-based**，m0 是 agent 问候；t0–t3 为 trial。[labels.jsonl](labels.jsonl) 记录全部 635 个失败及 65 个成功异常，正常成功不入标签文件。每条保留绝对轨迹路径、SHA256、原文短引文、工具调用及写入时间线、任务字段与行号、gold、reward breakdown、失败 NL 断言。评分总数以 `official_score` 为准，不能仅看 DB/NL 子项推断 fixed 总分。

规则仅作筛选。在失败侧，授权 STOP 规则筛出 ds 3、tb 46、lt 23、tbp 17、qw 2，其余 0；人工 U1 为 3、42、20、16、2。多筛的 8 条包含已经发生的 M2a、M8 或更早 U3；人工 U1 无规则漏项。见 [u1-crosscheck.json](u1-crosscheck.json)。成功侧对全部 1,525 个成功筛查 U6 与提前 TRANSFER；宽筛后重读，普通订单号澄清、偏好复述不直接计 U6，保守保留 20 条。

brief 所指仓库 `infra-thinking-audit.md` 当前不存在，实际读取同批已有 [v3 infra 审计](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/infra-case-study-v3data/REPORT.md)、其 [U1 规则实现](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/infra-case-study-v3data/lib.py:24) 和 [重跑实验记录](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/research/tau2/experiments/2026-10-18-user-sim-rerun.md)。来源替代不改变 v3 的标签定义。

### 2.3 总体结果

| 运行（缩写）                 | 成功 / 240 | 成功率 | agent 致败 | 用户致败 | benchmark 致败 | 成功 U6 | 成功 U7a |
| ---------------------------- | ---------- | ------ | ---------- | -------- | -------------- | ------- | -------- |
| DeepSeek thinking (`ds`)     | 180        | 75.00% | 47         | 10       | 3              | 3       | 5        |
| DeepSeek + patch (`dsp`)     | 182        | 75.83% | 49         | 1        | 8              | 1       | 4        |
| DeepSeek no-thinking (`dsn`) | 177        | 73.75% | 50         | 3        | 10             | 1       | 3        |
| Turbo (`tb`)                 | 142        | 59.17% | 44         | 52       | 2              | 4       | 5        |
| Turbo + patch (`tbp`)        | 166        | 69.17% | 44         | 27       | 3              | 1       | 5        |
| Lite (`lt`)                  | 164        | 68.33% | 42         | 32       | 2              | 3       | 12       |
| Lite + patch (`ltp`)         | 182        | 75.83% | 40         | 16       | 2              | 0       | 8        |
| GLM 5.3 (`gl`)               | 195        | 81.25% | 40         | 2        | 3              | 7       | 1        |
| Qwen3.5-4B (`qw`)            | 137        | 57.08% | 50         | 51       | 2              | 0       | 2        |

Airline、Retail 分开的运行×原因矩阵、全部用户码及 v3/patch 对照在 [matrix.md](matrix.md)；[cases.md](cases.md) 按代码列出每条轨迹。未打 patch 的五组，Retail 用户致败分别为 DeepSeek 6、Turbo 47、Lite 27、GLM 0、Qwen 37；主要模拟器差异仍集中在 Retail。GLM 的 195/240 最高，但有 7 个 U6、1 个提前 TRANSFER 成功，不能当作无噪声基准。

## 3. 可复现性：类别复现，逐题结果仍大幅变化

### 3.1 v3 → 重跑的失败来源与用户码

以下均为失败侧 primary；成功 U6/U7a 不混入用户致败。

| 模拟器            | 成功      | agent   | 用户    | benchmark | U1      | U3      | U4    | U4b   | U5    | U7a   |
| ----------------- | --------- | ------- | ------- | --------- | ------- | ------- | ----- | ----- | ----- | ----- |
| DeepSeek thinking | 173 → 180 | 48 → 47 | 16 → 10 | 3 → 3     | 13 → 3  | 2 → 1   | 0 → 0 | 0 → 1 | 0 → 0 | 1 → 5 |
| Turbo             | 152 → 142 | 36 → 44 | 51 → 52 | 1 → 2     | 41 → 42 | 3 → 5   | 0 → 0 | 4 → 4 | 0 → 0 | 3 → 1 |
| Lite              | 168 → 164 | 37 → 42 | 35 → 32 | 0 → 2     | 23 → 20 | 6 → 8   | 0 → 0 | 1 → 0 | 0 → 0 | 5 → 4 |
| GLM 5.3           | 182 → 195 | 43 → 40 | 9 → 2   | 6 → 3     | 0 → 0   | 4 → 0   | 0 → 0 | 4 → 0 | 0 → 0 | 1 → 2 |
| Qwen3.5-4B        | 140 → 137 | 40 → 50 | 57 → 51 | 3 → 2     | 0 → 2   | 51 → 45 | 1 → 1 | 3 → 1 | 1 → 0 | 1 → 2 |

这五组的 U2、U4a、U6、U7 在两轮失败侧均为 0；其他用户码均列于表中。成功 U6/U7a 的逐运行变化见 [matrix.md](matrix.md) 与原始计数 [v3-counts.json](v3-counts.json)、[counts.json](counts.json)。

Turbo 的 U1 41→42、Lite 23→20，复现了同句确认 STOP；Qwen 的 U3 51→45，仍是其主要用户缺陷。DeepSeek 的 U1 13→3，GLM 保持 0，说明不能照搬 v3 的发生次数。**“Qwen 从不发生 U1”不成立**：本轮有 2 条。未打 patch 五组的 M2a 从 97 增至 108，九组总计 196，是最稳定的 agent 缺口；修改用户 prompt 没有消除“政策不允许却转人工”的行为。

### 3.2 同题同 trial 的结果翻转及原因

配对键为 `(domain, task_id, trial)`；不是固定前缀，只能比较完整 rollout 的实际分歧。

| 模拟器            | 成→败 | 败→成 | 翻转总数 | 成→败后的 agent / 用户 / benchmark | 败→成前的 agent / 用户 / benchmark |
| ----------------- | ----- | ----- | -------- | ---------------------------------- | ---------------------------------- |
| DeepSeek thinking | 23    | 30    | 53       | 13 / 7 / 3                         | 15 / 12 / 3                        |
| Turbo             | 39    | 29    | 68       | 15 / 23 / 1                        | 6 / 22 / 1                         |
| Lite              | 32    | 28    | 60       | 14 / 16 / 2                        | 9 / 19 / 0                         |
| GLM 5.3           | 9     | 22    | 31       | 8 / 1 / 0                          | 12 / 8 / 2                         |
| Qwen3.5-4B        | 28    | 25    | 53       | 16 / 11 / 1                        | 6 / 17 / 2                         |

DeepSeek 的 30 条败→成中有 11 条原 U1；23 条成→败包括 4 条 U7a、4 条 M2a 和其他 agent/benchmark 问题。Turbo 两方向分别有 **20 条新增失败 U1、17 条原 U1 成功**，所以 U1 总数几乎不变也不代表同一批 episode 稳定复现。Lite 两方向对应 12 和 14 条。GLM 22 条改善来自原 U3 4、U4b 4、M1 3、M15 3 等；Qwen 25 条改善有 15 条原 U3，但仍有 6 条原成功变 U3。完整每码分布见 [matrix.md](matrix.md)，所有 1,200 对及另三组 patch、thinking 配对共 2,160 条，在 [pairs.jsonl](pairs.jsonl) 中逐对给出两边路径和致败理由。

### 3.3 哪些变化可能受 NL judge 影响

v3 与本轮的 DeepSeek 用户后端和 NL judge 都换了版本；当前 DeepSeek 用户响应 fingerprint 为 `e72d3c7f663d3e1383b00b3884cb4501`，旧审计为 `aeb56401ca74e127821c4f9126dcb669`。这证明用户服务响应标识不同，**不能用用户 fingerprint 代替 judge 的版本证据**。保存的 judge 元数据只有 model/host/调用配置，没有完整原始 judge 响应 fingerprint，因此无法独立固定 judge 权重版本。

在翻转中筛选“NL 子项变化、其他已记录 reward 子项为 1”，再排除同时受 fixed unexpected_transfer 影响的轨迹，得到 DeepSeek 3、Turbo 2、Lite 2、GLM 2、Qwen 0 条候选，见 [judge-sensitive-flips.json](judge-sensitive-flips.json)。候选涉及条件式补偿提议是否算已提出、乘客数量约束是否满足、政策理由是否说明等。例如 DeepSeek airline:13 t0 的 NL 解释断言从不满足变满足。**候选两边没有任何一对完整消息完全相同**：具体措辞和 agent 行为也变了，只能称 judge 敏感，不能说其中多少翻转由 judge 单独造成。本次不重评分，也不把明确写错或控制符提前终止全部推给 judge。

## 4. patch 效果：等待执行有效，仍要检查任务忠实度

### 4.1 实际改动与每码计数

patch 在用户 guideline 的 STOP 规则前加入：

> Agreeing to an action is not the same as the action being completed. If the agent offers to do something (e.g., cancel an order, process a refund), wait for the agent to confirm it is done before ending the conversation.

实际 diff 是 `simulation_guidelines.md` 与 `simulation_guidelines_tools.md` **各增加同一行**，前者第 14 行；完整 diff 在 [preflight.json](preflight.json)。任务和 gold 未改。

| 模拟器            | U1      | 用户致败总数 | 其他用户致败（不含 U1） | U3     | U4b   | U5    | 失败 U7a | 成功 / 240 |
| ----------------- | ------- | ------------ | ----------------------- | ------ | ----- | ----- | -------- | ---------- |
| DeepSeek thinking | 3 → 0   | 10 → 1       | 7 → 1                   | 1 → 1  | 1 → 0 | 0 → 0 | 5 → 0    | 180 → 182  |
| Turbo             | 42 → 16 | 52 → 27      | 10 → 11                 | 5 → 6  | 4 → 3 | 0 → 0 | 1 → 2    | 142 → 166  |
| Lite              | 20 → 0  | 32 → 16      | 12 → 16                 | 8 → 13 | 0 → 0 | 0 → 1 | 4 → 2    | 164 → 182  |

三组 patch 前后失败侧 U2、U4、U4a、U6、U7 都为 0；上表涵盖全部非零用户码。原始成功分别增加 2、24、18，即 0.83、10.00、7.50 个百分点。DeepSeek 用户致败少 9 个而净成功只增 2 个，因为同期 agent/benchmark 致败也变化；不能按用户缺陷减少量直接推算净得分。

### 4.2 原 U1 消失后发生了什么

| 模拟器            | 原 U1 消失 | 仍是 U1 | 其他配对新增 U1 | 消失后的成功 | 消失后的 agent 致败 | 消失后的其他用户致败 | 全部成→败 / 败→成 |
| ----------------- | ---------- | ------- | --------------- | ------------ | ------------------- | -------------------- | ----------------- |
| DeepSeek thinking | 3          | 0       | 0               | 3            | 0                   | 0                    | 21 / 23           |
| Turbo             | 35         | 7       | 9               | 32           | 1                   | 2                    | 27 / 51           |
| Lite              | 20         | 0       | 0               | 19           | 0                   | 1                    | 23 / 41           |

Turbo 的 35 条原 U1 消失，32 成功，余下为 retail:102 t0 的 M14、retail:26 t0 的 U3、retail:5 t1 的 U4b；7 条原 U1 仍发生，另有 9 条其他配对变成 U1（其中 8 条此前成功），因此最终为 16。Lite 的 20 条消失，19 成功，retail:5 t1 变为 U3。DeepSeek 的 3 条均成功。这些是配对去向，不是假设“把原消息里的 STOP 删去”后的结果。

[Lite retail:5 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-lite-plan-test-u0p0/trajectories/retail:5:trial0/d460d783493644d0839fbbbc6a4bae2c.json) 是清楚的 U1：用户按任务走完两次改主意分支，agent m20 仍在请求退瓶确认；m21：“Yes, that's all correct.”，同句附 `###STOP###`，退货工具尚未调用。[任务 reason_for_call](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:603) 允许这些改主意，缺陷发生在正确授权后提前结束。对应 [Lite + patch retail:5 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-lite-plan-u1patch-test-u0p0/trajectories/retail:5:trial0/1b86a989842b430bad8273cf12876a35.json) 中，m19 只授权，m20 调用退货工具，m21 返回成功，m23 才 STOP，reward=1。

patch 残留也很明确：[Turbo + patch retail:9 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-turbo-plan-u1patch-test-u0p0/trajectories/retail:9:trial0/485e60485fe24379960d602417f6436d.json) m18 请求交换确认；m19：“Yes, that all sounds right to me.”，随后同句 STOP，工具未执行，仍是 U1。见 [任务 reason_for_call](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:953) 及 patch guideline 对“等待完成”的新增规则。

### 4.3 其他失败形态、对话长度与上下文

Lite 的 U3 8→13，另出现 1 条 U5。[Lite + patch retail:71 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-lite-plan-u1patch-test-u0p0/trajectories/retail:71:trial2/e1148a8350f6408abbdaac31c0973d9a.json) m25：“I was originally going to change the shipping address to my Charlotte home and swap the desk lamp to black, but on second thought, let’s hold off on both of those.”。沿用 v3 边界：[任务 reason_for_call](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:6581) 的“只修改背包”是商品改主意分支，不能取消先前独立的 Charlotte 地址目标；该轨迹明确放弃地址后商品写入锁单，归 U3。retail:64 t1/t2 又把 pending 相机编成两周前已送达，促使 agent 无法执行原 pending 修改目标。它们在 patch 运行中出现，不足以证明由 patch 新引入。

[Lite + patch retail:102 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-lite-plan-u1patch-test-u0p0/trajectories/retail:102:trial1/c44a8d679e4f4291b8e4557c9e8676c6.json) 的用户 m25 在自己的确认后续写客服话语：m25：“Perfect, the item modification for order #W4219264 has been completed successfully.”，实际没有手表修改调用，还编造 purifier item ID 和价格；归既有 U5。[任务 task_instructions](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:8194) 要求扮演不耐烦的用户，并未授权冒充客服宣布操作成功。另一条 [Lite + patch retail:102 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-lite-plan-u1patch-test-u0p0/trajectories/retail:102:trial2/dfdff4b56fc44eb99ed41a1121555021.json) 是 agent 未调用交换却声称完成，归既有 M16。角色混淆与服务内容异常分开，未据此新增代码。

policy turns 指 actor 响应轮数，包含工具调用轮，分布如下；只统计 completed。

| 运行 | 平均 policy turns | p95 | 最大 | max_steps episode |
| ---- | ----------------- | --- | ---- | ----------------- |
| ds   | 12.58             | 21  | 25   | 0                 |
| dsp  | 12.52             | 20  | 27   | 0                 |
| dsn  | 12.77             | 20  | 28   | 0                 |
| tb   | 11.43             | 19  | 30   | 0                 |
| tbp  | 12.35             | 19  | 100  | airline:24:trial3 |
| lt   | 11.62             | 19  | 31   | 0                 |
| ltp  | 11.59             | 19  | 26   | 0                 |
| gl   | 13.64             | 22  | 40   | 0                 |
| qw   | 13.03             | 22  | 45   | 0                 |

patch 前后 DeepSeek 平均 12.58→12.52、Lite 11.62→11.59，没有整体拉长；Turbo 11.43→12.35，最大值升到 100。唯一 completed 的 max_steps 为 [Turbo + patch airline:24 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-turbo-plan-u1patch-test-u0p0/trajectories/airline:24:trial3/ca2f9dbc88be43b78626e933e3bfe2d5.json)：agent 持续枚举无关机场，M7，并非用户在任务完成后拒绝结束。被拒收的 actor context 超限尝试为 DeepSeek 3→7、Turbo 5→7、Lite 11→6；有增有减，不能说 patch 没有资源代价，也不能据一次完整重跑断言增加由 patch 导致。未发现需要单列“完成后用户一直不结束”的致败类别。

## 5. thinking 开关：控制符错误减少，认证缺信息更显著

| 指标                                | thinking 开 | thinking 关 |
| ----------------------------------- | ----------- | ----------- |
| success                             | 180         | 177         |
| agent                               | 47          | 50          |
| user                                | 10          | 3           |
| benchmark                           | 3           | 10          |
| U1                                  | 3           | 0           |
| U2                                  | 0           | 0           |
| U3                                  | 1           | 3           |
| U4                                  | 0           | 0           |
| U4a                                 | 0           | 0           |
| U4b                                 | 1           | 0           |
| U5                                  | 0           | 0           |
| U6                                  | 0           | 0           |
| U7                                  | 0           | 0           |
| U7a                                 | 5           | 0           |
| 成功 U6                             | 3           | 1           |
| 成功 U7a                            | 5           | 3           |
| 用户消息数                          | 1422        | 1506        |
| 含非空 reasoning_content 的用户响应 | 1394        | 0           |
| 用户正文平均字符数                  | 210.6       | 195.8       |
| 平均 policy turns                   | 12.58       | 12.77       |

表中 U 码为失败侧，成功标记另列。thinking 关后的 3 条用户致败都是 U3：airline:29 t1 放弃 JFK 直飞及到达时限，retail:5 t2 未遵循确认时改主意分支，retail:79 t2 选错必须保留的颜色。其余四条 retail:9 认证失败应归 B1，而不是把用户无法给出姓名认作 U3。

[DeepSeek no-thinking retail:9 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-deepseek-nothink-test-u0p0/trajectories/retail:9:trial2/ddb2e72b98774944aeb9c786c3c2debe.json) m5：“I’m not comfortable sharing my full name right now.”。单看措辞像主动拒绝，但 [任务 known_info](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:954) 只给账号与邮编，[任务 unknown_info](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:955) 声明不知道 email，完整姓名并未提供。[通用 guideline](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/user_simulator/simulation_guidelines.md:8) 要求未提供的信息视为未知、不得编造；从账号字符串猜姓名不是用户必须完成的隐含任务。[benchmark-variants.md](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/research/tau2/benchmark-variants.md:59) 明确记录 fixed-v3 只补了 retail:5 的姓名，其他题没有扩大修改。因此本轮同样的缺信息边界适用于 retail:9：no-thinking 四条、patch 一条都记 B1。即使把隐私措辞换掉，明确可用的认证信息仍缺失，不能据此声称关闭 thinking 导致四条用户能力错误。

thinking 开时 1,394/1,422 条用户响应有非空 `reasoning_content`，关时 0/1,506，实际返回与开关一致。正文平均字符数 210.6→195.8，但用户调用数 1,422→1,506、平均 policy turns 12.58→12.77；更短正文没有带来更少交互。配对有 25 条成→败、22 条败→成。失败侧 U3 仍有 1→3 的差异，但认证信息缺失另属 benchmark；no-thinking retail:9 t0 还保留 U7a 次要现象，primary 为 0 不表示提前 TRANSFER 完全消失。本次完整重跑无法分离开关、对话路径和后续 agent 行为的净效应，不足以推荐统一关闭 thinking。

## 6. infra：异常尝试单列，不算模拟器能力失败

### 6.1 服务异常和截断

以下限流、连接、context 均为未进入 completed 矩阵的拒收尝试；length 列为 completed 消息。run.log 同一 traceback 不重复计数，以 workflow failure 事件核对原始拒收文件。

| 运行 | 用户 API 限流 | actor 连接错误 | actor 超 context | 用户 length | actor length | 开场控制符 / 空回复 / 服务内容异常 |
| ---- | ------------- | -------------- | ---------------- | ----------- | ------------ | ---------------------------------- |
| ds   | 0             | 5              | 3                | 0           | 1            | 0 / 0 / 0                          |
| dsp  | 0             | 0              | 7                | 0           | 0            | 0 / 0 / 0                          |
| dsn  | 0             | 1              | 2                | 0           | 0            | 0 / 0 / 0                          |
| tb   | 9             | 0              | 5                | 0           | 0            | 0 / 0 / 0                          |
| tbp  | 1             | 0              | 7                | 0           | 0            | 0 / 0 / 0                          |
| lt   | 0             | 0              | 11               | 0           | 1            | 0 / 0 / 0                          |
| ltp  | 0             | 0              | 6                | 0           | 1            | 0 / 0 / 0                          |
| gl   | 0             | 0              | 1                | 0           | 0            | 0 / 0 / 0                          |
| qw   | 0             | 0              | 5                | 0           | 1            | 0 / 0 / 0                          |

共 **63 次拒收尝试：47 次 actor context 超限、10 次用户 API 限流、6 次 actor 连接错误**；不是 63 个额外失败题，重试完成后每个运行仍是 240 条。连接错误的前一用户请求已完成，尚无 judge 调用，故归 actor 连接层；不能算 DeepSeek 用户或 judge 服务失败。各次错误、run.log 行号在 [infra.json](infra.json)。

全部 completed 共 **13,069 条用户响应，finish_reason 均为 stop**；未发现开场控制符、空回复、用户 length 或经筛查需归服务故障的无关/协议垃圾内容。此结论针对已保存响应及人工复核候选，不保证未保存请求没有任何短暂异常。用户角色混淆、编造剧情另按行为标注，不能用“服务异常”豁免。

actor length 有 4 条：DeepSeek airline:37 t2 m30（仍成功）；Lite retail:102 t0 m36（更早 M14 已致败）；Lite+patch airline:32 t0 m18（更早 M11）；Qwen airline:44 t0 m86（更早 M14）。后三条不因为末尾截断而改成用户原因，也不覆盖先前已成立的致败原因。

### 6.2 raw_data.model 与 fingerprint

| 运行 | 配置 user_llm                | 实际 raw_data.model          | system_fingerprint               |
| ---- | ---------------------------- | ---------------------------- | -------------------------------- |
| ds   | deepseek-flash               | deepseek-flash               | e72d3c7f663d3e1383b00b3884cb4501 |
| dsp  | deepseek-flash               | deepseek-flash               | e72d3c7f663d3e1383b00b3884cb4501 |
| dsn  | deepseek-flash               | deepseek-flash               | e72d3c7f663d3e1383b00b3884cb4501 |
| tb   | doubao-seed-2-1-turbo-260628 | doubao-seed-2-1-turbo-260628 | None                             |
| tbp  | doubao-seed-2-1-turbo-260628 | doubao-seed-2-1-turbo-260628 | None                             |
| lt   | doubao-seed-2-1-lite-260915  | doubao-seed-2-1-lite-260915  | None                             |
| ltp  | doubao-seed-2-1-lite-260915  | doubao-seed-2-1-lite-260915  | None                             |
| gl   | glm-5.3                      | glm-5.3                      | None                             |
| qw   | qwen35-4b-user-step0         | qwen35-4b-user-step0         | None                             |

返回模型字符串与各自配置对应；`openai/` 是客户端 provider 前缀，DeepSeek 原始响应省略该前缀。DeepSeek 三组 fingerprint 一致，支持本轮三组共享同一返回服务标识。Turbo/Lite、GLM、Qwen 的 fingerprint 为 null，配置和模型名称对应只能核对路由标识，不能独立证明服务端权重未变；Qwen 的部署别名也不等同权重哈希。judge 版本的限制见 §3.3。

## 7. 用户问题造成的分数偏差：只做计数对比

| 运行 | 用户致败 / 240  | 占该运行全部失败 | 成功 U6 / 全部成功 | 成功 U7a / 全部成功 |
| ---- | --------------- | ---------------- | ------------------ | ------------------- |
| ds   | 10/240 = 4.17%  | 16.67%           | 3/180              | 5/180               |
| dsp  | 1/240 = 0.42%   | 1.72%            | 1/182              | 4/182               |
| dsn  | 3/240 = 1.25%   | 4.76%            | 1/177              | 3/177               |
| tb   | 52/240 = 21.67% | 53.06%           | 4/142              | 5/142               |
| tbp  | 27/240 = 11.25% | 36.49%           | 1/166              | 5/166               |
| lt   | 32/240 = 13.33% | 42.11%           | 3/164              | 12/164              |
| ltp  | 16/240 = 6.67%  | 27.59%           | 0/182              | 8/182               |
| gl   | 2/240 = 0.83%   | 4.44%            | 7/195              | 1/195               |
| qw   | 51/240 = 21.25% | 49.51%           | 0/137              | 2/137               |

U1 把守规矩的写前确认后续截掉，用户任务偏离则改变 agent 实际被评估的目标；因此原始零分中包含用户行为影响。另一方面，U6 的成功可能包含用户救回错误计划，提前 TRANSFER 的成功则可能避免暴露后续错误转接。九组共有 194 个用户致败、20 个 U6 成功、45 个提前 TRANSFER 成功；它们体现评分两侧受到的不同影响，不能合并成一个“纠正后的准确率”。

例如 [Lite airline:31 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-lite-plan-test-u0p0/trajectories/airline:31:trial3/6aca5dfbf60f4c7bb955e469015efe5e.json) reward=1，但 agent m14 刚把政策禁止修改的请求说成需要人工，用户 m15：“###TRANSFER###” 后结束，转接工具从未调用。该成功只证明截断时满足当前评分，不能证明 agent 随后会正确拒绝。U6 例子 [GLM 5.3 retail:97 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-glm53-test-u0p0/trajectories/retail:97:trial3/69b91238acff4ed1aec0f4bdfef7cc2b.json) 中，agent m14 询问是否把 LA 默认地址只改城市为 New York；用户 m15 指出 NYC 来源订单，并提醒先改地址避免商品写入锁单，后续才成功。这属于可见帮助，并不推出没有帮助就一定失败。任务允许的普通提醒仍需与执行方案纠正区分。

**不能把用户致败直接加回成功率，或把 U6/U7a 全部扣成失败。** patch 后的去向已经显示用户错误消失也可能仍败于 agent；同时也有原成功变失败、agent/benchmark 类型改变。这里没有固定前缀随机干预，没有“移除这句用户话”后的反事实执行。提供这些计数是为了识别分数里混入了哪些行为来源，不估计 actor 的真实无噪声成功率。

## 8. 边界、复核与局限

### 8.1 保持 v3 的责任边界，不为每种表象新增代码

- **agent 先读错，用户重复，不自动归用户。** 随机复核的 [Turbo + patch retail:101 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-turbo-plan-u1patch-test-u0p0/trajectories/retail:101:trial2/28f482b9ee6d4db28a33ec7c0407f0e0.json) 原标 U3；m12 是 agent 先把有 sneakers、helmet、purifier 的 W3445693 当成 purifier+speaker 订单，用户 m21 才重复该 ID。实际错误地址和商品来源由 agent 引入，更正为 M4，删除原次要码。
- **agent 漏参数仍是 agent。** [Qwen3.5-4B retail:86 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/selfhost-user/full-r-qwen35user/trajectories/retail:86:trial1/760be520462543f48a25381d9d4c4694.json) 的 agent 把来源地址 `Suite 501` 漏掉后让用户确认，不能把用户照着确认归 U3；记既有 M9（参数 grounding 错误）。M9 已在旧 capability codebook 中定义，只是 v3 未作 primary 使用，不是新增码。
- **gold 表示或选择过窄归 B1。** [DeepSeek + patch retail:42 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/api-user/r-deepseek-u1patch-test-u0p0/trajectories/retail:42:trial2/21e828ff17454dd78e4408db25b78dc4.json) 与 t3 按 [任务 known_info](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:4125) 的 Texas 写入，gold 要 TX；其余必要写入正确，不把相同地址的不同州表示归用户。Turbo retail:61 t2 选中满足蓝色且价格不超原价的选项，任务未要求最便宜，gold 却只接受另一 item；DeepSeek retail:100 t0、Qwen retail:100 t3 的可用塑料滑板也被 bamboo-only gold 拒绝。airline:44 含中转 7 小时的 KC18K6 不应按“≤3 小时才升舱”升舱，gold 仍要求；若实际轨迹有更早 agent 错误，B1 只记 secondary。
- **合法转接与政策拒绝分开。** DeepSeek 三种配置 airline:8 的指定 B2 失败在正确订票后追加合法的座位/出票请求，fixed unexpected_transfer 仍扣分。retail:32 因丢失平板合法转接，却与未揭示后续子目标冲突，沿用 v3 的 B1；M2a 则是本可明确政策拒绝而选择转人工。不能将所有转接失败都归一方。

### 8.2 固定种子 15 条重读与机器核验

按 Python `random.Random(20261018).sample(sorted(label_ids), 15)`，从 700 条标签抽 15 条重读任务、消息、写入与评分。**2 条致败原因改动、2 条记录变化**：上面的 Turbo+patch retail:101 t2 U3→M4，以及最终读回对 DeepSeek no-thinking retail:9 t2 的缺姓名边界纠正 U3→B1。抽样列表、前后理由、路径与哈希在 [audit-15.json](audit-15.json)；这是同一标注者重读，未进行独立标注一致性检验。

随机样本之外，定向复核修正了 Lite retail:71 t0/t3（U1→U3，U1 留作 secondary）及 Turbo+patch retail:5 t1（U3→U4b），见 [targeted-review-changes.json](targeted-review-changes.json)。最终核对“缺信息必须视为未知”的 guideline 与 fixed-v3 修补记录，另将五条 DeepSeek retail:9（四条 no-thinking、一条 patch）从 U3 改为 B1；其中 no-thinking t2 已计入上述随机样本，其余四条为定向更正，见 [missing-name-review.json](missing-name-review.json)。GLM retail:97 t3 的 U6 理由也按实际消息收紧为地址来源与执行顺序提醒，计数不变。早期措辞修订和初次抽样快照分别保存在 [post-audit-wording-changes.json](post-audit-wording-changes.json)、[audit-15-initial.json](audit-15-initial.json)；最终标签以 labels.jsonl 为准。

[validate.py](validate.py) 从原始轨迹核验失败覆盖、无重复、哈希、引文与消息编号、tool_calls、任务行号和 gold、计数与配对，并检查 45 个成功 U7a 的控制符发生在转接工具之前；先故意损坏一条内存引文，观察校验失败，再校验真实文件。结果见 [validation.json](validation.json)。这些检查证明文件与来源、计数一致，不能代替语义判断或独立复核。仓库保持 clean，未修改代码或文档。

### 8.3 真正影响解释的限制

1. 同一标注者、一次抽样复读；U3 与 B1、授权前的任务分支、用户帮助与正常澄清仍依赖语义边界。保留原话、实际工具状态和 v3 规则供复审，不把 15 条重读当可靠性系数。
1. v3 的用户后端、Turbo 接口及 NL judge 与本轮不完全相同；judge 缺原始 fingerprint，不能分离其净效应。本轮 patch/thinking 配对也未共享前缀，actor temperature=0 不能消除完整对话和推理服务差异。
1. U6 是保守筛查，可能漏掉隐蔽帮助，也可能包含 task 允许的强提醒；20 是本口径观察数。提前终止的成功无法揭示 actor 本会怎样继续。
1. 63 次拒收重试使 completed 样本带有完成条件；context 与网络异常的发生率不能直接作为用户能力错误率。各题四个 trial 也非四个独立任务，计数差不作为显著性或因果估计。

## 9. 交付与离线复算

主要交付为 [labels.jsonl](labels.jsonl)、[matrix.md](matrix.md)、[cases.md](cases.md) 与本报告；[pairs.jsonl](pairs.jsonl) 逐对保留翻转理由，[preflight.json](preflight.json)、[infra.json](infra.json)、[audit-15.json](audit-15.json) 保留核验依据。`decisions.json` 是本次人工判断源，`build.py` 生成标签、矩阵、索引与配对；`write_report.py` 渲染本报告。全部脚本仅访问已有本地证据。

在本目录运行 `PYTHONDONTWRITEBYTECODE=1 python validate.py` 可重新核验已交付文件；需要重建时依次运行 `build.py`、`write_report.py`、`validate.py`，均加上述环境变量。不调用推理接口。单一分析者完成了内容重读；没有独立 reviewer，也不宣称用户已验收。

## Claude 复核（2026-10-18）

- **标注方式**：Codex 在读过部分轨迹后，常按题批量给同一题的失败打同一个码（`annotate.py` 的 `group()`；700 条标签只有 117 种不同的 rationale）。固定种子重读 15 条改了 2 条。各码计数按近似值使用。
- **U1 计数可靠**：人工 U1（DeepSeek 3、Turbo 42、Lite 20、Turbo+patch 16、Qwen 2）与独立的规则筛选一致（规则多筛的 8 条是更早已有致败原因的 episode，没有漏项）。
- **patch 前后的用户码**（失败 episode 的致败原因）：

  | 运行 | patch 前 | patch 后 |
  | --- | --- | --- |
  | DeepSeek V4.1 | U1 3、U7a 5、U4b 1、U3 1 | U3 1 |
  | Turbo | U1 42、U3 5、U4b 4、U7a 1 | U1 16、U3 6、U4b 3、U7a 2 |
  | Lite | U1 20、U3 8、U7a 4 | U3 13、U7a 2、U5 1 |

  patch 消掉的是"提前结束"一类（U1，以及 DeepSeek 的 U7a），不影响 U3（违反任务目标或编造事实）；Lite 的 U3 反而从 8 增到 13，与"U1 去掉后对话继续、之后偏离任务"一致，逐条确认待做。
- brief 指向的 `infra-thinking-audit.md` 在 Codex 启动时尚未同步到 phai，它改读了产物目录里的同一份报告，内容相同。
