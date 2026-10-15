# 用户模拟器的影响

> 来源：原 README 第 9.1、16.15 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 Doubao 用户为什么拉低 Retail

fixed-v3 test 上（60 题 × 4 次，五个模拟器，§2）：Retail mean@4 DeepSeek-Flash 0.806，Doubao 2.1 Turbo
0.650，差 −15.6pp（按题配对 bootstrap 95% 区间 \[−26.3, −4.4\]）；Airline 两者差 0.05（Turbo
更高），落在运行间噪声内。Turbo 的 Retail 失败里 45 个由用户致败（DeepSeek 15 个），其中 39 个是 U1（授权操作的同一句里
STOP），DeepSeek 是 13 个。

U1 的形态：agent 已按 policy 给出完整、正确的确认，用户回 "Yes, I confirm… ###STOP###"，写操作还没执行，对话就结束，reward
为 0。 典型样例见 [v3 cases.md](user-sim-case-study-v3/cases.md) 和
[REPORT §4.1](user-sim-case-study-v3/REPORT.md)。

这是 Doubao 违反官方用户指令，不是 agent 的错：文本版
`data/tau2/user_simulator/simulation_guidelines.md:13-14` 要求 "continue the conversation
until the task is complete"，目标满足才发 `###STOP###`；agent 先确认再写是 retail `policy.md:16`
的要求。语音版 guideline 多一句 "Agreeing to an action is not the same as the action being
completed" （`simulation_guidelines_voice.md:42`），文本版没有，措辞有歧义。DeepSeek 在同一 prompt 下也犯 13
次，Turbo 约是它的 3 倍。

对训练的影响：训练 reward 来自同一个用户模拟器。换 Doubao 训练时，这类 episode 会把正确的确认行为记成 0 分。

## 2 用户模拟器评测总表（2026-10-12 起维护）

**口径。** actor 固定为训练前的 Qwen3.5-4B（non-thinking，policy temperature 0）；fixed v3 变体；官方 test
60 题（Airline 20、Retail 40）× 4 次。用户模拟器和 NL judge 都用各模型的默认模式：DeepSeek-Flash 和 Doubao 开
thinking，此时 temperature 不生效；Qwen3.5-4B 用户关 thinking、temperature 0。NL judge 固定为
DeepSeek-Flash（`9e7082dc` 起与用户解耦，见
[problem-and-plan.md §3](research-direction/problem-and-plan.md)）。mean@4 等于 pass^1。超出
32K 上下文的 episode 第一次出现就记 0 分并计入（6/4/3/1/3 条，依次为
DeepSeek、Turbo、Lite、GLM、Qwen；[ADR 0009](../../decisions/0009-official-tau2-runner-semantics.md)）。计分脚本
`scripts/rescore/rescore.py`。

| 用户模拟器             | 档位 | 模式            | 数据 | Airline mean@4 | Airline pass^4 | Retail mean@4 | Retail pass^4 | 合计 mean@4 | 合计 pass^4 |
| ---------------------- | ---- | --------------- | ---- | -------------: | -------------: | ------------: | ------------: | ----------: | ----------: |
| DeepSeek-Flash         | 中上 | thinking        | v3   |          0.525 |          0.350 |         0.806 |         0.600 |       0.713 |       0.517 |
| Doubao 2.1 Turbo       | 中下 | thinking medium | v3   |          0.575 |          0.250 |         0.650 |         0.300 |       0.625 |       0.283 |
| Qwen3.5-4B（自己部署） | 弱   | non-thinking T0 | v3   |          0.438 |          0.150 |         0.650 |         0.475 |       0.579 |       0.367 |
| Doubao 2.1 Lite        |      | thinking medium | v3   |          0.575 |          0.400 |         0.756 |         0.475 |       0.696 |       0.450 |
| GLM 5.3                |      | thinking        | v3   |          0.575 |          0.400 |         0.850 |         0.725 |       0.758 |       0.617 |
| Haiku / Luna           |      |                 |      |           待跑 |                |               |               |             |             |

数据：v3 是 2026-10-12 修过 7 道题的
fixed-v3（[benchmark-variants.md §1.2.1](benchmark-variants.md)）。最强与最弱用户的总分差 17.9pp。

**Doubao Lite 和 GLM 5.3（2026-10-13）。** 两者都走火山订阅接口（`/api/plan/v3`，key 在
`~/.config/tau2-doubao/ark-plan.env`），默认开 thinking，用户 temperature 0。GLM 关不掉 thinking（接口对
`thinking.type=disabled` 返回 400），但 temperature 0 在短 prompt 上能让输出固定；Lite 能关 thinking，但
temperature 0 不起作用。订阅接口会静默映射模型名（请求 `doubao-seed-2-1-lite-260628` 返回 260915），所以以轨迹里
`raw_data.model` 为准：两组分别是 `doubao-seed-2-1-lite-260915` 和 `glm-5.3`。GLM 第一次用 256 并发跑，约 5
分钟后被限流（"Requests are too frequent"），203 个 episode 失败作废；改用 32 并发重跑（pueue 495）后 0
次限流。单次同时发 128 个请求时两个模型都不限流，所以限的是持续请求量，不是瞬时并发；并发只影响 actor 推理的批次组合，不改变评测参数。

**已作废的 Doubao 运行。** 早期的 Doubao 评测（官方变体 10-09、fixed v2、v3
第一次）是在方舟服务异常或账户欠费时跑的（开场第一句就是控制符、空回复、无关回复、欠费排除），结果已作废，以上表的 v3 运行为准（Turbo 取 pueue
488），不要重跑。

**读数。**

- 换弱用户时掉分集中在 Retail：DeepSeek 0.806，GLM 0.850，Lite 0.756，Turbo 和 Qwen 都是 0.650。按题配对
  bootstrap（40 题）：Turbo 比 DeepSeek 低 15.6pp \[−26.3, −4.4\]，Turbo 和 Qwen 都比 GLM 低
  20.0pp（\[−32.5, −7.5\]、\[−33.1, −6.2\]）。
- Airline 的 mean@4 在 0.44–0.58，落在运行间噪声内：同配置（Qwen3.5-4B、fixed-v3 test、DeepSeek 用户、step
  0）的另一次运行（SAO 534 第 0 步）是 Airline 0.475、总体 0.696，与本表的 0.525、0.713 相差 5pp 和
  2pp，可作为运行间噪声的量级。
- 已知评分偏差：五组 v3 运行共 29 个成功 episode 在 agent 调用转人工工具前先收到用户的 `###TRANSFER###`，仍判成功（见
  [REPORT](user-sim-case-study-v3/REPORT.md)）。2026-10-12 决定暂不修题。
- 主表只报 test。base 运行（再加 train 99 题）只在需要看 train 题上的用户影响，或给 case study 扩样本时才跑。

**thinking 与 temperature（2026-10-12 核查）。** 473 的用户和 policy 都按 temperature 0 配置，4 次重复仍有
26/60 题结果不一致，而且几乎都在第一条用户消息就已分叉。原因是 `deepseek-flash` 默认开 thinking，服务商在 thinking 模式下不理会
temperature；Doubao 的 `reasoning_effort=medium` 同理。tau2 代码把 temperature 正确传到了请求里。直接调 API
实测：DeepSeek 传 `extra_body.thinking.type=disabled` 后三次输出逐字相同，经 LiteLLM 传也有效；Doubao 关掉
thinking 后仍有细微差别。

**关掉 thinking 后 DeepSeek 用户反而更好。** test 60（v2 数据）：Airline 0.412 / 0.200，Retail 0.900 /
0.750，合计 0.738 / 0.567（policy T=1 时合计 0.775 / 0.600）。对比同数据 thinking 的 Retail 0.819 /
0.525。原因待 case study 确认；setting 仍按下面的决定用默认模式，这组结果只作对照。

**setting 决定（2026-10-12）。** 用户模拟器和 NL judge 用各模型的默认模式（DeepSeek、Doubao 开
thinking）。闭源模型多数不支持关 thinking 或设 temperature，统一用默认模式最容易对齐。被训练的 actor 关 thinking，评测时
policy temperature 0。

**API 用户下，用户随机性是组内 reward 差异的主要来源；自部署 Qwen 用户不是。** GRPO 的组内 reward 差异里，只有 policy 造成的那部分是
agent 能学的信号。policy T=0 时，4 次结果不一致（有对有错）的 test 题数（共 60 题）：v3 五个模拟器依次为 DeepSeek 22、Doubao
Turbo 36、Lite 25、GLM 16、Qwen 自部署 24。DeepSeek 旧运行（v2 数据）的情形：

| 用户                     | policy T=0 | policy T=1 |
| ------------------------ | ---------: | ---------: |
| DeepSeek thinking        |      26/60 |      23/60 |
| DeepSeek non-thinking T0 |      18/60 |      19/60 |

API 用户下 policy 在 T=0 时基本确定（DeepSeek 运行中第一处分叉落在 actor 上的只有 3/492 对），不一致主要来自用户侧，policy
采样没有让它更乱。关掉 thinking 的 DeepSeek（pueue 481/482）回复里确实没有 reasoning（0/994 条），不一致的题从 26 降到
18，但 180 对重复里仍有 100 对在第一条用户消息就分叉：真实的长 prompt 下，DeepSeek 在 T=0、关 thinking 时也不完全确定，前面短
prompt 的"逐字相同"推广不到这里。所以"只有 policy 随机"一组做不出来，用户侧的随机性只能减小、不能消除。

自部署 Qwen 用户的情形相反：两侧都是 T=0，360 对重复轨迹的首处分叉有 188 对落在 actor 上，来自 SGLang
批量推理的数值不确定性。所以"用户随机性是大头"只适用于 API 用户。

**Case study：失败是谁造成的（v3，五个模拟器）。**
完整报告：[user-sim-case-study-v3/REPORT.md](user-sim-case-study-v3/REPORT.md)（Codex 标注，第 13
节是我的复核）。1,200 个 episode 里 385 个失败，每个只记一个致败原因（之后 reward 不可能再是 1 的那一步由谁造成）。标注基于重试后完成的
episode；严格计分下另有 17 条超出 32K 上下文的 episode 记 0 分（§2 表），这些没有重新标注，下表不含它们：

| 用户模拟器       | 成功 / 240 | agent | 用户 | benchmark | 用户侧主要问题                         |
| ---------------- | ---------: | ----: | ---: | --------: | -------------------------------------- |
| DeepSeek-Flash   |        173 |    48 |   16 |         3 | 同意操作的同一句里结束对话（U1）13     |
| Doubao 2.1 Turbo |        152 |    36 |   51 |         1 | U1 41                                  |
| Doubao 2.1 Lite  |        168 |    37 |   35 |         0 | U1 23、偏离任务 6、转接前发 TRANSFER 5 |
| GLM 5.3          |        182 |    43 |    9 |         6 | 偏离任务 4、提前结束 4；没有 U1        |
| Qwen3.5-4B       |        140 |    40 |   57 |         3 | 偏离任务说明（U3）51                   |

- 研究的问题存在：actor 不变，用户致败的失败从 9（GLM）到 57（Qwen），占全部失败的 44%。Retail 的 agent 失败在五者间是
  9–14，Retail 的分差基本来自用户。
- 各模拟器的毛病不同：Doubao 两个版本主要是 U1，Qwen 主要是不按任务条件走，GLM 最接近任务说明但不是零噪声。换模拟器不只是换噪声大小，也换了噪声种类。
- Airline 仍以 agent 为主，M2a 共 97 个，airline:6/26/35/45/48 在五个模拟器下都出现。
- "开场第一句就结束"在五组 v3 运行里都是 0。
- 没有新增分类代码。新发现一处 gold 错误：airline:44（见
  [benchmark-variants.md §1.2.1](benchmark-variants.md)）。

**证据。** 运行目录
`_artifacts/AReaL/codex-sao-math/user-sim-20261012/`：`api-user/deepseek-base-u0p0`（pueue
473）、`api-user/deepseek-test-u{1p0,0p1}`（474/475）；DeepSeek 旧运行
`tau2-eval-20261010-fixed-v2/step0-base-t0-deepseek-fixed-v2`。v3
五组：`api-user/v3-{deepseek,doubao,doubao-lite,glm53}-test-u0p0*`（Turbo 取 `-r2`，即 488；GLM
取 `-r2`）和 `step0-selfhost-user/full-qwen35user-v3data`。代码 `9e7082dc`，启动脚本
`scripts/user-sim/launch_api_user.sh`。
