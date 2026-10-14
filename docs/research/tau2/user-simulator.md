# 用户模拟器的影响

> 来源：原 README 第 9.1、16.15 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 Doubao 用户为什么拉低 Retail

同一题、同一 trial 配对比较（[step0-evaluations.md §2](step0-evaluations.md) 两组）：Airline 0.810 →
0.780，差 0.03 ± 0.03，不显著；Retail 0.851 → 0.649，差 0.20 ± 0.03（约 7σ），62 题变差、11 题变好。Retail
失败里 DB 不对的从 43 条增到 106 条，只挂 NL 断言的从 4 条增到 9 条， 失败的 `stop_reason` 都是 `user_stop`。NL judge
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

用户提前结束能解释 Retail 下降中的 [capability-gaps.md §1.5](capability-gaps.md)–17.4pp（66/456 到
79/455；补上写操作后仍可能在别的检查上失败，所以 17.4 是上界）。 例：`retail:6:trial1`，agent 列出 item、新
item、支付方式并请用户回复 yes，用户回 "Yes, I confirm I want to proceed with this exchange.
###STOP###"；`retail:109:trial1` 用户确认第 1 步就结束，另两个写操作没开始。

这是 Doubao 违反官方用户指令，不是 agent 的错：文本版
`data/tau2/user_simulator/simulation_guidelines.md:13-14` 要求 "continue the conversation
until the task is complete"，目标满足才发 `###STOP###`；agent 先确认再写是 retail `policy.md:16`
的要求。语音版 guideline 多一句 "Agreeing to an action is not the same as the action being
completed" （`simulation_guidelines_voice.md:42`），文本版没有，措辞有歧义。DeepSeek 在同一 prompt 下也犯 20
次，Doubao 约是它的 4.5 倍。

对训练的影响：训练 reward 来自同一个用户模拟器。换 Doubao 训练时，这类 episode 会把正确的确认行为记成 0 分。

## 2 用户模拟器评测总表（2026-10-12 起维护）

**口径。** actor 固定为训练前的 Qwen3.5-4B（non-thinking，policy temperature 0）；fixed v2 变体；官方 test
60 题（Airline 20、Retail 40）× 4 次。用户模拟器和 NL judge 都用各模型的默认模式：DeepSeek-Flash 和 Doubao 开
thinking，此时 temperature 不生效；Qwen3.5-4B 用户关 thinking、temperature 0。NL judge 固定为
DeepSeek-Flash（`9e7082dc` 起与用户解耦，见
[problem-and-plan.md §3](research-direction/problem-and-plan.md)）。mean@4 等于 pass^1。计分脚本
`scripts/user-sim/test60_table.py`。

| 用户模拟器             | 档位 | 模式            | 数据 | Airline mean@4 | Airline pass^4 | Retail mean@4 | Retail pass^4 | 合计 mean@4 | 合计 pass^4 |
| ---------------------- | ---- | --------------- | ---- | -------------: | -------------: | ------------: | ------------: | ----------: | ----------: |
| DeepSeek-Flash         | 中上 | thinking        | v3   |          0.550 |          0.350 |         0.806 |         0.600 |       0.721 |       0.517 |
| Doubao 2.1 Turbo       | 中下 | thinking medium | v3   |          0.588 |          0.300 |         0.656 |         0.325 |       0.633 |       0.317 |
| Qwen3.5-4B（自己部署） | 弱   | non-thinking T0 | v3   |          0.438 |          0.150 |         0.656 |         0.500 |       0.583 |       0.383 |
| Doubao 2.1 Lite        |      | thinking medium | v3   |          0.575 |          0.400 |         0.762 |         0.475 |       0.700 |       0.450 |
| GLM 5.3                |      | thinking        | v3   |          0.575 |          0.400 |         0.850 |         0.725 |       0.758 |       0.617 |
| Haiku / Luna           |      |                 |      |           待跑 |                |               |               |             |             |

数据列：v3 是 2026-10-12 修过 7 道题的
fixed-v3（[benchmark-variants.md §1.2.1](benchmark-variants.md)）。同模拟器 v2 → v3
的合计：DeepSeek 0.721 → 0.721，Qwen 0.596 → 0.583，变化在运行间噪声内；Doubao 0.550 → 0.633，变化大，原因见下一段。

**Doubao Lite 和 GLM 5.3（2026-10-13）。** 两者都走火山订阅接口（`/api/plan/v3`，key 在
`~/.config/tau2-doubao/ark-plan.env`），默认开 thinking，用户 temperature 0。GLM 关不掉 thinking（接口对
`thinking.type=disabled` 返回 400），但 temperature 0 在短 prompt 上能让输出固定；Lite 能关 thinking，但
temperature 0 不起作用。订阅接口会静默映射模型名（请求 `doubao-seed-2-1-lite-260628` 返回 260915），所以以轨迹里
`raw_data.model` 为准：两组分别是 `doubao-seed-2-1-lite-260915` 和 `glm-5.3`。GLM 第一次用 256 并发跑，约 5
分钟后被限流（"Requests are too frequent"），203 个 episode 失败作废；改用 32 并发重跑（pueue 495）后 0
次限流。单次同时发 128 个请求时两个模型都不限流，所以限的是持续请求量，不是瞬时并发；并发只影响 actor 推理的批次组合，不改变评测参数。

**Doubao 的"开场就结束"是方舟服务端的问题，不是模型行为。** 开场第一句用户消息就是 `###OUT-OF-SCOPE###` 之类控制符的 episode
数：v2 运行（476，base 656 个）20 个，v3 第一次（486，账户临近欠费时跑，240 个）10 个，其中 1 个回复是无关的"电商商品信息提取结果"，同时有
32 次空回复重试；充值后的 v3 重跑（488，240 个）0 个，空回复 0 次。代码、数据和配置都没变，所以 case study 里 Doubao 的 10 个
U4a（以及混入中文售后话术的回复）更可能来自服务端异常。Doubao 行取 488；486 因 2 个 episode 失败作废。

**读数。**

- 换弱用户时掉分集中在 Retail（v2 数据，即 case study 所用的运行）：DeepSeek 0.825 → Doubao 0.606 / Qwen
  0.669。按（题、trial）配对，DeepSeek 成功而 Doubao 失败 41 例、反向 6 例。
- Airline 三者在 0.44–0.51，落在运行间噪声内：同配置的 DeepSeek 旧运行为 Airline 0.438、Retail 0.812，与上表分别差
  0.075 和 0.013。
- [step0-evaluations.md §2](step0-evaluations.md)的 Doubao 结果（official 变体，Airline
  0.787）不可比：official 的 Airline 有"只说 Done
  也得分"的评分盲区（[benchmark-variants.md §3](benchmark-variants.md)）。
- 已知评分偏差：用户在 agent 调用转人工工具前发 `###TRANSFER###` 仍判成功，DeepSeek 下最多 11 个 episode（全扣掉为
  0.675，排序不变）。2026-10-12 决定暂不修题。
- 主表只报 test。base 运行（再加 train 99 题）只在需要看 train 题上的用户影响，或给 case study 扩样本时才跑。

**thinking 与 temperature（2026-10-12 核查）。** 473 的用户和 policy 都按 temperature 0 配置，4 次重复仍有
27/60 题结果不一致，而且几乎都在第一条用户消息就已分叉。原因是 `deepseek-flash` 默认开 thinking，服务商在 thinking 模式下不理会
temperature；Doubao 的 `reasoning_effort=medium` 同理。tau2 代码把 temperature 正确传到了请求里。直接调 API
实测：DeepSeek 传 `extra_body.thinking.type=disabled` 后三次输出逐字相同，经 LiteLLM 传也有效；Doubao 关掉
thinking 后仍有细微差别。

**关掉 thinking 后 DeepSeek 用户反而更好。** test 60：Airline 0.412 / 0.200，Retail 0.906 / 0.775，合计
0.742 / 0.583（policy T=1 时合计 0.775 / 0.600）。对比 thinking 的 Retail 0.825 / 0.525。原因待 case
study 确认；setting 仍按下面的决定用默认模式，这组结果只作对照。

**setting 决定（2026-10-12）。** 用户模拟器和 NL judge 用各模型的默认模式（DeepSeek、Doubao 开
thinking）。闭源模型多数不支持关 thinking 或设 temperature，统一用默认模式最容易对齐。被训练的 actor 关 thinking，评测时
policy temperature 0。

**用户随机性占了组内 reward 差异的大头。** GRPO 的组内 reward 差异里，只有 policy 造成的那部分是 agent 能学的信号。4 次结果不一致的
test 题数：

| 用户                       | policy T=0 | policy T=1 |
| -------------------------- | ---------: | ---------: |
| DeepSeek thinking          |      27/60 |      23/60 |
| Doubao thinking            |      36/60 |            |
| Qwen3.5-4B non-thinking T0 |      29/60 |            |
| DeepSeek non-thinking T0   |      17/60 |      19/60 |

policy 在 T=0 时基本确定（DeepSeek 运行中第一处分叉落在 actor 上的只有 3/492
对），所以用户的随机性单独就让四成以上的题结果不稳定，policy 采样没有让它更乱。关掉 thinking 的 DeepSeek（pueue 481/482）回复里确实没有
reasoning（0/994 条），不一致的题从 27 降到 17，但 180 对重复里仍有 100 对在第一条用户消息就分叉：真实的长 prompt 下，DeepSeek
在 T=0、关 thinking 时也不完全确定，前面短 prompt 的"逐字相同"推广不到这里。所以"只有 policy
随机"一组做不出来，用户侧的随机性只能减小、不能消除。Qwen 用户两侧都是 T=0 仍有 29 题不一致，来自 SGLang 批量推理的数值不确定性。

**Case study：失败是谁造成的（v3，五个模拟器）。**
完整报告：[user-sim-case-study-v3/REPORT.md](user-sim-case-study-v3/REPORT.md)（Codex 标注，第 13
节是我的复核）。1,200 个 episode 里 385 个失败，每个只记一个致败原因（之后 reward 不可能再是 1 的那一步由谁造成）：

| 用户模拟器       | 成功 / 240 | agent | 用户 | benchmark | 用户侧主要问题                         |
| ---------------- | ---------: | ----: | ---: | --------: | -------------------------------------- |
| DeepSeek-Flash   |        173 |    48 |   16 |         3 | 同意操作的同一句里结束对话（U1）13     |
| Doubao 2.1 Turbo |        152 |    36 |   51 |         1 | U1 41                                  |
| Doubao 2.1 Lite  |        168 |    37 |   35 |         0 | U1 23、偏离任务 6、转接前发 TRANSFER 5 |
| GLM 5.3          |        182 |    43 |    9 |         6 | 偏离任务 4、提前结束 4；没有 U1        |
| Qwen3.5-4B       |        140 |    40 |   57 |         3 | 偏离任务说明（U3）51                   |

- 研究的问题存在：actor 不变，用户致败的失败从 9（GLM）到 57（Qwen），占全部失败的 44%。Retail 的 agent 失败在五者间是
  9–14，Retail 的分差基本来自用户。把用户致败的失败和用户帮忙的成功（U6）从样本里删掉，五者成功率差距从 17.5 个百分点缩到 5.6。
- 各模拟器的毛病不同：Doubao 两个版本主要是 U1，Qwen 主要是不按任务条件走，GLM 最接近任务说明但不是零噪声。换模拟器不只是换噪声大小，也换了噪声种类。
- Airline 仍以 agent 为主，M2a 共 97 个，airline:6/26/35/45/48 在五个模拟器下都出现。
- "开场第一句就结束"在五组 v3 运行里都是 0，确认 v2 的 U4a 来自方舟服务异常。
- 没有新增分类代码。新发现一处 gold 错误：airline:44（见
  [benchmark-variants.md §1.2.1](benchmark-variants.md)）。
- v2 三模拟器版（Doubao
  用按量付费的旧运行）：[user-sim-case-study-20261012/REPORT.md](user-sim-case-study-20261012/REPORT.md)。

**证据。** 运行目录
`_artifacts/AReaL/codex-sao-math/user-sim-20261012/`：`api-user/deepseek-base-u0p0`（pueue
473）、`api-user/doubao-base-u0p0`（476）、`step0-selfhost-user/full-qwen35user-v2`（472）、`api-user/deepseek-test-u{1p0,0p1}`（474/475）；DeepSeek
旧运行 `tau2-eval-20261010-fixed-v2/step0-base-t0-deepseek-fixed-v2`。代码 `9e7082dc`，启动脚本
`scripts/user-sim/launch_api_user.sh`。
