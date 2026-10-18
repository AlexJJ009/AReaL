# Telecom full（D1）case study：失败来源、题目质量与训练题筛选

2026-10-18。分析对象是 D1 评测：未训练的 Qwen3-4B-Instruct-2507（non-thinking，T=0）在官方 `telecom_full`
2,285 题 × 4 次上的轨迹，fixed harness，用户模拟器为 DeepSeek 官方 API（全部为 10-17 换版后的 fingerprint
`e72d3c7f…`，见 [issue](../issues/2026-10-17-deepseek-user-backend-switch.md)）。逐条标注由 Codex（pueue 611）完成，Claude
复核并补了一项转人工筛查（§3.3）。

- Codex 完整报告：[codex-report.md](codex-report.md)；逐条案例：[cases.md](cases.md)
- 服务器产物：`_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/`（`labels.jsonl` 221 条阅读记录、
  `task_stats.csv` 逐题统计、`candidate_sets.json`、`topup_counterfactual.json` 等）

## 1 结论

1. **Telecom 的失败主要是 agent 能力不足，不是用户模拟器出错。** 从 7,347 条失败 episode 中等概率随机抽 60 条：agent 致败
   59 条，benchmark 致败 1 条，用户致败 0 条。用户致败比例的 95% Wilson 区间为 [0%, 6.0%]。这是抽样估计，不是"每条失败都是
   agent 的错"；它说明用户错误在 Telecom 里少到抽 60 条一条都碰不到，而 Airline / Retail 的 v3 case study 里用户致败约占 8%。
2. **有一类成批的题错：1,120 题"愿意充 2 GB"，评分却要求恰好充 2 GB。** 逐题重放确认：只充 1 GB 就能满足其余全部环境断言，只有
   金额断言失败。和 AReaL Telecom v1 第二轮审题发现的 50 题是同一个问题（[training-data.md §6](../training-data.md)）。这批题本轮
   整体排除，没有改题。
3. 没有发现 gold 不可达、空轨迹得分或 gold 工具报错的题。**但还没有做 §5 流程里的逐题审**（Codex 逐题 keep / fix / drop
   和对抗复核）；对推荐的 249 题，逐题审正在进行（§4.3）。
4. 对研究问题的含义：Telecom 能给训练增加 agent 侧的信号（大量 1–3/4 的题），但几乎不提供"用户出错"的样本。研究
   用户错误和 agent 错误的区分，样本主要还是来自 Airline / Retail。

## 2 通过率画像

| 项目                     | 数值                                 |
| ------------------------ | ------------------------------------ |
| 题数 / episode           | 2,285 / 9,140，infra 排除 0          |
| mean@4                   | 19.62%                               |
| 每题 0 / 1 / 2 / 3 / 4 次通过 | 1,440 / 378 / 159 / 135 / 173   |
| MMS 类题                 | 1,984 题（86.8%），通过率 16.3%      |
| 1 个故障 → 6 个故障      | 81.3% → 8.3%；7–9 个故障约 3–4%      |
| 官方 test 40 题          | 45.6%，比 full 高 26pp，不是 full 的缩小版 |

分数低主要来自多故障的 MMS 题：agent 要按顺序查出并修好飞行模式、APN、App 权限、数据开关、漫游、流量超额等多个故障，
中间漏掉任何一个就拿不到分。

## 3 失败归因：agent 错在哪里

### 3.1 怎么判

沿用 v3 的边界（[user-sim-case-study-v3/REPORT.md](../user-sim-case-study-v3/REPORT.md)）：用户违反场景或模拟规则才算用户致败（U）；
agent 的工具使用、推理、漏做子目标算 agent 致败；双方都按说明做了、评分仍不给分算 benchmark 致败（B）。每条失败只记一个主因。
agent 已经错误地宣布"没办法了"之后，用户顺着结束对话，不算用户提前 STOP。

### 3.2 agent 致败的三种表现

| 类别 | 表现                                               | 随机失败 60 条 | 四次全错题 30 条 |
| ---- | -------------------------------------------------- | -------------: | ---------------: |
| T3   | 漏做账户侧动作：没给流量充值、没开漫游             | 32 | 11 |
| T2   | 实体或工具语义错：修错线路、把正常状态读成故障     | 20 | 15 |
| T1   | 设备侧排障漏步、该转人工没转、来回循环耗尽步数     | 7  | 4  |
| B1   | 故障已修好，只因"恰好 2 GB"断言拿 0 分             | 1  | 0  |
| U    | 用户独立致败                                       | 0  | 0  |

典型例子（原文见 [codex-report.md §4.2](codex-report.md)）：

- **T3**：用户主动说"我愿意充 2 GB 流量"，agent 回答"充流量不影响 MMS"，拒绝充值，用户随后转人工。流量超额正是故障之一。
- **T2**：用户号码对应线路 L1002，agent 给 L1001 开了漫游，之后一直修不好。
- **T2**：Wi-Fi Calling 已经关掉、MMS 设置正常，agent 却说 MMSC 地址无效，转人工。这种情况官方评分其实是 1，fixed
  harness 因为多余的转人工判 0；4 条这样的轨迹逐条看过，都是 agent 误判状态，不是题错。

### 3.3 用户发 TRANSFER 而 agent 没调用转人工工具（Claude 补查）

D1 有 1,533 条失败里，用户发了 `###TRANSFER###`，但 agent 从没调用 `transfer_to_human_agents`。这个数量很大，需要排除
"工具调用没被解析"这类 harness 问题。按用户发 TRANSFER 前 agent 的最后一条消息分类（脚本
`phai:/data_storage/yl_test/lgx/tmp/claude-d1/transfer_screen.py`）：

| agent 最后一条消息                                       | 条数  |
| -------------------------------------------------------- | ----: |
| 正文宣告转人工（如 "YOU ARE BEING TRANSFERRED…"），没调工具 | 1,450 |
| 把转人工的工具调用 JSON 写在正文代码块里                 |    81 |
| 没提转人工                                               |     2 |

95% 是 agent 修不好、在正文里宣布放弃，用户按 guideline 回应 TRANSFER。就算真的调用了工具，故障也没修好，同样是 0 分，所以主因
仍是 agent 排障失败。81 条把 JSON 写进正文，是 Qwen3 的工具调用格式错误，属于 agent 侧。这一筛查不改变 §3.2 的结论。

### 3.4 用户侧还看到了什么

- STOP 与"yes / confirm / proceed"等授权词同句出现的 31 条：24 条成功、7 条失败；7 条失败都是 agent 先误诊或拒绝继续、用户
  按建议结束，不是 U1。扩大关键词又查了 26 条，也没有看到 U1。
- 14 次用户空回复（content 为空，内容在 reasoning 字段），都被 harness 拦下并整条重跑，不在最终轨迹里。

## 4 题目质量

### 4.1 和 Airline / Retail 的对照

Airline / Retail 的题错主要靠 §5 流程找出来：机械过滤 → Codex 逐题审 → 对抗复核 → 评测 → 失败轨迹复核
（[training-data.md §5](../training-data.md)）。Telecom 到目前为止做到了：

| 步骤                       | 官方 telecom_full（本轮） | AReaL Telecom v1（10-11） |
| -------------------------- | ------------------------- | ------------------------- |
| 机械检查（gold 1 / 空 0）  | 全部 2,285 题，0 失败     | 261 题，改 reward_basis 后全部通过 |
| 失败轨迹复核               | 抽读 90 条有主因的轨迹    | —                         |
| 跨题模板问题               | 发现"恰好 2 GB"，扫全量得 1,120 题 | 第二轮审题发现 50 题，改指令修复 |
| Codex 逐题审 + 复核        | 249 推荐题进行中（§4.3）  | 两轮完成                  |

### 4.2 不算题错的特征

以下官方题普遍存在，按 TELECOM_REVIEW2 的校准不算题错：ticket 写 "No Service" 但手机有信号（ticket 只有 5 个通用模板）、
persona 年龄与客户生日对不上、指令里写 `get_status_bar`、某条断言初始就成立、policy 规定的转人工。

### 4.3 推荐的训练题与逐题审

推荐 249 题：去掉与 test 同初始状态的 41 题和 1,120 道歧义题后，从 1–3/4 通过的题中按 1/2/3 次通过各抽 83 题。并入原 747 题后
Telecom 占 25%，每 epoch 996 题、8 步。

- pack：`dataset/tau2/tau2-official-telecom-pack-20261018-v1/`（Codex 616 构建；249 题与官方原题逐字一致，gold 1 / 空 0，
  与 test 和 v2 pack 无重合，训练加载检查通过）。
- 逐题审：pueue 620–622，三片各 83 题，校准沿用 TELECOM_REVIEW2，加上"去掉任一 gold 写操作后是否仍得分"的检查，并对照每题的
  D1 成功 / 失败轨迹。结论出来之前，这个 pack 不用于训练。

## 5 局限

- 60 条随机样本只能给出用户致败比例的上限（约 6%），不能证明一条用户错误都没有。
- 标注由同一个 Codex 完成，15 条重读无改动，但这是自查，不是独立一致性检验。
- D1 的用户是 10 月版 DeepSeek V4.1，与训练所用的火山 260910 不同版本；Airline / Retail 上两者表现接近（610 vs v3，配对差 −0.8pp），
  Telecom 上没有验证过。
