# 用户错误检测器：规则、gold、因果 judge 的离线评估（2026-10-18）

目的：按边界 β（检测器标记的用户边界取 β=0，其余取 1）需要一个检测器。这里在已有人工标注（v3/v4）和 critic 前向数据
（[critic/user-boundary-forward.md](../critic/user-boundary-forward.md)，PPO step 119 /
SAO step 59，30,083 个用户边界）上 离线比较三类检测器，不训练。Codex 执行，Claude 复核：

- 规则 + gold（pueue
  627）：`_artifacts/AReaL/codex-sao-math/tau2-detector-offline-20261018/`（`REPORT.md`、`flags.jsonl`、
  `audit.jsonl`）
- 因果 judge 第二轮（pueue 625 停止后由 630 完成，只用 DeepSeek 官方 `deepseek-flash`）：
  `_artifacts/AReaL/codex-sao-math/tau2-user-error-judge-20261018/round2/`。625 中途调用过
  `deepseek-v4-pro`，那部分结果 移到 `round2/superseded-pro/`，不进入任何结论。
- 第一轮 judge：[user-error-judge-validation/](../user-error-judge-validation/REPORT.md)

## 1 结论

1. **U7a 规则（用户发 TRANSFER 而此前没有转接工具调用）不能当用户错误检测器。** 标注集上只算失败轨迹时，precision 也只有 60% / 46%
   （v3 / v4）。训练前向数据里逐条审读的 8 个仅 U7a 命中的失败边界全部是 agent 的错：agent 算错价格或改错航班后在正文里说"建议转人工"，
   或者把转接工具调用写成正文 JSON，用户只是照着 TRANSFER。这和 Telecom 的转人工筛查一致
   （[telecom-case-study-d1 §3.3](../telecom-case-study-d1/REPORT.md)）。
1. **U1 规则在训练数据上也会误标 agent 的错。** 仅 R 命中（未被 gold 确认）的 U1 失败边界审了 22 个，12 个是用户错，9 个是 agent
   在用户 STOP 之前已经提议了错误方案（错商品、不可行的航班衔接、虚构的支付 ID）。这种情况下用户提前结束并没有让 agent 少拿分：照它的提议执行
   同样会失败。R（U1∨U7a）在训练失败边界上加权 precision 只有约 45%，算上成功轨迹为 37.6%，远低于标注集上的 69% / 58%。
1. **gold 检测器 G 的作用只是给 U1 加一道"提议是否正确"的检查，没有补上新的错误类型。** G 只在终止边界判断：已执行的写操作都符合 gold、还有
   gold 写操作待执行、agent 最后一条消息是确认请求，且某个待执行动作的全部参数都出现在最近 5 条 agent 文本里。标注集上 precision 97.5%
   / 94.7%，recall 46.4%。两版合并，正类里 R 和 G 都抓到 158 条，只有 R 抓到 46 条，只有 G 抓到 10 条（主要是 U4b），两者都漏
   148 条，其中 136 条是 U3（违反剧本、编造事实）。按"瑞士奶酪"的标准，G 没有补上 U3 这个最大的洞。训练数据上 G 只标了 38 个边界（PPO
   16、SAO 22，占 0.13%），审读 precision 85%。
1. **因果 judge C 是目前唯一能抓 U3 的检测器。** C 用 Flash 开 thinking，只判断用户的实质内容（不管 STOP / TRANSFER
   的时机），并要求 agent 先错时不归因用户。U3 recall 72.7% / 61.0%，agent 致败轨迹误标 3.4% / 4.9%。关 thinking 时
   U3 recall 只有 12%，不能用。 与只算失败轨迹的规则组合后，precision / recall 为 87.7% / 84.5%（v3）、73.7% /
   80.9%（v4），agent 误标 7.8% / 12.1%，剩下的误标 大多来自 U7a 那一支。第一轮 G+T 的 agent 误标约 30%。
1. **离线重算的 advantage 几乎看不出差别。** 以"agent 最后一轮的提议是否覆盖待执行 gold
   写操作的全部参数"（`proposal_correct`）为标签，β≡1 的 AUC 为 0.513（PPO）/ 0.516（SAO）。按 R、R∪G、加 G_agent
   否决把命中边界的 β 置 0，AUC 变化都在 ±0.004 以内，区间跨零；按严格 G 置 0 为 +0.004 / +0.010，但 G 和标签共用 gold
   参数覆盖，提升有构造上的相关性。统一 β=0.5 在 PPO 上 +0.028、在 SAO 上 −0.016， 方向不一致。原因是检测器只命中 0.1%–1.6%
   的边界，而且全在终止边界，对整体排序影响很小。

## 2 对方法的含义

- 629 实现的 `user_boundary_detector: rules`（U1∨U7a）**不能直接用于训练**：U7a 那一支主要是在给 agent 免责。按边界
  β 的代码 （`codex/sao-math` 的 `410ff779`）保留，检测器需要换。
- 现在能用的组合是：**U1 规则做候选，C 负责 U3**；U7a 去掉。
- **决定（用户，2026-10-18）：gold 不进入检测器。** 初步消融显示 G 抓到的几乎都是 U1，和规则重复，没有独立贡献。规则、judge
  各自独立评估、互不输入（规则结果不写进 judge prompt），只在最终标记处取 OR，并分别报告各自的命中、precision 和独有贡献。
- **待测：judge 能否单独覆盖 U1。** 当前 C 的 prompt 排除了控制符时机，所以 U1 只能靠规则。如果去掉这条排除、让 judge 同时判断控制符的
  judge 在 U1 上不漏检，且不带回第一轮的误报，就只保留 judge；否则规则管 U1、judge 管内容，职责分开。
- 前向数据上还没有 C 的边界级标记，所以第 1 节第 5 点里的 advantage 重算没有包含 C。C 输出首错消息编号，可以对到边界。
- 训练要等 C 的离线结果。前向数据里用户错误的绝对数量本身就少（U1 规则在 PPO 2,988 条 episode 里命中 114 个边界），即使检测器完全准确， 按边界
  β 改变的 advantage 也只占很小一部分。

## 3 复核记录

- 627：Claude 用 `flags.jsonl` 重算了各检测器在前向数据上的命中边界数（R 176 / 236、G 16 / 22、R∪G+否决 113 /
  166），与报告一致；按 审读层和命中规则拆分了 `audit.jsonl`（第 1、2 点的计数来自这里），逐条看了 17 条仅 R 命中、被判为 agent 错的理由。
- 630：确认 `run.py` 只剩两个 Flash 配置，`api_cache/` 只有 `flash_off`（1,688 条）和 `flash_on`（1,708
  条，含重试）；Pro 缓存最后 写入时间早于 630 启动，停止后没有新的 Pro 请求。这一轮 Flash 费用估算约 9.2 美元。
- 两份报告都是 Codex 单人审读，不是独立人工标注；v4 标签按题批量打，只当近似真值。规则组合使用最终 reward 门控，属于离线设计； 训练时检测发生在
  episode 结束后，也拿得到 reward。

## 4 局限

- 120 个审读边界里大部分是失败终止边界，分层权重放大了单条判断的影响；precision 区间不含审读者误差。
- `proposal_correct` 是文本参数覆盖的代理标签，和 G 共用信息。
- 前向数据是训练题上的 in-sample 轨迹，PPO / SAO 的 checkpoint 不同，差异不能归因于算法。
