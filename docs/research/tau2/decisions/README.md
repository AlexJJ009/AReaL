# τ² 研究决策记录

本目录记录 τ² 审计与训练中做出的决策，每份包含 Context / Decision / Consequences /
Status。编号只在本目录内有效。原因和证据留在链接的分析文档里，这里只写结论。

仓库级决策在 [docs/decisions/](../../../decisions/README.md)，下列几份与本目录互补，不在此重复：

- [ADR 0003](../../../decisions/0003-tau2-data-and-evaluation.md)：数据与评测。
- [ADR 0005](../../../decisions/0005-tau2-online-critic.md)：critic。
- [ADR 0009](../../../decisions/0009-official-tau2-runner-semantics.md)：官方 runner 语义。
- [ADR 0010](../../../decisions/0010-tau2-official-and-fixed-benchmark-variants.md)：official
  / fixed 变体。
- [ADR 0011](../../../decisions/0011-tau2-lost-episodes-in-training.md)：训练中丢失的 episode。

## 列表

- [0001](0001-user-simulator-deepseek-flash.md) 训练和评测的用户模拟器用 DeepSeek-Flash。早期默认于
  2026-10-10 采纳、10-13 复核；历史 Qwen 对照另有范围，当前仍遵守仓库 DeepSeek-only 约定。
- [0002](0002-nl-judge-decoupled.md) NL judge 与用户模拟器分开配置，固定为
  DeepSeek-Flash。状态：已采纳（`9e7082dc`，2026-10-13）。
- [0003](0003-simulators-default-thinking-mode.md) 用户模拟器和 NL judge
  用各模型的默认模式。状态：已采纳（2026-10-12）。
- [0004](0004-eval-protocol.md) 评测协议：Airline + Retail，temperature 0，每题 4 次，报 mean@4 与
  pass^k。早期 60 题协议于 2026-10-10 采纳；后续 PPO 扩展为含 Telecom 的 100 题，原指标定义保留。
- [0005](0005-transfer-rule-retail-exception.md) 转人工规则：Retail 中 gold 有写操作的题设例外，Airline
  不设。状态：已采纳（fixed v1 起，2026-10-10）。
- [0006](0006-fixed-v3-rejected-patches.md) fixed-v3 不采纳两项补丁：9 题 TRANSFER 说明和 retail:45
  追问。状态：已采纳（Claude review，2026-10-12）。
- [0007](0007-telecom-excluded-then-separate-pack.md) Qwen3.5 训练不用 Telecom；Telecom 数据单独建
  pack。状态：已采纳（2026-10-10，2026-10-11 补充）。
- [0008](0008-areal-data-filtering-policy.md) AReaL-tau2-data 的筛选策略：只留 reward
  正确的题，只允许指令侧小修。状态：已采纳（用户，2026-10-10）。
- [0009](0009-reply-cap-4096.md) 训练中保留 4096 的单次回复上限。状态：已采纳（用户，2026-10-10）。
- [0010](0010-pause-method-draft.md) 暂停方法草案（原 16.11），先找准现象。状态：已采纳（2026-10-12）。
- [0011](0011-research-object-unreliable-real-users.md)
  原研究对象为真实用户的不可靠反馈（2026-10-13）；10-19 起实验收窄为模拟器训练，10-24 反思与 10-30 结果限制了原主张。新方向待用户决定。
- [0012](0012-critic-head-init.md) critic head：w 零初始化、b
  初始化为训练集均值，同一学习率。状态：已采纳（2026-10-13，`7da3a4f3`、`275548de`）。
- [0013](0013-critic-data-mix.md) critic 训练数据配比：简单题降采样，保留全部有信号题。状态：已采纳（2026-10-13）。
- [0014](0014-sao-critic-510-step55.md) SAO 使用 critic 510 第 55 步的 checkpoint（看 test
  挑的）。状态：已采纳（用户，2026-10-14）；SAO 534 已完成。
- [0015](0015-sao-eval-save-every-10-steps.md) SAO 正式运行每 10
  步评测并存档。状态：已采纳（用户，2026-10-14；`b8d95027`）。
- [0016](0016-sglang-mem-fraction-0.5.md) critic mix 重跑把 SGLang `mem_fraction_static` 降到
  0.5。状态：已采纳（用户确认，2026-10-14）。
- [0017](0017-sglang-strip-tms-and-fp32-a-log.md) 启动 SGLang 前清掉 TMS 预加载变量；`A_log` 按 fp32
  传。状态：已采纳（`d8f72c51`、`6bf1d96e`，2026-10-14）。
- [0018](0018-qwen3-fsdp-config.md) Qwen3 沿用现有 FSDP 配置，不再加其他优化。状态：已采纳（2026-10-11）。
