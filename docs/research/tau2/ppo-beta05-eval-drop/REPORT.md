# 631（PPO + 全局 β=0.5）eval 回落诊断

2026-10-18。631 在 checkpoint 70 之后的 eval 比 PPO 基线低约 8.6pp，需要判断是实验崩了、infra 污染，还是策略退化。离线分析由
Codex（pueue 637）完成，只读日志和轨迹，不调用 API、不用 GPU；Claude 复核。

- Codex 报告：`phai:_artifacts/AReaL/codex-sao-math/tau2-ppo-beta-case-study-20261018/REPORT.md`，40 个配对案例在同目录
  `cases.md`，所有数字都有对应脚本（`analyze.py`、`train_reconcile.py`、`diagnose.py`、`reward_components.py`）。
- 分析快照：2026-10-18 21:05 PDT，631 训练到第 95 步。

## 1 结论

1. **631 没有崩。** 训练 reward、entropy、grad norm、截断比例都和基线一致（第 80–95 步 task reward 0.713 对
   0.736）。
1. **没有 infra 污染。** 631 共 3,584 条入分 eval episode，其中 infra 导致的零分为 0。三次火山配额卡顿期间的失败
   attempt 都被整条重跑或拒收，12,000 条训练导出和完成轨迹逐条对上，没有失败或半截轨迹进入训练 batch。controller
   超时后晚到的 48 条导出没有用于训练，这 48 条就是 rejected 计数。
1. **eval 回落是真的，主要在 Retail，有策略退化的迹象。** checkpoint 70/80 两轮合并差距 −8.62pp，按任务聚类的标准误
   2.34pp；631 自身前后窗口也下降了 5.08pp。Retail 在 checkpoint 70 低 16.9pp、80 低 14.4pp，失败多是数据库状态错误
   （DB fail 85 对 54、75 对 52），不是 NL judge 用词问题。
1. **差距不能归因到 β=0.5。** 631 和基线除了 β，还换了用户模拟器和 NL judge（火山 260910 对 DeepSeek 官方
   `deepseek-flash`），训练并发（24 对 128）、训练 timeout（1800 s 对 1200 s）和 SDK 重试次数（120 对 3）也不同。

## 2 关键证据

eval 文件名是 checkpoint 更新数，结果写进主日志要晚约 10 步：checkpoint 70 在第 79 步记入，80 在第 90 步记入。下表按
checkpoint 对齐。

成功率 %，631 / 基线：

| checkpoint | overall     | airline     | retail      | telecom     |
| ---------- | ----------- | ----------- | ----------- | ----------- |
| 50         | 47.3 / 48.0 | 26.3 / 18.8 | 52.5 / 55.0 | 52.5 / 55.6 |
| 60         | 46.9 / 47.5 | 18.1 / 17.5 | 55.0 / 59.4 | 51.9 / 50.6 |
| 70         | 42.3 / 51.0 | 27.5 / 20.0 | 45.0 / 61.9 | 46.9 / 55.6 |
| 80         | 41.5 / 50.0 | 13.8 / 21.3 | 49.4 / 63.8 | 47.5 / 50.6 |

每轮 400 条（airline 80、retail 160、telecom 160）；631 的 checkpoint 60 有 8 条 airline 被 infra 排除，按 392 条计。Claude
用 `evidence/async-eval/{70,80}.json` 复核了 70、80 两行。

40 个定向案例都是"基线同 checkpoint、同任务、同 trial 成功，631 失败"，主因分布：T2（实体、政策、工具语义错误）33，
T1 5，U1 2，infra 0。典型错误：

- 改地址任务连续两轮编造纽约地址（"1234 Broadway"），没有读另一张订单里的真实地址；
- 为取消一个音箱取消了整张订单；
- Telecom 把 L1002 的故障修到 L1001 上。

用户侧有可见差异：火山用户在确认时同句 STOP（U1）的候选在 checkpoint 70 是 28 条，基线 13 条；但到 checkpoint 80 两者
相同（15 对 15），成功率差距仍在，所以用户差异不能解释全部回落。

输出层面没有失控：两者的轮数都随训练增加，631 的正文工具 JSON 和重复文本不比基线多，没有语言切换，最新两轮没有
context limit。

## 3 决定

- 631 跑完，后续 checkpoint 90、100 的 eval 用来区分暂时回落和持续变差。
- 636（U1 检测门控、逐边界 β）和 631 用同一套火山设定，两者之间可以直接比较。
- 回答"β 是否优于标准 PPO"需要一个同为火山用户、同 judge、同并发和 timeout 的 β=1 PPO 基线。是否占 gpu8 跑（约 13
  小时）由用户决定。

## 4 局限

- 单 seed；比较窗口是看到曲线后选的，区间是探索性估计。
- 训练 batch 是用日志约束离线重建的，没有 optimizer 原生的 episode-ID 清单。
- 案例是机制取证，不是随机抽样；主因由单个 Codex 判定。
