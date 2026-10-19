# 631（PPO + 全局 β=0.5）eval 回落诊断

2026-10-18。631 在 checkpoint 70 之后的 eval 比 PPO 基线低约 8.6pp，需要判断是实验崩了、infra 污染，还是策略退化。离线分析由
Codex（pueue 637）完成，只读日志和轨迹，不调用 API、不用 GPU；Claude 复核。

- Codex
  报告：`phai:_artifacts/AReaL/codex-sao-math/tau2-ppo-beta-case-study-20261018/REPORT.md`，40
  个配对案例在同目录
  `cases.md`，所有数字都有对应脚本（`analyze.py`、`train_reconcile.py`、`diagnose.py`、`reward_components.py`）。
- 分析快照：2026-10-18 21:05 PDT，631 训练到第 95 步。

## 1 结论

1. **631 没有崩。** 训练 reward、entropy、grad norm、截断比例都和基线一致（第 80–95 步 task reward 0.713 对
   0.736）。
1. **没有 infra 污染。** 631 共 3,584 条入分 eval episode，其中 infra 导致的零分为 0。三次火山配额卡顿期间的失败 attempt
   都被整条重跑或拒收，12,000 条训练导出和完成轨迹逐条对上，没有失败或半截轨迹进入训练 batch。controller 超时后晚到的 48 条导出没有用于训练，这
   48 条就是 rejected 计数。
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

每轮 400 条（airline 80、retail 160、telecom 160）；631 的 checkpoint 60 有 8 条 airline 被 infra
排除，按 392 条计。Claude 用 `evidence/async-eval/{70,80}.json` 复核了 70、80 两行。

40 个定向案例都是"基线同 checkpoint、同任务、同 trial 成功，631 失败"，主因分布：T2（实体、政策、工具语义错误）33， T1 5，U1
2，infra 0。典型错误：

- 改地址任务连续两轮编造纽约地址（"1234 Broadway"），没有读另一张订单里的真实地址；
- 为取消一个音箱取消了整张订单；
- Telecom 把 L1002 的故障修到 L1001 上。

用户侧有可见差异：火山用户在确认时同句 STOP（U1）的候选在 checkpoint 70 是 28 条，基线 13 条；但到 checkpoint 80 两者 相同（15
对 15），成功率差距仍在，所以用户差异不能解释全部回落。

输出层面没有失控：两者的轮数都随训练增加，631 的正文工具 JSON 和重复文本不比基线多，没有语言切换，最新两轮没有 context limit。

## 3 决定

- 631 跑完，后续 checkpoint 90、100 的 eval 用来区分暂时回落和持续变差。
- 636（U1 检测门控、逐边界 β）和 631 用同一套火山设定，两者之间可以直接比较。
- 回答"β 是否优于标准 PPO"需要一个同为火山用户、同 judge、同并发和 timeout 的 β=1 PPO 基线。是否占 gpu8 跑（约 13 小时）由用户决定。

## 4 局限

- 单 seed；比较窗口是看到曲线后选的，区间是探索性估计。
- 训练 batch 是用日志约束离线重建的，没有 optimizer 原生的 episode-ID 清单。
- 案例是机制取证，不是随机抽样；主因由单个 Codex 判定。

## 5 β=0.5 改变了什么（pueue 644，2026-10-19）

离线分析（只读日志、轨迹和 checkpoint，无 GPU、无 API），Codex 报告：
`phai:_artifacts/AReaL/codex-sao-math/tau2-beta05-mechanism-20261019/REPORT.md`，配对案例在同目录
`cases.md`，入口 `run_analysis.sh`。Claude 用 `evidence/async-eval/{0,120}.json` 复核了 Retail
的终点和初始数字。

1. **Retail 差距一直持续到终点。** checkpoint 120 Retail 52.5% 对 66.25%，差 −13.75 \[−22.5, −5.6\]
   pp（按 task 聚类）；70–120 合并 −11.7 \[−18.2, −5.8\] pp。overall 45.25 对 50.5。训练集只有 Airline 和
   Retail，Telecom 的 eval 是跨域泛化。
1. **训练集上 Retail 也学得少，但没有回落。** Retail 训练成功率在 41–60 步开始落后（78.2 对 80.9），最后 20 步 80.8 对
   83.6，早于 checkpoint 70 的 eval 差距。eval 差距比训练差距大。
1. **β 的作用（代码层面已确认）。** 有终止用户边界时，actor advantage 对 $R$ 的直接系数是 β，其余经过 critic：
   $A_t = 0.5(R-U_\text{last}) + (U_\text{last}-V_t) + \dots$。约 90% 的 episode 以用户 STOP
   结束，所以几乎每条都被改写。 但两组都做 batch mean/std 归一化，整体缩放会被抵消，"惩罚减半"不等于梯度减半；真正改变的是 advantage
   中真实结局与 critic 估计的相对比重。终止边界占用户边界 TD 二阶矩的约 89%，中间用户轮的影响小。
1. **critic 没有全面变坏。** 631 的 explained variance 在 61–70 步落后（0.264 对
   0.365），终点恢复并超过基线（0.390 对 0.329）。新增 U 位置只占 critic loss token 的约 0.35%。训练时没有保存逐 token 的
   value，所以无法测失败轨迹上 $U_\text{last}$ 是否偏高，这是 H1（critic 看不出的错误只受到部分惩罚）证据链最关键的缺口。
1. **Retail 行为漂移部分符合 H1。** 相对基线，631 后期更多出现：写入了没有来源的地址（+2.2 pp）、无效的 item replacement
   数组（+3.8 pp）、与参考不符的整单取消（+3.2 pp），以及严格 U1 候选（+5.2 pp）。前两类和取消类错误工具未必报错， 用户也会道谢，只有 DB
   检查判失败，与 H1 一致；但无效 item 数组会直接报错，不属于"只有结局才能发现"的错误。工具调用数、 重复调用、未见 ID 写入和 transfer
   没有特异漂移。
1. **backend 混杂仍未排除。** 同一个初始 actor（checkpoint 0）在火山用户下 Retail 43.1%，官方 DeepSeek 下
   47.5%（−4.4 pp，区间含零）。NL judge 的直接影响很小：Retail eval 只靠 NL 判零分的 631 为 0 条、基线 4
   条；相同写操作序列的配对中 DB 判定无一不同。即使把 631 所有严格 U1 候选零分都改成成功，checkpoint 120 仍差 5 pp。
1. **结论。** β=0.5 改变 credit 的方式已在代码层面确认，Retail 行为漂移与"critic 看不出的错误被少罚"的解释一致，但真实训练中的
   因果归因置信度中低：训练和 eval 的用户、judge 都不同，也没有 value 可以还原 advantage。

决定性实验（644 只设计，未运行，均需 GPU）：

- 用同一个用户和 judge（豆包 Turbo + 261017）重评两边 checkpoint 60/70/120 的 actor，先只跑 Retail（6 个 actor
  × 160 = 960 episode）。若差距仍在，就排除了"只是 eval 用户不同"。
- 把保存的 critic（checkpoint 60/70 对应下一训练 batch）跑在 Retail 轨迹上，比较成功和失败轨迹的
  $U_\text{last}$，并在整个 batch 上复原归一化后的 advantage，直接检验 H1。checkpoint 路径和输入清单在
  `followup_plan.json`、`followup_inputs.jsonl`。
- 豆包下的 β=1 对照要保留 `train_user_boundary_values=true`，只改 β，才能把 β 和 U 监督分开。
