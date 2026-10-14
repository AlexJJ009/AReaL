# critic₀ 扩数据实验（2026-10-12，478–480 完成）

> 来源：原 README 第 16.14–16.14.2 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

**要回答的问题**：训练集从官方 train 扩到"AReaL pack v2 + 官方 fixed train"之后，Qwen3.5-4B 的 critic₀ 在官方
fixed test 上的 explained variance
能否变成正的。[problem-and-plan.md §4](../research-direction/problem-and-plan.md) 的方法线和 SAO
都以此为前提。

**上一次 critic₀ 的诊断**（`tau2-training-20260927-nonthinking/critic-formal-v1`，official 变体，含
Telecom，178 题 × 2 epoch = 24 次更新）：

| 验证步 | Airline EV | Retail EV | Airline MSE | Retail MSE |
| ------ | ---------- | --------- | ----------- | ---------- |
| 0      | −38.7      | −50.2     | 9.95        | 8.97       |
| 10     | −3.72      | −5.61     | 1.47        | 1.04       |
| 20     | −1.05      | −1.51     | 0.52        | 0.39       |
| 24     | −0.59      | −0.92     | 0.87        | 0.48       |

- 0/1 目标下，永远预测平均成功率的 MSE 只有 $p(1-p) \le 0.25$。最后一步的 MSE（0.87、0.48）仍高于 0.25，critic
  连平均值都没学到，属于**没训练完**，还谈不上"学不会区分"。
- 原因有三：只有 24 次更新；value head 是无 bias 的线性层、按 std 0.02 随机初始化，初始输出在 $\pm 3$ 左右（第 0 步 MSE 约
  10），而 Adam 以 5e-6 的学习率 24 步只能让每个权重移动约 1.2e-4，相对 0.02 的初始尺度几乎没动；每步梯度范数 16–1197，全部被裁剪到
  1，训练时 value 均值在 −0.2 到 3.4 之间摆动。
- 训练时每个 batch 只有 16 条，成功率高，有 5 个 batch 全为 1，训练侧的 EV 无定义。只看验证集的 EV。

**这次的设置**（代码 `04be68ab`，固定在 worktree `tau2-training-20261012-critic-data/src-04be68ab`
上运行，与上一次的 `tau2-training-20260927-nonthinking/critic-formal-v1` 同一命名方式）：

| 项       | 上一次                                                                                                                                                   | 这次                                                                                      |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 变体     | official                                                                                                                                                 | fixed（`fixed-v2` 数据目录）                                                              |
| 领域     | Airline、Retail、Telecom                                                                                                                                 | Airline、Retail                                                                           |
| 训练集   | 官方 train 178 题                                                                                                                                        | AReaL pack v2 648 题（打乱）在前，官方 fixed train 99 题（打乱）在后，每个 epoch 同一顺序 |
| 更新次数 | 24                                                                                                                                                       | 94（747 题 / batch 16，2 epoch）                                                          |
| 验证     | 官方 test 100 题                                                                                                                                         | 官方 fixed test 60 题，每 5 步一次                                                        |
| 其余     | 不变：actor 固定（lr 0）、policy temperature 1.0、critic lr 5e-6、MSE、冻结 attention、梯度裁剪 1；user 与 NL judge 都是 DeepSeek-Flash（temperature 0） | 同左                                                                                      |

**只改数据、不改 value head 的原因**：这次要回答的是"数据扩大后 EV 能否变正"。如果同时改 head 的初始化或学习率，EV
变正时就分不清是哪一项起了作用。若这次 EV 仍为负，下一步单独试 head 零初始化或给 head 更高的学习率。

**已知风险**：pack v2 在 temperature 0 下成功率
0.946（[step0-evaluations.md §5](../step0-evaluations.md)），每个 epoch 前 648 题的目标几乎全是
1，critic 容易学成"一律预测接近 1"；而官方 fixed test 的 Airline 只有 0.44
左右（[step0-evaluations.md §6.2](../step0-evaluations.md)）。所以 pack 部分能提供更多更新步数，但未必教会
critic 区分成败。验证报告要同时看 EV 和"MSE 对比 $p(1-p)$"，并按领域分开。

**成本估计**：1494 条训练 episode 加 60 条验证。按 pack v2 评测的用量（每条约 10K prompt token、1.4K completion
token）估算，约 16M prompt 和 2.2M completion token。上一次每步约 3.3 分钟，这次预计 5–6 小时，需要整机 8 张卡（4
张训练、4 张 rollout）；每 5 步存一次 checkpoint，每次约 15 GB，共约 280 GB。

**启动**：`tau2-training-20261012-critic-data/launch_critic.sh <name> {qwen35|instruct}`（副本在
`../scripts/critic-data`）。它只是给 `scripts/tau2/run.sh data critic` 固定环境变量和几个
override，训练脚本和 `config_critic_production.yaml` 与上一次相同。结果在
`<name>/tau2-critic-fit-evidence/validation-step-*.json`。

**其余配置的核对（2026-10-12）：**

- **batch 和步数**：每次更新 16 条 episode，与上一次相同；747 题 / 16 = 47 步每 epoch，2 epoch 共 94 步，是上一次的
  3.9 倍。
- **验证频率**：每 5 步一次，共 20 次。验证用的 60 条轨迹在第 0 步采集一次、之后复用，每次验证只是在 GPU 上前向加反向，不花 API。保持每 5
  步，便于看 EV 曲线。
- **user 与 judge**：`9e7082dc` 之后两者分开配置。`config_critic_production.yaml` 已显式写了
  `nl_judge_llm: deepseek-flash`、独立的 base URL、key 变量和 temperature 0，解析出的配置里两者都是
  DeepSeek-Flash，不走回退。
- **验证集**：数据目录是 `fixed-v2`，变体是 fixed，所以验证用的是修过的官方 test 60 题，不是 official 原版。
- **验证 rollout 的配置**：critic 训练用 `PPOTrainer.eval_rollout`，它复制 `rollout`
  段（`rl_trainer.py:375`），所以 parser、模板与训练一致；配置里没设的 `evaluation_rollout` 段不被使用。
- **Telecom 不加**：Qwen3.5 在官方 Telecom 和 Telecom pack 上分别是 1.000 和 0.993，目标几乎全为 1，没有信号；SAO
  也不训练 Telecom（[training-setup.md §1](../training-setup.md)），critic
  应与之后的训练用同一领域组合。Instruct 在 Telecom 上只有 0.38，信号多，但 Telecom pack 是另一个 pack，现在的代码只支持一个；两个
  actor 先用同样的数据，便于比较。

**Qwen3-4B-Instruct 的 critic**：同样设置，另跑一次。为它改了三处：

| 问题                                                                                                                        | 不改的后果         | 处理                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------- |
| 冻结 attention 的代码要求同时有 `self_attn` 和 `linear_attn`，Qwen3 是全 attention，没有后者                                | 启动即报错         | `04be68ab`：按模型类型选择，Qwen3.5 仍要求两者，其他模型只冻结 `self_attn`，并保留"一个都没有就报错" |
| HF cache 里 Instruct 只有第 3 个权重分片，另两个是中断的下载                                                                | 离线模式下加载失败 | 把 ModelScope 下载、sha256 与 HF 记录一致的两个分片链接进 HF cache                                   |
| critic 配置的 parser 是 `qwen3_coder`，Instruct 评测用的是 `qwen25`（[step0-evaluations.md §6.1](../step0-evaluations.md)） | 工具调用解析不一致 | 启动脚本对 Instruct 加 `rollout.agent.tool_call_parser=qwen25`                                       |

与 Qwen3.5 的已知差别：Qwen3 走通用的 `AutoModelForTokenClassification`，value head 带 bias；Qwen3.5
用专门的 adapter，head 无 bias。用小尺寸的 Qwen3 配置在 CPU 上确认了它能构建、冻结 12 个 `self_attn` 参数、输出形状
`[1, 5, 1]`；GPU 上的完整路径没跑过，所以先排一个 2 步的 smoke。

## 1 结果

smoke 478（2 步）正常结束；479（Qwen3.5，11:10–16:08）和 480（Instruct，16:08–20:04）都完成 94 步。验证集是
fixed test 60 题，EV 按 token 计算：

|                    | 总体 EV | Airline EV | Retail EV | 总体 MSE / 目标方差 |
| ------------------ | ------- | ---------- | --------- | ------------------- |
| 上一次，第 24 步   | −1.05   | −0.59      | −0.92     | 0.51 / 0.16         |
| Qwen3.5，第 90 步  | −0.04   | −0.08      | −0.35     | 0.28 / 0.22         |
| Qwen3.5，第 94 步  | +0.15   | +0.04      | −0.08     | 0.195 / 0.217       |
| Instruct，第 85 步 | −0.01   | −0.09      | −0.03     | 0.74 / 0.23         |
| Instruct，第 94 步 | −0.00   | −0.07      | −0.02     | 0.236 / 0.234       |

**结论：数据扩到 3.9 倍后，两个 critic 的 EV 都没有稳定变正。**

- Qwen3.5 第 94 步的 +0.15 不稳定，前一次验证是 −0.04。分领域看，两个领域的 MSE 都比直接预测本领域均值差（Airline 0.286 对
  0.245，Retail 0.131 对 0.122），总体 EV 的正值来自"Airline 比 Retail 难"这一领域差异。
- Instruct 第 30 步之后的 13 个验证点 EV 都在 −0.04 到 +0.01，第 94 步分领域 MSE 也略高于本领域方差（Airline 0.199 对
  0.186，Retail 0.255 对 0.246）。

**运行健康**：Qwen3.5 无 infra 失败；Instruct 有 3 次 DeepSeek 返回空 user 回复，都重试成功。日志里的
`Error in golden actions … not found` 与上一次 critic 逐条相同，来自官方 retail 题故意设计的"先给错邮箱/订单号"，不影响
reward。Instruct 有若干次工具调用 JSON 格式错误，按失败计分，是模型本身的格式问题。user 用量：Qwen3.5 15.5M prompt + 2.3M
completion token；Instruct 约 10M + 2M。

**训练过程中的现象：**

1. **Qwen3.5 的 pack 太简单**：训练 batch 成功率 0.81–1.00，94 步里 29 步整批为 1，这些步训练侧 EV 无定义；只有每个
   epoch 末尾的官方题降到 0.27–0.64。两个 epoch 里 pack 648 题有 581 题两次都对，pack 整体成功率 0.935，官方 train 为
   0.73。
1. **Qwen3.5 的预测水平来回漂**：训练侧 `old_value/avg` 常在 1.2–1.9（目标最大为 1），验证 bias 在 −0.68 到 +1.07
   之间。16 个验证点上，验证 bias 与当步训练 batch 的预测均值相关 0.82，与最近 3 个 batch 的目标均值相关 −0.00：漂移来自 critic
   自身，不是在追 batch 的成功率。
1. **Instruct 漂得少，学到的是训练集均值**：`old_value` 大多跟住约 0.6，第二个 epoch 训练 EV 在 0 到 +0.1；test
   成功率只有 0.37，验证 bias 多在 +0.1 到 +0.45，第 85 步跳到 +0.71。它的 head 带 bias，但 bias 零初始化、lr 5e-6
   下 94 步最多移动约 5e-4，不能解释它比 Qwen3.5 漂得少；差别更可能来自骨干结构（Qwen3 全 attention，Qwen3.5 混合 Gated
   DeltaNet），尚未验证。

## 2 两类问题

- **A：整体水平漂移。** SAO 的 advantage 约为 $R - V_t$，$V$ 整体偏高 0.6，所有 token 的 advantage 都被压低
  0.6，与 actor 做得好坏无关。A 不修，critic 有害。
- **B：领域内能否区分成败。** A 修好、B 不解决，$V$ 只是每个领域一个常数 baseline，相当于 REINFORCE 减均值，SAO 需要的逐 token
  credit assignment 落空：critic 无害，但也没用。

把每条验证轨迹的预测均值做最优仿射校准（同时调整体偏移和尺度）后看它与 reward 的 $R^2$，可以看轨迹之间的排序本身：

|          | 第 40–80 步 Airline $R^2$ | 第 40–80 步 Retail $R^2$ | 判断                                                                                                 |
| -------- | ------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------------- |
| Qwen3.5  | 0.28–0.50                 | 0.01–0.06                | Airline 的轨迹均值能排序，但 EV 低不是 A 造成的（[ev-case-study.md](ev-case-study.md)）；Retail 是 B |
| Instruct | 多数 0.00–0.17            | 多数 0.00–0.14           | 两个领域都是 B                                                                                       |

这个 $R^2$ 在同一批 60 条上拟合两个参数，偏乐观；每个领域只有 20/40 条，噪声大。它还用轨迹均值代替每个 token 的预测，用到了轨迹后面的信息，在线的
critic 做不到。验证报告里的 EV 本来就扣掉了整体偏移，A 不影响 EV；压低 EV 的是预测幅度太小和逐 token 的波动，见
[ev-case-study.md](ev-case-study.md)。
