# 实验记录：Qwen3-4B-Instruct 上的 PPO / GRPO / SAO 对比

| 项                 | 内容                                                                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 编号               | EXP-2026-10-15-qwen3-ppo-grpo-sao                                                                                                                  |
| 状态               | PPO 120 步（578 + 589）、GRPO 47 步、SAO `sao-formal-qwen3-v3-mse` 120 步已完成（2026-10-17）；120 步结果见 §4.0                                   |
| 设计 / 执行 / 批准 | Claude / Codex / GongxunLi                                                                                                                         |
| 代码               | `codex/sao-math`；每个运行用固定源码（见 §3 表）                                                                                                   |
| 产物目录           | `_artifacts/AReaL/codex-sao-math/` 下 `tau2-ppo-20261015/`、`tau2-sao-grpo-20261015/` 等（§3）                                                     |
| 相关文档           | [../critic/critic-accuracy-and-ppo.md](../critic/critic-accuracy-and-ppo.md)、[../ppo-training-budget-survey.md](../ppo-training-budget-survey.md) |

本记录是运行后整理的，没有预先登记假设。

## 1. 目的

在同一模型、同一题池、同一评测上，比较三种算法：

- 标准 PPO（带 critic）能不能训出效果；
- GRPO 在相同 episode 预算下的表现；
- SAO（critic 先更新两次再算 advantage）相对两者的差别。

这三组是"用户不可靠"研究方向的 baseline：先看清 baseline 为什么提升小，再决定方法要改什么。

## 2. 共同设置

- 模型：`Qwen/Qwen3-4B-Instruct-2507@cdbee75f`，actor 和 critic 同源。
- 训练题池：747 题 = AReaL-tau2-data 清洗后的 pack v2 648 题 + 官方 fixed train 99 题；Airline
  410、Retail 337，没有 Telecom（[../training-data.md](../training-data.md)）。
- 每步 128 条 episode，用户模拟器 DeepSeek。
- 评测：fixed-v3 test 60 题（Airline 20、Retail 40）+ Telecom test 40 题，每题 4 次，temperature
  0；训练前、每 10 步、结束时各一次。
- GPU：actor（PPO / SAO 时加 critic）在 0–3，rollout 在 4–6，评测在 7。
- `per_episode_microbatching: true`：每个 episode 单独一个 microbatch，与 Qwen3.5 的切法一致（见 §5.1）。

| 设置      | PPO                                              | GRPO-matched                        | SAO                                                      |
| --------- | ------------------------------------------------ | ----------------------------------- | -------------------------------------------------------- |
| 每步      | 128 题 × 1                                       | 16 题 × 8                           | 128 题 × 1                                               |
| 总量      | 8 epoch / 48 步，续训到 20 epoch / 120 步        | 1 epoch / 47 步 / 5976 条           | v1：48 步；v2：20 epoch / 120 步                         |
| actor lr  | 1e-6，constant                                   | 1e-6                                | 1e-6                                                     |
| advantage | GAE γ=λ=1，batch 白化                            | 组内 mean/std（eps 1e-5）           | length-adaptive GAE（α=1.5），不白化                     |
| clip      | 0.2 / 0.2                                        | 0.2 / 0.28（沿用 `grpo-formal-v2`） | direct DIS 0.3 / 5.0                                     |
| critic    | 全参数，lr 1e-5，value clip 0.5，ADR 0012 初始化 | 无                                  | attention 冻结，lr 5e-6，value clip 0.5，每步先更新 2 次 |
| offload   | TMS，actor 与 critic 轮流上 GPU                  | 关，`expandable_segments`           | 同 PPO                                                   |

## 3. 运行记录

| 运行                      | pueue | 源码                                                 | 结果                                                          |
| ------------------------- | ----- | ---------------------------------------------------- | ------------------------------------------------------------- |
| `ppo-formal-qwen3-v1`     | 571   | `413dfad9`                                           | 第 3 步 OOM                                                   |
| `ppo-formal-qwen3-v2`     | 575   | `ca4aad18`                                           | 第 2 步 OOM                                                   |
| `ppo-formal-qwen3-v3`     | 578   | `tau2-ppo-20261015/src-2a4f25bb`                     | 完成 48 步，约 7.5 h                                          |
| `grpo-matched-qwen3-v1`   | 584   | `tau2-sao-grpo-20261015/src-e223483c`                | 完成 47 步，约 6.6 h                                          |
| `sao-formal-qwen3-v1`     | 583   | 同上                                                 | step 0 评测后，第 1 步 `CUDA error: invalid argument`（§5.2） |
| `ppo-formal-qwen3-v3-ext` | 589   | `tau2-ppo-extend-20261016/src-2a4f25bb-continuation` | 从第 48 步续训到 120 步，运行中                               |
| `sao-formal-qwen3-v2`     | 590   | `tau2-sao-20261016/src-d8d040db`                     | 排队，排在 589 之后、D1 之前                                  |

## 4. 结果

### 4.0 120 步结果（2026-10-18 补）

数据：SAO case study（Codex，`tau2-sao-case-study-20261017/eval_curve.csv`）。SAO 实际运行是
`sao-formal-qwen3-v3-mse` （`tau2-sao-20261016/`）：critic 冻结 attention、lr 5e-6、不做 value
clip，每步先更新 critic 2 次；actor 用 length-adaptive GAE（α=1.5），不白化。评测是 fixed-v3 test，每题 4
次，mean@4。**训练域**是 Airline 20 + Retail 40（240 条 episode）；Telecom 40 题不在训练池里，单列。

| step | PPO 训练域 | PPO Airline | PPO Retail | PPO Telecom | SAO 训练域 | SAO Airline | SAO Retail | SAO Telecom |
| ---- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- | ---------- | ----------- |
| 0    | 0.396      | 0.237       | 0.475      | 0.456       | 0.346      | 0.212       | 0.412      | 0.444       |
| 10   | 0.354      | 0.175       | 0.444      | 0.537       | 0.354      | 0.188       | 0.438      | 0.438       |
| 20   | 0.371      | 0.200       | 0.456      | 0.487       | 0.321      | 0.163       | 0.400      | 0.438       |
| 30   | 0.362      | 0.150       | 0.469      | 0.456       | 0.396      | 0.212       | 0.487      | 0.487       |
| 40   | 0.400      | 0.175       | 0.512      | 0.450       | 0.404      | 0.287       | 0.463      | 0.463       |
| 50   | 0.429      | 0.188       | 0.550      | 0.556       | 0.388      | 0.175       | 0.494      | 0.525       |
| 60   | 0.454      | 0.175       | 0.594      | 0.506       | 0.396      | 0.212       | 0.487      | 0.469       |
| 70   | 0.479      | 0.200       | 0.619      | 0.556       | 0.442      | 0.225       | 0.550      | 0.506       |
| 80   | 0.496      | 0.212       | 0.637      | 0.506       | 0.458      | 0.237       | 0.569      | 0.469       |
| 90   | 0.483      | 0.188       | 0.631      | 0.512       | 0.400      | 0.200       | 0.500      | 0.531       |
| 100  | 0.504      | 0.212       | 0.650      | 0.581       | 0.379      | 0.138       | 0.500      | 0.487       |
| 110  | 0.496      | 0.237       | 0.625      | 0.588       | 0.429      | 0.212       | 0.537      | 0.444       |
| 120  | 0.521      | 0.237       | 0.662      | 0.481       | 0.458      | 0.237       | 0.569      | 0.463       |

GRPO-matched（47 步）训练域 0.379 → 0.408。

训练域按题配对 bootstrap（60 题，2
万次；`phai:/data_storage/yl_test/lgx/tmp/claude-boot/ar_boot.py`）：

| 比较                | 差值    | 95% CI          |
| ------------------- | ------- | --------------- |
| PPO 0 → 70          | +8.3pp  | \[+0.8, +16.3\] |
| PPO 0 → 120         | +12.5pp | \[+3.3, +21.7\] |
| PPO 70 → 120        | +4.2pp  | \[−2.1, +10.0\] |
| SAO 0 → 70          | +9.6pp  | \[+2.9, +16.3\] |
| SAO 0 → 120         | +11.3pp | \[+2.9, +19.2\] |
| SAO 70 → 120        | +1.7pp  | \[−5.0, +8.3\]  |
| GRPO 0 → 47         | +2.9pp  | \[−5.0, +10.8\] |
| PPO − SAO，step 0   | +5.0pp  | \[−2.5, +12.5\] |
| PPO − SAO，step 120 | +6.3pp  | \[−1.7, +14.2\] |

- PPO 和 SAO 在训练域都有显著提升，主要来自 Retail；Airline 没有可分辨的变化。
- 提升集中在前 70 步，70 → 120 两者都不显著。
- 同一个初始模型，PPO 与 SAO 的 step 0 相差 5pp，这是评测噪声的量级；两者 120 步的差（6.3pp）与它相当，分不出算法优劣。
- critic 在训练批次上更新前的 EV：PPO 第 1–20 步 0.07，第 101–120 步 0.34；SAO 分别为 0.08、0.33。
- **用户后端切换**：DeepSeek 官方 API 在 10-17 上午换了模型，PPO 全程用旧后端；SAO 的评测 0–50 步为旧、60 步混合、70–120
  步为新（[issue](../issues/2026-10-17-deepseek-user-backend-switch.md)）。step ≥ 70 的 PPO−SAO
  比较跨了用户版本。

以下 §4.1–§4.4 是 48 步时的记录。

### 4.1 评测

| 运行 | step 0 | 10    | 20    | 30    | 40    | 最终  |
| ---- | ------ | ----- | ----- | ----- | ----- | ----- |
| PPO  | 0.420  | 0.428 | 0.418 | 0.400 | 0.420 | 0.455 |
| GRPO | 0.410  | 0.403 | 0.430 | 0.428 | 0.470 | 0.468 |

按领域，step 0 → 最终：

| 运行 | Airline       | Retail        | Telecom       |
| ---- | ------------- | ------------- | ------------- |
| PPO  | 0.238 → 0.188 | 0.475 → 0.575 | 0.456 → 0.469 |
| GRPO | 0.213 → 0.225 | 0.463 → 0.500 | 0.456 → 0.556 |

- PPO 起点到最终 +3.5pp，按题配对 bootstrap 95% CI \[−2.0, +9.0\]，不显著。这套评测在 80% power 下的最小可分辨效应约
  8pp。
- GRPO 起点到最终 +5.75pp，没有做显著性检验；GRPO 评测的最小可分辨效应约 7.6pp。
- 两个运行的 step 0 是同一个初始模型，差 1pp，可以看作评测噪声的一个参照。
- Telecom 不在训练题池里，它的变化不能当作训练领域内的进步。

### 4.2 训练过程（PPO，两份 case study 一致）

- 同一批 747 题上，训练 reward 从 epoch 1 的 64.4% 升到 epoch 8 的 71.0%，配对 +6.6pp，CI \[2.8,
  10.2\]：policy 在见过的题上确实在学。
- 没有失稳：entropy 在 0.09–0.12，截断不到 0.1%，grad norm 只在第 19 步有一次尖峰。
- critic 在训练批次上更新前的 EV 从约 0 升到约 0.29；这是训练批次上的数，不能说明 critic 在新题上准。
- 每步只做一次梯度更新（1 个 minibatch），更新时新旧策略相同，ratio 恒为 1，clip 不起作用。

### 4.3 失败归因

两份独立 case study 的结论一致：

- 失败中 agent 自己的错约占 92%，用户的错约占 8%（Claude 146 条，Codex 训练 32 条 + 评测 48 条，均分层抽样）。
- 用户的错全部是同一种：在一条消息里说"Yes, proceed"并带 `###STOP###`，tau2 在 `orchestrator.py:582` 收到 STOP
  即结束，agent 已确认的写操作来不及执行。
- 因用户出错而被误判为 0 分的训练 episode 约 3%。
- 评测中重复之间的差异 99.8% 首先出现在用户消息上，所以用户模拟器的随机性主要放大评测噪声，而不是把 reward 判错。
- 6 个评测 checkpoint 盲标（每个 40 条），agent 错和用户错的占比都没有趋势。
- 有一个显著的行为变化：至少做一次写操作的 episode 从 74.6% 升到 84.2%，每条的写操作从 1.27 次到 1.69 次；没确认就写的比例（正则粗测）从
  6.2% 到 13.3%。
- 还发现一些得 1 分但做错的轨迹（扣款金额与确认不符、泄露禁止展示的地址），是评分标准没覆盖，不是用户造成的。

旧的"失败约 40% 归用户"出自
[../user-sim-case-study-v3/REPORT.md](../user-sim-case-study-v3/REPORT.md)：Qwen3.5-4B、五个用户模拟器合计、没有
Telecom，与本实验不可比；其中只看 DeepSeek 用户是 16/67。

报告：

- Claude：`tau2-case-study-20261016/claude/claude-report.md`、`claude-report-trend.md`
- Codex：`tau2-case-study-20261016/codex/codex-report.md`

### 4.4 训练量

48 步 × lr 1e-6 是可比工作里最少的：FACA（8B，GRPO，DeepSeek 用户，τ Airline+Retail）用同样的 lr 跑 120
步、15,360 条，平均 +5.9pp；AReaL 官方 tau2 配置跑 500 步、lr 1.7e-5。见
[../ppo-training-budget-survey.md](../ppo-training-budget-survey.md)。

## 5. 问题与修复

### 5.1 PPO OOM（571、575）

- 571：actor 和 critic 同时留在 GPU 上，`_should_offload_actor` 没有看 `critic.offload`。`413dfad9`
  让两者的上下 GPU 串行化。
- 575：清缓存（`ca4aad18`）不够。根因是 Qwen3 把多个 episode 拼成一个 32K token 的 microbatch，logits
  峰值翻倍；Qwen3.5 因为 GatedDeltaNet 本来就一个 episode 一个 microbatch。`2a4f25bb` 加了
  `per_episode_microbatching` 开关，v3 峰值约 54 GB。
- 这条路径（tau2 + 同卡 critic + TMS）是我们自己的组合，上游的 PPO 示例没有。

### 5.2 SAO 第一步崩溃（583）

`update_critic_before_actor`（`areal/trainer/ppo/update.py`）在 critic 更新两次后调用
`actor.compute_advantages`，但恢复 actor 显存的回调排在它之后。RPC 进入方法前就把数据搬上 GPU，而 actor 的 TMS
显存还在暂停状态，于是报 `CUDA error: invalid argument`。修复 `a79ed15d` 把回调提前；Codex 用原版源码在 GPU
上复现了同样的错误，修复后跑通 3 步。报告：`tau2-sao-fix-20261016/codex-report.md`。

### 5.3 续训的数据顺序（589）

恢复时 `DistributedSampler` 的 seed 和 epoch 没有保存，原代码恢复后 seed 变成 0。修复 `cd9e66a4` 恢复 seed 和
epoch，并在整 epoch 结束处显式进入下一个 epoch；CPU 验证续训的 72 批与从头跑 20 epoch 的第 48–120
批完全一致。报告：`tau2-ppo-extend-20261016/codex-report.md`。

### 5.4 SAO 的 advantage 用的是更新后的 value（调查，未改行为）

AReaL 在 critic 拟合两次后重新算 value，actor 用 $V_{\phi_2}$ 算 advantage；本地 `miles-values`
快照用的是拟合前的 value。SAO 论文 §3.2 只说 value 先更新"再用于 advantage 计算"，没有说明 Figure 4(a) 的 EV
在哪个时点测。后果：$R-V_{\phi_2}$ 是在同一批 return
上拟合过的残差。报告：`tau2-sao-grpo-prep-20261015/codex-report.md` §AReaL。

## 6. 决定（用户，2026-10-15 / 16）

- PPO 用 SAO 式的 microbatch 切分，不调低 32K microbatch 上限。
- GRPO 组大小 8、每步 16 题、1 个 epoch，与 PPO 的 episode 预算对齐；clip 沿用以往 GRPO 正式运行，不照抄 PPO。
- SAO critic 冻结 attention，critic lr 5e-6。
- PPO 从第 48 步续训到 20 epoch / 120 步；SAO 对齐到 120 步。
- 队列顺序：PPO 续训 → SAO → D1；D1 并发保持 256。

## 7. 未完成

- 589、590 的结果。
- GRPO 最终 +5.75pp 的配对显著性检验。
- critic 在评测轨迹上的 EV（用第 10、20、最终步的 critic checkpoint），区分"记住每题难度"和"学到状态价值"。
- D1 跑完后：Telecom 题目清洗和 Telecom 训练 pack。
