# τ² critic EV：训练暴露、分布差异与 ceiling 审计

## 结论与读数时间

训练数据确实缺少 Retail 失败，但这只解释了部分结果。479 的 674 条 Retail rollout 只有 29 次失败；510 的 mix 配比在本次读取的
1008 条训练 rollout 中，把 Retail 失败提高到 45 次，同时 Airline 失败达到 147 次。510 在固定验证轨迹上已经出现 Airline EV
0.179（step 55）、0.168（step 60），Retail 0.028（step 60）。这些数值比 issue 中记录的 step 50
更新，不能再概括成“Airline 最多 0.137、Retail 完全没有正值”。但它们仍来自同一批 20/40 条验证轨迹，不能证明泛化改善的统计显著性。

本分支只读代码与 run artifacts，只执行 CPU 标签统计；不访问其他 case study，不训练，不发 API 请求，不修改 run/repo。510
仍在增长：`data_summary.json` 的 train metrics 快照是 step 63、最近一次 validation 为 step 60；读取
rollout 时已存储 1008 条，即 63 × 16。checkpoint 的 recover 信息也是 step 60。没有 `summary.json` 与 exit
code，因此本报告不声称 80 步已经结束。其他分支或最终报告应按更新后的快照补充 510。

## 1. 来源、口径与复现

仓库根目录：`/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math`。artifacts
根目录：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math`。下面表格中 run
路径相对此 artifacts 根目录。

| run      | 路径                                                           |
| -------- | -------------------------------------------------------------- |
| previous | `tau2-training-20260927-nonthinking/critic-formal-v1`          |
| 479      | `tau2-training-20261012-critic-data/critic-qwen35-formal-v1`   |
| 480      | `tau2-training-20261012-critic-data/critic-instruct-formal-v1` |
| 499      | `tau2-training-20261013-critic-head/critic-head-qwen35-v2`     |
| 500      | `tau2-training-20261013-critic-head/critic-head-qwen35-mix`    |
| 510      | `tau2-training-20261013-critic-head/critic-head-qwen35-mix-r2` |

实际训练 rollout 位于各 run 的
`logs/root/<experiment>/<trial>/rollout/0/<rollout_task_id>.jsonl`，每条文件第 1 行包含
reward、domain、source_id、loss_mask 与 token IDs。官方题的 source_id 以 `tau2:` 开头，pack
题不以其开头。`data_train_labels_<run>.json` 保存逐条来源文件、行号和精简标签，未复制 token IDs 或对话。479 的 1494 条
rollout 正好覆盖 747 题各两次；500 原目录含失败前已生成但不一定消费的 rollout，不能把它们全算进 22 个 optimizer update。

训练读数来自 `tau2-critic-fit-evidence/train-metrics/step-*.json`；验证来自
`validation-step-*.json`。JSON 逐文件的完整读数保存在 `data_summary.json`。完整 rollout 统计脚本为
`data_audit.py`，固定 test ceiling 的修正脚本为 `data_ceiling_check.py`。第一个脚本的 `ceiling`
字段故意保留原始文件全量统计；实际用于 README 对照的是 `data_ceiling_fixed_test.json`。

复现命令（在本报告目录）：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math \
/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/.venv/bin/python data_audit.py

CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math \
/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/.venv/bin/python data_ceiling_check.py
```

510 正在运行，复跑会改变读数。原始 metrics 可以精确绑定更新步骤；原始 rollout 只能绑定当时已写出的采样，不自动证明已经消费。

## 2. 成败样本不是独立 token

| run      | Airline episodes / failures | Retail episodes / failures | 全成功 batch | optimizer updates |
| -------- | --------------------------: | -------------------------: | -----------: | ----------------: |
| previous |                     60 / 13 |                   148 / 17 |       5 / 24 |                24 |
| 479      |                   820 / 110 |                   674 / 29 |      29 / 94 |                94 |
| 480      |                   820 / 372 |                  674 / 195 |       0 / 94 |                94 |
| 499      |                    820 / 94 |                   674 / 29 |      30 / 94 |                94 |
| 500      |                    200 / 50 |                   165 / 17 |       1 / 22 |                22 |
| 510 快照 |                   572 / 147 |                   436 / 45 |       2 / 63 |                63 |

previous 另有 148 条 Telecom，全部成功。500 共生成 365 条，超过 22 × 16 = 352，因失败、恢复尝试或未消费尾部而不能视为 365
条训练暴露；这个表对 500 特意使用“生成的 episode”口径。其他整轮已完成 run 的计数与实际消费规模一致。

几十万 token 不等于几十万独立失败例子。terminal reward 广播给同一轨迹全部目标 token，信息单位首先是 episode，再是共享题目的 episode
cluster。479 的 Retail pack 只有 9 道题曾失败，官方 Retail 只有 15 道题曾失败；同一失败轨迹的 2000 个 token 会重复提供同一个
Monte Carlo 标签，也会揭示不同 prefix，但不会产生 2000 个独立终局标签。

具体 479 原始数据重算（episode success 与 token success 分列）：

| 来源 / domain    | episodes | 独立 tasks | failures / 曾失败 tasks | 混合成败 tasks | episode success | token success |
| ---------------- | -------: | ---------: | ----------------------: | -------------: | --------------: | ------------: |
| pack Airline     |      760 |        380 |                 73 / 58 |             43 |           0.904 |         0.803 |
| official Airline |       60 |         30 |                 37 / 22 |              7 |           0.383 |         0.355 |
| pack Retail      |      536 |        268 |                  11 / 9 |              7 |           0.979 |         0.978 |
| official Retail  |      138 |         69 |                 18 / 15 |             12 |           0.870 |         0.876 |

token success 以原始 `loss_mask` 的有效 token 数加权 reward。它用于估计 token MSE 的数据暴露，不代替训练 engine
实际构造的 target，尤其不处理 truncated 的 bootstrap；后者交由单独的 target 审计。

Airline pack 的长失败轨迹将 token success 从 0.904 拉到 0.803，因此“训练目标几乎全 1”对 Airline 比 episode
统计暗示的弱一些。Retail 两种权重下均明显偏向成功，稀少失败是更直接的限制。479 全部 Airline token success 为 0.744，Retail 为
0.953；499 分别为 0.799 与 0.953。510 快照分别降低到 0.630 与 0.866。

## 3. train/test 差异与对照的限制

| run      | Airline train episode success | Airline fixed validation | Retail train episode success | Retail fixed validation |
| -------- | ----------------------------: | -----------------------: | ---------------------------: | ----------------------: |
| previous |                         0.783 |      0.750（5 failures） |                        0.885 |     0.875（5 failures） |
| 479      |                         0.866 |     0.500（10 failures） |                        0.957 |     0.875（5 failures） |
| 480      |                         0.546 |     0.200（16 failures） |                        0.711 |    0.475（21 failures） |
| 499      |                         0.885 |      0.600（8 failures） |                        0.957 |     0.875（5 failures） |
| 500      |                         0.750 |      0.550（9 failures） |                        0.897 |     0.850（6 failures） |
| 510 快照 |                         0.743 |      0.600（8 failures） |                        0.897 |     0.775（9 failures） |

对 510，issue 中“Retail 大约 5/40 失败”的描述已不适用：它的固定 validation 9/40 失败。479/499 恰好都是 5，500 是
6，480 是 21。跨 run 的验证轨迹独立采样，而且 479/480 为 fixed-v2、499/500/510 为 fixed-v3，head
和数据更改不能组成严格的单因素实验。

更具体地说：479 的 pack Airline success 0.904，但官方 train Airline 只有 0.383，官方 test 单次读数
0.500；“Airline 不会学”混合了大量简单 pack 的最优值与更难官方题的最优值。510 官方 train Airline 的快照 success
0.302、pack 0.832，混合后仍与 validation 0.600 不同。重新配比减少了平均难度差，却未消除任务类型、失败机制、轨迹长度的差异。只有 success
rate 的差异无法证明 covariate shift 的具体机制。

480 共有 567 条失败、没有全成功 batch，但分 domain EV 仍接近零或为负。这排除了“整个问题仅因缺少失败”这种统一解释。它没有排除 Instruct 的
representation 不足、train/test 泛化差、优化不足或 per-prefix 标签噪声；Instruct actor/critic 与 Qwen3.5
同时不同，也不能当成纯类别比例对照。

## 4. optimizer 确实更新，但暴露很短

所有读取到的 train metrics 中，`optimizer_steps_since_init` 从 1 连续达到对应完成步数，`update_successful`
全部为 1。配置读数是 Adam 名称、constant lr $5\times10^{-6}$、weight decay 0、gradient clipping
1、`ppo_n_minibatches=1`，510 的参数 optimizer dtype 为 fp32。不是“训练没有调用 optimizer”或
“大量非有限梯度跳步”。

| run      | 未裁剪 grad norm 范围 | old value token 均值范围 | 最后 20 update 中有正 train EV | 最后 20 有定义 train EV 平均 |
| -------- | --------------------: | -----------------------: | -----------------------------: | ---------------------------: |
| previous |         16.76–1197.38 |             −0.210–3.383 |                              0 |                       −8.877 |
| 479      |          5.16–1568.21 |             −0.513–3.756 |                              8 |                       −0.037 |
| 480      |           1.50–411.41 |             −0.132–1.242 |                             13 |                        0.052 |
| 499      |            0.35–67.22 |              0.777–1.010 |                             12 |                        0.029 |
| 500      |            5.45–36.83 |              0.758–0.827 |                             15 |                        0.015 |
| 510 快照 |            6.50–63.26 |              0.657–0.895 |                             16 |                        0.116 |

这些 batch EV 是不同题目混合的训练前预测统计，不是固定训练集上的学习曲线；有正值可以支持“存在一些可预测结构”，不能独立证明能处理 unseen task，也不能量化
Retail 单领域训练 signal。

主要代码定位：`areal/engine/fsdp_engine.py:905` 中 grad norm 检查和 optimizer step；成功 update 才在
926/931 行递增计数。`areal/trainer/ppo/critic.py:98` 以 `ppo_n_minibatches` 构造
minibatch，`critic.py:157` 的 loss 默认 token mean；无 value clipping 时在 172–174 行计算 masked
squared error。`areal/engine/fsdp_utils/critic_freeze.py:41` 只冻结指定 attention 子模块，未冻结整骨干。

510 运行日志 `logs/root/tau2-critic-head/critic-head-qwen35-mix-r2/critic.log:215`
实际读数：frozen parameters 1,304,961,536、trainable parameters 2,900,792,321。它仍有约 29
亿可训练参数。63–94 次全局 optimizer 更新、每次仅 16 个终局标签，相比这个参数规模是很短的监督暴露；“增加 lr/epoch
能改善”是待检验假设，当前数据不能把更新不足与 attention 限制分开。

Adam 的单参数移动不能严格写成 $N\cdot lr$ 上界：一阶/二阶矩、bias correction、epsilon、随时间变化的 gradient clipping
都会改变有效步长。现有数值能排除舍入与跳步的粗略机制，不能证明某个 lr 必然够或不够。

checkpoint metadata 核对：previous recover `global_step=23`；479/480/499 为 93；500 为 19；510 为
59。它们分别对应完成步 24/94/94/94/20/60（0-based global_step）。500 metrics 22 但最新 recover 为 20，因此
checkpoint 不是 metrics 的最后一次 update。510 的 critic 导出 config
路径：`checkpoints/root/tau2-critic-head/critic-head-qwen35-mix-r2/critic/epoch2epochstep19globalstep59/config.json`，读出
`areal_value_head_bias=true`；注意 `default/.../config.json` 属于 actor 导出，不能用它判定 critic 是否带
bias。

## 5. ceiling 的可靠口径与 CPU signal check

源文件（下面所有 JSONL 原始标签在相应行的 reward/domain/task_id）：

- 473：`user-sim-20261012/api-user/deepseek-base-u0p0/episodes.jsonl`，原始 164 tasks × 4。
- 474：`user-sim-20261012/api-user/deepseek-test-u1p0/episodes.jsonl`，60 × 4。
- 475：`user-sim-20261012/api-user/deepseek-test-u0p1/episodes.jsonl`，60 × 4。
- 476：`user-sim-20261012/api-user/doubao-base-u0p0/episodes.jsonl`，原始 164 × 4。

473/476 原始文件包含 train 与 test；直接对整个文件运行 ceiling 脚本会算错 README 的实验口径。必须只保留与 475 相同的 60 个
`(domain, task_id)`。所有保留题恰好有 4 个
reward。仓库分析脚本：`docs/research/2026-10-09-tau2-high-baseline-audit/scripts/critic-data/ev_ceiling.py:20`
的分解与 32 行 bootstrap；README 的定义与限制在 `README.md:3267` 后。

| run              | Airline ceiling \[task bootstrap 95%\] |  Retail ceiling \[95%\] | 总体 ceiling |
| ---------------- | -------------------------------------: | ----------------------: | -----------: |
| 473 T0/T0        |                 0.550 \[0.272, 0.792\] | 0.105 \[−0.046, 0.245\] |        0.386 |
| 474 T0/T1        |                 0.566 \[0.306, 0.798\] | 0.172 \[−0.032, 0.403\] |        0.413 |
| 475 T1/T0        |                 0.614 \[0.341, 0.845\] |  0.204 \[0.025, 0.352\] |        0.491 |
| 476 Doubao T0/T0 |                 0.475 \[0.167, 0.722\] |  0.171 \[0.004, 0.333\] |        0.293 |

475 Retail 40 题中，24 题四次全对，16 题有对有错，0 题四次全错；Airline 20 题中 7 题全错、6 题全对、7 题混合。这说明 Retail 的
task-level 值更集中，难度排序需要区分很接近的成功概率；Airline 有更容易学习的两端题群。

CPU 可学习信号探针：每个任务用另外三次 rollout 的成功率预测留出的第四次，不拟合模型，不依赖 hidden states。指标仍使用
$1-\operatorname{Var}(R-\hat p)/\operatorname{Var}(R)$：

| 475 task-rate 预测器                          | Airline EV | Retail EV |
| --------------------------------------------- | ---------: | --------: |
| 用其他 3 次预测留出的 1 次                    |      0.486 |    −0.061 |
| 用同 4 次估计 rate，又评在同 4 次（泄漏示范） |      0.711 |     0.403 |
| noise-corrected oracle ceiling estimate       |      0.614 |     0.204 |

这不是模型能力测试：预测器拥有 test-task 身份和该题其他重复，critic 并不拥有这些标签。它展示了两个有用事实。第一，Airline 的 task-level
signal 用很朴素的统计方法就能提取。第二，Retail 只有三次估计 rate 时，估计误差可以吃掉全部约 0.20 的潜在 oracle EV；直接拟合同批
reward 会严重抬高表现。因此低 ceiling 加有限标签可以解释“浅层、少样本 task-rate 拟合弱”，但不能推出 LM 无法从语义共享或后续 prefix
获益。

### ceiling 不能识别的东西

这个分解有三个实质限制。

1. **task identity oracle 与可观测 prefix 不相同。** $\operatorname{Var}(\mathbb E[R\mid task])$
   描述知道任务身份、尚无其他随机信息的理想值。若最初 agent token 的 prefix 含不同的初始 user utterance，这些 utterance
   可能已揭示任务内成败风险；task-only 比例未必是该 prefix 的严格上界。需要确认每题重复的 initial prefix 是否固定，才能把其解释为首
   token ceiling。即便该条件成立，它也不是 token-mean EV 上界，因为之后会出现可预测事件。
1. **within-task variance 不是 user-only variance。** 同一题重复改变 actor 采样、远端 user thinking、NL
   judge、工具运行和服务端数值/并发状态。475 与 473 的差异不是同一个随机变量的受控开关，confidence interval 也重叠。数据支持“greedy
   actor 时终局仍不确定”，不支持精确归因“绝大部分一定来自 user”。judge thinking 与执行非确定性也未被单独干预。
1. **4 次重复不足以精确估计每题成功率。** 当前 bootstrap 对题组重采样，适合从当前 n 个 sampled task clusters
   推测相似任务群，但未独立刻画同一固定 60 题未来重复的条件不确定性。Airline ceiling 95% 区间宽达约 0.50；Retail 的 0.20
   也不宜当成精确常数。总体 ceiling 混入 Airline/Retail 的均值差，与单领域学习目标不等价。

“温度 0”还应按解析后的 thinking 路由解释：README `16.14.5` 已注明 DeepSeek 默认 thinking 时写入的 temperature 0
不生效。因此不能以配置里的 0 证明 user 逐次输出确定。

## 6. 可支持的判断与仍须实验的假设

**已有证据支持：** Retail pack 的失败覆盖少；Airline pack 与官方任务难度差明显；训练更新真实执行；head 初始化修掉了大幅随机输出；mix
在当前固定 validation 上已学到更多 Airline 结构；Retail ceiling 低而非零，三次 task-repeat 统计也不够稳定；跨 run
验证不是严格配对对照。

**尚不能识别：** 冻结 attention 与数据/优化哪项主导；更高 lr 是否更好；当前后缀是否含有 critic 能利用的成败事件；语义 representation
的线性可分性；真正的 user 因果噪声占比；510 的 final 80-step 与未来独立评测结果。token-level MSE、参数多、标签少都给出机制假设，当前
artifact 没有对应的干预实验。

廉价下一步应先固定已有训练轨迹，重复拟合其标签并同时评估固定训练 subset 与 unseen-task subset；如果训练 residual 明显下降而
held-out task 不改善，优先处理 generalization / 数据差异；如果训练也拟合不了，检查 optimizer、head
representation、attention 冻结或 objective。这个分支没有运行 hidden-state probe，也没有运行网络前向，因为 CPU
存量数据只有 token IDs 与文本，没有 frozen hidden states。实际 GPU/CPU 大模型 probe 的命令与预算应由总报告根据仓库入口列出。

## 7. 训练窗口与离线成本估计

`data_timing_loss_windows.json` 从原始 train metrics 分别取首 20 与读取时最后 20
步。窗口不是同一批任务重放，479/499 尤其受到 pack→官方固定顺序影响，不应把 MSE 跨窗口下降解释为固定数据上的拟合能力。

| run      | 首20 → 末20 batch MSE 平均 | 首20 → 末20 grad norm 平均 |
| -------- | -------------------------: | -------------------------: |
| previous |              2.766 → 1.202 |            240.05 → 161.39 |
| 479      |              2.919 → 0.233 |             262.09 → 81.43 |
| 480      |              3.658 → 0.329 |             211.88 → 99.60 |
| 499      |              0.087 → 0.120 |               9.24 → 16.87 |
| 500      |              0.182 → 0.169 |              15.23 → 14.52 |
| 510 快照 |              0.189 → 0.177 |              17.72 → 23.82 |

510 每批 `timeperf/critic_train_step` 约 10.5 秒（末 20 均值 10.528），`critic_values` 前向约 3.84
秒，rollout 约 120
秒。可直接核对的单步示例：`tau2-training-20261013-critic-head/critic-head-qwen35-mix-r2/tau2-critic-fit-evidence/train-metrics/step-000060.json`，`critic_train_step=10.379s`，`critic_values=3.811s`，`rollout=133.677s`。`critic_train_step`
聚合了 forward/backward/optimizer；不能进一步声称分别量到了 backward 的单独时延。

**估计，未执行：** 重放已有 1280 条 mix rollout、batch 16，一轮 80 update，沿用 4 张训练 GPU 和相近长度分布，纯 train
step 约 $80\times10.5=840$ 秒，即 14 分钟；若仍做每批 values 前向，再增加约 5
分钟。加初始化、onload/offload、定点验证与少量存档，保守预算约 20–40 分钟 wall time、1.3–2.7 GPU-hours，0 API
token。10 轮约 3–6 小时 wall / 12–24 GPU-hours；这个估计假设吞吐与已有 run 相近，未计大量导出 checkpoint
或数据读取瓶颈。冻结 hidden states 的线性 probe 会更便宜，但必须先付一次 representation 提取成本，不能把尚未存在的
hidden-state cache 当成已准备好。

### 追加核查：初始 user prefix 的变化是实测事实

`data_initial_user_check.py` 检查了 475 的 240 个 completed trajectory JSON。Airline
20/20、Retail 40/40 的任务在四次重复中都出现不同的第一条 user content；19/20 Airline 与 39/40 Retail
题四条内容全不相同。`data_initial_user_variation.json` 保存每个样本的 trajectory 路径、trial、SHA-256 与
reward，避免复制完整对话。

因此第 5 节“若 initial user utterance 变化”的条件已经成立，不能直接将 task-only ceiling 称为实际首个模型响应 token
的严格上界。真实 critic 输入可以同时包括 task 类型与这次具体 user 开场，而 $\mathbb E[R\mid task]$
忽略了后者。变化本身不证明这些内容能预测成败，但它推翻了“每题重复的第一 token 可观测 prefix 相同”这个前提。trajectory 的第一条 assistant
是固定 greeting，需要区分它与首次 model-generated response；对真正响应 user 的 prefix，task-only oracle
比例应称为任务层可解释方差估计。
