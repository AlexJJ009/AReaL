# SAO：mix pack + critic 510 第 55 步（2026-10-14，534 已完成）

> 来源：原 README 第 16.14.6 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

**设置（2026-10-13 用户确认）**。

| 项              | 设置                                                                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------------------------------- |
| actor           | Qwen3.5-4B（non-thinking），lr 1e-6                                                                                    |
| critic          | 510（critic-head-qwen35-mix-r2）第 55 步的 checkpoint（`epoch2epochstep14globalstep54`）；lr 5e-6，head 不单独设学习率 |
| 训练题          | mix pack 221 题 + 官方 fixed train 99 题 = 320 题（Airline 180、Retail 140）                                           |
| 采样            | 每题每个 epoch 4 条，2 个 epoch；每 epoch 1280 条，batch 64，共 40 步、2560 条 episode                                 |
| 更新顺序        | 每步先对 critic 做 2 次完整更新，再更新 actor 1 次（`critic_updates_before_actor: 2`）                                 |
| 评测            | fixed-v3 test 60 题 × 4 次，policy temperature 0；训练前和每 10 步一次，共 5 次                                        |
| user / NL judge | DeepSeek-Flash 默认 thinking                                                                                           |

- 选 mix 而不是 v2：v2 里约七成题对 Qwen3.5 几乎全对，SAO 中 advantage 接近 0，rollout 预算大多浪费；critic 510
  也是在这份分布上训练的。不另跑 v2 的 SAO，也不跑 GRPO 对照。
- **critic 用 510 第 55 步，是看 test 挑的（用户决定，2026-10-14）**。510 最后一步的验证 EV 回到约 0（Airline
  −0.005、Retail 0.001），第 55 步为 Airline 0.179、Retail 0.016。critic 的验证集就是 SAO 报分的 fixed-v3
  test 60 题，所以这是用 test 选模型：SAO 的 test 分数会偏乐观，报告时要注明；第 55 步的 0.179 本身误差约 ±0.1（critic EV
  case study）。
- 评测频率：原配置每 5 步一次会评 9 次、约 2160 条 episode，和训练本身一样多，所以改为每 10 步一次。

**代码**（`fccceaa1`，Codex 实现、Claude review）：新入口 `scripts/tau2/train_sao_pack.py` 和
`run.sh policy-pack mixed sao`。训练行是 pack 的全部题加官方 fixed
train，再走原来的按领域调度和打乱；要求每道题被调度的次数相同，否则拒绝启动。`examples/tau2/train.py` 只加了可选的
`train_rows_hook`，原 `policy` 入口设了 `train_task_pack` 会报错，其余行为不变。`_task_pack_rows` 移到
`examples/tau2/task_pack.py`，critic 训练与 SAO 共用。CPU 测试：新增 20 个，与 critic、task pack、SAO
合约、head 相关测试共 125 个通过；Codex 把 9 个新检查逐个改坏，对应测试都会失败。review 时改了一处：`_task_pack_rows` 的参数
`domains` 与模块函数重名，改名为 `selected_domains`。

**启动**：worktree `tau2-training-20261014-sao-mix/src-fccceaa1`，脚本
`launch_sao_pack.sh NAME {smoke|formal}`（副本在 `scripts/critic-data/`）。脚本检查 worktree 的
HEAD、设 `PYTHONPATH`、确认第 55 步的 checkpoint 和验证报告存在，并先跑 `check_critic_head.py`。smoke 跑 2
步，不在训练前评测，在第 2 步评测并存档。check-config（critic 为 510 第 55 步）：解析出的 critic 路径正确，每 epoch 1280
条，smoke 2 步，formal 40 步、2560 条 episode，验证集 60 题，head 预检通过。

**队列**：第一次 smoke（525）用的是未修 xccl 的代码，在加载阶段就停掉，没有调用 API。xccl
修复（[infra-issues.md §3](infra-issues.md)）合并后，worktree 改为 `src-6bf1d96e`、check-config
通过，smoke（526）重新排队，北京时间 17:00–18:08 跑完，exit 0：2
步都正常推送了权重（[infra-issues.md §3](infra-issues.md)），第 2 步评测 pass^1 总体 0.738、Airline
0.550、Retail 0.831。用户确认评测频率保持每 10 步一次。正式任务 527 于 21:27 启动后 1.5 分钟报错退出，没有训练也没有调用
API：异步评测要求存 checkpoint 和评测的频率一致（`areal/trainer/async_eval.py` 的
`_validate_save_eval_sync`），而 formal 模式只把评测改成了每 10 步，存档仍是每 5 步。smoke 模式两者都是
2，check-config 也不构建 trainer，所以之前没查出来。formal 模式改为同时设
`saver.freq_steps=10`，读回解析后的配置确认两者一致后重新排队（534）。

**结果（534，北京时间 10-14 21:31 → 10-15 05:37，40 步，exit 0）**。每 10 步在 fixed-v3 test 60
题上评测一次，每题 4 次。下表把评测时因对话超出 32K 上下文而被排除的 episode 记为失败；评测代码原本的 pass^k 直接不计这些 episode，不计时第
30、40 步的总体 pass^1 为 0.785、0.775。

| 第几步 | Airline pass^1 | Retail pass^1 | 总体 pass^1 | 总体 pass^4 | 超出上下文被排除的 episode | 轨迹长度中位数（token） |
| ------ | -------------- | ------------- | ----------- | ----------- | -------------------------- | ----------------------- |
| 0      | 0.487          | 0.819         | 0.708       | 0.500       | 0                          | 10832                   |
| 10     | 0.637          | 0.844         | 0.775       | 0.500       | 0                          | 10948                   |
| 20     | 0.575          | 0.844         | 0.754       | 0.500       | 1                          | 11435                   |
| 30     | 0.512          | 0.850         | 0.738       | 0.567       | 17                         | 11958                   |
| 40     | 0.500          | 0.806         | 0.704       | 0.467       | 25                         | 11656                   |

- **40 步之后没有可见的提升**：第 40 步总体 pass^1 为 0.704，与第 0 步的 0.708 相同。Airline 在第 10
  步最高（0.637），之后回落到 0.500。60 题的采样噪声 95% 半宽总体约 ±4.1pp、Airline 约
  ±6.4pp（[initial-audit.md §2.1](initial-audit.md)），第 10 步的升高还不能和噪声分开。
- **对话变长，超出上下文的 episode 增多**：第 0–20 步几乎没有，第 30、40 步分别有 17、25 条（共 240 条）。这些 episode 是
  policy 自己把对话拖过了 32K，评测代码把它们记为基础设施错误并从 pass^k 中去掉，会抬高分数。见
  [issues/2026-10-15-eval-context-overflow-excluded.md](issues/2026-10-15-eval-context-overflow-excluded.md)。
- **critic 在训练 batch 上的 EV**：前 10 步平均 0.34，后 10 步平均 0.55。这是训练 batch 上、包含题目之间差异的
  EV，不扣整体偏移，和 critic-only 验证集上的 EV（扣偏移、只看 test 题）不是同一个口径，不能直接比较。
- 训练 batch 的成功率在 0.72–0.94 之间，没有趋势。

数据：`tau2-training-20261014-sao-mix/sao-mix-formal/evidence/async-eval/{0,10,20,30,40}.json`
和 `logs/root/tau2-sao/sao-mix-formal-dedicated-eval/eval-rollout/`；重算脚本思路：已完成的 episode 按
`official_score` 计成败，`infra_excluded` 中的每条按失败补入同一道题。
