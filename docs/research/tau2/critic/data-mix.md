# critic 训练数据配比（2026-10-13）

> 来源：原 README 第 16.14.5 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

**依据**。把 479 的训练 rollout（temperature 1，每题 2 次）和
[step0-evaluations.md §5](../step0-evaluations.md) 的 pack 评测（temperature 0，每题 4
次）按题合并，每题 6 个样本。下表按 479 的 2 次统计：

| Qwen3.5      | 题数 | 成功率 | 2 次全对 | 2 次全错 | 有对有错 | 标签方差 |
| ------------ | ---: | -----: | -------: | -------: | -------: | -------: |
| pack Airline |  380 |  0.904 |      322 |       15 |       43 |    0.087 |
| pack Retail  |  268 |  0.979 |      259 |        2 |        7 |    0.020 |
| 官方 Airline |   30 |  0.383 |        8 |       15 |        7 |    0.236 |
| 官方 Retail  |   69 |  0.870 |       54 |        3 |       12 |    0.113 |

- 6 次全对的简单题有 526 道（Airline 284、Retail 242），其余 122 道有信号（Airline 96、Retail 26），这 122 道在
  temperature 1 下的成功率是 0.66。
- 简单题所有 token 的目标都是 1，只能教 critic 整体水平。整体水平已经交给 head 的
  bias（[value-head.md](value-head.md)），几道题就够；479 的 1494 条 episode 里约七成花在这类题上，每条约 11K
  DeepSeek prompt token。
- **降采样，不加权**。加权仍要把 rollout 跑完，费用不变，信息量也不变。
- **不整题删**。SAO 训练时 actor 仍会遇到简单题，critic 要对它们预测接近 1，所以保留一小部分。

**配比**。有信号的 122 道全部保留；简单题按（领域，keep/fix）分层随机抽 99 道，配额与各层简单题数成比例（Airline fix 22 / 117、keep
32 / 167，Retail fix 38 / 203、keep 7 / 39，种子 20261013）；再加官方 fixed train 99 道，共 320
行，batch 16 下每个 epoch 正好 20 步。按 479 的 temperature 1 成功率估算，标签均值 0.78（全量 pack 时 0.91），简单题占
31% 的 episode。

**顺序**。pack 题和官方题混在一起，每个 epoch 重新打乱（`train_dataset.shuffle=true`，sampler 按 seed 和 epoch
打乱）。479 是 pack 在前、官方在后、每个 epoch 同一顺序，每个 epoch 最后 7 个 batch 全是官方题。用同一个
`create_dataloader` 在 CPU 上核对：不打乱时官方题集中在最后 7 个 batch；打乱后每个 batch 含 2–9 道官方题，两个 epoch
的顺序不同。run 摘要里的 `source` 仍写 `task_pack_then_official_tau2_train_tasks`，指的是行的拼接顺序，采样顺序由
shuffle 决定。

**Pack**。`data-1/dataset/tau2/tau2-areal-pack-20261013-critic-mix-v1`（只读），由
`../scripts/critic-data/select_critic_mix.py`（`a88d480e`）从 pack v2 生成：两个领域的 `tasks.json`
和 `task_db.json` 只保留选中的 221 题，数据库文件原样复制，manifest 重算 sha256，`selection.json`
记录规则、输入路径、分层配额和题目清单。用 `examples/tau2/task_pack.py` 加载通过，221 题的任务定义和数据库路径与 v2
逐题一致。`tau2-areal-pack-latest` 仍指向 v2。

**Instruct 不调**。它在 pack 上成功率 0.656，2 次全对 333 道、全错 131 道、有对有错 184 道，标签本身是平衡的；每题只有 2
个样本，用来筛题噪声太大。它的问题是 pack（0.6）与 test（0.37）难度不同，但 SAO 训练时 critic
面对的是训练题，这个差异对实际使用的影响可能小于验证数字显示的。

**SAO 是否共用这份配比（未定）**。critic 在 SAO 里面对的是 SAO 的训练题，两边的分布应当一致。若 SAO 仍用全部 747 题，critic
应对简单题加权（约 526 / 99 ≈ 5 倍）来补偿；若 SAO 也用这 320 题，不需要加权。简单题上 $R = 1$、$V \approx 1$，advantage
接近 0，对 actor 也几乎没有信号。

**实验**。worktree `tau2-training-20261013-critic-head/src-a88d480e`，启动脚本
`launch_critic_head.sh NAME {v2|mix}`（副本在 `../scripts/critic-data`），入口
`data critic-head`，head 与骨干的 lr 都是 5e-6（[value-head.md](value-head.md)），其余与 479 相同：

| 模式 | 训练数据                     | 顺序                | epoch / 步数 | bias 初值 | 对比对象                |
| ---- | ---------------------------- | ------------------- | ------------ | --------- | ----------------------- |
| v2   | pack v2 648 + 官方 train 99  | pack 在前，固定顺序 | 2 / 94       | 0.91      | 479：只差 head          |
| mix  | mix pack 221 + 官方 train 99 | 混合，每 epoch 重排 | 4 / 80       | 0.78      | v2 模式：只差数据和顺序 |

v2 模式看 Airline 能否逼近 0.6 的上限（[ev-ceiling.md](ev-ceiling.md)）；Retail 的上限低，不期望大的正 EV。两个模式的
check-config 都通过：v2 为 747 行、94 步，mix 为 320 行、80 步，解析出的配置里 bias、shuffle 和 epoch 数与上表一致，不设
`critic_head_lr`。

**队列**（pueue `gpu8` 组）：

- 498 smoke（mix 数据、tune 模式、2 步、验证 dev 20 题，head lr 1e-4）2026-10-13 02:13–02:38 PDT
  跑完，退出码 0。第 1 步 所有 token 的预测都等于 bias 初值 0.78，起点 MSE 0.26（479 第 0 步约 10）；checkpoint 带
  `score.bias`，SAO 的 `check_critic_head.py` 通过。它也暴露了 head lr 1e-4
  的问题（[value-head.md](value-head.md)），所以正式实验去掉了这个设置。
- 499（v2）和 500（mix）依赖 498 成功，延迟到北京时间 10 月 14 日 00:35 才入队，赶上 DeepSeek
  的夜间低价时段。启动脚本在入队前改成不设 head lr、读 `dataset/tau2/` 下的 pack，任务命令不变。
- user 和 NL judge 与 [user-simulator.md §2](../user-simulator.md) 评测相同，都是 DeepSeek-Flash
  默认 thinking（配置里写的 temperature 0 不生效）。policy 保持 temperature 1，与 SAO 训练时的采样一致，也与
  [ev-ceiling.md](ev-ceiling.md) 中 475 的上限口径一致；它和评测规范的 policy temperature 0 不同，这是有意为之。

**499 结果（v2 模式，2026-10-13 19:26–00:27 北京时间，94 步，exit 0）**。 验证集是 fixed-v3 test 60 题；EV
用去掉整体偏移后的残差方差计算，偏移单独列出：

| 验证步 | 总体 EV | 总体偏移 | Airline EV | Airline 偏移 | Retail EV | Retail 偏移 |
| -----: | ------: | -------: | ---------: | -----------: | --------: | ----------: |
|      0 |   0.000 |    +0.16 |      0.000 |        +0.30 |     0.000 |       +0.07 |
|     25 |   0.021 |    +0.22 |      0.015 |        +0.35 |     0.007 |       +0.14 |
|     50 |   0.058 |    +0.09 |      0.004 |        +0.14 |    −0.002 |       +0.06 |
|     55 |   0.074 |    +0.11 |      0.015 |        +0.10 |     0.003 |       +0.11 |
|     70 |   0.000 |    +0.24 |      0.002 |        +0.38 |    −0.001 |       +0.15 |
|     94 |   0.033 |    +0.12 |      0.025 |        +0.24 |    −0.001 |       +0.05 |

- 零初始化 $w$ 去掉了 479 那种随机 head 带来的噪声，EV 从 0 起步、全程不再是大负数；训练时预测均值在 0.89–0.98，没有 479 的
  1.2–1.9。
- 区分成败的能力没有改善：Airline EV 最高 0.025，离 0.61 的上限很远；Retail 在 0 附近。总体 EV 的正值主要来自领域之间的差异。
- 偏移随数据顺序起伏：每个 epoch 第 41–47 步训官方题，偏移在第 50 步降到 +0.09；回到 pack 题后升回 +0.24。训练集（pack 0.93）比
  test（Airline 0.60）容易，Airline 整体偏高 0.24–0.38。

**500（mix 模式）失败，重跑为 510**。500 于北京时间 10 月 14 日 00:27 启动，01:55 跑到第 22 步（第 2 个 epoch）时，GPU
4–7 上的一台 SGLang 服务器在动态申请显存时 OOM 退出（`torch_memory_saver` 报 `CUresult error: 2`，配置为
`mem_fraction_static=0.65`、上下文 32768、每台并发 4）。训练端没有发现服务器已退出，之后约四分之一的 episode 仍发往它，连接失败
464 次；02:13 第 2 个 epoch 内有 3 道题两次被拒，超过 `max_skipped_tasks_per_epoch=2`，任务退出（exit 1）。499
用同一配置跑了 5 小时没有出现。报错来自 `torch_memory_saver`，而这台 SGLang 配置里没开 memory
saver：[infra-issues.md §3](../infra-issues.md) 查明 SGLang 一直继承着训练端的 TMS 钩子，OOM
很可能与此有关，但没有复现，未证实。失败前第 20 步：Airline EV 0.016、偏移 +0.33，Retail EV 0.019、偏移 +0.02，与 v2
模式同期接近（Airline 0.006、Retail 0.004）。

- **从第 20 步 resume 失败（506）**。critic 训练脚本支持 recover，第 20 步恰好是第 1 个 epoch 结束，验证 rollout
  也有带摘要校验的缓存。但 AReaL 的 recover 流程总会把 actor 权重经 NCCL（`weight_update_mode: xccl`）推给
  SGLang，这一步在 Qwen3.5-4B 上触发 `illegal memory access`，GPU 4、5 的服务器退出，任务随即手动终止。critic-only
  训练的正常步不推权重，所以 498–500 没有遇到。2026-09-27 的 SAO 运行 `sao-mixed-v1` 也在这一步出现同样的错误。另外，resume 后
  epoch 打乱用的是采样器默认种子 0 而不是 `seed=42`（`Tau2CriticOnlineTrainer._init_impl` 只在不 recover
  时设种子）。
- **重跑**：510（`critic-head-qwen35-mix-r2`）从头训练，设置与 500 相同，只把 `mem_fraction_static` 降到
  0.5（用户确认），每张推理卡多留约 12 GB 给动态显存。500 目录里两次尝试的日志改名为
  `run.attempt1.log`、`run.attempt2-resume.log` 保留。
- **待修**：SGLang 服务器退出后训练端不会察觉，继续向它分发请求，直到跳过上限才报错；critic-only 的 recover 不必推送未变的 actor 权重。

**510 结果（mix 模式 r2，2026-10-14 11:13–15:59 北京时间，80 步，exit 0）**。验证集是这次重新采的 fixed-v3 test
60 题，与 499、500 的不是同一批轨迹，跨 run 只能粗比：

| 验证步 | Airline EV | Airline 偏移 | Retail EV | Retail 偏移 |
| -----: | ---------: | -----------: | --------: | ----------: |
|      0 |      0.000 |        +0.07 |     0.000 |        0.00 |
|     20 |      0.069 |        +0.02 |     0.008 |       +0.08 |
|     40 |      0.020 |        −0.02 |     0.025 |       +0.11 |
|     55 |      0.179 |        +0.01 |     0.016 |       +0.08 |
|     70 |      0.150 |        −0.16 |     0.004 |       +0.07 |
|     80 |     −0.005 |        +0.04 |     0.001 |       +0.13 |

- 运行健康：OOM 没有复现，没有题被跳过；2 个 episode 因 DeepSeek 返回空 user 回复被拒，重放后成功。
- Airline 在第 55–70 步到 0.15–0.18，高于 v2 全程最高的 0.025，但最后两个验证点掉回 0；Retail 全程在 0 附近。
- **SAO 约定用的最后一步 checkpoint，在验证集上的 EV 约为 0**。中间步更好，但按 test 挑步等于用 test
  选模型（[sao-runs.md](../sao-runs.md)）。每个领域只有 20/40 条轨迹：第 60 步 Airline EV 0.168 的 95% 区间是
  \[−0.04, 0.35\]，第 20 → 60 步的配对变化 +0.10 的区间是 \[−0.05, 0.23\]，与 0
  分不开（[ev-case-study.md](ev-case-study.md)）。

启动前发现并处理了两处：

- **数据目录换成 fixed-v3**。当前代码的 `fixed` 变体校验 v3 的任务哈希，v2 目录启动即报错。v3 比 v2 多修了 7
  道题（[benchmark-variants.md §1.2.1](../benchmark-variants.md)），所以 v2 模式与 479 的对比不只差
  head；[user-simulator.md §2](../user-simulator.md) 里同一用户模拟器 v2 → v3 的合计变化在运行间噪声内。
- **固定的 worktree 原本没有真正隔离**。worktree 里的 `.venv` 是指向主 worktree 的软链接，其中的 `areal.pth` 把主
  worktree 加进 `sys.path`，第一次 check-config 的 traceback 显示 `train_critic.py` 是从主 worktree
  导入的。启动脚本现在先设 `PYTHONPATH=$SRC`，之后 check-config 里的模块路径都在 `src-a88d480e` 下。479/480 在
  20:04 前跑完，`04be68ab` 之后第一处代码提交是 20:48 的 `7da3a4f3`，已提交的代码一致；当时主 worktree
  有没有未提交的代码改动，已无法追溯。`launch_critic.sh` 和 `launch_ceiling.sh` 没有这一行，以后复用要补上。
