# 训练设置：题目划分、运行约定与训练侧处理

> 来源：原 README 第 12.1–12.6 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

本文 §1–2 记录 2026-10-10 起的早期 fixed-v3、Airline/Retail 训练设置；§3 的失败处理另有算法适用范围。 后续 pack、Qwen
用户和 Telecom 实验见[训练数据](training-data.md)及
[10-24 实验记录](experiments/2026-10-24-qwenuser-no-telecom.md)，不能套用这里的题数或用户设置。

## 1 早期题目划分

不用 Telecom，用 fixed 变体去重后的训练集（[benchmark-variants.md §1.4](benchmark-variants.md)）。

| 模式           | 训练集                                      | 训练中评测                                      |
| -------------- | ------------------------------------------- | ----------------------------------------------- |
| formal（早期） | 99 题：Airline 30、Retail 69                | 官方 test 60 题：Airline 20、Retail 40          |
| tune           | 从这 99 题切出 79 题：Airline 24、Retail 55 | dev 20 题：Airline 6、Retail 14，同样来自训练集 |

- **tune 的 dev 切法**：每个领域先排序，再用 `Random("42:<domain>")` 打乱，取 `round(n × 0.2)` 题。代码会检查 dev
  与训练集不重叠。
- **划分校验**：官方划分的题数和 sha256 必须与 `examples/tau2/contracts.py` 的 `OFFICIAL_SPLIT_SNAPSHOTS`
  一致，否则拒绝启动。去重在这一步之后执行。
- **为什么不用 Telecom**：训练前 Telecom 就是 320/320 全对，166 个 GRPO 组里 160 个全对，没有组内对比，却占了约 42% 的
  rollout（[ADR 0003](../../decisions/0003-tau2-data-and-evaluation.md)）。
- **哪些结果是 held-out**：训练只在 train 上做，test 只记录，不用来选 checkpoint。base 集包含 train，所以训练后的 base
  分数不是 held-out 结果，test 60 题才是。

## 2 运行与配置约定

早期运行用 DeepSeek-Flash 做用户模拟器和 NL judge，temperature 0，用户模拟器不设回复上限，key 从
`TAU2_DEEPSEEK_ENV_FILE` 读取。运行显式传 `econfig.benchmark_variant=fixed`，并把 `TAU2_DATA_DIR`
指向对应 fixed 数据目录（当时为 fixed-v3，[benchmark-variants.md §1.5](benchmark-variants.md)）。

本地代码中，`examples/tau2/config_grpo.yaml` 和 `config_critic_production.yaml` 默认 `official`，
`config_ppo.yaml` 覆盖为 `fixed`；目录与变体不符时拒绝启动。PPO 由 `examples/tau2/ppo_data.py` 合并 fixed
Airline/Retail 官方训练题与配置的 packs，评测固定为 Airline 20、Retail 40、Telecom 40，共 100 题； 不是上述 99/60
题设置。历史 Qwen 用户训练见实验记录；当前用户与 judge 的批准模型范围仍以仓库 `AGENTS.md` 的 DeepSeek-only 约定为准，换模型或
endpoint 需用户批准。

- **critic 的 epoch**（ADR 0005）：actor 固定（lr 0）。每个 epoch 给每道训练题现采一条 temperature 1 的
  episode，只更新 critic。每个 epoch 的最后一个 batch 只有 3 题（`drop_last: false`）。

- **critic 的验证**：开头用训练的 `gconfig`（temperature 1，每题 1 次）采一次 60 条固定轨迹，之后每次验证都复用这批轨迹。验证测的是
  critic，不是 policy，所以没有 mean@4 / pass^k。

- **当时的资源分配**：8 张 A100-80GB。

  - critic：rollout 用 4 卡，actor 和 critic 共用另外 4 卡，阶段之间 offload。
  - GRPO / SAO：rollout 3 卡，独立评测 1 卡，actor 4 卡。

- **运行目录**：每次运行都要用新的 `TAU2_RUN_ROOT`，因为 `recover.mode=auto` 会续跑同一目录。

## 3 训练侧的处理（已实现）

以下失败分类和重试预算沿用官方 runner 语义；存活成员阈值按组大小计算。GRPO 的 n=8 时阈值为 4， PPO / critic 每题一条时为 1，不应把“至少 4
条”套到单 rollout 训练。具体实现见 `examples/tau2/agent.py`、`utils.py` 和 `evaluation.py`。

- **单次回复上限**：保留 4096，以节省资源（用户决定）。已知代价：上一轮 12% 到 17% 的 Airline 训练 episode
  至少有一次回复被截断（[recuris-and-leaderboard.md §4](recuris-and-leaderboard.md)）。
- **训练墙钟超时**：保留这条 episode，标记为 truncated，reward 0，不重跑（ADR 0009 修订）。如果重跑，只会保留按时跑完的
  episode，会系统性地去掉慢的长轨迹。
- **组内成员丢失**（[ADR 0011](../../decisions/0011-tau2-lost-episodes-in-training.md)）：
  - 上下文超限和墙钟超时记 0 分，照常训练。
  - 其他异常对单个成员从头重跑，最多 3 次。仍然失败的成员从组里去掉，不计入 baseline，也不训练。
  - 存活成员 ≥ $\lceil n/2 \rceil = 4$ 时，在存活成员上归一化。不足 4 个时，这道题放到本 epoch 末尾重跑一次；再被拒就记为
    skipped。每个 epoch 超过 2 题 skipped，训练中止。
  - 用户第一句就结束、policy 一次都没被调用的 episode，按同样的方式丢弃。
  - critic 每题只有一条 episode，所以在同一步内重跑一次，预算相同。
  - 新增训练指标：`tau2_batch/epoch_deferred_tasks`、`tau2_batch/epoch_skipped_tasks`。
- **还没实现**：按题统计 infra 失败率，以及失败与轮数的相关性。这是判断上述处理是否偏向短对话的依据。

同类框架的做法（定阈值 4、预算 2 的依据）：

调研了 AReaL upstream、verl、OpenRLHF、slime、ROLL、SkyRL、rLLM、Agent Lightning、TRL、NeMo-RL、ART 和
DAPO。下表的结论来自代码，不是文档：

| 做法                                              | 谁在用                                                                              | 偏差                                                            |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 先分类：模型造成的失败记 0 分，infra 失败另行处理 | AReaL upstream（`_classify_agent_failure`）、NeMo-RL、SkyRL harbor                  | 依赖分类是否准确                                                |
| infra 失败对单个成员做有上限的重跑                | NeMo-RL（行级 3 次，再整组）、SkyRL（每条 2 次）、rLLM（3 次）                      | 对与轨迹无关的瞬时故障无偏；对随轮数增长的故障，仍偏向短对话    |
| 在剩余成员上算 baseline                           | AReaL upstream（`min_usable_group_size` 默认 1）、verl v1、ART、TRL（NaN 不计入）   | 失败与轨迹有关时会有幸存者偏差                                  |
| 丢掉整组，用别的 prompt 补                        | AReaL upstream 系统错误、OpenRLHF、ROLL、DAPO 动态采样                              | 难任务被少采；破坏有限 epoch 的"每题一次"                       |
| infra 失败记 0 分照常训练                         | AReaL upstream tau2 示例、Agent Lightning、slime 示例                               | 把 infra 失败当成策略失败，惩罚失败前的动作，并推动策略缩短对话 |
| 预算用尽就中止训练                                | NeMo-RL 默认 `max_skipped_prompts=0`、ART `max_exceptions=0`、rLLM `raise_on_error` | 不静默吞掉问题，代价是训练会停                                  |

**"至少剩 2 个就保留"的问题**：

- 没有一个框架用这个阈值作为默认值，最接近的是 rLLM 的 `min_trajs_per_group=2`，默认关闭。
- 2 个成员、二值 reward 时，advantage 只有两种情况：全 0，或者 ±1，噪声很大。
- 一个组丢了 6 个成员，本身就说明任务或环境有系统性问题；这个阈值会把这个信号掩盖掉。

## 4 训练结果验证不了什么

- **与 Recuris 或榜单比绝对分数**：用户模拟器不同（[benchmark-variants.md §3.2](benchmark-variants.md) 第 1
  项），也没有 Telecom。fixed 的分数本来就不能与榜单比。
- **小于噪声的提升**：test 只有 60 题。同一配置两次重测相差约
  3pp（[step0-evaluations.md §3](step0-evaluations.md)）；采样噪声的 95% 半宽，合计约 ±4.1pp，Airline 约
  ±6.4pp（[initial-audit.md §2.1](initial-audit.md)）。 几个点的变化不能当作训练有效。
- **对 Doubao 或 gpt-5.2 用户的泛化**：训练和评测都用 DeepSeek 用户，换用户后的效果需要另跑评测。
- **早期 critic 对 SAO 的具体贡献**：MSE / EV 只说明 critic 拟合了 temperature 1 下的回报。它对 SAO 有没有帮助，只能从
  SAO 与 GRPO 的对比里间接看出，而这个对比还混着两种算法的其他差异。
- **NL judge 带来的 reward 噪声有多大**：Airline 有 26 道题按 NL 断言计分，judge 就是用户模拟器所用的模型。
