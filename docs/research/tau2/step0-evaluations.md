# 训练前（step 0）评测

> 来源：原 README 第 8、13、15.2–15.4 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 训练前评测总表

所有未训练模型的 step 0 评测汇总在这两张表里，新模型按同一口径评测后加一行。两张表分开放：表 A 是官方 τ² 题，用来和官方 / 论文比；表 B 是筛选后的
AReaL 训练数据，用来判断这份数据对某个模型有没有训练信号。

**共同口径**：policy temperature 0；DeepSeek-Flash 做用户模拟器和 NL judge（temperature 0）；每题 4
次；Airline / Retail 用 `benchmark_variant=fixed`（fixed v2 数据，§4）。"有对有错"的题，GRPO
的组内才有分数差异；训练时按 temperature 1 采样，这类题会更多，但还没测过。

**评测入口**：

- 表
  A：`scripts/qwen3-2507/launch_eval.sh NAME MODEL GPUS VARIANT DOMAINS THINKING PARSER`（Qwen3.5
  的配置见 §4，Qwen3-2507 的配置见 §6.1）。
- 表 B：同一个脚本，加环境变量 `TASK_PACK=_cache/tau2-areal-pack-latest`（或
  `tau2-areal-telecom-pack-latest`）和 `TAU2_SRC=<不早于 1b17bc3f 的固定 worktree>`。产物在
  `tau2-eval-20261012-qwen3-2507-packs/`。

**表 A：官方 base 集**（Airline 50、Retail 114、Telecom 114 题，train + test）

| 模型                   | 领域                       | 变体                | mean@4 | pass^4 | 4 次全对 | 有对有错 | 4 次全错 | 来源 |
| ---------------------- | -------------------------- | ------------------- | -----: | -----: | -------: | -------: | -------: | ---- |
| Qwen3.5-4B             | Airline                    | fixed v2            |  0.388 |  0.229 |       11 |       19 |       20 | §4   |
| Qwen3.5-4B             | Retail                     | fixed v2            |  0.849 |  0.634 |       73 |       39 |        2 | §4   |
| Qwen3.5-4B             | Telecom                    | official            |  1.000 |  1.000 |      114 |        0 |        0 | §6.2 |
| Qwen3-4B-Instruct-2507 | Airline                    | fixed v2            |  0.225 |  0.060 |        3 |       21 |       26 | §6.2 |
| Qwen3-4B-Instruct-2507 | Retail                     | fixed v2            |  0.410 |  0.096 |       11 |       72 |       31 | §6.2 |
| Qwen3-4B-Instruct-2507 | Telecom                    | official            |  0.377 |  0.167 |       19 |       60 |       35 | §6.2 |
| Qwen3-4B-Thinking-2507 | Airline                    | fixed v2            |  0.245 |  0.060 |        3 |       20 |       27 | §6.2 |
| Qwen3-4B-Thinking-2507 | Retail                     | fixed v2            |  0.654 |  0.351 |       40 |       65 |        9 | §6.2 |
| Qwen3-4B-Thinking-2507 | Telecom                    | official            |  0.373 |  0.246 |       28 |       25 |       61 | §6.2 |
| （下一个模型）         | Airline / Retail / Telecom | fixed v2 / official |      — |      — |        — |        — |        — |      |

Airline / Retail 在 official 变体下的分数见 §6.2；official 与 fixed 的差别见
[failure-taxonomy.md §3](failure-taxonomy.md)、§4。

**表 B：筛选后的 AReaL-tau2-data**（Airline / Retail 用 pack v2，648
题，[training-data.md §4](training-data.md)；Telecom 用 pack v0 / v1，261
题，[training-data.md §6](training-data.md)）

| 模型                   | 领域                       | pack        | 题数 | mean@4 | pass^4 | 4 次全对 | 有对有错 | 4 次全错 | 来源                                    |
| ---------------------- | -------------------------- | ----------- | ---: | -----: | -----: | -------: | -------: | -------: | --------------------------------------- |
| Qwen3.5-4B             | Airline                    | v2          |  380 |  0.926 |  0.818 |      311 |       63 |        6 | §5                                      |
| Qwen3.5-4B             | Retail                     | v2          |  268 |  0.976 |  0.922 |      247 |       21 |        0 | §5                                      |
| Qwen3.5-4B             | Telecom                    | telecom v0  |  261 |  0.993 |  0.973 |      254 |        7 |        0 | [training-data.md §6](training-data.md) |
| Qwen3-4B-Instruct-2507 | Airline                    | v2          |  380 |  0.593 |  0.289 |      110 |      211 |       59 | 本节                                    |
| Qwen3-4B-Instruct-2507 | Retail                     | v2          |  268 |  0.787 |  0.459 |      123 |      138 |        7 | 本节                                    |
| Qwen3-4B-Instruct-2507 | Telecom                    | telecom v1  |  261 |  0.380 |  0.222 |       58 |       94 |      109 | 本节                                    |
| Qwen3-4B-Thinking-2507 | Airline                    | v2          |  380 |  0.740 |  0.489 |      186 |      168 |       26 | 本节                                    |
| Qwen3-4B-Thinking-2507 | Retail                     | v2          |  268 |  0.913 |  0.761 |      204 |       62 |        2 | 本节                                    |
| Qwen3-4B-Thinking-2507 | Telecom                    | telecom v1  |  261 |  0.307 |  0.176 |       46 |       64 |      151 | 本节                                    |
| （下一个模型）         | Airline / Retail / Telecom | 当前 latest |      |        |        |          |          |          |                                         |

Qwen3.5 的 Telecom 用的是 v0。v1 只在 50 题里改了一句"要恰好 2 GB"，Qwen3.5 在 v0 上已经 0.993，所以没有重评。

**读表**（2026-10-12，表 B 的四个 Qwen3 运行都没有 infra 排除）：

- **Qwen3.5-4B 在 AReaL 数据上没有训练信号**：v2 的 648 题里只有 84 题有对有错（13%），Telecom 261 题里只有 7 题。
- **在 v2 上，Instruct 的信号最多，Thinking 的分数更高**：
  - Instruct 有 349 题有对有错（54%），4 次全错 66 题；
  - Thinking 的 mean@4 是 Airline 0.740、Retail 0.913，比 Instruct（0.593、0.787）高，有对有错 230
    题（35%），Retail 几乎做满（204 / 268 题 4 次全对）。
- **两个 Qwen3 全错的题都集中在修过的 Airline 题**：Instruct 在 Airline fix 上 0.463、全错 47 题，Thinking
  0.673、全错 17 题；keep 上分别是 0.721 和 0.806。这些题的 gold 是多步改签、降舱，Qwen3.5
  能做对（§5），所以是模型能力不够，不是题错。
- **AReaL 的 Airline / Retail 比官方题容易，Telecom 不是**：
  - Airline / Retail：Instruct 在官方题上是 0.225 / 0.410，在 v2 上是 0.593 / 0.787；Thinking 是
    0.245 / 0.654 对 0.740 / 0.913。
  - Telecom：Instruct 在官方题上是 0.377，在 AReaL 上是 0.380；Thinking 是 0.373 对 0.307。AReaL 的
    Telecom 题和官方一样要在一次对话里排除多个故障，Qwen3
    卡在同一处：长程排障做不完（[capability-gaps.md §2.2](capability-gaps.md)）。
- **Telecom 对 Qwen3 有信号，但难题太多**：Instruct 有对有错 94 题、全错 109 题；Thinking 全错 151 题（58%）。直接拿来做
  RL，大半题拿不到正样本，需要按难度挑题，或先做示范数据 warm-up（[capability-gaps.md §2.1](capability-gaps.md) 第 5
  条）。
- **在 v2 上练出的提升能不能迁移到官方题，要训练后用表 A 的口径评测才知道。**

**费用**：表 B 的四个 Qwen3 运行共用了约 108 CNY 的 DeepSeek 余额，其中 Telecom 占大头：Telecom 对话长，一个模型跑 261 题
× 4，用户侧约 150M prompt token。

## 2 官方 runner 口径下的 step 0 评测（同日补充）

两组都用 commit `31634dee`：未训练的 Qwen3.5-4B（non-thinking），base 集（Airline 50 + Retail 114 题，不含
Telecom），policy temperature 0，每题 4 次，用户模拟器 temperature 0、无回复上限，32K context，评测不设单次回复上限。

| 用户模拟器       | 领域    | episode（排除） | mean@4 = pass^1 | pass^2 | pass^4  |
| ---------------- | ------- | --------------- | --------------- | ------ | ------- |
| DeepSeek-Flash   | 合计    | 656（0）        | 0.838           | 0.752  | 0.646   |
| DeepSeek-Flash   | Airline | 200（0）        | 0.810           | 0.733  | 0.660   |
| DeepSeek-Flash   | Retail  | 456（0）        | 0.851           | 0.760  | 0.640   |
| Doubao 2.1 Turbo | 合计    | 656（2）        | 0.691           | 0.548  | 0.426\* |
| Doubao 2.1 Turbo | Airline | 200（1）        | 0.783           | 0.703  | 0.633\* |
| Doubao 2.1 Turbo | Retail  | 456（1）        | 0.651           | 0.480  | 0.336\* |

\* Doubao 有 2 个 episode 在重试时遇到 Ark 账户欠费，被排除；pass^4 只在其余 162 题上算（Airline 49、Retail 113）。

- 只换用户模拟器，Retail 从 0.851 降到 0.651，Airline 只差 0.03。Doubao 用户下的 Retail（0.651）已经接近 Recuris
  报告的 68.0；Airline（0.783）仍高于 75.3。
- 旧代码下同配置的 DeepSeek 评测是 0.794，新代码 0.838。区别是评测不再截断 4096 token 的回复（旧代码有 32 个 episode
  被截断）和重试规则；本次无排除。两者和用户模拟器本身的随机性各占多少，没有逐题核对。
- Doubao 评测里 3 个 episode 是用户第一句就
  stop，按官方打分，没有中止评测。后来查明是方舟服务异常（[user-simulator.md §2](user-simulator.md)）。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261009/step0-base-t0-deepseek-official/`
和
`.../step0-base-t0-doubao21turbo-official/`（`summary.json`、`episodes.jsonl`、`trajectories/`）。

复现脚本：`scripts/official-runner/fault_user_proxy.py`（PRD 第 6 节的故障注入
smoke）、`scripts/official-runner/metrics_table.py` （从 `episodes.jsonl`
重算[step0-evaluations.md §2](step0-evaluations.md)的 mean@4 / pass^1/2/4，在仓库根目录运行）。

## 3 上一轮：official 与 fixed v1（2026-10-10）

fixed v1 是加入
[benchmark-variants.md §1.2](benchmark-variants.md)–[benchmark-variants.md §1.4](benchmark-variants.md)
之前的版本，只有 #411、#571、#322 和第一条转人工规则（没有 Retail 例外）。

配置与第 2 节相同：未训练的 Qwen3.5-4B，base 集 164 题，每题 4 次，policy 和 DeepSeek-Flash 用户都是 temperature
0。代码是 `1d14d8dc`，从一个单独固定在这个提交的 worktree 运行。两个变体先后在同一台 8 卡机上跑完，各用约 26 分钟和 31 分钟。

| 变体     | 领域 / 划分   | 题数 | 排除 | mean@4 = pass^1 | pass^2 | pass^4 |
| -------- | ------------- | ---- | ---- | --------------- | ------ | ------ |
| official | 合计          | 164  | 0    | 0.808           | 0.705  | 0.561  |
| official | Airline       | 50   | 0    | 0.780           | 0.710  | 0.620  |
| official | Airline train | 30   | 0    | 0.775           | 0.717  | 0.667  |
| official | Airline test  | 20   | 0    | 0.787           | 0.700  | 0.550  |
| official | Retail        | 114  | 0    | 0.820           | 0.703  | 0.535  |
| official | Retail train  | 74   | 0    | 0.848           | 0.739  | 0.568  |
| official | Retail test   | 40   | 0    | 0.769           | 0.637  | 0.475  |
| fixed v1 | 合计          | 164  | 1    | 0.690           | 0.571  | 0.460  |
| fixed v1 | Airline       | 50   | 1    | 0.428           | 0.297  | 0.224  |
| fixed v1 | Airline train | 30   | 1    | 0.364           | 0.228  | 0.138  |
| fixed v1 | Airline test  | 20   | 0    | 0.525           | 0.400  | 0.350  |
| fixed v1 | Retail        | 114  | 0    | 0.805           | 0.692  | 0.561  |
| fixed v1 | Retail train  | 74   | 0    | 0.845           | 0.750  | 0.635  |
| fixed v1 | Retail test   | 40   | 0    | 0.731           | 0.583  | 0.425  |

fixed v1 排除了 1 条 episode，`airline:14:trial1`：4 次尝试都超过了 32K 上下文。它所在题的 pass^4 只按 3 次有效
trial 无法计算，所以 Airline 的 pass^4 只在 49 题上算。

**fixed v1 的下降几乎全在 Airline，主要来自转人工规则**：

- Airline 从 0.780 降到 0.428。fixed v1 运行中，转人工规则之前的分数是 0.744，规则把 199 条 episode 里 63
  条原本成功的记为 0，扣了 31.7pp。剩下约 3.6pp 是 #411 新增的 NL 断言计分，这个量与下面的运行间噪声同级，不能单独确认。
- Retail 几乎不变（0.820 → 0.805）。转人工规则扣了 17 条成功（3.7pp），规则之前的分数是 0.842，高于 official。
- 这和 [failure-taxonomy.md §3](failure-taxonomy.md) 的估计（Airline 约 30pp
  来自转人工）一致。也就是说，修复后这个 4B 模型的 Airline 真实水平约为 0.43，Airline train 只有 0.36。

**训练信号的变化**：Airline train 的 30 题从 0.775 降到 0.364，GRPO 组内会出现更多有对比的组，训练有更多可学的东西。Retail
train 基本不变。

**同一配置两次运行的差异**：

- official 的 0.808 比 10-09 同配置的 0.838 低 3.0pp。按题配对 bootstrap 的 95% 区间是 \[−5.9,
  −0.3\]pp（Airline \[−8.0, +1.5\]，Retail \[−6.4, +0.4\]）。
- 两次之间评测路径的代码没有改动：只增加了只对 fixed 生效的转人工规则、只对训练生效的超时处理，以及变体校验。
- 164 题中有 64 题的 4 次平均分变了。temperature 0 仍然不是确定的：SGLang 的批处理和 DeepSeek API 都会引入差异。
- 所以同一 checkpoint 重测本身就有约 3pp 的波动，与 [initial-audit.md §2.1](initial-audit.md) 节中两次 step
  0 相差 4.58pp 是同一量级。比较训练前后，至少要看 test 上的配对差和区间，不能只看点估计。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261010/step0-base-t0-deepseek-{official,fixed}/`。对比脚本：`scripts/official-runner/variant_compare.py`。

## 4 本轮：fixed v2

配置与 §3 相同：未训练的 Qwen3.5-4B，base 集 164 题，每题 4 次，policy 和 DeepSeek-Flash 用户都是 temperature
0。代码 `b31e8eb5`，在固定于该提交的 worktree 中运行，数据目录 `…-fixed-v2/data`。official 没有重跑：它的数据和评分都没变，沿用
§3。

| 变体     | 领域 / 划分                   | 题数 | 排除 | mean@4 = pass^1 | pass^2 | pass^4 |
| -------- | ----------------------------- | ---- | ---- | --------------- | ------ | ------ |
| fixed v2 | 合计                          | 164  | 4    | 0.708           | 0.612  | 0.512  |
| fixed v2 | Airline                       | 50   | 2    | 0.388           | 0.283  | 0.229  |
| fixed v2 | Airline train                 | 30   | 2    | 0.356           | 0.233  | 0.179  |
| fixed v2 | Airline test                  | 20   | 0    | 0.438           | 0.358  | 0.300  |
| fixed v2 | Retail                        | 114  | 2    | 0.849           | 0.756  | 0.634  |
| fixed v2 | Retail train                  | 74   | 2    | 0.868           | 0.782  | 0.667  |
| fixed v2 | Retail 训练池（去重后 69 题） | 69   | 2    | 0.880           | 0.792  | 0.672  |
| fixed v2 | Retail test                   | 40   | 0    | 0.812           | 0.708  | 0.575  |

排除 4 条，按题计算的 pass^4 因此只覆盖 160 题：

- `airline:10:trial3`、`airline:34:trial2`：4 次尝试都超过 32K 上下文，和 v1 的 `airline:14` 同类。
- `retail:59:trial3`、`retail:99:trial3`：`MissingAttemptRecord`，见
  [proxy 的 session 容量泄漏](infra-issues.md)。

**Airline 和预期一致：基本不变**。0.428 → 0.388（−4.0pp）。fixed v2 对 Airline 只改了反向转人工规则，这条规则只影响
airline:13， 这次没有清零任何 Airline episode。所以这 4pp 是运行间噪声，在 §3 测得的约 3pp 的量级内。正向转人工规则在 Airline
清零了 75 条原本成功的 episode（v1 是 63 条）。

**Retail 没有降低，反而升高了**。0.805 → 0.849（+4.4pp）。按题分组归因（114 题，每组为 4 次平均分之和的变化）：

| 题组                                                               | 题数 | v1 → v2  | 折合 Retail pp |
| ------------------------------------------------------------------ | ---- | -------- | -------------- |
| 20 道打了补丁的题                                                  | 20   | +3.50 题 | +3.1           |
| v1 被正向规则清零、v2 按 Retail 例外恢复的题（1、32、43 等 10 题） | 10   | +2.25 题 | +2.0           |
| 其余未改动的题                                                     | 84   | −0.75 题 | −0.7（噪声）   |

- **补丁让分数上升，是补丁本身的性质**。补丁去掉的是"做对了也判错"的失败：

  - 38：v1 是 0/4，v2 是 4/4，补丁写明了取消原因；
  - 64：0/4 → 4/4，补丁加了"多款符合时选最便宜的"；
  - 59、60、63、66、110 也都升到了 4/4。

  这些失败原本就不是模型能力的问题，留着只会给训练加噪声，不会增加可学的空间。

- **本该降分的三题没有降**。57、65 改为必须查订单，但这个模型本来就会查：57 在 v1、v2 都是 4/4，65 从 3/4 升到 4/4。25 要求转人工，v2
  是 2/4，与 v1 相同。 "什么都不做也得分"的漏洞对这个模型的分数影响很小。

- **Retail 例外恢复了 v1 误清零的分数**（+2.0pp），这和
  [benchmark-variants.md §1.3](benchmark-variants.md) 的判断一致。正向规则在 Retail 不再清零任何
  episode。反向规则清零了 2 条（retail train）。

**训练集上的组内对比**（temperature 0，每题 4 次，按 train 池统计）：

| 领域          | 题数 | v1：全对 / 全错 / 有对比 | v2：全对 / 全错 / 有对比     |
| ------------- | ---- | ------------------------ | ---------------------------- |
| Airline train | 30   | 4 / 11 / 15              | 5 / 12 / 13                  |
| Retail 训练池 | 69   | 45 / 0 / 22              | 47 / 0 / 22（另 2 题有排除） |

- Retail 训练池有 47/69 题在 temperature 0 下 4 次全对。可学的空间主要在 Airline，以及 Retail 剩下的约 22 题。
- 训练用 temperature 1、每组 8 条，组内对比会比这里多。上一轮 GRPO（含 Telecom、官方数据）训练期 Retail reward 是
  0.868，72.5% 的组没有组内优势（[initial-audit.md §2.2](initial-audit.md)）。所以"Retail
  提升空间小"在训练里同样存在。
- 去掉不公平的失败后 Retail 分数上升，是修复的预期结果。如果想要更多 Retail 训练空间，办法只能是加更难的题或换更难的用户模拟器，不能靠让评分变得不公平。
  本轮没有做这类改动。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261010-fixed-v2/step0-base-t0-deepseek-fixed-v2/`。
对比脚本是 `scripts/official-runner/variant_compare.py`，按领域、划分和训练池输出 mean@4 /
pass^k，以及转人工规则清零的条数。

## 5 AReaL pack v2（2026-10-11）

口径与 [training-data.md §3](training-data.md) 第三步的 v1 评测相同：未训练的 Qwen3.5-4B，temperature
0，DeepSeek-Flash 做用户和 NL judge，每题 4 次，`benchmark_variant=fixed`，代码固定在 `1b17bc3f`。pack 是
`_cache/tau2-areal-pack-20261011-v2`（[training-data.md §4](training-data.md)，648 题）。

- 第一次运行（pueue 447）时 DeepSeek 余额耗尽，690 条 episode 报 `Insufficient Balance`，被当作 infra
  错误排除。充值后把涉及的 197 题 4 次全部重跑（pueue 448），按题替换合并，最终 2592 条 episode 全部有分。
- DeepSeek 用户侧：两次运行共 25.9M prompt token、3.6M completion token（含第一次运行里被替换掉的 98 条）。v1 的
  813 题用了 33M / 5.0M。
- 产物：`tau2-eval-20261011-areal-pack/step0-areal-pack-v2-merged/episodes.jsonl`（合并结果）。两次运行的原始目录
  `step0-areal-pack-v2-t0-deepseek`、`…-rerun` 和合并脚本 `merge_v2_rerun.py` 都在同一目录下。

| 范围                   | 题数 | mean@4 = pass^1 |    pass^2 |    pass^4 | 4 次全对 | 有对有错 | 4 次全错 |
| ---------------------- | ---: | --------------: | --------: | --------: | -------: | -------: | -------: |
| v1 全部                |  813 |           0.912 |     0.859 |     0.783 |      637 |      161 |       15 |
| v1 中 v2 保留的 648 题 |  648 |           0.934 |     0.887 |     0.815 |      528 |      114 |        6 |
| v1 中 v2 丢掉的 165 题 |  165 |           0.827 |     0.747 |     0.661 |      109 |       47 |        9 |
| **v2 合计**            |  648 |       **0.946** | **0.912** | **0.861** |      558 |       84 |        6 |
| v2 Airline             |  380 |           0.926 |     0.881 |     0.818 |      311 |       63 |        6 |
| v2 Airline keep        |  191 |           0.969 |     0.946 |     0.911 |      174 |       16 |        1 |
| v2 Airline fix         |  189 |           0.882 |     0.816 |     0.725 |      137 |       47 |        5 |
| v2 Retail              |  268 |           0.976 |     0.955 |     0.922 |      247 |       21 |        0 |
| v2 Retail keep         |   48 |           0.964 |     0.931 |     0.875 |       42 |        6 |        0 |
| v2 Retail fix          |  220 |           0.978 |     0.961 |     0.932 |      205 |       15 |        0 |

**结果**：

- **丢掉的题分数更低**（0.827 对保留题的 0.934）。v1 里 4 次全错的 15 题，有 9 题在 v2 被丢掉，说明大部分全错的题是题本身有问题。v2 剩下
  6 题 4 次全错，都是 Airline（airline_361、365、437、505、710、871），还没看轨迹。
- **同一批题，修过之后结果更稳**：有对有错从 114 降到 84，pass^4 从 0.815 升到 0.861，主要在 Retail（0.851 →
  0.922）。第三轮复核在指令里写明了"替谁办理"和地址写法，用户模拟器少了自由发挥的空间。
- **单题的变化多是噪声**：648 题里 505 题分数不变，83 题升、59 题降。temperature 0 下 DeepSeek 用户仍有随机性，单题对比没有意义。
- **最难的是修过的 Airline 题**（0.882，keep 是 0.969）：替换后的 gold 多是多步的改签、降舱。

**对后续实验设置的含义**：

1. **v2 对 Qwen3.5-4B 太容易，不适合拿它直接做 RL**。86% 的题 4 次全对，GRPO 只有 84 题（13%）的组内有分数差异；按训练时
   temperature 1 采样，差异会多一些，但这一点没测。
1. **AReaL 的题比官方 test 容易得多**。同一模型、同一口径下，官方 fixed v2 base 集 Airline 0.388、Retail
   0.849（§4），AReaL v2 是 0.926、0.976。AReaL 的 Airline 题多是单一预订上的一两步操作，官方 Airline
   题要在多个预订之间推理 policy。所以在 AReaL 上练出的分数提升，不一定能迁移到官方 Airline。
1. **训练模型的候选是更弱的 Qwen3-4B-2507**（[capability-gaps.md §2.3](capability-gaps.md)）。它在官方
   fixed 上 Airline / Retail 只有 0.225–0.245 / 0.410–0.654，v2 的难度对它可能正合适。下一步先用本节口径在 v2 上评测
   Instruct 和 Thinking 两个版本，看有对有错的题有多少。DeepSeek 费用参照本节：Qwen3.5 跑 648 题约 26M prompt
   token，Thinking 的对话更长，会更多。
1. **Telecom**：Qwen3.5-4B 在官方 Telecom 上 456/456
   全对（[capability-gaps.md §2.1](capability-gaps.md)），AReaL Telecom 也一样容易（0.993，见
   [training-data.md §6](training-data.md)）。按上一条，Telecom 只对更弱的模型有用。

## 6 Qwen3-4B-2507 的 step 0 评测（2026-10-11）

对象：未经训练的 Qwen3-4B-Instruct-2507（non-thinking）和 Qwen3-4B-Thinking-2507（thinking）。
目的：Qwen3.5-4B 在 fixed v2 的 Retail、Telecom 上已接近满分（§4）。这里确认两个 Qwen3
模型的可提升空间，并弄清分数低的原因：是模型能力不足还是题目问题，如果是模型，具体缺哪种能力。另外核查我们为 Qwen3.5 做的加速配置有哪些对 Qwen3 仍然可用。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-eval-20261010-qwen3-2507/`

- 每个运行目录下有 `summary.json`、`episodes.jsonl`、`trajectories/`
- `analysis/` 存放逐 episode 特征（`feat-*.jsonl`）和 actor probe 日志

脚本在本目录 `scripts/qwen3-2507/`：

- 启动：`launch_eval.sh`、`lane_a.sh`、`lane_b.sh`
- 特征与分类：`extract_episodes.py`、`aggregate.py`、`compare.py`、`write_diff.py`、`telecom.py`
- 查看轨迹：`view.py`
- Actor probe：`actor_probe.py`
- 模型下载与校验：`fetch_models.py`

### 6.1 配置

| 项                    | 设置                                                                                                              |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 题集                  | base（train + test），每题 4 次，policy temperature 0。fixed：Airline 50 + Retail 114；official：再加 Telecom 114 |
| 用户模拟器 / NL judge | DeepSeek-Flash，temperature 0，同 §4                                                                              |
| Instruct              | `enable_thinking=false`；concat 模板，与 Qwen3.5 评测相同；`tool_call_parser=qwen25`                              |
| Thinking              | `enable_thinking=true`；hf 模板 + individual 导出；`tool_call_parser=qwen25`                                      |
| 代码                  | `a6abe781`，在固定于该提交的 worktree 上运行                                                                      |
| 模型                  | Instruct `cdbee75f`，Thinking `768f209d`；从 ModelScope 下载，4 个 LFS 文件的 sha256 都与 HF 一致                 |

- **Telecom 不分变体。** fixed 只改了 Airline 和 Retail，Telecom 只在 official 里跑一次。Qwen3.5 的
  Telecom 对照是这次按相同配置新跑的。
- **与模型卡分数不可比。** 模型卡用 gpt-4.1 类用户模拟器、采样解码；这里是 DeepSeek 用户、temperature 0。

**为支持 Thinking 改的代码（`a6abe781`）。** Thinking-2507 的模板在生成前缀里就放了 `<think>`，所以模型输出里只有
`…</think>答案`。改之前有三个问题：

- proxy 不把这段识别为推理，整段思维链作为 `content` 返回；
- 用户模拟器会读到 agent 的思维链；
- 带工具调用的回复同时带文本，orchestrator 会判为错误。

改动内容：

- `ReasoningStrippedLLMAgent` 在 tau2 侧去掉 `</think>` 及之前的内容，完整回复保留在 `raw_data`；
- proxy 把只出现 `</think>` 的情况也识别为推理段；
- 每轮用 hf 模板完整渲染，历史里不保留之前各轮的思维链；
- 训练仍要求 `enable_thinking=false`。

三处都有单元测试，分别故意破坏后对应测试会失败。

已知限制：带工具调用的轮次，`raw_data` 里也没有思维链，所以 case study 只能看纯文本回复轮的推理。

### 6.2 分数

mean@4 = pass^1。

| 模型          | 变体     | 领域    | pass^1 | pass^2 | pass^4 | test  | train |
| ------------- | -------- | ------- | ------ | ------ | ------ | ----- | ----- |
| Qwen3.5-4B    | official | Airline | 0.780  | 0.710  | 0.620  | 0.787 | 0.775 |
| Qwen3.5-4B    | official | Retail  | 0.820  | 0.703  | 0.535  | 0.769 | 0.848 |
| Qwen3.5-4B    | fixed v2 | Airline | 0.388  | 0.283  | 0.229  | 0.438 | 0.356 |
| Qwen3.5-4B    | fixed v2 | Retail  | 0.849  | 0.756  | 0.634  | 0.812 | 0.867 |
| Qwen3.5-4B    | official | Telecom | 1.000  | 1.000  | 1.000  | 1.000 | 1.000 |
| Instruct-2507 | fixed    | Airline | 0.225  | 0.107  | 0.060  | 0.200 | 0.242 |
| Instruct-2507 | fixed    | Retail  | 0.410  | 0.235  | 0.096  | 0.438 | 0.395 |
| Instruct-2507 | official | Airline | 0.330  | 0.203  | 0.140  | 0.250 | 0.383 |
| Instruct-2507 | official | Retail  | 0.445  | 0.288  | 0.158  | 0.481 | 0.426 |
| Instruct-2507 | official | Telecom | 0.377  | 0.240  | 0.167  | 0.481 | 0.321 |
| Thinking-2507 | fixed    | Airline | 0.245  | 0.140  | 0.060  | 0.263 | 0.233 |
| Thinking-2507 | fixed    | Retail  | 0.654  | 0.494  | 0.351  | 0.619 | 0.672 |
| Thinking-2507 | official | Airline | 0.655  | 0.533  | 0.400  | 0.650 | 0.658 |
| Thinking-2507 | official | Retail  | 0.651  | 0.501  | 0.333  | 0.669 | 0.642 |
| Thinking-2507 | official | Telecom | 0.373  | 0.313  | 0.246  | 0.506 | 0.301 |

- **排除与超时。** 这次的 5 个运行都没有 infra 排除。Thinking official 有 7 条 1800 s 墙钟超时（Telecom 6
  条、Airline 1 条），按官方语义记 0 分。
- **成本。** 每轮生成的中位数：Thinking 956 token，Instruct 85，Qwen3.5 200。Thinking 单轮最长 7.5K
  token，没有截断，也没有超出上下文。

temperature 0 下各题 4 次的结果分布（全对 / 有对有错 / 全错）汇总在 §1 表 A。

### 6.3 题目与评测因素

**用户在确认句里结束对话。** agent 请求确认后，DeepSeek 用户回 "Yes, please proceed. ###STOP###"，写操作来不及执行。这和
[user-simulator.md §1](user-simulator.md) 节 Doubao 的现象相同。

判定规则：

- 该 episode 失败；
- gold 有写操作，agent 没写；
- 最后一句用户消息带肯定词，并紧接 `###STOP###`。

| 运行              | Retail 条数 | Retail mean@4 | 去掉这些 episode 后 |
| ----------------- | ----------- | ------------- | ------------------- |
| Qwen3.5 fixed v2  | 12          | 0.848         | 0.871               |
| Instruct fixed    | 42          | 0.410         | 0.452               |
| Thinking fixed    | 41          | 0.654         | 0.718               |
| Instruct official | 48          | 0.445         | 0.498               |
| Thinking official | 34          | 0.651         | 0.704               |

- Qwen3 模型更常请求确认，所以受影响更多，但扣除后差距仍在。
- Airline 每个运行有 0–6 条，Telecom 为 0。
- 这些 episode 不该作为负样本进入训练，与 [capability-gaps.md §1.3](capability-gaps.md) 第 2 项是同一个问题。

**转人工漏洞（[failure-taxonomy.md §3](failure-taxonomy.md)）。** Thinking 在 Airline 大量转人工：fixed
规则清零 95 条，其中 57 条在 official 评分下是 1 分。official 0.655 与 fixed 0.245 之间的 41pp
差距就来自这里。Instruct 很少转人工，而是直接执行修改，所以两个变体只差 10pp。

**没有发现新的题目缺陷。** 精读过的失败轨迹都能对应到 agent 的具体错误。
