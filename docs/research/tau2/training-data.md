# 训练数据：AReaL-tau2-data 的核查、筛选与流程

> 来源：原 README 第 12.7–12.11、13.4 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 Retail 训练信号不足：社区的做法（2026-10-10 调研）

**问题**：fixed v2 下，Retail 训练池 69 题里有 45 题在 temperature 0 下 4 次全对（另有 2 题缺 1 次 trial
的记录）（[step0-evaluations.md §4](step0-evaluations.md)）。

- 对 GRPO 来说，全对的组 advantage 为 0，没有梯度。
- 对 PPO / SAO 来说，critic 能给全对轨迹的中间步骤非零的 advantage。但这些题上从来没有失败样本，critic 学不到"哪里会失败"，信号很弱。所以
  PPO 也只是比 GRPO 好一些，题太简单的问题仍然存在。

下面的结论来自调研 agent，标注说明来源：\[代码/论文\] 表示读过原文，\[摘要\] 表示只读了摘要或 README。两个 HF 数据集的存在、license 和 DB
文件，我用 HF API 独立核对过；其余没有独立核对。

**现成的、能直接当 RL 任务用的数据**：

| 数据                                                                                                         | 规模                                           | 内容                                                               | License     | 用于我们的代价                                                                                     |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------------------------- |
| [inclusionAI/AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)（AReaL-SEA 论文） | RL 任务：Airline 1148、Retail 563、Telecom 271 | 每题带 gold actions 和 `db_path`。Retail 用 3–4 个新合成的 DB      | Apache-2.0  | 要能按任务加载各自的 DB；现在 `agent.py` 只用默认 DB。数据质量见 §2                                |
| [wwwenlin/skillscaffold-tau-rl](https://huggingface.co/datasets/wwwenlin/skillscaffold-tau-rl)               | 每个领域 300 题                                | 上面那份数据的子集：用 Qwen3-8B 每题跑 8 次，只保留通过 1–6 次的题 | Apache-2.0  | 格式与上面相同。它是按 8B 筛的，要用我们的 4B 重新筛                                               |
| tau-bench v1 Retail train                                                                                    | 500 题                                         | 与 tau2 Retail 共用同一个 DB                                       | MIT         | **不推荐**：调研 agent 实测，tau2 Retail test 的 41 个写操作 order_id 中有 26 个出现在 v1 train 里 |
| APIGen-MT-5k、CoVe-12k                                                                                       | 5k / 12k 条轨迹                                | 只有轨迹，没有 gold 或 DB                                          | NC / Apache | 只能用于 SFT                                                                                       |

**不换数据也能做的**：

- **训练时丢掉全对或全错的组**（DAPO 的 dynamic sampling）。AReaL-SEA 在 Airline 30B 上做过消融：pass^1 65.0 →
  70.5，pass^4 40 → 52 \[论文\]。
- **离线按 pass-rate 筛题**：用当前模型每题跑 8 次，只留通过率在 1/8 到 6/8 之间的题，即 Skywork-OR1、POLARIS 的做法
  \[摘要\]。
- **训练时用户模拟器用 temperature 1**：AReaL-SEA 训练时 agent 和用户都用 1.0 \[论文\]。我们的训练配置里，用户是
  temperature 0。
- tau2 自带的 `PersonaConfig` 只有 verbosity 和 interrupt 两项，提不了多少难度 \[代码\]。

**同类工作**：

| 工作        | 模型    | 训练数据                           | Retail / Airline                   |
| ----------- | ------- | ---------------------------------- | ---------------------------------- |
| AReaL-SEA   | 30B-A3B | 自己合成的任务和新 DB              | 54.2 → 75.0 / 56.0 → 70.5 \[论文\] |
| MUA-RL      | 32B     | tau1 Retail 115 题 + Airline 50 题 | 67.3 / 45.4 \[摘要\]               |
| AgentScaler | 4B      | 合成环境，只做 SFT                 | 56.1 → 62.3 / 52.0 → 56.0 \[论文\] |

这些工作的训练数据都不只是官方的 train split。

**建议**（未实施，需要决定）：

1. 先做零成本的两项：GRPO 丢掉全对或全错的组，训练时用户模拟器用 temperature 1（评测仍用 0）。
   - 代价：每步有效样本变少，可能要加大 batch。
   - 与已经批准的训练计划不同，需要批准。

## 2 AReaL-tau2-data 的数据质量（2026-10-10 核查）

**结论**：这份数据可以用，但不能整份直接用。官方没有披露数据问题或标注误差；我们的机械检查找到约 2% 的题 gold 确定是错的，算上有歧义的题至少
6.8%，Retail 偏高。接入前要先按下面的检查过滤，并按 pass-rate 重新筛题。

**来源与披露**（调研 agent 读了 HF 数据卡、HF 讨论区、[AReaL 仓库](https://github.com/inclusionAI/AReaL)和
AReaL-SEA 论文）：

- 合成流水线没有开源。有人在 AReaL issue #915 里问过，没有公开代码。
- 论文没有给出噪声率、和官方 test 的去重方法，也没有局限性一节。HF 讨论区没有人报告数据问题。
- 所以下面的问题都是我们自己查出来的，不是官方披露的。

**结构**（我核对过）：

- RL 任务共 1982 题：Airline 1148、Retail 563、Telecom 271。`evaluation_criteria` 存成 JSON
  字符串，每题有自己的 `db_path`。
- Airline 有 543 题用官方 DB，其余用 `new_db_1/2/3`；Retail 全部用 `new_db_1–4`。
- Airline 和 Retail 的 1711 题全部用默认 reward_basis（DB + COMMUNICATE），而 `communicate_info`
  都是空的，`nl_assertions` 都是 null。也就是说，实际只检查最终 DB 的 hash。

**机械检查**（脚本与结果见下面的归档；除标注"我核对过"的，数字来自调研 agent）：

| 检查                | 结果                                                                             | 含义                                                                      |
| ------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| gold 写操作重放     | 0 条抛错。故意破坏一条 gold 后，检查会报错                                       | gold 都能执行                                                             |
| 什么都不做也得 1 分 | Airline 55 题、Retail 15 题                                                      | 和官方 Retail 25、57、65 同类，训练时奖励不作为                           |
| 重复题              | Airline 有 228 道多余的重复题，其中 Mia Li 的 HKEG34 一个预订就有 152 题         | 训练分布严重偏向少数几个用户                                              |
| 与官方 test 重叠    | 3 道 Airline 题与某道 test 题写同一个预订；14 道与 test 同一个用户。没有文本重叠 | 这 17 道要去掉                                                            |
| 支付方式没写        | Retail 61 题：用户有多种支付方式，指令没说用哪一种，gold 却固定了一种            | 和我们 [benchmark-variants.md §1.2](benchmark-variants.md) 修过的问题同类 |
| 新 DB 内部不一致    | `new_db_1/2` 有 668 / 777 个预订的航线与航班表不符；55 道 gold 用到这些坏记录    | agent 按航班表推理会与 gold 不一致                                        |

**case study**：

| 题                                                | 问题                                                                                                                                                                                                       | 判断                         |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| airline_56（同类共 30 题，含 63、664、810、1045） | 指令三次要求用 `credit_card_4421486`（Visa 尾号 7447）付款，gold 用的是 `credit_card_1955700`（尾号 1907）。tau2 信用卡没有额度限制，不存在换卡的理由。**我核对过**                                        | gold 确定错误                |
| airline_718（同类共 5 题，含 781、844）           | 用户说"昨天订的"，要求按 24 小时免费取消。gold 取消的 3 个预订创建于 2024-05-14 14:00–14:30，policy 的当前时间是 2024-05-15 15:00，已超过 24 小时；都是 economy、没买保险，policy 不允许取消。**我核对过** | gold 违反 policy，确定错误   |
| airline_54                                        | 要求最早的航班。最早的 10:15 航班没有座位；有座位的最早时刻是 10:45，有两个航班并列，gold 只认 `CSPD2YO`                                                                                                   | 有歧义                       |
| retail_18（同类含 98、141）                       | 用户有信用卡和 PayPal，指令没提支付方式，gold 用信用卡补差价。**我核对过**                                                                                                                                 | 有歧义，agent 问用户也问不出 |

**误差估计**：

- 确定错误：35 / 1711 = 2.0%（Airline 3.0%）。
- 确定错误加上有歧义：6.8%（Retail 约 11%，Airline 4.6%）。这是下限，只计入了机械检查能发现的类型。
- 推测的真实比例：Retail 10–20%，Airline 5–10%。这是调研 agent 的推测，没有抽样验证。
- 对比：官方 Retail 我们修了 20 题（114 题中的 18%），所以官方数据并不比它干净多少；差别是官方的问题我们已经逐题查过。

**与 fixed 变体的关系**：#322 会改变 13 道 Retail 题的 gold hash。但 gold 是在线重放出来的，不是存好的 hash，所以 fixed
变体可以直接用在这份数据上。

归档：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/areal-tau2-data-audit-20261010/`（`scripts/`
是检查脚本，`out/` 是每题的检查结果，原始数据从 HF 下载）。

## 3 AReaL-tau2-data 的筛选与修复：v1（2026-10-10 → 10-11）

**决定**（用户，2026-10-10）：从 Airline / Retail 的 1711 题里筛出一个 reward 正确的子集用于训练（按
[training-setup.md §1](training-setup.md) 不用 Telecom）。

- 能修的自己修，修不好或超出修复范围的直接丢掉。
- 留下的题，不论原样保留还是修过，都要用评测证明可用。
- 评测时 agent 用本机 4B，用户模拟器仍用 DeepSeek。

**第一步：机械过滤**（脚本，零成本），结果在 `out/bucket_mechanical.json`：

| 类别               | Airline | Retail | 处理                                                                                                               |
| ------------------ | ------: | -----: | ------------------------------------------------------------------------------------------------------------------ |
| 直接丢弃           |     353 |     17 | 重复题（每组留一道）、什么都不做也得分、gold 只有转人工、用到 DB 坏记录、与官方 test 同用户或同写入目标            |
| 修复候选           |      35 |     63 | 支付方式与指令矛盾（18）、24 h 取消违规（5）、"最早航班"有歧义（12）；Retail 支付方式没写（61）、取消原因没写（2） |
| 机械检查未发现问题 |     760 |    483 | 机械检查只能查出已知类型的问题，仍要逐题人工审                                                                     |

后两类共 1341 题进入第二步。一题命中多条规则时，丢弃优先。

**第二步：逐题 case study**（本机 Codex 订阅，7 个 `codex exec` 并行，按 DB 分片，每片 146–251 题）：

- 每题的结论只能是 keep、fix、drop 之一，并附证据（记录 id、字段值、policy 条款）。

- fix 只允许两种改法：给用户指令补一句用户本来就知道或能决定的信息，或者在指令明确时替换 gold。超过约两处小改动就 drop； 没有把握时也 drop。

- 每个 keep 和 fix 都要用工具 `score.py` 在该题自己的 DB 上重放，满足：

  - gold 得 1 分、空轨迹得 0 分、gold 写操作不报错；
  - 去掉任何一个 gold 写操作都得 0 分。

- 不允许把题修成 gold 不写 DB 的拒绝题，因为那样什么都不做也能得分。

- **试跑**（2026-10-10，6 题，约 8 分钟）：1 keep（airline_2）、1 fix（airline_56）、4
  drop（airline_718、54，retail_18、549）。

  - 已知的三类错误都查出来了：换卡、24 h 取消违规、"最早航班"有歧义。
  - 另外查出两处 DB 引用断裂，重放时不会报错：retail_18 订单里的商品 `3949674330` 不在该产品的 variant
    列表里（**我核对过**）；airline_54 的预订引用了航班表里没有的 `VNOB4BW`。
  - airline_56 的修法：gold 改用指令要求的卡；指令里写死的 `HAT084` 是 04:00
    起飞，不在用户要求的时间段内，改成"最便宜的直飞商务舱"，唯一解是 HAT229。
  - 试跑还发现 `score.py` 的一个 bug：retail 的 #322 定价补丁丢了 WRITE 标记，导致"去掉单个写操作"的检查漏掉了
    `modify_pending_order_items`。已修复，并在审题规则里加了一条：重放成功不等于引用的记录都存在，要自己核对。

- **全量结果**（7 个分片，2026-10-10 23:46 → 10-11 00:07，模型 gpt-6-astra）：1341 题全部有结论，逐题结论在
  `out/s*.jsonl`。

  | 领域    | 进入第二步 | keep | fix | drop |
  | ------- | ---------: | ---: | --: | ---: |
  | Airline |        795 |  305 | 227 |  263 |
  | Retail  |        546 |  119 | 250 |  177 |

  - drop 的主要原因：Airline 是 DB 引用断裂（178）和 gold 与指令矛盾（43）；Retail 是 reward 太弱（56，含不作为得分）、gold
    与指令矛盾（48）和 DB 不一致（44）。
  - 机械检查"没发现问题"的 1243 题里，只有 424 题原样 keep，其余 819 题被 fix 或 drop。机械检查漏掉的问题远多于查出的。
  - fix 的改法：320 题只在指令里补一句（主要是指定支付方式、变体、航班），156 题替换了 gold。
  - **我的独立重放**：用 `score.py` 重放全部 424 道 keep 和 477 道 fix，都满足 gold 1 分、空轨迹 0 分、写操作不报错。有 4
    道 keep 的 gold 包含一次不改变 DB 的行李更新（airline_90、259、414、514），不影响 reward。

- **抽查发现的风险**：我抽读了 8 道 fix。只补指令的改法都合理；但有几道替换 gold 的改法，依据的是审题 agent 自己推断的 policy 规则，官方
  gold 从不这么做。例如 airline_533、873：降舱后，对超出新额度的托运行李补收费用；airline_225：先升舱，再借此改航班。这类 gold
  放进训练，会奖励与官方评测不一致的行为。

- **第二轮对抗复核**（2026-10-11 00:12 → 00:29，5 个分片，模型 gpt-6.1-sol，规则见 `REVIEW.md`）：

  - 复核范围是全部 477 道 fix，外加随机抽样的 100 道 keep。
  - 复核者先自己判断，再看第一轮的证据。依据 policy 的结论要逐字引用原文，复核结束时还会校验引用确实是原文的子串。没把握就拒绝。

  | 对象                               | 题数 | 通过 | 拒绝 | 主要拒绝原因                                                                                 |
  | ---------------------------------- | ---: | ---: | ---: | -------------------------------------------------------------------------------------------- |
  | Airline fix，只改指令              |  143 |  132 |   11 | 补丁文本不合格、gold 不唯一                                                                  |
  | Airline fix，替换 gold             |   84 |   53 |   31 | 依据推断出的 policy（Airline 全部 fix 中因此被拒 32 道，含我抽查到的 airline_533、873、225） |
  | Retail fix，只改指令               |  222 |  195 |   27 | 补丁文本不合格（Retail 全部 fix 中 23 道），比如替 agent 规定怎么处理                        |
  | Retail fix，替换 gold              |   28 |   20 |    8 | DB 不一致、gold 不唯一                                                                       |
  | keep 抽样（Airline 79、Retail 21） |  100 |   98 |    2 | 都是"商品不存在"那一类，见下                                                                 |

  - 结论：替换 gold 的修法拒绝率是 35%，只改指令的是 10%。keep 抽样的错误率约 2%，按二项分布的 95% 区间是 0.2%–7%。

- **"商品不存在"类**：retail_97、127 的指令要求替换用户订单里根本没有的拼图。按这个特征机械扫描 keep 和 fix，共找到 24 道。

  - 第一轮把它们当作"用户记错了"：agent 查无此物，只做地址修改，gold 仍然唯一。复核拒绝了其中 15 道，理由是指令说的目标做不到。
  - **处理**：24 道全部去掉。这类题的目标本来就有争议，去掉最稳妥；以后想要，可以从 `out/` 里的结论恢复。

- **进入评测的候选集**（当时写在 `out/candidates_after_review.json`，现已由 §5 的 decisions 文件取代并删除）：

  | 领域    | 原样 keep，未复核 | 原样 keep，已复核 | fix，已复核 | 合计 |
  | ------- | ----------------: | ----------------: | ----------: | ---: |
  | Airline |               226 |                79 |         185 |  490 |
  | Retail  |                93 |                19 |         211 |  323 |

  - 进入的条件：两轮都没被拒绝，且不属于"商品不存在"类。
  - 共 813 题，是官方 train 99 题的 8 倍多。

**第三步：评测证明**（2026-10-11，用户批准）

1. **按任务加载 DB**（`1b17bc3f`）：

   - `examples/tau2/task_pack.py` 定义 task pack：每个领域一份 `tasks.json` 和 `task_db.json`，加上
     DB 文件和带每个文件 sha256 的 manifest。
   - `task_pack.install` 包装 registry 里的环境构造函数，让它接受 `task_db`。episode 和官方 evaluator 的
     gold 重放都通过它建环境，每次都拿到一份 新的 DB。
   - `scripts/tau2/evaluate.py --task-pack <pack>` 评测 pack 里的全部题。结果里的 `split` 记为 keep /
     fix，方便分开看分数。

1. **生成 pack**：`scripts/tau2/build_areal_task_pack.py` 按候选集生成 pack，并应用每道 fix 的指令修改和 gold
   替换。

   - 当前 pack：`_cache/tau2-areal-pack-20261011-v1`，Airline 490 题、Retail 323 题。

1. **打分检查**：`scripts/tau2/check_task_pack.py` 对每道题，在它自己的 DB 上构造 gold 和空轨迹，用官方
   `evaluate_simulation` 加上 fixed 的转人工规则打分。

   - 结果：813 题全部是 gold 1 分、空轨迹 0 分。
   - 这个检查能失败：改用默认 DB 打分（`--default-db`），402 题不通过，多数是 gold 重放报错。仍然通过的 411 题里，340 道
     Airline 题用的 DB 与官方 DB 逐字节相同，其余的新 DB 题碰到的记录与官方 DB 相同（抽查过）。
   - 单元测试 `tests/test_tau2_task_pack.py`：让 `task_db` 失效，测试就会失败。

1. **评测**（2026-10-11 03:11 → 04:16，pueue 任务 437）：口径与
   [step0-evaluations.md §4](step0-evaluations.md) 相同。本机 Qwen3.5-4B 用 temperature
   0，DeepSeek-Flash 做用户和 NL judge，每题 4 次， `benchmark_variant=fixed`，`sglang:d8p1t1`，并发
   256，代码固定在 `1b17bc3f`。

   - 3252 条 episode 全部有分，其中 44 条（Airline 40、Retail 4）超出 32K 上下文，按首次超出记 0 分。DeepSeek
     用户侧实际用了 33M prompt token、5.0M completion token。
   - 产物：`tau2-eval-20261011-areal-pack/step0-areal-pack-v1-t0-deepseek/`。

   | 范围         | 题数 | mean@4 = pass^1 | pass^2 | pass^4 | 4 次全对 | 有对有错 | 4 次全错 |
   | ------------ | ---: | --------------: | -----: | -----: | -------: | -------: | -------: |
   | 合计         |  813 |           0.903 |  0.844 |  0.763 |      620 |      177 |       16 |
   | Airline      |  490 |           0.884 |  0.817 |  0.727 |      356 |      119 |       15 |
   | Airline keep |  305 |           0.924 |  0.880 |  0.816 |      249 |       49 |        7 |
   | Airline fix  |  185 |           0.818 |  0.714 |  0.578 |      107 |       70 |        8 |
   | Retail       |  323 |           0.933 |  0.885 |  0.817 |      264 |       58 |        1 |
   | Retail keep  |  112 |           0.902 |  0.839 |  0.759 |       85 |       26 |        1 |
   | Retail fix   |  211 |           0.949 |  0.909 |  0.848 |      179 |       32 |        0 |

   - **可用性**：797 / 813 题至少通过一次，说明 gold 可以达到。4 次全错的 16 题（Airline 15 道，Retail 只有
     retail_353）不能直接认定为题错，要看轨迹。
   - **这份数据对当前模型偏容易**：同一模型、同一口径下，官方 fixed v2 base 集是 0.694（Airline 0.350，Retail 0.844，见
     [step0-evaluations.md §4](step0-evaluations.md)），这里是 0.903。 temperature 0 下，76% 的题
     4 次全对。
   - **Retail 的 fix 比 keep 还高**（0.949 对 0.902）：fix
     往往是在指令里写明支付方式、变体、航班，让用户能直接说出唯一答案。这正是修复要做的，但也让题变容易了。 Airline 的 fix 反而低于 keep（0.818 对
     0.924），因为替换后的 gold 多是多步的改签、降舱。

1. **失败轨迹复核**（pueue 任务 439，Codex gpt-6.1-sol，规则见 `EVAL_REVIEW.md`）。任务 438 因为 codex 等待
   stdin 挂了 4 小时，没有产出，入口加了 `< /dev/null` 后重跑。

   - 复核的 4 次全错的 15 题：9 道是 agent 的错，题本身没问题；6 道是题的错。
   - 从"有对有错"的 161 题里随机抽的 40 题：15 道是 agent 的错，25 道是题或用户模拟器的错。

## 4 v1 pack 审查（2026-10-11，结论：不能直接训练）

用户的标准：题可以简单，但不能是错题，也不能是做不完的题。按这个标准，v1 的 813 题**还不能用**。证据有三部分：

**1. 独立抽审**（Claude，与 Codex 不是同一个模型；分层抽 32 题，结果在 `scratch/claude_audit.jsonl`）：

| 类别                  | 抽查数 | 正确 | 错误                                        | 存疑                                                             |
| --------------------- | -----: | ---: | ------------------------------------------- | ---------------------------------------------------------------- |
| gold 被替换的 fix     |     12 |   11 | 0                                           | 1（retail_104：gold 只有在先完成取消、礼品卡余额够了之后才成立） |
| 只改指令的 fix        |     10 |    9 | 1（retail_375：指令里的手表在 DB 里不存在） | 0                                                                |
| 原样 keep、只审过一轮 |     10 |   10 | 0                                           | 0                                                                |

**2. 失败轨迹复核**（§3 第 5 步）找出了几类系统性的问题。下面按根因归类，并用脚本统计了每类在整个 pack 里有多少题：

| 根因                                                                                                          | pack 内题数 | 处理                                                 |
| ------------------------------------------------------------------------------------------------------------- | ----------: | ---------------------------------------------------- |
| HKEG34 / Mia Li 题族：指令写的是 Mia 本人的行程，DB 里的乘客却是 Amelia，用户模拟器会要求改乘客               |          54 | 整族丢弃                                             |
| 模板套用：指令自己承认"这里的手表其实是相机"之类                                                              |          19 | 丢弃                                                 |
| 复核判为题错的其他题，加上 airline_654（gold 改了 basic economy 的航班，违反 policy）、retail_104、retail_375 |  合计约 100 | 丢弃                                                 |
| 国家字段：gold 写的是 "USA"，指令没写，用户会说 "United States"，DB hash 因此不同                             |          99 | 补一句"国家写 USA"，或丢弃                           |
| 乘客列表顺序：gold 的顺序固定，指令没写                                                                       |          43 | 补一句列出顺序，或丢弃                               |
| 用户模拟器在 agent 执行最后一步之前就发 `###STOP###`                                                          |           — | 是 DeepSeek 用户的行为噪声，不是数据错误，官方题也有 |

"整族丢弃""模板套用"和"其他题错"三行合起来去重后是 100 题。

**3. 机械 policy 检查**（全 pack）：取消资格（24 h、商务舱、保险及理由、航司取消）、basic economy 不能改航班、新航班存在且可订。只查出
airline_654 一道违规。其余被标出的都是误报：gold 里先订再改的新预订；6 道靠保险取消的题都给了健康理由。

**结论**：

- **评测通过不等于题对**。模板套用的 19 题里有 13 题 4 次全对：用户模拟器把错误的描述"翻译"回来了。所以 4/4 只能说明 gold 可达。
- **问题主要出在两处：**
  - 只审过一轮的部分：第一轮的 keep，以及"有对有错"里没被抽到的 121 题；
  - 只补一句指令、但没处理场景其余部分矛盾的 fix。 替换 gold 的修法经过第二轮的逐字引用规则筛选后，抽查是可靠的。
- **"100 题"的构成**：HKEG34 题族 54 + 模板套用 19 + 失败轨迹复核判为 drop / 重修、且不在前两类里的 26 + retail_104 =
  100（airline_654、retail_375 已在 26 题里）。后来做了两处修正：
  - 26 题里有 11 题是用户模拟器提前 `###STOP###`，不算题错，放回保留；
  - 国家字段（99）和乘客顺序两类改为补一句事实，不丢弃。

**v2 的做法与结果**（2026-10-11，用户确认：国家、乘客顺序补一句事实；用 Codex gpt-6.1-sol 复核）：

1. 机械修正（`tools/decisions.py`）：丢掉 HKEG34 题族、模板套用题和复核判错的题；凡是 gold 写地址的，指令补"Whenever you
   give an address, the country is USA (say exactly "USA")."；凡是 gold 改或订多名乘客的，补"When you
   list the passengers, list them in this order: …"。按 gold 的写操作判定，所以补顺序的是 106
   题，比上表按失败轨迹统计的 43 题多。得到 724 题的 staging 版本，全部 gold 1 分、空轨迹 0 分。
1. 第三轮复核（`REVIEW3.md`，7 个分片，每题附 v1 的 4 次轨迹作证据）：724 题全部有结论，keep 525、fix 123、drop 76。
   - drop 里 64 道是"场景与 DB 不符"，在 HKEG34 之外又找到同类：例如 retail_489 的指令一直把订单里的鼠标叫作手表，airline_137
     的指令说 Ivan 改"我的行程"，预订上却没有他。
   - 76 道 drop 里有 61 道在 v1 评测中 4 次全对，再次说明 4/4 只证明 gold 可达。我抽查了 6 道 4/4 的 drop、5 道
     fix，判断都成立。
   - 123 道 fix 都只改指令（替换 74、追加 43、两者都有 6），没有替换 gold。
1. **v2 pack**：`_cache/tau2-areal-pack-20261011-v2`，Airline 380（keep 191、fix 189）、Retail
   268（keep 48、fix 220），共 648 题；`check_task_pack.py` 结果是全部 gold 1 分、空轨迹 0 分。

## 5 筛选流程（现行做法）

这一节记录现在怎么筛题、为什么这么做、用哪些脚本。§2–§4 是过程记录；以后筛新数据（包括 Telecom）照这一节做。

**原则**：

- **标准**：题可以简单，但不能是错题，也不能是做不完的题（infra 原因或题本身的原因都算）。没有把握就丢，错留一道题会把错误的 reward 带进 RL
  训练，错丢一道只少一道题。
- **评测通过不等于题对**。gold 可达只说明 reward 能拿到，不说明题面、DB 和 gold 互相一致。v1 里模板套用的 19 题有 13 题 4/4，v2
  第三轮丢掉的 76 题有 61 题 4/4。评测失败轨迹的用处是发现问题类别。
- **逐题审查看不出跨题的模板问题**。HKEG34 题族两轮 Codex 都没发现，是从失败轨迹里看到个例后，用脚本扫全量才找出来的。发现一个个例，就写规则扫整个
  pack，再把这一类写进 rubric。
- **修复只改指令，尽量不改 gold**。补一句用户本来就知道的事实（国家写法、乘客顺序、支付方式），或替换一个写错的名称。v1 第二轮里，替换 gold 的修法被拒
  35%，只改指令的被拒 10%，多数是 agent 自己推断出 policy 里没有的规则。
- **每次修改都要重放验证**：gold 1 分、空轨迹 0 分、去掉任一 gold 写操作得 0
  分。检查本身也要能失败：`check_task_pack.py --default-db` 和单元测试里有意破坏的检查都能变红。
- **用户模拟器提前 `###STOP###` 不算题错**，除非是场景本身诱发的。
- **先拿官方题校准"什么算错"**。官方题本来就有的写法（通用模板的 ticket、persona 年龄、工具名的叫法），不能当作题错。Telecom 第一轮因为
  rubric 没写这一条，97 道 drop 里绝大多数是这类误判（§6）。写 rubric 前先在官方题里数一数：某个特征官方题里有多少。
- **rubric 不能跨领域照搬**。Airline 的"gold 没有转人工、用户却要求转人工就丢"搬到 Telecom 就错了：Telecom 的 policy
  规定问题解决不了时要转。

**步骤与脚本**：

| 步骤              | 做什么                                                                                                   | 脚本或规则                                                                                          |
| ----------------- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1. 机械过滤       | 重复题、什么都不做也得分、gold 只有转人工、DB 坏记录、与官方 test 同用户或同写入目标                     | 工作目录 `tools/common.py`、`tools/mech_check.py`（policy 规则：取消资格、basic economy 改航班等）  |
| 2. Codex 逐题审   | keep / fix / drop，附证据；fix 用 `score.py` 自查                                                        | `PROMPT.md`；`tools/dossier.py`（打印题目和相关 DB 记录）、`tools/score.py`（在该题 DB 上重放打分） |
| 3. Codex 对抗复核 | 先独立判断再看上一轮证据；policy 结论必须逐字引用                                                        | `REVIEW.md`（第二轮）、`REVIEW3.md`（第三轮，列出已知故障类别并附评测轨迹）                         |
| 4. 评测           | 口径同 [step0-evaluations.md §4](step0-evaluations.md)：本机 4B、temperature 0、DeepSeek 用户、每题 4 次 | `scripts/tau2/evaluate.py --task-pack`                                                              |
| 5. 失败轨迹复核   | 找到第一个偏离 gold 的写操作，判断是 agent 错还是题错                                                    | `EVAL_REVIEW.md`                                                                                    |
| 6. 汇总与构建     | 把各轮结论合成一份 decisions 文件，构建 pack，检查 gold 1 / 空轨迹 0                                     | `tools/decisions.py`，`scripts/tau2/build_areal_task_pack.py`，`scripts/tau2/check_task_pack.py`    |

**派发 Codex**：

- `tools/run_codex.sh <rubric> <shard> <输出前缀>`，由 pueue 运行（`codex-review` 组）。分片结论缺 id
  时自动续跑，最多 6 轮。
- 模型 `gpt-6.1-sol`，reasoning effort high。必须显式写 `-m`：本机 Codex 的默认模型是
  `gpt-6-astra`，第一轮就是因为没写 `-m` 才用了它。
- stdin 必须接 `/dev/null`，否则 `codex exec` 会一直等输入（v1 的 pueue 任务 438 因此挂了 4 小时）。
- 在 prompt 里明确禁止开 subagent。全局 `~/.codex/AGENTS.md` 要求 Codex 主动开 subagent，第三轮 7 个分片因此额外开了
  20 个；`--disable multi_agent` 和 `agents.max_threads` 都去不掉这个工具（codex-cli 0.162.1）。
- `tools/supervise.sh <输出前缀> <分片>…` 放在后台跑，监督整轮：等 pueue
  任务结束，检查每个任务都成功、每片结论不缺不重、格式合法，然后汇总、构建候选 pack 并打分。只有全部通过才返回 0，主 agent 只需要等它结束。

**版本**：

- 每个版本只读保存在 `data-1/dataset/tau2/tau2-areal-pack-<日期>-v<N>`，里面带
  `decisions.json`，记着每道题为什么保留、改了什么、为什么丢，供追溯和写 ADR 用。`_cache` 名义上是可再生成的缓存，pack 不能再生成，所以
  2026-10-13 起放在 `dataset/`。`code/_cache/` 下同名路径是指向 `dataset/tau2/` 的软链，本文其他地方和旧脚本、旧 run
  的 `pack.txt` 里写的 `_cache/...` 路径都仍然有效。
- `tau2-areal-pack-latest` 是指向当前版本的软链（现在指向 v2），`dataset/tau2/` 和 `_cache/` 下各有一个，脚本默认读
  `_cache` 下的那个。出新版本时两个软链都要改。
- Telecom 单独建 pack：`dataset/tau2/tau2-areal-telecom-pack-<日期>-v<N>`，当前版本的软链是
  `tau2-areal-telecom-pack-latest`（现在指向 v1）。
- critic 训练用的子集 pack（[data-mix.md](critic/data-mix.md)）也放在 `dataset/tau2/`。修过的
  tau2-bench 数据目录（fixed-v2、fixed-v3）还在 `_cache/`，许多 pueue 命令写死了这些路径，等队列空了再搬。

**工作目录**
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/areal-tau2-data-screen-20261010/`：`data/`（原始数据），四份
rubric 加 Telecom 的 `TELECOM_REVIEW.md`、`TELECOM_REVIEW2.md`，`tools/`（上表的 7 个脚本，加 Telecom
的 `telecom_mech.py`、`score_telecom.py`、`decisions_telecom.py`），`shards/`，`out/`（各轮逐题结论和
decisions 文件），`scratch/`（Codex 的验证文件和 Claude 抽审），`logs/`。旧的启动脚本和中间产物已经删掉。

## 6 AReaL Telecom pack（2026-10-11）

**为什么现在要 Telecom**（用户，2026-10-11）：[training-setup.md §1](training-setup.md) 没用 Telecom
是因为它对 Qwen3.5-4B 太简单；换更弱的 agent 模型后，它就有训练价值。筛选按 §5 的流程走，pack 与 Airline / Retail 分开。

**机械过滤**（`tools/telecom_mech.py`，271 → 261）：

- AReaL 的 Telecom 数据（271 题）用的 DB 和官方 `db.toml` 逐字节相同。题目不与官方 test 集的初始状态重合；有 251 题与官方
  `telecom_full`（2285 题）同一个初始状态，只是 persona 文本不同。我们评测用的是 base 集，不用 `telecom_full`，所以保留，并在
  `out/telecom_mech.json` 里标出。
- **reward 设置有问题**：205 题没写 `reward_basis`，默认是 DB + COMMUNICATE。没有一题有
  `communicate_info`，所以 reward 实际只比较 agent 侧 DB 和用户设备状态的 hash。官方评测器重放 gold 后不调用
  `sync_tools`，而真实对话里每次工具调用都会同步，设备状态（比如 `mobile_data_usage_exceeded`）因此永远对不上：其中 136 题的
  gold 本身就得不了 1 分。官方 Telecom 题都用 ENV_ASSERTION。
- **处理**：195 题改用 ENV_ASSERTION，改完后 gold 1 分、空轨迹 0 分。剩下 10 题改了也区分不开 gold 和空轨迹，丢掉。为此 pack
  的 fix 加了 `reward_basis` 这一种改法（`6d477635`）。
- 代码：`check_task_pack.py` 按 tau2 的 `verify_task` 先应用 `initial_state`，再把每个 gold
  动作放在对应的一方（agent 或用户）重放（`0634cfe6`）。

**评测 v0**（Qwen3.5-4B，口径同 [step0-evaluations.md §5](step0-evaluations.md)，代码
`6d477635`，261 题 × 4）：mean@4 0.992，pass^2 0.985，pass^4 0.973；254 题 4 次全对，7 题有对有错，没有 4
次全错的题。DeepSeek 用户侧用了 157M prompt token，是 v2 的 6 倍：排障对话轮数多，每轮都带完整历史。

**Codex 审题**：

- **第一轮**（`TELECOM_REVIEW.md`）：keep 79、fix 85、drop 97。我抽查后发现 drop 多半不对，这些"题错"官方题里也普遍存在：

  | 第一轮判为题错的特征                                     |    第一轮引用它的条数 | 官方 2285 题中                             |
  | -------------------------------------------------------- | --------------------: | ------------------------------------------ |
  | ticket 写 "No Service"，手机其实有信号                   |            57 道 drop | 132 题同样如此（ticket 只有 5 个通用模板） |
  | persona 年龄与客户生日对不上                             | 49 道 drop、60 道 fix | 全部用同一个客户，760 题写 41 岁           |
  | 指令里写 `get_status_bar`（工具名是 `check_status_bar`） |             50 道 fix | 2285 题全都这么写                          |
  | 有一条断言初始就成立                                     |             4 道 drop | 15 题有                                    |

  另外 6 道 drop 是 rubric 照搬了 Airline 的转人工规则。97 道 drop 里 95 道在 v0 评测中 4 次全对。

- **第二轮**（`TELECOM_REVIEW2.md`，复核第一轮的 182 道 drop 和 fix）：rubric 写明上面这些不算题错。结果没有 drop，132
  题 keep，50 题 fix。50 道 fix 都是同一个改法：原指令说"愿意充 2 GB"，reward 却要求恰好 2
  GB，用户若少充也能解决问题，却拿不到分。改成"要恰好 2 GB"。

**v1 pack**：`_cache/tau2-areal-telecom-pack-20261011-v1`，261 题（50 题改了指令，195 题改了
reward_basis），全部 gold 1 分、空轨迹 0 分。v1 只改了 50 题的一句话，Qwen3.5 在 v0 上已经 0.992，所以没有用 Qwen3.5
重评。

**对后续实验的含义**：Telecom 对 Qwen3.5-4B 没有训练信号。Qwen3-4B-2507 在官方 Telecom 上约
0.37（[capability-gaps.md §2.1](capability-gaps.md)），所以 Telecom 是否有用取决于 Qwen3 在这份数据上的分数，见
[step0-evaluations.md §1](step0-evaluations.md) 表 B。DeepSeek 费用要按 Telecom 的对话长度估算：一个模型跑
261 题 × 4 约 150M prompt token。

## 7 官方 Telecom full 的 249 题 pack（2026-10-18）

Qwen3-4B-2507 在官方 `telecom_full`（2,285 题 × 4，D1）上 mean@4 19.6%，1,440 题四次全错。筛题依据和失败归因见
[telecom-case-study-d1/REPORT.md](telecom-case-study-d1/REPORT.md)：

- 排除与官方 test 同完整初始状态的 41 题（只按 test id 排除会漏 1 题），以及 1,120 道"愿意充 2 GB / 评分要求恰好 2 GB"的题
  （与 §6 第二轮的 50 题同类，本轮不修）。
- 从剩下 1–3/4 通过的题中按 1/2/3 次通过各抽 83 题，共 249 题。
- pack：`dataset/tau2/tau2-official-telecom-pack-20261018-v1`（`_cache` 下有同名软链）。题目与官方原题逐字一致，gold 1 / 空 0，
  与 test、v2 pack 无重合。与 v2 合并后每 epoch 996 题（Airline 410、Retail 337、Telecom 249），batch 128 时 8 步；配置覆盖项见
  `_artifacts/AReaL/codex-sao-math/tau2-telecom-pack-20261018/REPORT.md`。
- **状态：§5 的逐题审进行中**（pueue 620–622，`tau2-telecom-pack-20261018/review/`），结论出来之前不用于训练。
