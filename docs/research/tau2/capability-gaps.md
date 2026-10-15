# 模型能力缺口与训练期问题 case study

> 来源：原 README 第 14、15.1、15.5、15.7、15.8 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 Qwen3.5-4B 的能力缺口与训练期问题 case study（2026-10-11）

本节的对象是 Qwen3.5-4B 在官方 base 集上的表现。当前候选训练模型 Qwen3-4B-2507 的能力缺口在 §2.2，按最新结果维护；§1.3
的训练期问题与模型无关，仍然适用。

目的：区分两件事——未训练的 Qwen3.5-4B 自身的能力缺口（每类该用 RL、加数据还是 harness 解决），以及在这个交互式场景做 RL
时训练信号本身的问题。只用已有记录，没有新采样或付费调用。

数据：fixed v2 的 step 0（[step0-evaluations.md §4](step0-evaluations.md)）为主，official / fixed
v1 的 step 0 交叉核对；训练期信号用 2026-09-27 GRPO formal-v2 的 3,088 条 T=1 rollout（386 组 ×
8，official 评分）。SAO 只落盘了 step 0 的 126 条、每题 1 条，做不了组内分析。

分工：Airline、Retail、训练信号各一个 agent，另一个做文献调研。我抽查了每份报告的关键证据 （airline:44 t0、airline:45
t0–3、retail:37 t0、retail:20 t0、GRPO 组内首句用户消息），都与报告一致；其余标注没有逐条复核。

产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-capability-gap-20261011/`
（`BRIEF.md` 是分类口径；`out/{airline_gap,retail_gap,train_signal,lit_scan}.md` 是各自报告，`.jsonl`
是逐条记录； `out/pass_table.json` 是每题在全部运行里的通过数）。

### 1.1 结论

1. **Airline 的头号缺口是"用转人工代替拒绝"（M2a）：69 条、25 题，占 fixed v2 step 0 的 Airline 完成 episode（198
   条）的 34.8%。**64/69 条 agent 先正确列出了不能取消的理由， 然后调用 `transfer_to_human_agents` 而不是直接拒绝；9
   条用户明确说过不要转。53/75 条引用了 policy 第 141 行 "already been flown … transfer is needed"，52/69
   条说"API 不能取消"，把第 149 行 "The API does not check…" 读反了。 所以这是**行为选择**问题，不是资格推理问题。去掉 8
   条"用户先发 `###TRANSFER###` 而侥幸得分"后，Airline 真实 pass^1 约 0.31（fixed v2 严格计分的 0.350 再扣
   8/200）。
1. **Retail 的缺口不是长程或多实体**。gold 写操作数、订单数与通过率的 Spearman ρ 为 +0.17 / +0.15；只算模型失败时， 1
   个写操作的题失败率 11.2%，≥3 个的 6.0%。有区分度的是：按约束挑选（10.2% vs 4.0%），以及误解工具语义加报错后乱补救。
1. **新发现的一类：工具语义误解（M14）+ 报错后静默改参数重写（M15）**。例：airline:44 认为没有工具能查航班时刻
   （`search_direct_flight` 其实返回），3/4 trial 如此；retail:37 t0 以为一单只能调一次 modify，把不变的 item
   也填进去， 报错后不问用户就把背包换成更贵的款（\[30\]→\[32\]）。M15 在 T=1 失败中占 6.9%、成功中 1.4%。
1. **几乎所有失败过的题都在别的运行或 T=1 中成功过。**真正"从来做不对"的只有 airline:7（benchmark）、45（每次推理都对，然后转人工）、
   47（保险只覆盖健康/天气，0/8，T=1 1/18）。也就是说，按 4B 的现状，大多数缺口是**稳定性**问题（RL 能磨），不是能力缺失。
1. **训练信号里只有约 58% 的组内对比在教真本事**（95% CI 46–73%；Airline 88%，Retail 只有 42%）。其余：用户模拟器失败 29%、
   gold 错或歧义 10%、转人工骗分 3.5%。
1. **约 66% 的负梯度落在没有导致失败的 token 上**。人工定位的决定性错误轮中位数是第 9/15 轮，加上噪声失败的全部 token。

### 1.2 模型能力缺口与补救途径（fixed v2 step 0，按首要原因）

| 机制                                          | Airline pp | Retail pp | 补救                                        | 依据                                                                                                        |
| --------------------------------------------- | ---------- | --------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| M2a 用转人工代替拒绝                          | 34.8       | 0.9       | R-RL（用 fixed reward）+ R-HARNESS          | 推理已正确，只差最后一步的动作；T=1 下多数题有组内对比。接近 0 的题（27、39、43、47、49）用转人工门控更便宜 |
| M14/M15 工具语义误解、报错后静默重写          | 2.0        | 3.5       | R-HARNESS + R-RL                            | 工具描述写清返回字段、自动剔除 old==new 的项；成功样本会回头问用户，组内有对比                              |
| M1 政策资格规则（保险覆盖原因、同款换货）     | 4.5        | 0.7       | 43/44/18 R-RL；47 R-DATA 或资格检查器       | 47 是唯一稳定做不对的能力缺失                                                                               |
| M5 支付/退款路由                              | 3.0        | 1.8       | R-RL；airline:21 R-HARNESS                  | 成功样本都按原支付方式退                                                                                    |
| M4 多实体、M8 遗忘已读事实或子目标            | 3.0        | 1.3       | R-HARNESS（已读事实 + 待办写操作的 ledger） | airline:42 唯一的成功是用户替 agent 推理的                                                                  |
| M7 枚举/比较、M6 算术                         | 1.0        | 1.1       | R-HARNESS                                   | retail:20 列出 261.11 却选 258.71，3 个 trial 一样；`calculate` 每 200 条只用 4–11 次                       |
| M11 先写后报价                                | 2.0        | 0.2       | R-HARNESS                                   | 写前自动算差价并要求明确 yes                                                                                |
| M3/M12 写完就转人工、不回答追问               | 0.5        | 1.1       | R-RL                                        | T=1 成功率 > 90%                                                                                            |
| 非模型：benchmark、用户模拟器、转人工规则误伤 | 7.5        | 6.4       | R-REWARD                                    | 见 §1.4                                                                                                     |

R-DATA 的需求很小：只有 airline:47 一类，以及 airline:44 这种训练集没覆盖的工具用法。加数据的主要理由仍是
[training-data.md §1](training-data.md) 的"训练题太简单"， 不是补能力。

### 1.3 训练期问题（交互式 RL 本身，与能力缺口分开）

按对训练的危害排序（P = 训练范式，B = benchmark/环境）：

| #   | 问题                                                              | 证据                                                                                                       | 现有手段能否解决                                                                                                                       |
| --- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | \[P+B\] 用户模拟器噪声占 29% 的组内对比（API 用户），T=0 也消不掉 | 29% 的组内对比是"正常 agent vs 用户 STOP/偏离"，Retail 42%；同组 8 条的首句用户消息 386/386 组都不全相同   | 动态采样反而保留这些组（噪声恰好让组变成混合组）；fixed 只修 gold；用户 T=1 会更糟                                                     |
| 2   | \[P\] 用户 STOP 惩罚"写前确认"                                    | 5 条重复确认后被 STOP 的失败拿负优势，不确认的样本拿 1                                                     | 都不针对。reward 与 policy 直接冲突，训练久了会学会少确认                                                                              |
| 3   | \[P\] 序列级优势的信用分配                                        | 约 66% 负梯度落在非因果 token；决定性错误轮中位 9/15                                                       | 都不改变。需要 turn 级信用；SAO critic 的 EV 在 0 附近（Airline 0 到 +0.18，Retail 约 0，[critic/](critic/README.md)），目前没证明有用 |
| 4   | \[B\] official 评分奖励转人工                                     | 应拒绝任务里转人工样本净优势 +14.9；占 Airline T=1 成功的 45%（198/510 条是骗分）                          | fixed 的转人工规则直接解决，但误伤 policy 要求转人工的已飞航段（airline:37、9，3.0pp）                                                 |
| 5   | \[B\] gold 错让组内排序反转                                       | 10% 的对比；retail:98（G68）6 条遵从指令的样本全得 0                                                       | fixed v2 已修 98（[benchmark-variants.md §1.2](benchmark-variants.md)）；AReaL-tau2-data 引入新题时会再出现                            |
| 6   | \[P\] 二值 reward 抹掉部分进度                                    | 225 条有 gold 写的失败中，44 条 DB 全做对（其中 16 条是 airline:7）、66 条做了一部分，与一个都没做的同得 0 | 按写操作给部分分能缓解，但会放大 gold 歧义                                                                                             |
| 7   | \[P\] 有效样本少                                                  | 平均每步 8 组只有约 2.3 个混合组、约 1.3 个有效对比组                                                      | 动态采样提高混合组比例，但带着同样比例的噪声                                                                                           |

4096 截断不是独立的失败原因：组内被截断与未截断的样本成功率无差异（+0.011 ± 0.095）。

### 1.4 可能的论文问题（结合文献调研，待核实）

文献调研只读了摘要，arXiv 编号未逐个打开确认，引用前要核实。

1. **用户模拟器噪声下的优势估计**（§1.3 第 1、2 项）。文献里有人度量过换用户带来的分数差（UserProxyBench 2609.38043，15.2 分），
   没看到在训练中处理用户 STOP/偏离的方法。候选做法：用户在 agent 请求确认后 STOP 时把样本按 infra 失败剔除；同一前缀对用户重采样取平均。
1. **turn 级信用分配**（第 3 项）。约 66% 的负梯度落在非因果 token 上，需要 turn 级信用；SAO critic 想证明有用，至少要打赢"把结局
   advantage 均匀分给各轮"这个零信息基线。
1. **转人工捷径与"拒绝并留在对话里"**（§1.2 M2a、§1.3 第 4 项）。step 0 的 fixed 评分下 Airline 仍有 34.8% 的
   episode 因此失败（M2a），official reward 还在强化它；fixed 规则是 硬规则，会误伤合规的转人工。一个按 policy 判断"该不该转"的
   reward 是空缺。
1. **部分进度 reward**（第 6 项）。按 DB 终态字段 diff 给分、并防止回滚刷分，文献调研里没找到做过的。

### 1.5 局限

- Airline 非转人工的 46 条失败读了约 40 条；Retail 69 条读了约 45 条，14 条按同题同路径规则核实。训练信号的 320 条标注只抽查了 2 组。
- "确认后立刻 STOP"的 pp 是可追回的上限。
- 训练期结论全部来自 GRPO、official 评分、旧数据；fixed v2 下训练信号的噪声比例需要新一轮训练后重算。
- 观测到的 Airline 转人工率从 epoch 1 到 2 下降 13.7pp（CI −22.5 到
  −5.0），但权重变化接近随机游走（[initial-audit.md §2.3](initial-audit.md)），原因未知，不能当作训练纠正了 hacking。
- Retail T=1 episode 计数在两个脚本里不同（1145 vs 994，后者按 `episode_id` 去重），没有查清。

## 2 Qwen3-4B-2507 的能力缺口（2026-10-11、10-12）

### 2.1 结论

1. **两个 Qwen3 的可提升空间都大得多**（[step0-evaluations.md §1](step0-evaluations.md)）：
   - 官方 fixed：Instruct 的 Airline / Retail mean@4 是 0.225 / 0.410，Thinking 0.245 /
     0.654，Qwen3.5 0.350 / 0.844。Telecom 上两个 Qwen3 都约 0.37，Qwen3.5 全对。
   - 筛选后的 AReaL v2：Instruct 0.593 / 0.787，Thinking 0.740 / 0.913，Qwen3.5 0.918 /
     0.976。AReaL Telecom：Instruct 0.380、Thinking 0.307，Qwen3.5 0.992。
1. **分数低主要是模型问题。** 用户在确认句里结束对话、转人工漏洞这两项评测因素扣除后，与 Qwen3.5 的差距仍有
   15–42pp（[step0-evaluations.md §6.3](step0-evaluations.md)）。AReaL case study 的 353
   个单元里只有 29 个是非模型原因，没有题错（§2.2）。
1. **缺口集中在多步执行**：Airline 和 Telecom 的成功率随 gold 写操作数、故障数下降，Qwen3.5 几乎不受影响（§2.2）。
1. **四个具体缺口**（§2.2）：
   - Telecom 选错线路：查非来电的线路，得出"账户没问题"，不充流量、不开漫游，最后转人工。官方和 AReaL 数据上都一样，Qwen3.5 从不出错。
   - Instruct 把工具调用写成正文、或没执行就声称已完成。
   - Instruct 的 Airline 政策规则和时间计算（24 小时取消、行李只能加不能减）。
   - Thinking 的 Airline 退款路由：不先问退款方式就执行降舱。
1. **对训练设置的启示**（由失败分类推断，未经实验验证）：
   - Instruct 的组内信号最多，但文本工具调用要先用解析容错或少量 SFT 处理；
   - 来电线路、日期差、资格检查适合放进 harness；
   - Telecom 故障多的题两个模型几乎全错，需要按难度挑题或示范数据 warm-up，示范应覆盖遍历完所有实体、按手册走完排障、不把转人工当退路。
1. **加速配置：沿用现有 FSDP 配置，不再加其他优化**（2026-10-11 决定）。
   - FSDP + FA2 + gradient checkpointing + packing 能直接跑通，在测过的组合里最快。
   - Ulysses、tree training、archon 也能启动，但都不更快（[infra-issues.md §2](infra-issues.md)）。

### 2.2 能力缺口（2026-10-12 更新：官方 base 集 + 筛选后的 AReaL 数据）

本节是 Qwen3-4B-2507 两个模型当前的能力缺口，以后对着最新结果改这一节。Qwen3.5-4B 的缺口分析在第 1 节，那是另一个模型，不再重复。

**数据**：

- 官方 base 集的 step 0（[step0-evaluations.md §6.2](step0-evaluations.md)），2026-10-11 的分析；
- AReaL pack v2 和 Telecom v1 的 step 0（[step0-evaluations.md §1](step0-evaluations.md) 表
  B），2026-10-12 新做的 case study：353 个"模型 × 题"单元，包括 v2 上两个模型全部 4 次全错的题（Instruct
  66、Thinking 28），以及随机抽的有对有错题（v2 上 54 / 52 题，Telecom 上每个模型 40 道全错、30 道有对有错），Qwen3.5 的 13
  个失败单元作参照。
- 分工：7 个 Codex（gpt-6.1-sol）分片逐条读轨迹，找出"之后 reward 不可能再是 1"的那一步，按第 1 节的 M 码分类，并与通过的 trial
  或其他模型对比；我汇总，并用脚本在全部 episode 上核对频率。
- 产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-capability-gap-20261012-qwen3/`：`BRIEF.md`（给
  Codex 的背景与分类口径），`units.jsonl`，`out/case_*.jsonl`（逐条结论）和
  `case_*.summary.md`，`out/tables.json`、`out/gap_table.json`（`scripts/tables.py`、`gap_table.py`
  生成）。

**按 gold 写操作数 / 故障数的成功率**：

| 数据                  | 模型                          |             1 个写 |             2 个写 |            ≥3 个写 |
| --------------------- | ----------------------------- | -----------------: | -----------------: | -----------------: |
| 官方 Airline（fixed） | Instruct / Thinking / Qwen3.5 | 0.31 / 0.62 / 0.67 | 0.17 / 0.12 / 0.62 | 0.07 / 0.11 / 0.39 |
| 官方 Retail（fixed）  | Instruct / Thinking / Qwen3.5 | 0.52 / 0.72 / 0.82 | 0.30 / 0.52 / 0.90 | 0.19 / 0.55 / 0.92 |
| AReaL Airline v2      | Instruct / Thinking / Qwen3.5 | 0.66 / 0.79 / 0.96 | 0.41 / 0.54 / 0.81 | 0.32 / 0.61 / 0.89 |
| AReaL Retail v2       | Instruct / Thinking / Qwen3.5 | 0.80 / 0.95 / 0.95 | 0.79 / 0.89 / 0.99 | 0.78 / 0.90 / 0.99 |

| Telecom 注入的故障数 |    1 |    2 |    3 |    4 |    5 |    6 |    7 |    8 |   ≥9 |
| -------------------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 官方，Instruct       |    — | 0.59 | 0.38 | 0.51 | 0.37 | 0.16 | 0.09 | 0.06 | 0.04 |
| 官方，Thinking       |    — | 0.60 | 0.48 | 0.51 | 0.31 | 0.02 | 0.00 | 0.00 | 0.00 |
| AReaL v1，Instruct   | 0.82 | 0.75 | 0.55 | 0.63 | 0.26 | 0.24 | 0.19 | 0.20 | 0.12 |
| AReaL v1，Thinking   | 0.88 | 0.74 | 0.49 | 0.43 | 0.15 | 0.14 | 0.09 | 0.03 | 0.10 |

Qwen3.5 在两份 Telecom 数据上每一档都接近 1。AReaL 的 Retail 题与写操作数无关，Airline 和 Telecom 随步数增加明显下降。

**缺口分类**（AReaL case study，按首要原因；数字是抽样单元数，不是总体比例；"全错/有对有错"分开计数，格内写合计）：

| 缺口                                   | 代码        | Instruct Airline | Thinking Airline | Instruct Retail | Thinking Retail | Instruct Telecom | Thinking Telecom | 主要补救         |
| -------------------------------------- | ----------- | ---------------: | ---------------: | --------------: | --------------: | ---------------: | ---------------: | ---------------- |
| 选错对象（Telecom 里是选错线路）       | M4          |                2 |                5 |               4 |               1 |               18 |               21 | R-HARNESS / R-RL |
| 子任务没做完、排障没走完               | M8、T1      |                4 |                1 |               7 |               2 |               13 |               15 | R-RL             |
| 账户侧动作缺失（充流量、开漫游）       | T3          |                — |                — |               — |               — |               17 |                8 | R-RL / R-DATA    |
| 用转人工收尾                           | M2、M2a、M3 |                2 |               12 |               1 |               1 |                9 |               11 | R-RL             |
| 政策资格规则                           | M1          |               22 |                4 |               0 |               0 |                3 |                1 | R-HARNESS / R-RL |
| 时间与算术、枚举比较                   | M6、M7      |               16 |                8 |               1 |               0 |                0 |                0 | R-HARNESS        |
| 支付 / 退款路由                        | M5          |               10 |               17 |               0 |               1 |                0 |                0 | R-RL             |
| 参数 grounding、工具语义、读错设备输出 | M9、M14、T2 |               13 |                0 |              10 |               3 |               10 |               14 | R-RL             |
| **声称做了但没执行（新）**             | M16、M17    |               11 |                2 |               2 |               2 |                0 |                0 | R-RL             |
| 该问没问、确认流程                     | M10、M11    |                4 |                5 |               0 |               0 |                0 |                0 | R-RL             |
| 非模型：用户模拟器、评分               | N1、N3      |                6 |                9 |               5 |               7 |                0 |                0 | R-REWARD         |
| 单元合计                               |             |               90 |               63 |              30 |              17 |               70 |               70 |                  |

**主要发现**：

1. **Telecom 的头号缺口是选错线路，这是新发现**。客户 C1001 名下有两条线路，来电号码 555-123-2002 对应 L1002；两个 Qwen3 常去查
   L1001。L1001
   的流量没用完、漫游正常，模型据此判断"账户没问题"，于是不充流量、不开漫游，最后转人工。一条错误连起了上表的选错对象、账户侧动作缺失、用转人工收尾三类。
   - 用脚本在全部 episode 上核对，账户工具只碰过非来电线路的 episode：

     | 模型     | AReaL Telecom v1            | 官方 Telecom          |
     | -------- | --------------------------- | --------------------- |
     | Instruct | 331 / 1044（其中 284 失败） | 118 / 456（104 失败） |
     | Thinking | 547 / 1044（448 失败）      | 205 / 456（177 失败） |
     | Qwen3.5  | 0                           | 0                     |

   - 例：instruct:telecom_13 t0 先 `get_customer_by_phone("555-123-2002")`，接着
     `get_details_by_id("L1001")`、`enable_roaming(C1001, L1001)`，流量没处理就转人工（我核对过轨迹）。thinking:telecom_118
     唯一通过的 trial 1 回头改查 L1002，看到 15.1/15 GB 后才充了流量。

   - 补救：harness 里按来电号码直接给出线路，最便宜；RL 也可行，Qwen3.5 每次都选对。

   - 这也解释了官方集上的统计：需要充流量的题，`refuel_data` 只调用了 Thinking 6/197、Instruct 48/200 次（Qwen3.5
     200/200）。失败里有相当一部分不是不会充，而是在错的线路上看到"不需要充"：只碰过错线路的 episode 占 Instruct 失败的 44%（284 /
     647）、Thinking 的 62%（448 / 723）。AReaL v1 上同样的统计是 Instruct 66 / 408、Thinking 14 /
     408。
1. **"声称做了但没执行"是新的一类（M16 / M17）**，主要是 Instruct：
   - M17：把工具调用写成正文里的 JSON，`tool_calls` 为空，工具没执行。例：instruct:airline_617 t2 第 16、18 条消息是
     `{"name": "update_reservation_flights", "arguments": …}` 纯文本（我核对过）。
   - M16：没调用写工具，或写操作报错之后，仍告诉用户"已成功修改"。例：thinking:airline_439 说"升舱已确认"，全程没有
     `update_reservation_flights`；instruct:retail_432 换货报 "Payment method not found"
     后照样回复"已完成"。
   - 脚本核对 M17：Instruct 在 v2 有 53 条 episode 出现（43 条失败），在官方 fixed 有 16 / 656
     条（全部失败）；Thinking 两边各 1 条。旧版"工具调用格式没问题"只查了空消息，不准确。
   - Instruct 在 Telecom 上还会把用户手机上的工具（`check_network_status` 等）写成 `<tool_call>`
     文本自己调用，1044 条里 449 条出现过；它随后多会改口让用户去做，所以不一定致命，单独记一笔。
1. **Instruct 的 Airline：政策规则和时间算术**。4 次全错的 59 题里，M1 17 题、M6/M7 15 题。典型是 24
   小时取消规则：instruct:airline_179 把 24h15m 前订的预订当成"24 小时内"取消了；airline_361 三个模型（包括
   Qwen3.5）都把 18h50m 前的预订算成"超过 24 小时"。其次是行李：instruct:airline_430 把托运行李从 6 件减到 3 件，policy
   规定只能加不能减。补救：时间差和资格检查交给 harness，比让 4B 模型自己算可靠。
1. **Thinking 的 Airline：支付 / 退款路由**（17 个单元，4 次全错的 26 题里 8
   题）。降舱前不先问退款去哪，直接用原支付方式执行，退款打到礼品卡，用户随后要求退到信用卡时已经改不回来。例：thinking:airline_669、1008、40，4
   次都是同一种失败。Qwen3.5 和 Instruct 的通过 trial 都先问退款方式。另一类是转人工（12 个单元），与官方集上的结论一致。
1. **Retail 基本是稳定性问题**：Thinking 在 v2 Retail 已经 0.913。剩下的失败主要是只做完部分子任务（Instruct 228 条失败里
   88 条、Thinking 93 条里 54 条）和用户在确认句里结束对话（36 / 17 条，非模型原因）。
1. **没有发现题目缺陷**：353 个单元没有一个判为题错（N2），非模型原因的 29 个都是用户模拟器提前结束或评分细节。这与
   [training-data.md §5](training-data.md) 的筛选结果一致。
1. **补救的分布**：多数单元判为 R-RL（同一题其他 trial 或其他模型能做对）；判为 R-HARNESS 的集中在选错对象、政策规则和时间算术；R-DATA 只有
   18 个单元，都是 Telecom 故障很多、四个模型都做不完的题。

**仍然成立的旧结论**（官方 base 集，2026-10-11）：

- 列表没遍历完：Thinking airline:22 t1 只查了第一个预订，思维链写 "picked the first one"；retail:54 t1 6
  个订单只查了 5 个。
- Instruct 不查证就行动：retail:101 t1 把 policy 里的格式示例 `#W0000000` 当订单号去查；Retail 写操作失败里参数错 81
  条，主要是 `new_item_ids`。
- 该拒绝的 Airline 题（0 个 gold 写操作）三个模型都只有 0.11–0.24，与模型代际无关，同 §1.2 的 M2a。
- Thinking 的 policy 推理不完整：airline:32 t0 只记住 "basic economy 不能改"，没想到可以先升舱再改签。

**对训练的含义**：

- **Instruct 做 RL 的信号最多**（v2 上 54% 的题有对有错），但"写成文本的工具调用"要先处理：它让该成功的 episode
  直接失败，属于格式问题，可以用工具调用解析的容错或少量 SFT 解决，不必交给 RL。
- **Telecom 的选错线路**是一条确定性的子技能，harness 里给出来电对应的线路即可。不加 harness 的话，RL
  要能学会它，前提是组内有选对线路的样本：Instruct 有 135 条 episode 碰过 L1002，Thinking 只有 55 条。
- **Airline 的时间和资格计算**放进 harness（日期差、资格检查器），比让模型在 RL 里学更省。

**局限**：case study 的单元是抽样，表中数字只说明类别的相对大小；每个单元以第一条失败 trial 为准，164 个单元的几次失败方式不同。我核对了
instruct:airline_617、instruct:telecom_13 两条轨迹，以及选错线路和 M17 的全量统计，其余 Codex 结论没有逐条复核。

### 2.3 后续：在 AReaL-tau2-data 上训练 Qwen3

**依据。** [training-data.md §3](training-data.md) 的 813 题对 Qwen3.5-4B 太容易：mean@4
0.912，temperature 0 下 78% 的题 4 次全对。两个 Qwen3 模型在 fixed 上比 Qwen3.5 低
15–44pp，这份数据的难度对它们可能正合适。

**已测（2026-10-12）。** 改为在筛选后的 v2（648 题）和 Telecom v1 上评测，结果在
[step0-evaluations.md §1](step0-evaluations.md) 表 B。Instruct 在 v2 上 54% 的题有对有错，难度合适。

**训练前要处理的两点：**

- [step0-evaluations.md §6.3](step0-evaluations.md) 的"用户在确认句里结束对话"会让 Qwen3 产生更多假负样本；
- Qwen3 critic 路径还没验证（[infra-issues.md §2](infra-issues.md)）。

### 2.4 局限

- 失败分类靠规则抽取（`aggregate.py`、`write_diff.py`）。人工精读了约 10 条轨迹，主要类别各 1–2 条，没有逐条标注。
- temperature 0 同配置重测约有 3pp
  波动（[step0-evaluations.md §3](step0-evaluations.md)）。本节比较的差距都在 15pp 以上，不受影响。
- Actor probe 只跑了 3 步、用的是合成数据，吞吐只能做相对比较。
