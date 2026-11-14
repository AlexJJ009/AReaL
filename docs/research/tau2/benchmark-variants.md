# benchmark 变体与官方 runner 对照

> 来源：原 README 第 7、10.5、11 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 official 与 fixed 两个变体（固定版本定义）

决策记录见 [ADR 0010](../../decisions/0010-tau2-official-and-fixed-benchmark-variants.md)。

- `official` 是 tau2-bench `b7ea907` 原样，用于和榜单比较。
- `fixed` 用于训练。

早期比较同时报告 official 与 fixed。用 `econfig.benchmark_variant` 选择变体：GRPO 和 critic production
配置默认 `official`，PPO 配置默认 `fixed`；各次实验的题集和比较范围见对应运行记录。

### 1.1 fixed 包含什么

| 层           | 内容                                                             | 来源                                                                        |
| ------------ | ---------------------------------------------------------------- | --------------------------------------------------------------------------- |
| 数据（上游） | Airline 44 题、Retail 105 题的 reward_basis；替换 retail 64、105 | PR #411、#571，pin 的 head commit 记在 `examples/tau2/benchmark_fixes.json` |
| 数据（我们） | 27 道题的补丁（Airline 5、Retail 22），见 §1.2                   | 2026-10-10 Retail 核查及 2026-10-12 用户模拟器轨迹核查                      |
| 环境         | `modify_pending_order_items` 按每个 item 自己的 variant 定价     | PR #322                                                                     |
| 评分         | 两条转人工规则，见 §1.3                                          | 我们                                                                        |
| 训练集       | Retail train 去掉与 test 重复的 5 题，见 §1.4                    | 我们                                                                        |

- **当时的上游核查**：在 §2 记录的范围内，`b7ea907` 之后的 main 未改动 Airline 和 Retail 的数据、工具或评估器。
- **#580 没有作为代码合入**。它的作用是：gold 写操作执行抛错时，DB 判定不算通过。fixed 下，`check_gold_replay.py` 没有查出抛错的
  gold 写操作，合入它不会改变任何分数。
- **社区里不采用的修改**：
  - Amazon verified 对 49、18、91、21 的改动：49 的新 gold 不在订单里，18、91 的目标商品缺货，21 把 product id 当成了
    item id；
  - #561：它会替换 `pyproject.toml`；
  - #392 中把用户模型默认换成 gpt-5.5、修改转人工停止条件的部分；
  - AGI-Eval revised：只改了用户 prompt。

### 1.2 我们自己的 Retail 补丁

依据是两次 step 0（official 和 fixed v1，各 456 条 Retail episode）的失败轨迹：逐条比较 agent 实际的写操作和 gold
的写操作。补丁和每题的理由记在 `benchmark_fixes.json` 的 `patches` 里。社区调研的中间结果存档在
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-retail-audit-20261010/`。

| 问题                                                                                             | 题号（划分）                                 | 补丁                                                                                                                                | 两次运行中直接因此失败的 episode |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 指令没写支付或退款方式，gold 用的是原支付方式 PayPal                                             | 16、63（train）；60、61（test）              | 用户指令加一句：被问到时选 PayPal                                                                                                   | 3                                |
| 指令里的 "the other card" 指向 gold 没用的那张卡                                                 | 98（train）                                  | 改为"尾号 8484 的 Mastercard"                                                                                                       | 4                                |
| 指令没写取消原因，gold 是 "no longer needed"                                                     | 59、66（train）；38（test）                  | 用户指令加一句：被问到原因时说 no longer needed                                                                                     | 10（38 题 8 条全挂）             |
| 64（#571 的替换题）：预算内有两款 4K 防水相机，gold 选了较便宜的黑色，agent 选了与原机同色的银色 | 64（test）                                   | 指令加一句：多款符合时选最便宜的                                                                                                    | 4（fixed v1 中 4/4）             |
| 什么都不做也得 1 分                                                                              | 25、57（train）；65（test）                  | 25：gold 加上转人工（用户要求退到 Amex，账户里没有 Amex），并加 COMMUNICATE 检查追踪号；57、65：改为 DB+ACTION，要求 agent 查过订单 | 25 有 2 条转人工被清零           |
| gold 里的转人工是从 25 抄来的：summary 写的是 Amex 退款，26 的用户根本没提                       | 26（test）                                   | 删掉这个 gold 转人工                                                                                                                | —                                |
| NL 断言只是重述写操作，而 judge 只看对话文本                                                     | 105–107、109、110、113（train）；111（test） | reward_basis 只保留 DB                                                                                                              | 2（110 只因这条断言失败）        |

**另两处核查**：

- 82 不是指令矛盾。用户在两个订单里各有一台平板，较贵的那台是用 gift card 付的，退款回不到信用卡。按指令，这时应整单退到 gift card，这正是 gold
  的做法。
- 在 fixed v1 中，10、12、50 已经按 DB+ACTION 计分，必须转人工才能得分，不属于"什么都不做也得分"。

**不修的**：

- 5–9 等 23 道题的 known_info 只给了 user id 和邮编（#337/#356）。v2 核查未发现认证失败；v3 新证据确认 retail:5
  卡在缺失姓名，现只补该题姓名，其他题未扩大修改。
- 20、18、79 等题是 agent 选错了变体，gold 是对的，属于正常难度。
- 30–32、54、69 的取消原因能从指令推出来。v3 新证据显示 retail:55 的财务困难并不排除合法的 ordered by mistake；现明确使用 gold
  的 no longer needed。
- 40（test）和 99（train）的指令只说"信用卡"，用户却有两张。两次运行中没有因此失败的 episode。
- airline:7 的金额有歧义（#495），Airline `payment_history` 的 hash 对列表顺序敏感（#514）。

#### 1.2.1 fixed-v3：用户模拟器轨迹确认的修补（2026-10-12）

依据 [user simulator case study 第 6 节](user-sim-case-study-20261012/REPORT.md)，逐条核查了
`user-sim-20261012/fixed-v3/evidence.json` 的 39 条轨迹。补丁清单逐题记录轨迹路径和消息编号（从 0
开始）。原则：任务说明只补缺失的一两句，原目标和 policy 不变；一个按 policy 正确行事的 agent 在修改前后都应该能通过。

| 题号             | 问题                                                  | 改动                                                     | 层                  |
| ---------------- | ----------------------------------------------------- | -------------------------------------------------------- | ------------------- |
| airline:29       | 新预订是否买保险、用哪张卡没写，gold 只认一种         | 用户说明补：新预订不买保险，用原预订的 Visa 尾号 6521    | 用户说明            |
| retail:5         | known_info 没给姓名，gold 认证要求姓名                | known_info 补 Mei Kovacs                                 | 用户说明            |
| retail:55        | 取消原因没写，gold 只认 no longer needed              | 被问到原因时说 no longer needed（同 §1.2 的 38、59、66） | 用户说明            |
| airline:37       | NQNU5R 已飞，policy 第 141 行要求转人工，gold 没有    | gold 在升舱后加转人工                                    | gold                |
| airline:26/31/48 | 拒绝类题 DB 不变，空轨迹也能得分（同 §1.2 的 57、65） | ACTION 要求查询目标预订；所有引用轨迹都已有这一步        | gold / reward_basis |

**审查后没有采纳的修改**（Codex 初版 `fab5e7b9` 提出，Claude review 后回退）：

- **9 道题追加"请求转人工后要等真正转接再发 ###TRANSFER###"**（airline 13/26/31/32/48，retail
  27/32/65/94）。tau2 的全局用户规则 `simulation_guidelines.md` 第 15 行已经规定被转接后才发
  TRANSFER，用户提前发是模拟器违反协议（case study 的 U7a），不是题目问题。只在 9 道题里重复这条规则，其余题不受约束，等于针对一批题打补丁。提前
  TRANSFER 的影响记在 [user-simulator.md §2](user-simulator.md) 节的已知偏差里。
- **retail:45 让用户在只有一个选项时追问其他选项**。数据库里有 3 款可用的 canister 吸尘器，agent
  只列一款属于枚举不全（M7），这道题本来就在考这一点；case study 里的 B1 标注应改为 agent 失败。

`apply_transfer_rule` 没有改。只改用户说明的题，旧轨迹不能验证新说明，需要在 v3 上重新采样。airline:37 不改用户说明，离线重打分时 8
条完成升舱和转接的旧轨迹从 0 变 1，1 条没有转接的从 1 变 0。Codex 的逐题报告（含被回退的部分）见
`_artifacts/AReaL/codex-sao-math/user-sim-20261012/fixed-v3/REPORT.md`。

**待修：airline:44（2026-10-13 v3 case study 发现，未改）。** 任务说"含中转 ≤3 小时的航班才升舱"，KC18K6 是 HAT300
MSP 06:00 → HAT215 CLT 13:00，含中转 7 小时，gold 却升舱 KC18K6；NL 断言 "Agent updates KC18K6 to
business" 和总价 1380–1390 也按升舱算。GLM t2 按任务正确排除 KC18K6 后得
0（`user-sim-case-study-v3/REPORT.md` 6.1）。最小改法是删掉 KC18K6 的升舱 action 和对应断言、重算总价区间，需要用
gold replay 验证。

### 1.3 转人工规则

两条规则都在官方 evaluator 之后执行（`benchmark_variant.apply_transfer_rule`）：

1. agent 转了人工而 gold 没有转，记 0 分。例外：Retail 中 gold 有写操作的题。
1. gold 有转人工而 agent 没有转，记 0 分。gold 含转人工的题是 Airline 13、37，Retail 10、12、25、50。

**为什么只在 Retail 设例外**（依据 fixed v1 的 step 0）：

- Retail 有 17 条原本成功的 episode 被转人工规则清零，其中 13 条在有 gold
  写操作的题上（1、32、43、46、47、51、53、93、94、101）。 抽读到的都是写操作已经完成后，用户提出 agent
  做不了的要求，比如"收不到邮件，拿不到退货单""能不能加急退款"，agent 才转人工。这符合 Retail policy 的"超出能力范围时转人工"。
- Airline 有 59 条被清零，其中 12 条在有写操作的题上（24、32、35、37、39）。抽读到的是在回避拒绝：35 的用户明确说 "I don't want
  to be transferred"，agent 应该拒绝全额退款，却转了人工；当时将 37、39 归为同类。所以 Airline 不设领域级例外。v3 逐条核查纠正了
  37：NQNU5R 已飞，policy 第 141 行明确要求转人工，故补该题 gold；35、39 的处理不变。
- 代价：两个领域的规则不一致；例外的依据是抽读，不是 evaluator 能检查的条件。

**转人工漏洞的量级**（Qwen3.5-4B step 0、DeepSeek 用户、base 集 164 题，超出上下文的 episode 记 0 分）：Airline 在
official 上 0.745（[step0-evaluations.md §3](step0-evaluations.md) 的 official 运行），在 fixed
v2 上 0.350，差约 39pp。其中转人工规则清零了 200 条 Airline episode 里 73 条原本成功的（36.5pp），其余约 3pp 是 #411 的
NL 断言计分和运行间噪声。别的文档里出现的 31.5、34.8、约 30pp 等数字量的是别的东西（按各文档说明），不要和这个差距直接比较。

### 1.4 训练集去重

Retail train 的 72、96、37、41、67 与 test 的 71、97、36、42、68 是同一用户，gold 写操作完全相同（逐条核对），因此从训练中去掉。

- test 不变。去重后 train 是 Airline 30 + Retail 69 = 99 题。
- 实现方式是 `get_tau2_dataset(variant="fixed")` 在训练集里去掉这些题。benchmark 本身不变，`--task-set base`
  仍然评全部 164 题；报告中另给 `retail/train_pool`（69 题）。
- 近似重复的题没有去掉，因为 gold 不同：31↔32、93↔94、6/7/8↔5/9、63↔62、13/14↔12、34↔33、25↔26。

### 1.5 生成与校验步骤

1. **生成数据目录**：`python scripts/tau2/build_fixed_data.py <official data> <out>`，生成的 digest
   必须等于 `benchmark_variant.TASK_FILE_SHA256["fixed"]`。

   - 当前目录：`_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data`。
     Airline digest 为
     `25086da158eedb97aee3bce2bc131d6acea094267bee6fd1c71933f74d90ef74`； Retail 为
     `5b7a8fa30a7615b4e8f8d1fc21a66fb85ba6d337a06ce50f4a9565258098301f`。
   - v2 目录 `…-fixed-v2/data` 保留且未改写。使用历史代码 `eb09aa24`，设置 `TAU2_DATA_DIR`
     为该目录、`benchmark_variant=fixed` 即可复现原实验；当前 v3 代码会拒绝 v2 digest。 v2 Airline digest 为
     `ec04aec07306c8fe640b40905e1e7b9b7c88e55132cdcf9cc5448daa1798c5a4`； Retail 为
     `03a79539bff56afefcad29eece5bcf55884a3bda67bd60b862134e64ea4c005d`。 要重建 v2，用
     `eb09aa24` 的构建脚本和补丁清单，在新输出目录执行相同 build 命令。
   - 更早的 `…-fixed-pr411-pr571/data` 保留，只用于复现
     [step0-evaluations.md §3](step0-evaluations.md)。
   - Telecom 三版均链接官方文件，digest 为
     `37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4`。

1. **gold
   重放**：`TAU2_DATA_DIR=<fixed> python scripts/tau2/check_gold_replay.py fixed`，要求没有抛错的写操作。当前是
   0 条；另有 16 条读操作失败，读操作不影响 DB。

1. **补丁题打分检查**：运行本目录的 `scripts/official-runner/check_patched_scoring.py`。

   - 做法：对 27 道补丁题（Airline 5、Retail 22）各构造三条合成轨迹（空轨迹、gold、去掉转人工的 gold），用真实 evaluator
     打分，NL 断言打桩为通过。
   - 要求：gold 得 1 分，空轨迹得 0 分。v3 全部满足；airline:37 的 gold 去掉转人工得 0。
   - 这个检查能失败：换到上一版数据上运行，会报出 25、57、65。

1. **单元测试**：

   - `tests/test_tau2_benchmark_variant.py`：补丁操作、两条转人工规则、Retail 例外、固定的 digest；
   - `tests/test_tau2_official_runtime.py` 全文件。

   v3 数据目录下两份测试共 46 个通过。task:37 的正向/反向检查绑定真实补丁；临时删掉新增 gold 转人工，两个用例都会失败。

   分别破坏反向转人工规则、Retail 例外和训练集去重，对应的测试都会失败。

1. **启动校验**：评测、训练和 critic 启动时都核对 `TAU2_DATA_DIR` 的 digest，与变体不符就直接报错。

## 2 官方和社区是否已经修过（2026-10-10 调研）

**官方没有修。**

- 我们 pin 的 `b7ea907` 与 2026-10-10 的 main 相比，`src` 和 `data` 的 diff 为空。这一条是调研 agent clone
  后 diff 得到的，我没有独立复核。
- Airline 和 Retail 的 `tasks.json` 只在 v1.0.0 改过一次，即官方的
  [τ³ task fixes](https://taubench.com/blog/tau3-task-fixes.html)（Airline 27 题、Retail 26
  题，来自 Amazon 的 τ-Bench Verified），这批修复已经包含在 pin 里。
- Airline 的 `reward_basis` 从来没有包含 NL_ASSERTION。维护者在
  [#224](https://github.com/sierra-research/tau2-bench/issues/224)
  表示会处理，[RFC #129](https://github.com/sierra-research/tau2-bench/issues/129) 计划让
  no-action agent 得 0 分，两者都没有实现。

**社区修复**：以下 PR 全部 open，未合并。

| 来源                                                                               | 修什么                                                                                       | 覆盖 [failure-taxonomy.md §2](failure-taxonomy.md) / [failure-taxonomy.md §3](failure-taxonomy.md) 的哪项 | 核实情况                                                                        |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [PR #411](https://github.com/sierra-research/tau2-bench/pull/411)                  | Airline 23 道不写 DB 的拒绝题改为 `[DB, NL_ASSERTION]`，按实际 criteria 修剪 Retail 的 basis | 转人工假阳性，间接覆盖                                                                                    | 改动已读；断言里没写"不得转人工"，judge 能否拦住没人测过                        |
| [PR #322](https://github.com/sierra-research/tau2-bench/pull/322)                  | `modify_pending_order_items` 按每个 item 自己的 variant 写价格                               | retail:100 环境 bug                                                                                       | 本地验证：打补丁后两种顺序 hash 相同、价格正确                                  |
| [PR #571](https://github.com/sierra-research/tau2-bench/pull/571)                  | retail:105 换成礼品卡付得起的商品；删去 retail:64 中执行不了的一步                           | retail:105、retail:64                                                                                     | 本地验证：原 gold 执行抛错，新 gold 执行成功                                    |
| [PR #580](https://github.com/sierra-research/tau2-bench/pull/580)                  | gold 的写操作执行抛错时，不再给 DB=1                                                         | 防止同类 gold 不可行的题                                                                                  | 改动已读                                                                        |
| [KnowWhenToHandOff](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2) | 训练 reward 对不必要的转人工 −0.5、漏转 −1.0，基于 tau2 train split                          | 转人工（只改训练）                                                                                        | 只读了 model card：Qwen3-4B 过度转人工从 46% 降到 10%，成功率持平；代码仓库 404 |
| [ABC](https://arxiv.org/abs/2507.02825) patch                                      | 原版 tau-bench：不写 DB 的题要求调用指定的读工具                                             | do-nothing agent                                                                                          | 没有移植到 tau2；拦不住"先读再转人工"                                           |

Amazon 的 [tau2-bench-verified](https://github.com/amazon-agi/tau2-bench-verified) 比我们的
pin 旧，内容已经被官方吸收。AGI-Eval 的
[tau2-bench-revised](https://github.com/AGI-Eval-Official/tau2-bench-revised) 只改了用户
prompt。两者都没有修转人工问题。

**调研时没有任何来源修过的**（除 airline:7 和 #514 外，现已在 fixed 变体中由我们修复，见第 1 节）：

- 评测端针对转人工的直接规则；
- airline:7 的金额歧义（只有 issue
  [#495](https://github.com/sierra-research/tau2-bench/issues/495)）；
- retail:38 的取消原因、retail:60 的支付方式、retail:110 的空操作断言；
- Airline `payment_history` 的列表顺序
  hash（[#514](https://github.com/sierra-research/tau2-bench/issues/514)）。

另外 NL judge 本身有已报告的缺陷，例如
[#554](https://github.com/sierra-research/tau2-bench/issues/554) 中空回复会被判通过。把
NL_ASSERTION 放进训练 reward，这些缺陷会直接变成 reward 噪声。

**转人工规则的边界**（official 数据）：gold 动作含 `transfer_to_human_agents` 的只有 Airline 1 题（13）和
Retail 4 题（10、12、26、50）。Airline 有 23 题、Retail 有 7 题没有 gold 写操作。fixed 的现状见 §1.3。

分类脚本和逐条结果：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/audits/tau2-failure-taxonomy-20261010/`。其中
`auto.json` 的 `final` 字段是逐条分类，`table.py` 的 `MAN` 字典记录精读时的人工改判。

## 3 当前实现与官方 `tau2 run` 的差异

### 3.1 起因

2026-10-09 的一次评测跑到 74/656 个 episode 时中止。原因是用户第一句就 stop，policy 一次都没被调用，旧代码把这种 episode
当导出错误处理，连续重试失败后中止整轮。官方 tau2 会正常给这个 episode 打分。修复后整个调度层改为照搬官方 runner，见
[ADR 0009](../../decisions/0009-official-tau2-runner-semantics.md) 和
[PRD](../../prd/2026-10-09-tau2-official-runner.md)。

### 3.2 当前对照（HEAD `c70a9800`，官方 pinned `b7ea907`）

逐项对照官方源码。官方路径相对 `site-packages/tau2`。

**已一致**：

- **重试**：任何异常都整段重跑，最多再试 3 次，间隔 1 s（`runner/progress.py`）。
- **infra 排除**：重试用尽记 `INFRASTRUCTURE_ERROR`，不进指标，并报数量和原因。
- **打分**：只有 AGENT_STOP / USER_STOP 用官方 evaluator 打分，其余终止原因记 0；用户首句就 stop 也正常打分。
- **max_steps / max_errors**：200 / 10。
- **orchestrator 行为**：首条 agent 消息、非 solo、不做通信校验，都与官方相同。
- **用户模拟器**：直接调用官方实现，空回复时整段重跑，不在原地重试。
- **评测回复长度**：policy 单次回复不设上限，只受剩余上下文限制。

仍不同的项：

| #   | 维度                 | 官方                                                | 我们（评测）                                                                                                    | 我们（训练）                                                                | 状态               | 影响                                                                                                      |
| --- | -------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------- |
| 1   | 用户模拟器模型       | gpt-4.1，temperature 0；榜单推荐 gpt-5.2            | DeepSeek-Flash（默认）或 Doubao，temperature 0，top_p 1，超时 120 s                                             | 同评测                                                                      | 未对齐             | 最大的一项：只换用户，Retail 差约 20pp（[user-simulator.md §1](user-simulator.md)）；与榜单数字不可直接比 |
| 2   | NL judge             | gpt-4.1，temperature 0                              | DeepSeek-Flash，temperature 0，与用户模拟器分开配置（`9e7082dc` 起，[user-simulator.md §2](user-simulator.md)） | 同评测                                                                      | 未对齐             | 只影响含 NL_ASSERTION 的任务，约 1pp（[user-simulator.md §1](user-simulator.md)）                         |
| 3   | seed                 | `random.seed(300)` 为每个 trial 生成 seed           | 不传                                                                                                            | 不传                                                                        | 未对齐（决定不改） | 只影响可复现性；provider 是否支持 seed 未核实                                                             |
| 4   | policy `num_retries` | 3                                                   | 0，重试放在 episode 层                                                                                          | 同评测                                                                      | 未对齐（决定不改） | 一次传输错误会让整段 episode 重跑                                                                         |
| 5   | 上下文               | 不设窗口                                            | 32K；溢出记 `context_limit`，0 分并计入，不重试                                                                 | 溢出记 `context_limit`，0 分并计入                                          | 有意不同           | 官方没有窗口，不会出现这种结局；SAO 534 第 30、40 步各有 17、25 条（共 240 条）                           |
| 6   | 墙钟超时             | 默认不设                                            | 1800 s，超时按官方 TIMEOUT 记 0 分                                                                              | 1200 s，超时记 truncated、0 分                                              | 有意不同           | 实测最长 651 s，评测基本不触发                                                                            |
| 7   | policy 生成设置      | temperature 0                                       | temperature 0，thinking 关闭，concat 模板加 qwen3_coder 解析                                                    | temperature 1，单次回复 4096 上限                                           | 有意不同           | concat 拼接与服务端完整渲染的 token 差异未量化                                                            |
| 8   | 领域                 | 榜单覆盖全部领域                                    | 只跑 Airline + Retail                                                                                           | 同评测                                                                      | 有意不同           | 不能和榜单总分比，只能按领域比                                                                            |
| 9   | `pass^k` 的 k        | k 裁到每题最少有效 trial 数，排除 1 条就没有 pass^4 | 只报 k=1/2/4，每个 k 在有效 trial 数 ≥ k 的题上平均，并报题数                                                   | 同评测                                                                      | 有意不同           | 无排除时两种算法结果相同；有排除时多报出一个 pass^4                                                       |
| 10  | 失败 episode 处理    | —                                                   | 同官方                                                                                                          | 丢弃重试用尽的成员，存活 ≥4 的组照常训练，否则 epoch 末重跑一次（ADR 0011） | 训练有意不同       | 存活成员可能偏向短对话；按题失败率尚未统计                                                                |
| 11  | 用户工具             | `get_user_tools(include=task.user_tools)`           | `get_user_tools()`，不按任务过滤                                                                                | 同评测                                                                      | 未对齐（新发现）   | Airline/Retail 没有用户工具，无影响；重新加 Telecom 前要改                                                |
| 12  | `evaluate.py` 题集   | 榜单用 base 全量，≥4 trials                         | `--task-set` 默认 test，需显式传 `base`                                                                         | 训练中评测用 test（formal）或 dev                                           | 默认值不同         | 忘传 `--task-set base` 的结果和榜单不可比                                                                 |
| 13  | 题目与评分           | `b7ea907` 原样                                      | official 与 fixed 两个变体都报（第 1 节）                                                                       | fixed                                                                       | 有意不同           | fixed 的数字不能和榜单比                                                                                  |

10-09 旧对照表中已经解决的项：重试、最终失败排除、评测时的上下文溢出与墙钟、原地重试、用户 512 上限 （`afb42082`），评测去掉 4096
回复上限（`31634dee`），汇总改为官方 pass^k（`afb42082`、`c70a9800`）。

**未核实**：

- 官方遇到上下文溢出时的行为是从代码推断的。
- 文本模式下官方的 `persona_config` 是否为 None。
- concat 模板与服务端完整渲染的 token 差异。

### 3.3 指标口径（2026-10-10 起）

所有评测都报同一组指标：离线 `evaluate.py`、训练中的 validation 和 critic 之后的 policy 训练。指标是
mean@4、pass^1、pass^2、pass^4，按合计和领域分开报，不再报 pass@k。

- **mean@4**：每题 4 次的成功率先在题内平均，再在题间平均。没有排除时等于 pass^1 和 `avg_reward`。
- **pass^k**：官方 $\binom{c}{k}/\binom{n}{k}$，即同一题 k 次全部成功的概率，再在题间平均。

训练中评测的 `eval_gconfig.n_samples` 是 4，所以 pass^4 有定义。实现见 `examples/tau2/official_runner.py`
的 `REPORTED_PASS_HAT_KS`。
