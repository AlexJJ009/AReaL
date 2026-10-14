# 两组 step 0 评测的失败分类（2026-10-10）

> 来源：原 README 第 10.1–10.4 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

对象是[step0-evaluations.md §2](step0-evaluations.md) 的两组评测，共 308 条失败：DeepSeek 106 条，Doubao
202 条。处理方式：

- 全部失败先按规则自动分类；
- 人工精读约 90 条（DeepSeek 约 40、Doubao 约 50）；
- 只因 NL 断言失败的 episode 逐条核对；
- 另读了约 20 条成功样本。

NL judge 与用户模拟器是同一个模型、同一套配置（[benchmark-variants.md §3.2](benchmark-variants.md) 第 2
项），所以下表放在同一大类里，但分开计数。

## 1 infra

数字精确，来自逐次尝试记录和 run.log。

| run × 领域    | 完成/计划 | 失败 | 排除 | 失败的尝试（按原因）                            | 重试后救回（其中成功） |
| ------------- | --------- | ---- | ---- | ----------------------------------------------- | ---------------------- |
| DeepSeek 航空 | 200/200   | 38   | 0    | 上下文超限 13                                   | 11（9）                |
| DeepSeek 零售 | 456/456   | 68   | 0    | 上下文超限 6，连接 1                            | 7（6）                 |
| Doubao 航空   | 199/200   | 43   | 1    | 上下文超限 12，Ark 欠费 2，连接 1               | 9（8）                 |
| Doubao 零售   | 455/456   | 159  | 1    | 上下文超限 11，用户空回复 2，Ark 欠费 2，连接 1 | 10（6）                |

- 两条被排除的 episode 是 `airline:10:trial0` 和 `retail:99:trial2`：都先上下文超限两次，再遇上 Ark 欠费两次。
- 上下文超限共 42 次，prompt 在 32,791–33,178 token，都比 32,768 多一点，全部被重试吸收。
- TIMEOUT 和 MAX_STEPS 都是 0。所有完成的 episode 都以 `user_stop` 结束，包括 `###TRANSFER###` 和
  `###OUT-OF-SCOPE###`。
- 完成集里的失败没有一条是 infra 造成的。
- 重试会偏向短轨迹：救回的 episode 里 DeepSeek 18 条成功 15 条，Doubao 19 条成功 14 条。估计 pass^1 因此偏高不到 1pp。

## 2 主分类

"精确"是逐条确认的；"估计"是规则分类，再用精读样本校正。

| 类别       | 子类型                           | DeepSeek | Doubao | 口径 | 例子                                                                                                   | 改进途径                          |
| ---------- | -------------------------------- | -------- | ------ | ---- | ------------------------------------------------------------------------------------------------------ | --------------------------------- |
| benchmark  | gold 不可行                      | 4        | 4      | 精确 | retail:105：礼品卡 \$17 付不了 \$21.10，agent 正确拒绝，NL 断言却要求完成换货                          | 无，记为上限损失                  |
| benchmark  | gold 写死取消原因                | 3        | 1      | 精确 | retail:38：用户说 "ordered by mistake"，gold 是 "no longer needed"                                     | 同上                              |
| benchmark  | gold 动作顺序错                  | 2        | 3      | 精确 | retail:64                                                                                              | 同上                              |
| benchmark  | 任务有歧义                       | 3        | 4      | 精确 | airline:7：问的是 "other upcoming flights"，gold 把要取消的两单也算进金额（\$1628 vs \$708），8 条全挂 | 同上                              |
| benchmark  | 未指定支付方式                   | 5        | 3      | 估计 | retail:60：任务没写支付方式，用户自己选了礼品卡，gold 是 paypal                                        | 同上；归 benchmark 还是用户有争议 |
| benchmark  | 环境实现 bug                     | 2        | 0      | 精确 | retail:100：`modify_pending_order_items` 把最后一个 variant 的价格写给所有 item，DB hash 对顺序敏感    | 同上                              |
| benchmark  | NL 断言要求空操作                | 0        | 2      | 精确 | retail:110：默认地址已经是新地址，断言仍要求 "modify user address"                                     | 同上                              |
| 用户模拟器 | 确认后立刻 STOP                  | 21       | 88     | 估计 | retail:17："Yes, I confirm… ###STOP###"，agent 还没执行写操作                                          | 换用户模型，或在 prompt 层修      |
| 用户模拟器 | 首轮即 OUT-OF-SCOPE              | 0        | 3      | 精确 | retail:109：第一条用户消息就是 `###OUT-OF-SCOPE###`                                                    | 换用户模型                        |
| 用户模拟器 | 中途 OUT-OF-SCOPE                | 0        | 5      | 估计 | airline:29                                                                                             | 换用户模型                        |
| 用户模拟器 | 编造或偏离指令                   | 3        | 4      | 估计 | retail:79（Doubao）编造"另一只瓶子是黑色"；retail:5（DeepSeek）known_info 里有姓名却拒绝提供           | 换用户模型                        |
| NL judge   | 判错                             | 0        | 0      | 精确 | DeepSeek 的 4 条 NL-only 失败都是 retail:105；Doubao 的 9 条 judge 都按字面判对                        | 当前不是瓶颈                      |
| 模型       | 参数、物品或支付方式错           | 25       | 46     | 估计 | airline:18：5 张票的退款全打到一张卡，应原路退回；retail:49 选错 variant                               | 训练                              |
| 模型       | 违反政策的写操作，或该转人工没转 | 13       | 15     | 估计 | airline:47：不符合条件的 basic economy 照样取消                                                        | 训练                              |
| 模型       | 不该转人工却转了                 | 21       | 17     | 估计 | airline:44：本该升舱，却转了人工                                                                       | 训练，但见 §3                     |
| 模型       | 漏步骤、漏传达、错误拒绝、幻觉   | 4        | 7      | 估计 | retail:110：错称 pending 订单不能修改；retail:39 编造地址                                              | 训练                              |
| infra      | —                                | 0        | 0      | 精确 | 见 §1                                                                                                  | 充值 Ark；32K 是有意设定          |

按类别折算的 pass^1 损失（pp；合计 = 100 − pass^1）：

| run × 领域    | benchmark | 用户模拟器 | 模型 | 合计 |
| ------------- | --------- | ---------- | ---- | ---- |
| DeepSeek 航空 | 1.5       | 0.5        | 17.0 | 19.0 |
| DeepSeek 零售 | 3.5       | 5.0        | 6.4  | 14.9 |
| Doubao 航空   | 2.0       | 2.0        | 17.6 | 21.6 |
| Doubao 零售   | 2.9       | 21.1       | 11.0 | 34.9 |

- 两种用户下模型自己的失败差不多：航空约 17pp，零售 6–11pp。
- 两组评测的差距几乎全部来自 Doubao 确认后立刻 STOP，与 [user-simulator.md §1](user-simulator.md) 的精读结论（90
  条中 79 条）一致。
- "确认后立刻 STOP"的判定条件是：agent 最后一次确认里列出了 gold 的全部 id。抽检 8 条有 1 条误判，所以 21 和 88 可能各偏高
  5–10%。即使用户不提前结束，agent 后面的步骤也未必都对，所以这部分 pp 是可追回的上限。
- 每条失败只标一个原因，取最先导致失败的那个。例如 retail:98 同时有支付方式错和提前 STOP。
- Doubao 有 14 条 episode 的用户中途改说中文，其中 5 条失败，已计入上面各类。

## 3 假阳性：转人工骗过 DB 评分

这是这次发现的最大问题，比任何一类失败都大。

**机制**：航空任务的 `reward_basis` 是 DB + COMMUNICATE，NL 断言不计分。以 airline:0 为例，它没有 gold 写操作，唯一的
NL 断言是 "Agent should refuse to proceed with the cancellation"。在这类"应该拒绝"的任务上，agent
只要转人工，DB 不变，就判成功。

**规模**：

- DeepSeek 航空无写操作任务上的成功有 86 条，其中 63 条转了人工，约占航空 pass^1 的 31.5pp。
- Doubao 航空是 55 条，约 27.6pp。
- 零售两组各 5 条。

**样本**：抽读 10 条（airline 0、5、6、26、27、34、41、45、48 和 retail 65），agent
都是在回避，没有按政策拒绝。airline:45 的用户明确说了不要转人工。直接计数：DeepSeek 航空 162 条成功里有 75 条调用过
`transfer_to_human_agents`，Doubao 是 156 条里 63 条。

**影响**：

- 这是官方评分规则本身的漏洞，榜单上的数字同样受影响，所以不算我们与官方的差异。
- 但它直接影响训练：训练 reward 用的是同一套 DB 评分，会强化"不确定就转人工"。§2 里的"不该转人工却转了"就是同一个倾向的另一面。
- 从航空 0.81 里扣掉这部分，实际按政策解决的比例约 0.5。这也解释了为什么 4B 模型的航空分数接近 Opus
  4.5（[recuris-and-leaderboard.md §3](recuris-and-leaderboard.md)）。

**其他检查**：

- 已知的"回一句 Done. 就得分"盲点，在这两组评测里没有出现：无写操作任务上的成功都有工具调用，用户轮数都大于 2。
- 没看到 NL judge 放水：成功样本里带数字的 NL 断言，数字都能在 agent 回复里找到。

## 4 可提升的方向

| 方向                           | 涉及的失败                      | 量级（pp，DeepSeek / Doubao）     | 做法                                                                                                                                       | 是否改变与官方的可比性    |
| ------------------------------ | ------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| 训练用的用户模拟器             | 确认后 STOP、OUT-OF-SCOPE、编造 | 零售 5 / 21                       | 训练用 DeepSeek（已定），不用 Doubao                                                                                                       | 不改评测                  |
| 训练 reward 罚转人工           | 转人工假阳性，以及不该转却转    | 航空约 30 的假阳性 + 约 10 的失败 | 已做：fixed 变体的转人工规则（[benchmark-variants.md §1.3](benchmark-variants.md)），训练用 fixed                                          | 只改 fixed，official 不变 |
| 训练（模型能力）               | 参数、物品、支付、政策违规      | 航空约 17，零售 6–11              | GRPO / SAO（[training-setup.md](training-setup.md)）                                                                                       | —                         |
| 评测时附报一个去掉转人工的视图 | 转人工假阳性                    | 航空约 30                         | 由 fixed 变体代替：评测同时报 official 与 fixed（13）                                                                                      | 不改官方指标              |
| benchmark 本身                 | gold 不可行、歧义、环境 bug     | 1.5–3.5                           | fixed 已修大部分（[benchmark-variants.md §1.1](benchmark-variants.md)–[benchmark-variants.md §1.2](benchmark-variants.md)）；official 不改 | 只改 fixed                |
| infra                          | Ark 欠费；32K 边缘超限          | 排除 2 条                         | Ark 充值并换 key；32K 不改                                                                                                                 | —                         |
