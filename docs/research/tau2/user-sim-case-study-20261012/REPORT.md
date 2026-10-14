# 三种用户模拟器下的失败归因：哪些是用户模拟器的问题，哪些是 agent 要补的能力

2026-10-12。标注：Codex（gpt-6.1-sol，pueue 477）逐条完成；复核与重写：Claude，按类别读了代表轨迹，见第 2.4 节。Codex
的原始报告保留在 `_artifacts/AReaL/codex-sao-math/user-sim-20261012/case-study/REPORT.md`。

## 1. 结论

1. **Retail 上换弱用户的掉分，几乎全部是用户模拟器造成的。** 三个模拟器下 Retail 的 agent 失败数基本相同（DeepSeek 13、Doubao
   14、Qwen 12）；多出来的失败都记在用户头上：Doubao 比 DeepSeek 多 35 个失败，同时多 36 个用户原因；Qwen 多 25 个失败，同时多
   27 个用户原因。
1. **三个模拟器坏的方式不一样**：
   - DeepSeek 最好，但 Retail 仍有 13 个失败是"用户在同意操作的同一句话里结束了对话"（U1），工具还没执行，episode 就判零。
   - Doubao 主要是**提前结束**：U1 31 个，开场第一句就结束对话 10 个，中途缩小目标后结束 3 个。
   - Qwen3.5-4B 主要是**不忠于任务说明**（U3，50 个）：agent 问到它不知道的信息时自己编一个，或者不按任务里的条件分支走。
1. **Airline 主要暴露 agent 的问题。** 头号缺口仍是
   [capability-gaps.md §1.2](../capability-gaps.md)的"资格判断正确，却用转人工代替拒绝"（M2a）：59
   个失败，三个模拟器下各约 20 个， 其中 6 道 Airline 题在三个模拟器下都失败在这里。其余 agent 缺口（工具语义、报错后乱改、退款路由）也都是
   [capability-gaps.md §1.2](../capability-gaps.md) 节已有的类别，**这次没有发现新的 agent 能力缺口**。
1. **agent 缺口的测量基本不受模拟器影响**：有 9 个"题 × 缺口"组合在三个模拟器下都出现，例如 airline:45 的 M2a、airline:44 的
   M14、retail:36 的 M15。用户模拟器影响的是 Retail 的总分，不太改变 agent 缺口的排序。
1. **用户模拟器也会让分数偏高**：5 个成功 episode 是用户把 agent 从死路上拉回来的（U6）；19 个成功 episode 是用户在 agent
   调用转人工工具之前就发了 `###TRANSFER###`，评测仍判成功（第 6.3 节）。

## 2. 怎么做的

### 2.1 数据

- actor 固定为训练前的 Qwen3.5-4B（non-thinking，policy temperature 0）；fixed v2 变体；官方 test 60
  题（Airline 20、Retail 40）× 4 次。
- 三个用户模拟器：DeepSeek-Flash（thinking）、Doubao 2.1
  Turbo（thinking，reasoning_effort=medium）、自己部署的 Qwen3.5-4B（non-thinking，T=0）。NL judge 都是
  DeepSeek-Flash。
- 每个模拟器 240 个 episode，共 720 个，失败 272 个。分数见
  [user-simulator.md §2](../user-simulator.md)总表。

### 2.2 每个失败只记一个"致败原因"

一个失败 episode 里常有好几个问题。例如 agent 先问了一个本可以自己查到的地址，用户接着编了一个假地址，agent
又照着假地址写入了数据库。如果每个问题都计数，三类加起来会超过失败总数，也就没法回答"Retail 多掉的分有多少是用户造成的"。所以每个失败只记一个**致败原因**：之后
reward 就不可能再是 1 的那一步，由谁的哪种行为造成。其余问题记为次要原因，保留在 `labels.jsonl` 的 `secondary` 字段里，不进入计数。

致败原因在哪一方，按下面的规则判断：

| 归为      | 判断规则                                                                                                    |
| --------- | ----------------------------------------------------------------------------------------------------------- |
| 用户      | 用户违反了自己的任务说明（scenario）或用户模拟器的通用规则，而 agent 在那之前的行为符合 policy              |
| agent     | agent 违反 policy、用错工具、漏做子目标；用户此前的行为符合任务说明                                         |
| benchmark | 用户和 agent 都按各自的说明做了，评分仍然判零：任务说明没写清楚而 gold 只认一种做法，或评分规则误伤合法行为 |

三条边界规则：

- 用户在 agent 出错**之后**才偏离，归 agent。例如 agent 已经执行了错误的写操作，用户随后的混乱不改变归因。
- 用户按任务说明本来就该说错话、该改主意时，不算用户错误。例如 airline:2 任务要求用户故意报错乘客人数。
- 用户说错话但被纠正、没有影响结果的，不作为致败原因。

### 2.3 编码

- agent 侧沿用 [capability-gaps.md §1.2](../capability-gaps.md) 和
  [capability-gaps.md §2.2](../capability-gaps.md)的 M 码，这次出现了
  M1、M2a、M3、M4、M5、M7、M8、M11、M12、M14、M15、M16。
- 用户侧用 [user-simulator.md §2](../user-simulator.md)提出的
  U1–U7。这次细分出三个子码：U4a（开场第一句就结束）、U4b（没做完就结束或缩小目标）、U7a（agent 调用转人工工具之前，用户就发了 TRANSFER）。
- benchmark 侧：B1（任务说明没写清，gold 只认一种做法）、B2（fixed 的转人工规则误伤 policy 要求的转人工）。

### 2.4 复核

- Codex 按固定种子随机抽了 10 条重读，致败原因改动 0 条，补了 1 个次要原因。
- 我读了 17
  条代表轨迹：[deepseek retail:49 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:49:trial0/dedf5721279141a082455c5db9f348d3.json)、[doubao retail:17 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:17:trial0/9853b3065bf84c878f54fb47400afa90.json)、[doubao retail:9 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:9:trial0/71f2b6e3b65e4db189d47dfa37bebbb5.json)、doubao
  airline:29
  t1、[doubao airline:13 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:13:trial0/32959f497918421cb1a5d41c5ed7c727.json)、[deepseek retail:64 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:64:trial3/98cca3afb69346588b1ab2973fe09302.json)、[qwen retail:33 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:33:trial0/5a58e53c9c3740e2b18d0f6b46690b8d.json)、qwen
  retail:90
  t1、[qwen airline:25 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/airline:25:trial0/e5c8e19c584740d093f0f2351b2cc171.json)、[deepseek airline:45 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:45:trial2/8c63a311d7a44de3987f4a1ca615deee.json)、[doubao airline:2 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:2:trial0/d7f426baef674e9792b433e2adad584a.json)、deepseek
  airline:44
  t2、[deepseek retail:36 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:36:trial1/9af990dd77b943618723c508251613ae.json)、[deepseek airline:18 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:18:trial2/f967d5f941cf44cb91f5fa5b8e90b66f.json)、deepseek
  airline:29
  t1、[deepseek retail:111 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:111:trial1/bddba86a1f1f4e13b0410d3db5218f98.json)、[qwen retail:39 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:39:trial3/22314bd801654f1793e1fb08b31224b3.json)。归类都成立，下文的例子都来自这些轨迹。
- 其余约 250 条标注没有逐条复核。

## 3. 总表

致败原因计数（每个失败只计一次）：

| 领域    | 用户模拟器 | 成功 / 总数 | agent | 用户 | benchmark |
| ------- | ---------- | ----------: | ----: | ---: | --------: |
| Airline | DeepSeek   |     41 / 80 |    33 |    0 |         6 |
| Airline | Doubao     |     35 / 80 |    35 |    6 |         4 |
| Airline | Qwen3.5-4B |     36 / 80 |    26 |   16 |         2 |
| Retail  | DeepSeek   |   132 / 160 |    13 |   13 |         2 |
| Retail  | Doubao     |    97 / 160 |    14 |   49 |         0 |
| Retail  | Qwen3.5-4B |   107 / 160 |    12 |   40 |         1 |
| 合计    | DeepSeek   |   173 / 240 |    46 |   13 |         8 |
| 合计    | Doubao     |   132 / 240 |    49 |   55 |         4 |
| 合计    | Qwen3.5-4B |   143 / 240 |    38 |   56 |         3 |

按代码细分：

| 代码     | 含义                                          | DeepSeek | Doubao | Qwen |
| -------- | --------------------------------------------- | -------: | -----: | ---: |
| U1       | 同意操作的同一句里结束对话                    |       13 |     31 |    1 |
| U3       | 偏离任务说明                                  |        0 |      7 |   50 |
| U4a      | 开场第一句就结束                              |        0 |     10 |    0 |
| U4b      | 没做完就结束，或缩小目标                      |        0 |      3 |    0 |
| U2/U4/U5 | 错给信息 / 无理由要求转人工 / 角色混淆        |        0 |      2 |    2 |
| U7a      | agent 调用转人工工具之前，用户就发了 TRANSFER |        0 |      2 |    3 |
| M2a      | 资格判断正确，却用转人工代替拒绝              |       21 |     20 |   18 |
| M14      | 误解工具能力或订单状态                        |        9 |      7 |   10 |
| M15      | 工具报错后不问用户就改参数重写                |        4 |      2 |    2 |
| M5       | 支付 / 退款路由错误                           |        2 |      3 |    5 |
| M8/M16   | 漏做子目标 / 声称做了但没执行                 |        1 |      9 |    0 |
| M1       | 政策资格规则                                  |        5 |      5 |    0 |
| 其他 M   | M3、M4、M7、M11、M12                          |        4 |      3 |    3 |
| B1       | 任务说明没写清，gold 只认一种做法             |        5 |      0 |    2 |
| B2       | 转人工规则误伤合法转人工                      |        3 |      4 |    1 |

## 4. 用户模拟器的问题

### 4.1 U1：同意操作的同一句里结束对话（45 个：DeepSeek 13、Doubao 31、Qwen 1）

**背景。** Retail 和 Airline 的 policy 都要求 agent 在任何写操作前列出操作内容，并取得用户明确的 "yes"。tau2
的用户模拟器规则是：任务完成时输出 `###STOP###` 结束对话。U1 就是用户把"我同意执行"当成了"任务已完成"，在同一句话里既说 yes 又发
STOP。对话就此结束，agent 再也没有机会调用工具，数据库没变，episode 判零。

**例子。**

- [deepseek retail:49 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:49:trial0/dedf5721279141a082455c5db9f348d3.json)。第
  20 条消息，agent 说明要把耳机 2757705742 换成同订单里最便宜的 1646531091，请用户确认。第 21 条用户回复："Yes, please
  proceed with exchanging item_id 2757705742 for item_id 1646531091. Refunding the
  \$26.48 to the gift card is fine. ###STOP###"。换货工具一次都没调用。
- [doubao retail:17 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:17:trial0/9853b3065bf84c878f54fb47400afa90.json)。任务是把订单
  #W8665881 的地址从 Suite 640 改成 Suite 641。agent 在第 12 条列出新地址并请求确认，第 13 条用户回复："Yes, that's
  correct. Please update the address for order #W8665881 to Suite 641. ###STOP###"。

**为什么归用户。** agent 在这两个例子里的做法完全符合 policy，而且正是 policy 要求的写前确认。用户违反了"任务完成才 STOP"的规则。

**分布。** 集中在 Retail 的单步写操作题：retail:5 出现 5 次，retail:70 出现 4 次，retail:108、26、9 各 3
次。DeepSeek 作为最好的模拟器也有 13 次，占它全部 240 个 episode 的 5%。Qwen 几乎不犯这个错（1 次），它的问题在 U3。

**对训练的影响。** U1 惩罚的恰恰是正确行为：agent 越守规矩地先确认，越容易撞上 U1 拿到零分。不先确认就直接执行的 agent
反而能得分。[capability-gaps.md §1.3](../capability-gaps.md)已经在 GRPO 训练数据里观察到这个冲突。

### 4.2 U3：偏离任务说明（57 个：Doubao 7、Qwen 50）

Qwen 的 50 个 U3 可以分成五种，各自的成因不同：

| 子类                   | Qwen | Doubao | 说明                                                       |
| ---------------------- | ---: | -----: | ---------------------------------------------------------- |
| 编造任务里没有的事实   |   19 |      3 | 编地址、编航线、给错预订号或乘客                           |
| 放弃或缩小任务目标     |   12 |      0 | 遇到阻碍就改成取消整单、只问价、换一个目标                 |
| 不按任务里的数值条件走 |   11 |      2 | 超预算仍同意、价格超上限仍同意                             |
| 确认时该改主意却没改   |    6 |      2 | 任务要求"agent 请求确认时改成只换台灯"，用户直接同意原方案 |
| 说出任务禁止透露的信息 |    2 |      0 | 提前说出应该保留的物品                                     |

**编造事实的例子。**

- [qwen airline:25 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/airline:25:trial0/e5c8e19c584740d093f0f2351b2cc171.json)。任务是"给朋友订一张和我现有预订完全一样的票"，现有预订是
  DTW→SEA，任务里没有直接写出航线。agent 在第 6 条问出发地和目的地，并在括号里举例"IATA code, such as SFO"、"such as
  JFK"。用户第 7 条回复："This is a round-trip flight. The origin is SFO and the destination is
  JFK. The outbound flight is on March 15th"，照抄了 agent 举的例子，还编了日期。
- [qwen retail:33 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:33:trial0/5a58e53c9c3740e2b18d0f6b46690b8d.json)。任务要求把默认地址改成订单里显示的西雅图父母家地址，并且"不想在聊天里说出来"。agent
  在第 14 条请用户提供完整地址。用户第 15 条一边说 "I don't want to share the exact street address in this
  chat"，一边给出 "1234 Maple Avenue, Apt 4B, Seattle, WA 98101"。正确地址在订单里，是 517 Lakeview
  Drive。agent 照着假地址写入了数据库。

**数值条件的例子。**
[qwen retail:90 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:90:trial1/e55014b6f9e74b329ea322ba28707f1c.json)。任务写明：10x
变焦的相机如果超过 3000 美元，就以 ordered by mistake 为由取消订单。agent 在第 20 条明确报出新款价格 3066.23，用户第 21
条回复："Yes, I confirm. Please go ahead with the modification."。

**为什么归用户，以及 agent 的那一半责任。** 编造事实那一类，常常是 agent 先问了自己本可以查到的信息：retail:33 的地址就在 agent
已经读过的订单里，airline:25 的航线就在用户的现有预订里。agent 的这个行为记为次要原因
M14。致败原因仍归用户，因为任务说明明确要求用户不提供这些信息，弱模拟器却编了一个。

同一类题在三个模拟器下暴露的问题不一样，第 5.2 节有对照：DeepSeek 用户坚持不说，于是暴露出 agent 不会自己查（agent 失败）；Qwen
用户编了一个假地址，于是记成用户失败。

### 4.3 U4a / U4b：提前结束（Doubao 13 个）

- **U4a，开场第一句就结束（10 个）。**
  [doubao retail:9 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:9:trial0/71f2b6e3b65e4db189d47dfa37bebbb5.json)：agent
  问候 "Hi! How can I help you today?"，用户第 1 条回复只有 `###OUT-OF-SCOPE###`。这道题的任务说明是完整的：身份
  mei_kovacs_8020、邮编 28236、要换水瓶和台灯，没有缺信息。10 个 U4a 都出现在 Doubao 下，涉及 retail:9（3 次）、108（2
  次）、42、56、65、71 和 airline:32。**原因未查明**：可能和 Doubao 的 thinking
  有关，比如推理用完了输出预算、正文为空，但这只是猜测，需要看原始响应才能确认。
- **2026-10-12 补充：U4a 很可能是方舟服务端异常。** 同样代码和配置下，充值后的 Doubao 重跑（pueue 488，240 个
  episode）开场就结束的是 0 个，空回复重试 0 次；账户临近欠费时的一次运行（486）是 10 个，还出现过一条无关的"电商商品信息提取结果"。所以这 10 个
  U4a 不宜算作 Doubao 作为用户模拟器的能力问题（[user-simulator.md §2](../user-simulator.md)）。
- **U4b，缩小目标后结束（3 个）。**
  [doubao retail:108 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:108:trial3/9711cd545660412aae8f0b00fe3e6a5f.json)：任务是退掉除平板以外的物品，用户把目标改成了"问一下能退多少钱"，得知金额
  346.93 后就发了 STOP。

### 4.4 U5：推理内容漏进回复（Doubao 1 个）

[doubao airline:29 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:29:trial1/5aafbb377e6b448ca883dce25a1c73b3.json)。agent
在第 2 条问用户 ID，用户第 3 条回复是一大段中文自言自语："用户现在需要提供用户ID吗？不对，等下，之前的scenario是啥？……"，最后以 OUT-OF-SCOPE
结束。用户 ID raj_brown_5782 就写在任务说明里。这看起来是 Doubao 把 thinking 内容输出到了正文。只有 1 例，但说明 thinking
模式的模拟器有这种失败方式。

### 4.5 U7a：转人工的协议错误（5 个：Doubao 2、Qwen 3）

tau2 的规则是：agent 调用 `transfer_to_human_agents` 之后，用户才输出 `###TRANSFER###`。U7a
是用户自己请求转人工的同时就发了 TRANSFER，agent 还没来得及调用工具。

[doubao airline:13 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:13:trial0/32959f497918421cb1a5d41c5ed7c727.json)：任务要求"如果
agent 说改不了，就要求转人工"，gold 也是转人工。agent 在第 12 条解释了改不了的原因，用户第 13 条回复："please go ahead and
transfer me to a human agent … ###TRANSFER###"。工具没被调用，fixed 评分记为 missing_transfer，判零。

5 个都在 airline:13 上。agent 在这道题上的做法是对的，所以归用户。

### 4.6 U6：成功 episode 里用户帮了 agent（5 个）

U6 不是失败，所以不进入上面的计数。它说明有些成功并不是 agent 自己做出来的。

[deepseek retail:64 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:64:trial3/98cca3afb69346588b1ab2973fe09302.json)。用户想把相机换成更好的。订单状态是
pending，按 policy 只能 modify，不能 exchange。agent 在第 16 条列出订单，没有给出可行的办法。用户第 17 条主动提出："If that
order is still pending, can we switch it to a waterproof camera at the same price
instead?"，agent 才转去用 modify，最后成功了。gold 的做法正是 modify。

其余 4
例：[doubao retail:64 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:64:trial3/fb96c60c9af944819bd2ae06f3028344.json)、[deepseek retail:97 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:97:trial1/30f9effe8e9744a98a2b570f3f4e390d.json)
和
[qwen retail:97 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:97:trial3/98cba302b66a487d9e352c661a6d0ce8.json)
是同一类，都是用户提出用 modify 代替 exchange，或者提醒 agent
从另一张订单里读地址。[qwen retail:5 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:5:trial1/55cc445caea44044b379f82be2dccdca.json)
是用户直接引用了任务说明里的分支。

**意义。** agent 在这几题上的真实缺口是 M14（不知道 pending 订单可以 modify），但在强用户下它被掩盖了。所以用户越"聪明"，agent
的分数越可能偏高。

### 4.7 三个模拟器的问题画像

| 模拟器                 | 主要问题                              | 次要问题                      | 对 agent 分数的影响                    |
| ---------------------- | ------------------------------------- | ----------------------------- | -------------------------------------- |
| DeepSeek-Flash         | U1（13）                              | 偶尔帮 agent 找到出路（U6 2） | Retail 少量误判零分；分数略偏高        |
| Doubao 2.1 Turbo       | 提前结束：U1 31、U4a 10、U4b 3        | 推理内容漏进回复、条件判断错  | Retail 掉 0.22                         |
| Qwen3.5-4B（自己部署） | 编造事实、不按条件走、放弃目标：U3 50 | 转人工协议错误（U7a 3）       | Retail 掉 0.16；agent 会被引向错误写入 |

## 5. agent 的能力缺口

### 5.1 和之前缺口分析的对照

| 缺口                                 | 代码                 | 本次 DeepSeek / Doubao / Qwen | [capability-gaps.md §1.2](../capability-gaps.md)（官方 base 集）  | 三个模拟器下同一题都出现       |
| ------------------------------------ | -------------------- | ----------------------------- | ----------------------------------------------------------------- | ------------------------------ |
| 资格判断正确，却用转人工代替拒绝     | M2a                  | 21 / 20 / 18                  | 头号缺口，Airline 34.8pp                                          | airline:24、26、31、35、45、48 |
| 误解工具能力或订单状态               | M14                  | 9 / 7 / 10                    | 已有                                                              | airline:44                     |
| 报错后不问用户就改参数重写           | M15                  | 4 / 2 / 2                     | 已有                                                              | retail:36                      |
| 支付 / 退款路由                      | M5                   | 2 / 3 / 5                     | 已有                                                              | airline:18                     |
| 漏做子目标 / 声称做了没执行          | M8、M16              | 1 / 9 / 0                     | 已有（M16 来自 [capability-gaps.md §2.2](../capability-gaps.md)） | —                              |
| 政策资格规则                         | M1                   | 5 / 5 / 0                     | 已有                                                              | —                              |
| 写前没报价、没传达指定信息、选错对象 | M11、M12、M4、M7、M3 | 4 / 3 / 3                     | 已有                                                              | —                              |

这次没有出现 [capability-gaps.md §1.2](../capability-gaps.md)以外的新类别。数量排序也和
[capability-gaps.md §1.2](../capability-gaps.md)一致：M2a 远多于其他，其次是工具语义（M14/M15）。

### 5.2 M2a：资格判断正确，却用转人工代替拒绝（59 个）

**现象。** 用户的请求按 policy 不能办，比如 basic economy 不能改、超过 24 小时又没有保险不能退。agent 能正确说出不能办的理由，然后调用
`transfer_to_human_agents`，而不是直接拒绝。fixed 评分把这种"不该转的转人工"判零。

**例子。**
[deepseek airline:45 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:45:trial2/8c63a311d7a44de3987f4a1ca615deee.json)。任务说明写着用户"under
no circumstances you want to get transferred to another agent"。agent 查到预订 PEP4E0 是 basic
economy、没有保险、10 天前订的，第 6 条自己列出了"超过 24 小时、basic economy、没有保险"，紧接着调用了转人工。用户第 9 条抗议："No,
please don't transfer me!"。

**稳定性。** 同样 6 道 Airline 题在三个模拟器下都失败在 M2a 上，说明这是 agent
自己的行为习惯，和用户怎么说无关。[capability-gaps.md §1.1](../capability-gaps.md)分析过成因：agent 把 policy 第
141 行"已飞航段需要转人工"推广到了所有办不了的请求。

**补救。** RL（fixed 评分已经能区分转与不转），或者在 harness 里加转人工门控。

### 5.3 M14 / M15：工具语义误解和报错后乱改（34 个）

- **以为工具查不到。**
  [deepseek airline:44 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:44:trial2/d6d0357a878f4fe2a8d758a6da962e38.json)：任务需要比较航班时长。agent
  只调了 `get_flight_status`，返回的是 "available"，第 18 条就说"I don't have a tool to get the
  actual flight times"，然后转人工。实际上 `search_direct_flight`
  的返回结果里有起降时间（[capability-gaps.md §1.1](../capability-gaps.md) 节已核对）。三个模拟器下这道题都这样失败。
- **以为必须让用户口述地址。** 好几道 Retail 题的任务设定是：用户要用的地址在另一张订单里，用户不愿意说出来。agent 的正确做法是去订单里查。deepseek
  retail:111 t1 里，DeepSeek 用户第 23、25、27 条三次坚持"不想说，请查我的订单"，agent 始终没找到，最后转人工，记为 agent
  失败（M14）。[qwen retail:39 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:39:trial3/22314bd801654f1793e1fb08b31224b3.json)
  里，agent 已经读到了正确地址 157 Oak Street，却仍坚持要用户亲口说出完整地址。同类题在 Qwen 用户下多变成用户编地址（第 4.2 节）。所以这里
  agent 有一个真实缺口：**不会用已读到的信息去填工具参数**。这个缺口在强用户下表现为 agent 失败，在弱用户下被记成了用户失败。
- **把 exchange 拆成先 return 再 exchange。**
  [doubao retail:102 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:102:trial1/de6d4c720b0a4a86892a88350b385cfa.json)：先调
  return，订单状态变成 return requested，第二步 exchange 报错 "Non-delivered order cannot be
  exchanged"，换货目标就此做不成了。
- **报错后乱改（M15）。**
  [deepseek retail:36 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:36:trial1/9af990dd77b943618723c508251613ae.json)：用户预算
  1131，要把所有商品换成最便宜的款。agent 第 30 条把本来就是最便宜款、不用换的伞和背包也填进了参数，工具报错 "new item id should be
  different"。agent 没有去掉这两项，而是不问用户就把它们换成了更贵的款，总价变成 1146.89，超出预算，订单也锁死了。三个模拟器下这道题都这样失败。

**补救。** harness：在工具描述里写清楚返回哪些字段，自动剔除 old==new 的项。RL：成功的 trial 会回头问用户，组内有对比。

### 5.4 M5：退款路由（10 个）

[deepseek airline:18 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:18:trial2/f967d5f941cf44cb91f5fa5b8e90b66f.json)：用户要把
5 个商务舱预订都降为经济舱，退款退回各自的原支付方式。agent 第 20–28 条五次调用
`update_reservation_flights`，`payment_id` 全部填
credit_card_2929732，丢掉了礼品卡和另一张信用卡的原支付路由。三个模拟器下这道题都这样失败。

### 5.5 M8 / M16：漏做子目标（10 个，9 个在 Doubao 下）

doubao airline:2 的 4 次 trial 都是这样：用户投诉航班延误，NL 断言要求 agent "check that the flight was
indeed delayed"。agent 查了预订，但没有调用航班状态工具核实是否真的延误，就开始讨论补偿政策。同一道题在 DeepSeek、Qwen 下的失败原因是
M1/M2a 和用户偏离，所以 M8 集中在 Doubao 下可能和对话走向有关，不一定说明 agent 在 Doubao 下更差。

## 6. benchmark 的问题

### 6.1 B1：任务说明没写清，gold 只认一种做法（7 个）

[deepseek airline:29 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:29:trial1/6bdd7a712d084bc185c248de713ad6fd.json)–t3：用户取消旧预订后重新订票。任务说明只写了"因为这次旅行买了保险，想免改签费"，没有说新预订要不要买保险。用户第
15 条说 "I'd like to keep travel insurance on the new booking"，agent
照办（`insurance: "yes"`），gold 却写死了 `insurance: "no"`，数据库比对失败。用户和 agent 都没有违反说明。

同类：[qwen retail:55 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:55:trial3/ba46462c091e44dbba93f14a43cbc6ff.json)
的取消原因任务没指定，gold 只认 "no longer needed"。

deepseek retail:45 t2 原标为 B1，fixed-v3 审查时改判为 agent 失败（M7）：数据库里有 3 款可用的 canister
吸尘器，agent
只列了一款，而这道题考的正是列全选项（[benchmark-variants.md §1.2.1](../benchmark-variants.md)）。所以 B1 实际是 6
个。

### 6.2 B2：转人工规则误伤合法转人工（8 个）

全部在 airline:37 上。agent 正确完成了升舱，然后按 policy 第 141 行为已飞航段转人工。gold 里没有转人工，fixed
的转人工规则就把这个合法的转接判成
unexpected_transfer。[capability-gaps.md §1.3](../capability-gaps.md)已经记录过这个问题。

### 6.3 用户先发 TRANSFER 仍被判成功（不进入失败计数）

用户发 `###TRANSFER###` 时 agent 还没调用转人工工具，episode 却得了 1 分。我扫描了全部成功 episode：DeepSeek 11
个、Doubao 4 个、Qwen 4 个，集中在 airline 26/31/32/37/48 和 retail 27/32/65/94。这是上限：如果 agent
本来就该拒绝，用户转人工后离开，数据库没变，这个成功是合理的。即使全部扣掉，DeepSeek 也还有 0.675，排序不变。按 2026-10-12 的决定暂不修题。

## 7. 对研究的意义

**用户模拟器侧的研究问题**，按影响大小排列：

1. **U1，"同意"被当成"完成"。** 连最好的 DeepSeek 都有 5% 的 episode 撞上，而且它惩罚的是 policy
   要求的写前确认。训练时可以用工具状态区分两种情况：用户说 STOP 时，是否还有已获授权但没执行的写操作。不能简单删掉所有带 STOP
   的样本，因为拒绝类任务本来就不需要写操作。
1. **弱模拟器不忠于任务说明（U3）。** 编造事实的情况常常由 agent 的提问引出。所以 U3 的比例同时取决于模拟器和 agent，换 agent 后要重新测。
1. **提前结束（U4a）。** Doubao 特有，原因待查。如果是 thinking 输出预算的问题，可以通过配置修好，就不算模拟器的能力问题。
1. **成功侧的偏差（U6 和 6.3 节的 TRANSFER）。** 会让分数偏高，单看成功率看不出来。

**agent 侧要补的能力**，和 [capability-gaps.md §1.2](../capability-gaps.md)的结论一致：

1. M2a：拒绝后继续处理允许的部分，不要用转人工收尾。占 Airline 失败的大头，RL 或转人工门控都能处理。
1. M14：利用已读到的信息填参数（订单里的地址、预订里的航线），读懂工具返回的字段。
1. M15：工具报错后重新核对用户的约束，不擅自更换写入参数。
1. M5：每个预订按各自的原支付方式退款。

**对评测协议的含义**：Retail 分数对用户模拟器很敏感，同一个 actor 在 0.61–0.83 之间变动；Airline 分数对模拟器不敏感。报告 Retail
结果时必须固定并写明用户模拟器。

## 8. 局限

- **单人标注。** Codex 一个人标完 272 条，没有第二个标注者，所以算不出标注一致性。我只核对了 17 条代表轨迹。
- **配对比较不是因果证明。** 同一题同一 trial 在三个模拟器下的对话，从第一句用户消息起就不同；DeepSeek 和 Doubao 开着
  thinking，本身就不确定。所以"A 成功、B 失败"的配对只用来找分歧在哪里，不能证明差异完全由用户造成。也没有做删掉某句用户话再重放的实验。
- **三个模拟器的配置不同。** DeepSeek、Doubao 开 thinking，Qwen 不开。比较的是三套完整配置，差异不能全部归到模型强弱上。
- **用户原因的计数不等于能追回的分数。** 124 个用户原因的失败，如果换成完美用户，不一定都会成功，因为 agent 在后面的步骤里可能还会出错。

## 附录：文件与证据

- **全部标注的轨迹索引：[cases.md](cases.md)**，按代码分组，每条都链接到轨迹 JSON。JSON 的 `messages` 数组就是对话，编号从 0
  开始；`reward_info` 是评分细项。

- 逐条标注：`_artifacts/AReaL/codex-sao-math/user-sim-20261012/case-study/labels.jsonl`（277
  行：272 个失败 + 5 个 U6）。每条记录有消息编号、原文引文、任务说明、gold 操作、评分细项和轨迹路径。消息编号从 0 开始，0 是 agent 的开场问候。

- 模拟器 × 原因矩阵与两两配对统计：[matrix.md](matrix.md)。

- 运行目录（轨迹在 `<目录>/trajectories/<领域>:<题号>:trial<N>/`）：

  - DeepSeek：`user-sim-20261012/api-user/deepseek-base-u0p0`（pueue 473）
  - Doubao：`user-sim-20261012/api-user/doubao-base-u0p0`（pueue 476）
  - Qwen3.5-4B：`user-sim-20261012/step0-selfhost-user/full-qwen35user-v2`（pueue 472）
