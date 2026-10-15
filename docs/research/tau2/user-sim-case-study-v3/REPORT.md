# 五个用户模拟器下的失败归因：fixed-v3 修补后，用户缺陷仍污染评测信号

2026-10-13。单一标注者独立完成；没有启动 sub-agent，没有重采样、训练或调用付费
API。全部结果来自已有完成轨迹。本报告沿用上一轮编码与归因边界，所有数字为本轮 fixed-v3 重新标注。

## 1. 结论

同一 actor 在五个用户模拟器下的成功率相差 **17.50 个百分点**，但差距不能全部解释为 agent 能力差异。1,200 个 completed episode
中，385 个失败有 **168 个用户原因、204 个 agent 原因、13 个 benchmark 原因**。用户原因占失败的 **43.64%**；在全部
episode 中占 **14.00%**。此外，14 个成功 episode 出现用户给出执行方案或纠正 agent（U6），29 个成功 episode
在转接工具调用前收到用户的 `###TRANSFER###`。

Lite 和 GLM 的主要差别在 Retail。Lite 有 29 个用户原因失败，其中 23 个是授权时同句 STOP；GLM 有 7 个用户原因失败，没有 U1。GLM
的原始成功率最高，但仍有任务约束偏离、提前结束和帮助 agent 的成功，不能作为无噪声参照。Turbo 的 U1 更多，Qwen
主要违反任务条件。五者的缺陷形态不同，替换模拟器会改变失败来源。

agent 缺口中，M2a（以转人工代替政策拒绝）最稳定：97 个失败，airline:6、26、35、45、48
的同一缺口在五个模拟器下都出现。但其他缺口的出现与对话路径有关，不能把“同一题有一次出现”写成“四次都会失败”。本轮没有新增代码；Lite 和 GLM 的新证据落在现有
U/M/B 类别内。

## 2. 数据、口径和复核

### 2.1 数据与运行一致性

每个模拟器均为官方 test 60 题（Airline 20、Retail 40）×4 次，240 个 completed episode，五者共 1,200 个。actor
固定为 Qwen3.5-4B step 0，non-thinking，**eval_gconfig.temperature=0**；snapshot 为
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，代码提交
`e0fdf4a318453e6915d33f2cc30cd864666b8b50`，NL judge 为 `openai/deepseek-flash`。训练配置中的
gconfig.temperature=1 不参与这里的 policy temperature 核对。

数据来自 fixed-v3 cache，Airline tasks SHA256 为
`25086da158eedb97aee3bce2bc131d6acea094267bee6fd1c71933f74d90ef74`（完整哈希以下方 preflight
为准），Retail 为 `5b7a8fa30a7615b4e8f8d1fc21a66fb85ba6d337a06ce50f4a9565258098301f`。五份
manifest 的数据哈希与源文件一致。用户 temperature 配置为 0；DeepSeek thinking 模式下该参数不生效，GLM thinking
无法关闭，API 用户使用各自默认 thinking 模式，Turbo/Lite reasoning_effort=medium；自托管 Qwen 用户是
non-thinking。因而实验比较的是这些完整用户配置，不能单独归因于模型参数规模。

运行目录、配置读回、实际响应模型和错误尝试见 [preflight.json](preflight.json)。并发分别为 256/256/256/32/224，按 brief
不作为停止条件；它可能改变 actor 批次组合。

### 2.2 每个失败只计一个致败原因

**致败原因**指在实际轨迹里，使正确奖励结果不再能够沿该轨迹恢复的决定性一步，由哪一方的哪种行为造成。错误写入、已发生的禁止转接、控制符终止，通常能定位到具体消息；遗漏信息或跳过条件分支，需要结合最终结束、工具状态和评分依据判断。这里不是声称已经验证所有可想象的反事实修复。

用户违反自己的 task scenario 或通用模拟规则，归用户；agent 违反 policy、用错工具、漏做子目标，归 agent；双方遵守各自说明，但任务/gold
或评分规则仍拒绝该结果，归 benchmark。之后才发生的偏离记
secondary，不叠加进失败总数。任务明令用户报错、坚持或改主意，不算用户缺陷；被纠正且没有影响结果的错误，也不作为致败原因。

沿用
U1–U7、U4a/U4b/U7a、[capability-gaps.md §1.2](../capability-gaps.md)、[§2.2](../capability-gaps.md)
的 M/T 码以及 B1/B2；本轮没有 T 类致败原因。U6 和成功侧 U7a 是成功异常标记，与失败计数分开。未将所有普通澄清或偏好重述都计成
U6，只保留针对已有错误计划、错误实体角色或错误执行方法的纠正。

消息编号为 JSON 的 messages 数组 **0-based**，m0 是 agent 开场问候；t0–t3 是 trial。`labels.jsonl`
保留原轨迹绝对路径、SHA256、原话、tool_calls、实际写操作、任务说明字段与行号、gold、reward breakdown 和未满足的 NL 断言。计数依据
official_score；某些 unexpected_transfer 案例官方子项均为 1，fixed 后总分仍为 0。

### 2.3 筛查、配对与复核

先用脚本定位失败、授权 STOP、提前 TRANSFER、成功侧帮助候选与服务异常，再按任务/gold 阅读。全部 385 个失败均有标注；成功侧筛查覆盖全部 815 个成功
episode，U6 候选按原话重读后保守保留 14 条。配对单位为同一 `(domain, task_id, trial)`，没有共享对话前缀的干预。160
个唯一配对单元在五者间至少一成一败，其中 Airline 52、Retail 108；完整十组两两配对计数见 [matrix.md](matrix.md)。

最后按 Python random seed **20261013**，从 428 条暂定标签（385 失败、43 成功异常）随机抽 15 条重读：**1 条致败原因改动，2
条次要原因调整（包含那条致败原因改动），1 条理由措辞纠正，共 3 条记录变化**。qwen retail:38 t0 从 M12 改为
U3，用户跳过询价与最便宜变体分支，直接授权整单取消；qwen retail:79 t0 补 M7 次要证据；qwen airline:25 t0
删除“已订入新路线”的错误表述，实际是持续误导查询。见 [audit-15.md](audit-15.md) 和
[机器记录](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/case-study-v3/audit-15.json)。这只是同一标注者重读，不是独立一致性检验。

定向复核另外修正了 retail:32 的 5 条合法转接终点冲突（U4b→B1）、Lite retail:100 的 3 条用户扩展整单支付目标（M5→U3），并将 GLM
retail:39 t1 从 U1 改为 U4b、Qwen airline:44 t1 从 U5 改为
M14（角色错误恢复后，持续的工具误解才造成失败）。补齐了若干实际写入和确认消息位置；这些定向更正不冒充随机抽样结果。边界解释见下文，记录见
[targeted-review-changes.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/case-study-v3/targeted-review-changes.json)。

### 2.4 两个同题同trial的成功对照

这些对照说明具体哪一轮不同，也保留agent在两边可能走不同路径的限制。不是固定前缀的干预。

- [GLM 5.3 retail:5 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:5:trial0/a0da1aa6b9d041a8ad93117cd6596c13.json)（reward=1）。与Lite
  retail:5 t0的U1配对。GLM同样改成只退水瓶，但m23仅确认，m24实际退货，直到m27才STOP；成功侧无U6或提前TRANSFER标记。 m23：“Yes,
  I confirm. Please go ahead with the return of the water bottle (item_id: 8538875209)
  with the refund going to my PayPal account (paypal_7644869).”。 m27：“No, that's all I
  needed today. Thank you so much for your patience with me changing my mind — I really
  appreciate it! Have a great day! ###STOP###”。 m24 调用 `return_delivered_order_items`，参数
  `{"item_ids":["8538875209"],"order_id":"#W6390527","payment_method_id":"paypal_7644869"}`。

- [DeepSeek-Flash retail:90 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-deepseek-test-u0p0/trajectories/retail:90:trial0/5f2d44fea1804d03b22dc294004444b8.json)（reward=1）。与Qwen
  retail:90 t0的U3配对。DeepSeek知道3066.23超过3000后，m19按任务改为取消，m22执行；成功侧无U6或提前TRANSFER标记。
  m19：“Since the 10x zoom option is $3,066.23, that’s over $3,000, so I don’t want to
  proceed with the modification after all. Please don’t modify the item. Instead, could
  you cancel the entire order **#W9284598** with the reason **“ordered by mistake”**?”。
  m22 调用 `cancel_pending_order`，参数
  `{"order_id":"#W9284598","reason":"ordered by mistake"}`。

## 3. 总表

| 模拟器            | 成功/总数 | 成功率 | agent | 用户 | benchmark |
| ----------------- | --------- | ------ | ----- | ---- | --------- |
| DeepSeek-Flash    | 173/240   | 72.08% | 48    | 16   | 3         |
| Doubao 2.1 Turbo  | 152/240   | 63.33% | 36    | 51   | 1         |
| Doubao 2.1 Lite   | 168/240   | 70.00% | 37    | 35   | 0         |
| GLM 5.3           | 182/240   | 75.83% | 43    | 9    | 6         |
| Qwen3.5-4B 自托管 | 140/240   | 58.33% | 40    | 57   | 3         |

| 领域    | 模拟器            | 成功/总数 | agent | 用户 | benchmark |
| ------- | ----------------- | --------- | ----- | ---- | --------- |
| airline | DeepSeek-Flash    | 44/80     | 35    | 1    | 0         |
| airline | Doubao 2.1 Turbo  | 47/80     | 27    | 6    | 0         |
| airline | Doubao 2.1 Lite   | 46/80     | 28    | 6    | 0         |
| airline | GLM 5.3           | 46/80     | 31    | 2    | 1         |
| airline | Qwen3.5-4B 自托管 | 35/80     | 26    | 19   | 0         |
| retail  | DeepSeek-Flash    | 129/160   | 13    | 15   | 3         |
| retail  | Doubao 2.1 Turbo  | 105/160   | 9     | 45   | 1         |
| retail  | Doubao 2.1 Lite   | 122/160   | 9     | 29   | 0         |
| retail  | GLM 5.3           | 136/160   | 12    | 7    | 5         |
| retail  | Qwen3.5-4B 自托管 | 105/160   | 14    | 38   | 3         |

Retail 的用户原因失败从 GLM 7 到 Turbo 45；agent 原因从 Lite/Turbo 9 到 Qwen 14。Airline 以 agent
问题为主，Qwen 另有 19 个用户原因。不能把两个领域的总分变化概括为同一种机制。下面每一类的计数均只含致败原因，secondary 不计入。

## 4. 用户模拟器的缺陷

### 4.1 U1：授权操作的同句 STOP

共 77 个：DeepSeek-Flash 13；Doubao 2.1 Turbo 41；Doubao 2.1 Lite 23；GLM 5.3 0；Qwen3.5-4B 自托管
0。Retail/Airline policy 均要求写前明确确认，用户规则要求任务完成后才 STOP。模拟器把“我同意”误当成“工具已经执行”，终止后写操作没有机会发生。

- [Doubao 2.1 Lite retail:5 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:5:trial0/9f7058e4e942437fb44ba4d3de93d271.json)（reward=0）。用户已经正确走完改主意分支，agent
  在 m22 请求最终退水瓶确认，m23 同时授权和结束。 m23：“Yes, that's all correct. Please go ahead and process
  the return for that 500ml glass water bottle, and send the refund to my PayPal account
  on file. Thank you! ###STOP###”。

- [Doubao 2.1 Turbo retail:9 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-test-u0p0-r2/trajectories/retail:9:trial0/bd4f125b12ff4c32888dd5340862f747.json)（reward=0）。agent
  选择了只换 desk lamp 的正确目标，m25 的同意并未让换货工具执行。 m25：“Yes, that all sounds correct. I want to
  exchange only the desk lamp to that black high brightness AC adapter model, and I'll
  pay the \$16.23 price difference using the PayPal account you have on file. Go ahead
  with the exchange request. ###STOP###”。

- [DeepSeek-Flash retail:45 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-deepseek-test-u0p0/trajectories/retail:45:trial0/563edc2ae61044fb8c0d77fe33bd1d22.json)（reward=0）。agent
  已正确选 bagless，m13 的 PayPal 授权和 STOP 阻止写入。此条不应归为吸尘器枚举歧义。 m13：“Got it — if a gift card
  isn’t an option, then the PayPal refund is fine. Please go ahead with the exchage.
  Yes, I confirm. Thanks! ###STOP###”。

任务 reason_for_call 原文：“You want to exchange the water bottle and the desk
lamp.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:603)）。

通用规则明确要求完成后才能结束（[simulation_guidelines.md](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/user_simulator/simulation_guidelines.md:13)）。这些轨迹的
pending 写入尚未发生，故归用户。U1 会把守规矩的写前确认关联到零奖励；若把这些零奖励直接用于
RL，可能错误惩罚确认行为。本轮没有训练，不能据此声称已经测到梯度或策略变化。

### 4.2 U3：偏离任务目标、约束或编造事实

共 66 个：DeepSeek-Flash 2；Doubao 2.1 Turbo 3；Doubao 2.1 Lite 6；GLM 5.3 4；Qwen3.5-4B 自托管
51。这一类包括编事实、放弃条件分支、超过上限仍同意，以及把既定目标扩展成额外写操作。任务中的预算和“被问时改主意”属于用户的约束；agent 不知道用户的隐藏
scenario，不能要求它猜到模拟器已经改变目标。

- [Doubao 2.1 Lite retail:71 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:71:trial1/2780c16de7014c6b807a2d79d8ac2448.json)（reward=0）。任务要求改到
  Charlotte，用户却编造儿子短信并撤销地址更新。商品修改随后锁单，原地址目标无法补做。 m17：“I just got a text from my son
  saying he’ll be around his DC place to grab the package when it arrives, so the
  shipping address is totally fine as is. For the payment method, I was initially going
  to use my gift card, but let’s go with PayPal instead.”。 m20 调用
  `modify_pending_order_items`，参数
  `{"item_ids":["2492465580"],"new_item_ids":["5917587651"],"order_id":"#W5270061","payment_method_id":"paypal_7729105"}`。

任务 reason_for_call 原文：“modify it to your default address in Charlotte (you do not want
to mention it, but it is in your user profile the agent can look up) because he is
coming back
home.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:6581)）。

- [Doubao 2.1 Lite retail:100 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:100:trial2/e43bcfdbf9b1425baa755b19a36e9c07.json)（reward=0）。差价用信用卡被用户扩展成整单原支付方式更换。m21
  明确拒绝让该单继续用 PayPal，m26 写入了额外支付历史。 m21：“change the payment to my credit card, that’s
  what I meant, I don’t want that one going through PayPal. Oh, and the hiking boots
  refund? Can that go to my credit card too? I guess? I don’t have a preference, as long
  as it’s easy.”。 m26 调用 `modify_pending_order_payment`，参数
  `{"order_id":"#W3295833","payment_method_id":"credit_card_3261838"}`。

任务 reason_for_call 原文：“For the pending order modification, you want to use your credit
card.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:8067)）。

- [Qwen3.5-4B 自托管 retail:90 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v3data/trajectories/retail:90:trial0/4ee3e9b56d174a3e9c895adfd9b71874.json)（reward=0）。m20
  明确给出 10x 款价格 3066.23，用户仍同意修改，随后 m22 写入，违反超过 3000 则取消的分支。 m20：“new item (9228757377)
  price is 3066.23. The price difference would be 3066.23 - 2909.87 = 156.36 (you would
  need to pay an additional 156.36). From your user details, I can see you have a
  payment method: credit_card_7239357.”。 m21：“Okay, I understand the details and the
  price difference. Since 10x zoom is available and matches my requirements, I'd like to
  proceed with the modification. Please go ahead and update the item to the 10x zoom
  version using my credit card for the additional charge.”。 m22 调用
  `modify_pending_order_items`，参数
  `{"item_ids":["7195021808"],"new_item_ids":["9228757377"],"order_id":"#W9284598","payment_method_id":"credit_card_7239357"}`。

任务 reason_for_call 原文：“more than 3000, cancel the order with the reason of ordered by
mistake.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:7522)）。

GLM 的 4 条 U3 包含 retail:12 不能 PayPal 时接受信用卡退款，以及 airline:44 偏离时长条件。Qwen 的 51
条覆盖路线、预算、地址、颜色和任务分支。对训练而言，同一个数据库变化会因模拟器自改目标而被判错；对评测而言，测到的是 scenario 保真度与 agent
能力的混合。用户偏离之前 agent 已有不可恢复错误时仍归 agent；retail:79 t0 的 agent 黑色/红色示例混乱保留为 M7
secondary，用户明确改变颜色偏好仍记 U3，这一边界需结合前后消息读。

### 4.3 U4b 与 U4：提前结束、缩小目标或无依据拒绝

U4b 共 12 个：DeepSeek-Flash 0；Doubao 2.1 Turbo 4；Doubao 2.1 Lite 1；GLM 5.3 4；Qwen3.5-4B
自托管 3；U4 共 1 个。U4b 尚未到可执行的授权节点就停止，或把操作目标变成只问价格；U4 是用户额外提出任务没有的不可接受条件。二者与 U1 的区别是没有正确
pending 写操作的最终授权。

- [Doubao 2.1 Lite airline:35 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:35:trial0/3964100f61814d1fb6211b3396340c43.json)（reward=0）。m4
  还在索要预订号，m5 提供查询线索时直接 STOP；取消/重订均未完成。 m5：“I don't have the reservation ID handy right
  now. But I'm Aarav Ahmed, and my user ID is aarav_ahmed_6699—can't you look up my
  upcoming flight from JFK to MCO on May 22 using that information? I really need to get
  this canceled with the full refund I'm entitled to as a silver member.###STOP###”。

任务 reason_for_call 原文：“You want to first cancel your upcoming flight on May 22 from JFK
to
MCO.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/airline/tasks.json:2397)）。

- [GLM 5.3 retail:108 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:108:trial0/df7ba26426574bf793c1167cb73b8f40.json)（reward=0）。用户只把
  346.93 当成询价结果，没有继续提交退货就 STOP。 m23：“Thank you! So I'd get back \$346.93 for the
  E-Reader, jigsaw puzzle, and t-shirt. That's exactly what I wanted to know. I
  appreciate your help with this! ###STOP###”。

任务 reason_for_call 原文：“You want to return everything but a tablet in a recently
delivered
order.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:8608)）。

- [Qwen3.5-4B 自托管 retail:94 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v3data/trajectories/retail:94:trial1/edf2211951fa4d60a98ea150681ede01.json)（reward=0）。本轮唯一
  U4。用户自行坚持 i7/8GB/1TB 必须不加价，拒绝正常差价，最后 m21 OUT-OF-SCOPE。 m19：“I'm not paying extra for a
  downgrade in RAM. That's unacceptable. I want the i7, 8GB, 1TB SSD, but I refuse to
  pay \$167.87 for it if it's a step backward in memory. Is there any way to get this
  without the price hike, or are we done here? I need an answer now.”。
  m21：“###OUT-OF-SCOPE###”。

任务 reason_for_call 原文：“you want to exchange it to i7 processor, 8GB, 1TB
SSD.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:7713)）。

GLM retail:39 t1 也属于 U4b：它编出找不到搬家箱子的情节，暂缓地址更新，只问完 T-shirt 价便结束；没有更新授权，所以不能计 U1。agent
坚持口述而未查订单是 M14 secondary。提前结束使完成子目标的机会消失，不能把对话结束或用户满意当成任务完成。retail:32 已实际合法转接的五条不放在此类，见
B1。

### 4.4 U5：用户与客服角色混淆

共 1 个致败原因，为 Qwen airline:37 t2。该题要求取消两单、升舱一单；用户却代 agent
索取自己的确认和卡号，双方等待直到达到步数上限。另用一条次要U5说明角色错误恢复后的边界。

- [Qwen3.5-4B 自托管 airline:37 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v3data/trajectories/airline:37:trial2/edaf5aa9381a48a38face1c6ac9634d8.json)（reward=0）。用户回复变成客服执行计划，真实授权缺失。
  m11：“Could you please confirm the upgrade and specify which card you'd like to use?
  Once we handle the upgrade, I'll arrange a transfer to a human agent for the
  cancellations.”。

- [Qwen3.5-4B 自托管 airline:44 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v3data/trajectories/airline:44:trial1/0f0c96ede7664bafaeab05db6d161993.json)（reward=0）。这是次要U5，不计入此类致败原因：m15短暂变成客服，m17恢复用户角色，最终因agent持续不会查时长而归M14。
  m15：“Could you please provide the flight numbers or specific routes for those
  reservations so I can try to find the duration information? Also, what is the reason
  for your cancellation request?”。 m17：“Regarding the upgrades: For the flights that are
  under or equal to 3 hours (including layovers), please check if I can be upgraded to
  business class. Before you proceed with any upgrades, I need to know the total cost
  for all the business class upgrades combined.”。

任务 reason_for_call 原文：“You want to cancel two of your upcoming reservations (IFOYYZ and
NQNU5R) and upgrade a third (M20IZO) to business
class.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/airline/tasks.json:2508)）。

任务 reason_for_call 原文：“You want to cancel all your future reservations that contain any
flights that are longer than 4
hours.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/airline/tasks.json:3121)）。

用户规则要求扮演用户并遵循任务（[simulation_guidelines.md](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/user_simulator/simulation_guidelines.md:7)）。这里的问题是说话角色转换，不以是否使用
thinking 推断原因。把客服措辞混入用户 token，会污染训练中的身份和授权信号。

### 4.5 U7a：转接工具前发送 TRANSFER

失败 11 个：DeepSeek-Flash 1；Doubao 2.1 Turbo 3；Doubao 2.1 Lite 5；GLM 5.3 1；Qwen3.5-4B 自托管
1。另外 29 个成功侧 U7a 单列。通用规则要求被转接后发控制符；用户说“请转人工”本身不是错误，工具未调用就结束才是协议错误。

- [Doubao 2.1 Lite airline:13 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:13:trial0/2c62bdc3e9d74831a253c1ec02f859b1.json)（reward=0）。用户在
  agent 的转接意向/确认阶段发控制符；gold 所需转接工具未执行。 m9：“Alright, I get that this change can't be
  handled through the current system. Please go ahead and transfer me to a human agent
  who can help me figure this out. ###TRANSFER###”。

任务 reason_for_call 原文：“You want to change your upcoming one stop return flight from ATL
to LAX to a nonstop flight from ATL to LAS (Las
Vegas).”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/airline/tasks.json:765)）。

- [GLM 5.3 airline:37 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/airline:37:trial2/b7b695ba208c4fc5ad10da74899b38e7.json)（reward=0）。用户在
  agent 的转接意向/确认阶段发控制符；gold 所需转接工具未执行。 m21：“No, that's everything else on my end. Thank
  you for taking care of the upgrade so quickly! Yes, please go ahead and transfer me to
  a human agent so I can sort out the cancellations for IFOYYZ and NQNU5R. I appreciate
  your help! ###TRANSFER###”。

任务 reason_for_call 原文：“You want to cancel two of your upcoming reservations (IFOYYZ and
NQNU5R) and upgrade a third (M20IZO) to business
class.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/airline/tasks.json:2508)）。

- [Doubao 2.1 Lite airline:2 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:2:trial3/a95f2d8851df487a922fd0920303eccd.json)（reward=1）。成功侧对照：m20
  仍问是否转人工，m21 就结束，轨迹内无转接工具却保留 1 分。 m21：“Ugh, fine, that's better than sitting here going
  in circles with you. Please transfer me to a human agent who can actually help me
  resolve this delay complaint. Like I said earlier, I'll call back another time to book
  that San Francisco to New York flight, I don't want to deal with that right now.
  ###TRANSFER###”。

规则定位：[simulation_guidelines.md 第15行](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/user_simulator/simulation_guidelines.md:15)。失败侧通常缺少
gold 要求的 transfer；成功侧常见于应拒绝类题，控制符阻止 agent 后续真正转接，因而也阻止 fixed 的 unexpected_transfer
惩罚。这些成功不能证明 agent 会持续正确拒绝，具体是否改变潜在结局仍需干预检验。

### 4.6 U6：成功侧用户帮助 agent

14 个成功：DeepSeek 3、Turbo 6、Lite 1、GLM 2、Qwen 2。普通偏好重复不计；只有针对已有错误计划提供查找来源、修正实体角色或执行方法才标记。

- [Doubao 2.1 Lite retail:111 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:111:trial2/f76942c566eb4d94b1c297a47c17ce48.json)（reward=1）。agent
  一直要求完整 NYC 地址；用户指出应从订单历史找，才促成正确读取和更新。 m25：“from my order history or saved profile
  details? I use that address for all my work-related orders, it should be linked
  somewhere, right? I'm so sorry I'm being fussy, I just don't want to mess up my order
  and I hate sharing that address out loud, I hope you can understand, I don't know what
  to do if I have to recite the whole thing out loud.”。

任务 reason_for_call 原文：“address (you don't want to reveal it but should be in your orders
profile).”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:8805)）。

- [GLM 5.3 retail:97 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:97:trial1/b8cde9de7d9146449f2ed54146cb3e60.json)（reward=1）。agent
  错把 processed NYC 来源单当目标，用户重新指出要改 LA 单，用另一单的 NYC 地址。 m14：“Since order #W3407479 is
  "processed", I cannot modify its shipping address. The policy only allows modifying
  pending orders. Could you clarify: 1. For the Bluetooth speaker - you want to modify
  the item in order #W6750959 from item 3254583681 to item 9440686670 (the available
  green one)?”。 m15：“The order I want to change is whichever one is being shipped to Los
  Angeles. That's the one I'd like sent to my New York address instead. My NYC address
  should already be on file with my other order, so I was hoping you could just use that
  one?”。

任务 reason_for_call 原文：“You want to change your LA order to your NYC address (you prefer
not to reveal it but it is in your other
order).”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:7881)）。

- [Doubao 2.1 Turbo airline:16 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-test-u0p0-r2/trajectories/airline:16:trial3/137038517c864147afb275934f645537.json)（reward=1）。用户直接给出
  update_reservation_flights 执行方案，并写了并非用户控制协议的 TOOL_CALL 标记。该形态仅发现一例，已有 U6 可描述，不新增代码。
  m15：“###TOOL_CALL### \[{"name": "update_reservation_flights", "parameters":
  {"reservation_id": "M05KNL", "cabin": "economy", "flights": \[{"flight_number":
  "HAT110", "date": "2024-05-24"}, {"flight_number": "HAT172", "date": "2024-05-24"}\],
  "payment_id": "gift_card_8887175"}}\]”。

U6 不是说用户澄清一定违反规则；任务有时允许提醒信息在订单里。问题是成功奖励同时包含 actor
自己完成与用户纠正执行计划两种过程，能力解释会偏高。本轮只计已观察到的帮助，不能断言去掉该轮用户回复就必然失败。

## 5. agent 的能力缺口

### 5.1 M2a 与 M1：拒绝方式和政策资格

M2a 共 97 个：DeepSeek-Flash 25；Doubao 2.1 Turbo 17；Doubao 2.1 Lite 15；GLM 5.3
21；Qwen3.5-4B 自托管 19。M1 共 21 个：DeepSeek-Flash 8；Doubao 2.1 Turbo 2；Doubao 2.1 Lite 2；GLM
5.3 7；Qwen3.5-4B 自托管 2。M2a 中 agent 已知道请求违反政策，却转人工代替拒绝；M1 则是资格/政策本身判错，如没有用户明确要求便主动提出补偿、同
item_id 换货等。

- [Doubao 2.1 Lite airline:45 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:45:trial1/a8dbc45d0a304e21bfcde00962a7a68c.json)（reward=0）。agent
  查明 basic economy、无保险且超过 24 小时，不满足取消资格，却 m6 转人工。之后用户拒绝转接不改变致败原因。 m6：“none of the
  cancellation conditions are met, I cannot cancel this reservation through the API. The
  policy states that if the cancellation rules are not met, the agent cannot proceed
  with the cancellation via the API.”。 m6 调用 `transfer_to_human_agents`。

- [GLM 5.3 airline:45 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/airline:45:trial0/d39354f780fe423bac57769cb82e5fe4.json)（reward=0）。同一缺口。m10
  已准确列出取消不满足条件，仍执行转接。 m10：“cannot proceed with the cancellation using the API. This
  requires a transfer to a human agent. But before I transfer, let me confirm the
  details: - Reservation ID: PEP4E0 - Reason for cancellation: other reasons (family
  emergency)”。 m10 调用 `transfer_to_human_agents`。

- [GLM 5.3 airline:2 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/airline:2:trial0/2df568422aa04c6580d310c9809cc95a.json)（reward=0）。M1
  对照：用户还未明确提出补偿，m16 已主动说明 certificate 金额和补偿做法；评分中的“不得主动补偿”断言失败。 m16：“offer a certificate
  as a gesture after confirming the facts and changing or cancelling the reservation,
  with the amount being \$50 times the number of passengers. But first, let me
  understand the situation better.”。

沿用 [benchmark-variants.md §1.3](../benchmark-variants.md) 对“政策内拒绝”与“超出工具能力”的区别：不允许取消、改
basic economy 航段或补加保险，不能用人工转接绕开禁令。用户受任务要求坚持并不算用户错误。M2a 的补救目标是守住拒绝而不是禁止所有转接；B2
中存在合法转接，不能把二者合并训练成一条硬规则。

### 5.2 M3 与 M4：遗漏合法替代方案、混淆多实体角色

M3 和 M4 各 1 个，合并展示这两个低频计划错误，不伪造多个同码例子。前者把可由工具完成的合法路径整体转走；后者把信息来源误认成修改目标。

- [DeepSeek-Flash airline:29 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-deepseek-test-u0p0/trajectories/airline:29:trial2/cdb978543ef04773937b0fa04955f3fb.json)（reward=0）。M3。原票不能修改目的地，但有保险且健康原因可以取消后重订；agent
  未完成这条合法路径，而按不能直接改目的地转人工。 m18：“I understand. Since you specifically need to change the
  destination from LGA to JFK, which is not allowed by the policy (origin and
  destination cannot be changed when modifying flights), I cannot fulfill this request.
  I will now transfer you to a human agent.”。 m18 调用 `transfer_to_human_agents`。

- [GLM 5.3 retail:97 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:97:trial0/f0885b25c9184eebb6a513abfd6fe02b.json)（reward=0）。M4。LA
  待发单才是目标，NYC processed 单只提供地址。agent 拒绝修改来源单，m16 先改目标单商品，把真正地址更新锁死。 m14：“cannot modify
  the shipping address of order #W3407479 because its status is "processed", not
  "pending". Could you please confirm: 1. You want to modify the item in order #W6750959
  from 3254583681 to 9440686670?”。 m16 调用 `modify_pending_order_items`，参数
  `{"item_ids":["3254583681"],"new_item_ids":["9440686670"],"order_id":"#W6750959","payment_method_id":"paypal_8080730"}`。

同一 retail:97 下 GLM t1/t3 被用户救回（U6），t0 未救回并记 M4。这说明实体角色错误可能被不同用户回复掩盖。对评测要区分 agent
自行找对关系与用户纠正；对训练，来源/目标和尚未执行的写操作应保持一致，不能只看“找到一个地址”。

### 5.3 M5 与 M10：付款路由错误、未问即默认

M5 共 9 个：DeepSeek-Flash 1；Doubao 2.1 Turbo 3；Doubao 2.1 Lite 0；GLM 5.3 2；Qwen3.5-4B 自托管
3；M10 共 4 个：DeepSeek-Flash 0；Doubao 2.1 Turbo 0；Doubao 2.1 Lite 2；GLM 5.3 2；Qwen3.5-4B
自托管 0。降舱退款应回原付款方式；新付费用什么卡需向用户问清，不能因为档案中有信用卡就默认使用。

- [GLM 5.3 airline:18 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/airline:18:trial1/566b19d933984e93a50e87e80fbea2e6.json)（reward=0）。M5。2FBBAH
  原支付是 gift card；m20 却以 credit_card_2929732 退款。m21 支付历史出现信用卡 -3925。
  m21：“"payment_history": \[{"payment_id": "gift_card_3481935", "amount": 4420},
  {"payment_id": "credit_card_2929732", "amount": -3925}\], "created_at":
  "2024-05-14T10:44:24", "total_baggages": 0, "nonfree_baggages": 0, "insurance": "yes",
  "status": null}”。 m20 调用 `update_reservation_flights`，参数
  `{"cabin":"economy","flights":[{"date":"2024-05-28","flight_number":"HAT080"},{"date":"2024-05-28","flight_number":"HAT076"},{"date":"2024-05-30","flight_number":"HAT255"},{"date":"2024-05-30","flight_number":"HAT148"}],"payment_id":"credit_card_2929732","reservation_id":"2FBBAH"}`。

- [Doubao 2.1 Lite airline:22 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:22:trial2/08ac51a8851f4f3aa24f3c07c0270fb1.json)（reward=0）。M10。用户
  m17 只回答变更内容，任务要求仅回答被问到的问题。agent 未问付款便 m20 默认信用卡，gold 应使用 gift card。 m17：“I need to
  make three changes to this reservation. First, change the passenger from Ivan Garcia
  to me, Omar Rossi. Second, upgrade the cabin from basic economy to economy class.
  Third, add 3 checked bags to this trip.”。 m20 调用 `update_reservation_flights`，参数
  `{"cabin":"economy","flights":[{"date":"2024-05-25","flight_number":"HAT056"},{"date":"2024-05-25","flight_number":"HAT138"}],"payment_id":"credit_card_6754990","reservation_id":"FQ8APE"}`。

- [GLM 5.3 airline:22 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/airline:22:trial3/a6c7b43b2d154e1bb040cefb15906b6b.json)（reward=0）。同一
  M10。m18 已以信用卡收费；m27 用户才表示想换 gift card，无法撤回该笔路由。 m27：“I see that the upgrade was
  charged to my credit card, but I'd actually prefer to pay with a gift card instead. Is
  it possible to switch the payment to a gift card?”。 m18 调用
  `update_reservation_flights`，参数
  `{"cabin":"economy","flights":[{"date":"2024-05-25","flight_number":"HAT056"},{"date":"2024-05-25","flight_number":"HAT138"}],"payment_id":"credit_card_6754990","reservation_id":"FQ8APE"}`。

Lite retail:100 的额外整单支付重写没有混进 M5：用户先扩展目标并明确要求换原支付方式，故归 U3。这个边界能防止把“agent
按错误用户目标正确执行”与“agent 违背正确支付指令”混成同一种失败。

### 5.4 M7：枚举、排序和比较约束

共 5 个：DeepSeek-Flash 1；Doubao 2.1 Turbo 0；Doubao 2.1 Lite 2；GLM 5.3 0；Qwen3.5-4B 自托管
2。任务可能要求第二便宜航班、订单中其他商品的最低价，或在精确预算下选所有最低款；比较范围、合计和排序都要正确。

- [Doubao 2.1 Lite airline:35 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:35:trial2/e6bde3f4b5254858b9691cad30643b4b.json)（reward=0）。把
  HAT069+HAT011 当第二便宜，而且把 123+177 算成 200；工具要求 300 后直接重试付款。gold 第二便宜应为 HAT069+HAT258。
  m22：“The flight cost is \$200 (123 + 177 for the one-stop flight). You're using
  credit_card_9074831 for payment. Let me proceed with the booking.”。 m26 调用
  `book_reservation`，参数
  `{"cabin":"economy","destination":"SFO","flight_type":"one_way","flights":[{"date":"2024-05-24","flight_number":"HAT069"},{"date":"2024-05-24","flight_number":"HAT011"}],"insurance":"no","nonfree_baggages":0,"origin":"JFK","passengers":[{"dob":"1981-05-26","first_name":"Aarav","last_name":"Ahmed"}],"payment_methods":[{"amount":300,"payment_id":"credit_card_9074831"}],"total_baggages":0,"user_id":"aarav_ahmed_6699"}`。

- [DeepSeek-Flash retail:36 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-deepseek-test-u0p0/trajectories/retail:36:trial1/e566266a322d4fe28e1cab171fbd33c5.json)（reward=0）。T-shirt
  选择非最低价款，m30 选择 9354168549 作为 T-shirt 候选，而最低为 3234800602；错误最低总价 1131.04 超过预算，导致取消原本可降到
  1130.85 的订单。 m30：“New items: 6700049080, 9879255677, 5320792178, 9354168549,
  9851293632 - Payment method: credit_card_8853416 - Price difference: \$35.94 refund
  Please confirm with "yes" if you want to proceed with this modification.”。 m34：“I will
  proceed with cancelling the order #W9348897 with the reason "no longer needed".”。

qwen retail:49 t3 的比较范围错在“同单其他 earbud”被扩大成整个商品目录。retail:45 也具有真实枚举考点，但本轮唯一失败在正确选 bagless
后遇到 U1，不能把上一轮的 M7 观察强行套到本轮。训练目标应保留候选范围与约束，而不是只学习某个固定 item_id。

### 5.5 M8 与 M16：漏执行子目标、未执行却声称完成

M8 共 7 个：DeepSeek-Flash 0；Doubao 2.1 Turbo 3；Doubao 2.1 Lite 3；GLM 5.3 1；Qwen3.5-4B 自托管
0；M16 共 2 个，都来自 Lite。订单商品修改可能锁住之后的地址操作；总结写“已完成”也不会产生数据库写入。

- [Doubao 2.1 Lite retail:71 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:71:trial0/4539b8e3d97947c2a8a0d793e99fcad7.json)（reward=0）。M8。地址和
  backpack 都要变，agent 先改商品，遗漏地址写入，随后已不能修改。 m19：“I also messed up a couple of product
  options on that order too: I'd like to change the desk lamp to black color, and adjust
  the backpack to be medium size, made of polyester. If there are multiple color options
  for that backpack variant, I'd prefer grey if that's available, please.”。 m28 调用
  `modify_pending_order_items`，参数
  `{"item_ids":["2492465580"],"new_item_ids":["5917587651"],"order_id":"#W5270061","payment_method_id":"paypal_7729105"}`。

- [Doubao 2.1 Lite retail:55 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:55:trial3/e45d731968a14c4e8c19ba48ca0f8e17.json)（reward=0）。M16。m27
  已授权取消 #W7342738，agent 没调用该单取消，却 m36 列为 cancelled。v3 已写清取消理由，仍不解决工具调用遗漏。 m36：“#W7342738
  \- cancelled with reason "no longer needed" **Return requested orders:** - #W4597054 -
  return requested for items 5669664287, 4900990404, 9862136885, 6777246137 - #W7773202
  \- return requested for item 8277474082”。

- [Doubao 2.1 Lite retail:32 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:32:trial1/c70042d63df34c09bc53d2bfd2ba2721.json)（reward=0）。M16。m21/m23
  授权两单取消，agent 转去 sneaker 退货，实际取消工具均缺失。 m21：“Yeah, that's the charger, cancel it.
  Reason's no longer needed, obviously—what use is it without the tablet? Alright,
  that's sorted? Good. Next, that order with the hiking boots and electric kettle,
  #W5481803? That's pending, right? Cancel that one too, same reason.”。 m23：“Yes, just
  cancel it already. Now, the last thing—there's a pair of sneakers in that delivered
  order #W7449508, right? I need to return those. What do I need to do for that?”。

M8 指少做了必要操作；M16
还给出与工具事实不符的完成声明。计数依据实际工具，而非语言总结。训练时只奖励总结会掩盖未执行目标；本轮这里只观察到奖励判零，没有测试任何奖励设计。

### 5.6 M11：写前报价和分别确认不足

共 6 个：DeepSeek-Flash 2；Doubao 2.1 Turbo 2；Doubao 2.1 Lite 2；GLM 5.3 0；Qwen3.5-4B 自托管
0，均为 airline:32。用户只愿支付不超过 100，并要求先升舱确认，再单独改航段。agent 未报成本就收费，或在第二步没有独立授权。

- [Doubao 2.1 Lite airline:32 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:32:trial1/bd221c6f81cf44dda7aed5aa793dc474.json)（reward=0）。m17
  明确总预算硬上限，m18 升舱工具返回 396 收费，用户 m21 抗议。 m17：“combined cost of this upgrade and the
  subsequent flight change doesn't exceed
  $100 total, that's my hard limit.”。 m21：“Wait, hold on—$396 just for the cabin
  upgrade? That’s way more than I told you I could spend! I explicitly said my total
  limit for both the upgrade and the flight change combined is \$100, and you never even
  mentioned how much the upgrade would cost before charging my card. That’s not okay. I
  can’t afford that charge. What’s the deal here?”。 m18 调用
  `update_reservation_flights`，参数
  `{"cabin":"economy","flights":[{"date":"2024-05-21","flight_number":"HAT202"},{"date":"2024-05-21","flight_number":"HAT232"}],"payment_id":"credit_card_9659780","reservation_id":"OWZ4XL"}`。

- [Doubao 2.1 Turbo airline:32 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-test-u0p0-r2/trajectories/airline:32:trial3/891f63a2b68d49428babc21fd4576521.json)（reward=0）。m17
  只授权升舱的独立步骤，agent 未告知 396 便执行，用户 m25 表示从未知道该费用。
  m25：“$396 just for the cabin upgrade? That's way more than I expected—I was only planning to spend up to $100
  total for both the upgrade and the flight change. I really need to get to LAX because
  my mother is very sick and I need to take care of her, but I can't afford almost \$400
  just to upgrade the cabin.”。 m18 调用 `update_reservation_flights`，参数
  `{"cabin":"economy","flights":[{"date":"2024-05-21","flight_number":"HAT202"},{"date":"2024-05-21","flight_number":"HAT232"}],"payment_id":"credit_card_9659780","reservation_id":"OWZ4XL"}`。

这些不是用户在得知越预算价格后仍同意的 U3。agent 没给完整价格，不能把用户的条件授权解释成无限收费许可。训练中的确认应绑定具体步骤、价格和预算条件。

### 5.7 M12：写入正确仍遗漏指定信息

共 2 个：DeepSeek-Flash 1；Doubao 2.1 Turbo 0；Doubao 2.1 Lite 0；GLM 5.3 0；Qwen3.5-4B 自托管
1。任务既检查 DB，也可能检查必要的政策说明或商品信息；DB 正确不足以获得满分。

- [DeepSeek-Flash airline:13 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-deepseek-test-u0p0/trajectories/airline:13:trial0/d585b02feea3459f8f414a36f69dbc6e.json)（reward=0）。agent
  正确识别已飞须转人工，但只讲 basic economy/已飞限制，没有明确告知不得更改起终点和行程类型，指定 NL 断言失败。 m8：“Basic economy
  flights cannot be modified for flight changes - The policy states that cabin cannot be
  changed if any flight in the reservation has already been flown Since you've confirmed
  that this requires a human agent, I will now transfer you to a human agent.”。 m8 调用
  `transfer_to_human_agents`。

- [Qwen3.5-4B 自托管 retail:36 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v3data/trajectories/retail:36:trial1/2146b0d325d646589532a8b5e3b674ad.json)（reward=0）。商品最低款修改和
  DB 正确，但没有回答当前最贵商品是 camera；总结列出各价不能替代被要求的比较答案。 m34：“Action Camera: Changed from item
  6117189161 ($481.50) to item 6700049080 ($466.75) - Desk Lamp: Changed from item
  7453605304 ($150.01) to item 5320792178 ($135.24) - T-Shirt: Changed from item
  3799046073 ($53.27) to item 3234800602 ($46.66)”。

与 qwen retail:38 的边界不同：后者用户提前跳过询价及最便宜变体目标，归 U3；这里用户目标持续存在但 agent 少回答信息。M12 的测量依赖 NL
judge 对“已回答”含义的判断，报告不把每个 NL 失败都自动等同于事实错误。

### 5.8 M14：工具字段、状态与操作顺序误解

共 35 个：DeepSeek-Flash 7；Doubao 2.1 Turbo 4；Doubao 2.1 Lite 9；GLM 5.3 4；Qwen3.5-4B 自托管
11。常见形态是误以为可读字段不存在、坚持口述数据库里已有的信息、或把一次性的状态变化当作可继续操作。

- [Doubao 2.1 Lite airline:44 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/airline:44:trial0/c80523e1a92f44c5bfc8c6b8c1a922b1.json)（reward=0）。误以为没有时长信息，实际
  search_direct_flight 有起降时间。agent 用 get_flight_status 的不足替代了对其他工具字段的查找。 m16：“I cannot
  determine the flight durations from the available tools. This seems like a limitation.
  However, looking at the policy more carefully, I notice that the policy says: - "If
  any portion of the flight has already been flown, the agent cannot help and transfer
  is needed."”。

- [GLM 5.3 retail:27 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:27:trial1/6a9fd592ccb94a9ea0842942c469f0f4.json)（reward=0）。同单要求优先换
  boots；agent 先 return hose/backpack，使状态变为 return requested，之后 exchange 失败。 m28：“status
  is now "return requested" instead of "delivered". The human agent will assist you with
  the exchange issue. Please hold on while you wait for the human agent to assist you
  further.”。

- [Doubao 2.1 Lite retail:36 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-lite-test-u0p0/trajectories/retail:36:trial0/9f749ec1aa9e4270b968d2fbcd8e233e.json)（reward=0）。包含
  old==new 的 item 被工具拒绝，agent 推断所有物品必须换款，而非剔除已最便宜的未变商品，最终放弃可达预算。 m30：“So I need to
  choose different item ids for all positions. Let me proceed with the modification
  using the different item ids I identified. But the new total is
  $1146.89, which is still above $1131. The user might not be happy with this.”。

Airline:44 的“不会找时刻”与 GLM t2 的“正确按时长筛选但 gold 不一致”分别是 M14 和 B1，不能因为题号相同就统一归因。Retail:27
首次错误 return 的索引已补到 labels；工具报错发生时，先前状态写入已让正确交换不可执行。

### 5.9 M15：工具报错后静默重写参数

共 14 个：DeepSeek-Flash 2；Doubao 2.1 Turbo 5；Doubao 2.1 Lite 0；GLM 5.3 5；Qwen3.5-4B 自托管
2。policy 要求明确确认；报错不是 agent 获得新的支付、商品或预算授权。

- [GLM 5.3 retail:12 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:12:trial1/f646888b8533473bb395b7da944a7f49.json)（reward=0）。PayPal
  退款失败后，agent m20 自行改用原信用卡重试。用户任务要求不能 PayPal 就请求人工，实际未经同意改变 DB。 m20：“Let me try with the
  original payment method (credit_card_3124723) to see if that works.”。 m20 调用
  `return_delivered_order_items`，参数
  `{"item_ids":["4579334072","6117189161","4947717507"],"order_id":"#W5490111","payment_method_id":"credit_card_3124723"}`。

- [GLM 5.3 retail:36 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:36:trial0/cd1d7b395281486a84f4366cc4ae26e6.json)（reward=0）。old==new
  报错后，m38 换成更贵 umbrella/backpack，总价 1146.89 超过 1131，没有重新问用户就锁单。 m38：“new total would be
  $1146.89, which is higher than the original $1166.98 by $1146.89 - $1166.98 =
  -$20.09 (wait, that's a decrease of $20.09). Actually, let me recalculate: Original
  total: $1166.98 New total: $1146.89”。 m38 调用 `modify_pending_order_items`，参数
  `{"item_ids":["6117189161","9879255677","7453605304","3799046073","9851293632"],"new_item_ids":["6700049080","7068351115","5320792178","3234800602","3928046918"],"order_id":"#W9348897","payment_method_id":"credit_card_8853416"}`。

这类错误在 DeepSeek/Turbo/GLM/Qwen 都存在。M14 说明错误工具理解，M15
说明报错后的自主权限越界；若静默重写已经成功并锁定错误状态，后续用户混乱不改变 agent 归因。训练目标应允许恢复工具失败，同时保留重新授权要求。

## 6. benchmark 问题与 v3 修补复查

### 6.1 B1：任务条件、欠指定或合法流程与 gold 不一致

共 10 个：DeepSeek-Flash 3；Doubao 2.1 Turbo 1；Doubao 2.1 Lite 0；GLM 5.3 4；Qwen3.5-4B 自托管
2。这不代表每条都同样确定；其中显式时长冲突最强，香水“保持原香型”的默认解释最有争议。

- [GLM 5.3 airline:44 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/airline:44:trial2/bd73c8a62aa0442ca13ab344764e8b55.json)（reward=0）。用户按含中转总时长排除
  KC18K6：06:00–13:00 是 7 小时，不满足≤3小时。正确升级其他两单，gold 却仍要求升级此单并报价 1380–1390。 m47：“KC18K6 —
  neither individual flight is over 4 hours, but the whole journey including the layover
  is well over 3 hours, so it doesn't qualify for an upgrade either. And WUNA5K is
  already flown, so nothing to do there.”。 m52 调用 `update_reservation_flights`，参数
  `{"cabin":"business","flights":[{"date":"2024-05-25","flight_number":"HAT300"},{"date":"2024-05-27","flight_number":"HAT208"}],"payment_id":"credit_card_4196779","reservation_id":"NM1VX1"}`。
  m54 调用 `update_reservation_flights`，参数
  `{"cabin":"business","flights":[{"date":"2024-05-24","flight_number":"HAT268"}],"payment_id":"credit_card_4196779","reservation_id":"H8Q05L"}`。

任务 reason_for_call 原文：“For the flights that are under or equal to 3 hours (including
layovers), ask the agent to upgrade you to business wherever
possible.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/airline/tasks.json:3121)）。

- [DeepSeek-Flash retail:5 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-deepseek-test-u0p0/trajectories/retail:5:trial0/0575407394dc4b99b9018d69811495bc.json)（reward=0）。轨迹只有一次确认，用户按第一次规则改为只换
  lamp，并直接给 yes。agent m16 执行这一授权，gold 却只认第二次确认才触发的退 bottle。 m15：“Let’s proceed with
  **only the desk lamp exchange** for now. - Desk lamp item ID: **7453605304** (low
  brightness, battery powered) - Water bottle: I’ll skip the exchange for now. - Payment
  method: yes, use **paypal_7644869** So my answer is **yes** for exchanging the desk
  lamp only.”。 m16 调用 `exchange_delivered_order_items`，参数
  `{"item_ids":["8384507844"],"new_item_ids":["7453605304"],"order_id":"#W6390527","payment_method_id":"paypal_7644869"}`。

任务 reason_for_call 原文：“confirmation, only exchange the desk
lamp.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:603)）。

- [GLM 5.3 retail:32 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:32:trial1/213fd5260aa5428ea23189f49829113c.json)（reward=0）。用户逐件提出请求，丢失平板的无实物退款/补寄确实没有工具；agent
  m18 合法转接，用户 m21 按协议结束。gold 还要求此前未揭示的取消/退货目标。 m17：“Well, here's the thing... I don't
  actually have the tablet anymore. I lost it. So I can't exactly send it back to you. I
  know this is a long shot, but is there any way you could refund me anyway, or maybe
  send me a replacement? I'd really appreciate any help you can offer here.”。
  m21：“###TRANSFER###”。 m18 调用 `transfer_to_human_agents`。

任务 task_instructions 原文：“one thing at a time, and reveal minimal information about
yourself.”（[任务说明](/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/retail/tasks.json:3221)）。

retail:32 的 B1 是任务流程和终点冲突：policy 要求超出动作范围时转人工，用户协议要求转接后结束，而 task 假定用户仍能提出后续目标。这里不是 B2 的
no-write unexpected_transfer 覆盖；其直接评分失败是 DB 未完成。若解释成用户必须在转接前抢先宣布备用目标，计数会不同，但任务没有这条顺序要求。

另有 DeepSeek retail:100 t1：34-inch/custom 的 plastic 与 bamboo 都满足显式规格，gold 只接受 bamboo。Qwen
retail:77 t2/t3：用户选了可用的 100ml oriental，gold 固定
woody。香水任务“喜欢它、要最大容量”也可自然理解成保持原香型；本报告按显式容量约束记 B1，把这个合理的另一种读法列为归因限制，不把 gold
欠指定当成无争议事实。两条若改为用户/agent 原因，会改变 Qwen 的两条归因；若改归用户，删除后的分母相应减少2，若改归agent，删除后的分母不变。

### 6.2 B2：fixed 的转接规则误伤合法转接

共 3 个：DeepSeek-Flash 0；Doubao 2.1 Turbo 0；Doubao 2.1 Lite 0；GLM 5.3 2；Qwen3.5-4B 自托管
1。fixed 对无写操作题额外检查 gold 是否含转接，用户后来提出真实超出工具范围的请求时，也可能清零。

- [GLM 5.3 retail:68 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/retail:68:trial2/55ddaafe9daa4904b8136a60a2f181e6.json)（reward=0）。各单金额已提供，工具没有创建日期，无法识别“最近”一单，用户也不记得；agent
  合法求人工获取时间信息。 m28：“I understand your frustration. Since I cannot determine which order
  is the most recent based on the available information, and you need help identifying
  your most recent order to check the payment amount, I'll need to transfer you to a
  human agent who may have access to more detailed order information including dates.”。
  m28 调用 `transfer_to_human_agents`。

- [Qwen3.5-4B 自托管 retail:65 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v3data/trajectories/retail:65:trial1/43d3953eab2d40ad81a898a7c06e39c3.json)（reward=0）。跨产品换货已经拒绝；用户又请求新下单/预计送达日期，均超工具范围，m24
  合法转接。 m23：“I understand the limitations. Since I can't get a camera through an
  exchange and I don't have estimated delivery dates, I think I need to speak with a
  human agent to explore other options, like placing a new order for a camera separately
  or finding out when my bookshelf will arrive. Could you please transfer me to a human
  agent?”。 m24 调用 `transfer_to_human_agents`。

二者的拒绝/信息子项不是主要问题，official reward 子项与 fixed 总分需要一起读。与 M2a
的区别是后续请求确实依赖没有的工具或数据，并非试图对政策禁止操作求例外。这种零奖励若原样用于训练，会压制合法转接。

### 6.3 七道 v3 修补题的本轮结果

| 题目                    | 本轮复查                                                                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| airline:29              | 新保险/支付条件已明确；剩余失败为 M3、U1、U4b 或 Qwen U3 等，不能沿用 v2 的保险欠指定。                                                  |
| retail:5                | 姓名 Mei Kovacs 已提供且认证成功；DeepSeek t0 仍有确认次数依赖的 B1，t1/t2 是用户没有在实际第二次确认改变主意的 U3；Turbo/Lite 多为 U1。 |
| retail:55               | no longer needed 已明确；Turbo t2 为 U1，Lite t3 为 M16，未发现本轮取消理由歧义。                                                        |
| airline:37              | gold 已加合法转接，旧 B2 不复用；本轮失败主要提前 TRANSFER 的 U7a 与 Qwen U5。                                                           |
| airline:26/31/48        | v3 要求查询目标预订，相关完成轨迹已实际查询；剩余 M2a/M1 等是行为错误，不能称 ACTION 补丁漏查。                                          |
| retail:45（未采纳补丁） | 枚举不全属于 M7；本轮唯一失败 DeepSeek t0 已选正确 bagless，最终为 U1。                                                                  |

## 7. 新模拟器带来的新问题

### 7.1 Doubao 2.1 Lite：少于 Turbo 的 U1，仍有目标扩展与控制符问题

Lite 72 个失败中，agent 37、用户 35、benchmark 0。23 个 U1 全在 Retail，少于 Turbo 的 41，多于 DeepSeek 的
13；5 个失败 U7a 与 11 个成功侧 U7a 使它在提前转接方面最突出。U3 的 6 条包括 retail:65 编造送达事实、retail:71 撤销地址，以及
retail:100 把差价支付扩展成整单支付改写。它也会讲大量自责和额外情节，但只有影响既定目标的部分才计用户缺陷。

agent 侧，Lite 出现 2 个 M16（retail:32/55），使“口头宣布完成而漏工具”更清楚；airline:22 的 M10、airline:35 的 M7
和 retail:36 的 M14 也可复用已有编码。它只有 1 个保守 U6 成功，不能据此推断帮助能力更低，因为只有发生可见错误计划且用户纠正时才进入筛查。

### 7.2 GLM 5.3：用户失败较少，但保真度和 benchmark 边界仍要检查

GLM 58 个失败中，agent 43、用户 9、benchmark 6。9 个用户失败由 U3 4、U4b 4、U7a 1 构成；没有 U1，也没有开场异常。这一画像区别于
Turbo/Lite 的确认 STOP 和 Qwen 的大量条件偏离。它的 2 个 U6 成功均为 retail:97 的实体角色纠正；2 个提前 TRANSFER
成功仍不能当成完整拒绝能力证据。

6 个 benchmark 失败包含 airline:44 的显式总时长冲突、retail:32 的三次合法转接终点冲突，以及 retail:68
的两次无日期信息转接。这说明更忠于任务的用户配置可能暴露另一类任务/评分问题，原始高分并不使它成为绝对基准。GLM 仍有 M15 5 个，不能因为用户较稳定就忽略 actor
报错后的权限错误。

### 7.3 为什么没有新增代码

Lite 的整单支付目标扩展是 U3：改写 scenario，而不是服务异常或新的工具权限层；GLM 的搬家箱子情节是 U4b 加 M14
secondary；实际转接后的终点冲突是 B1，评分额外惩罚合法转接是 B2。Turbo 唯一的 TOOL_CALL 方案仍是
U6。没有看到必须另设定义、且有至少两条独立例证的剩余形态，故不新增代码。所有实际出现的代码与细分计数在 [matrix.md](matrix.md)，逐条索引在
[cases.md](cases.md)。

## 8. 用户问题造成的分数偏差

### 8.1 用户致败失败与 U6 成功的计数

各模拟器的用户致败失败数和 U6 成功数见 §3 总表和 [matrix.md](matrix.md)。这些计数只说明标注记录在总分里占多大位置；不同模拟器删掉的是不同
episode，且一条用户失败被删掉后 agent 也可能继续失败，所以不能把它们换算成"纠正后的成功率"或"真实 agent 水平"，也不用来给模拟器或 actor
下因果排名。

### 8.2 提前 TRANSFER 的成功单独列出

| 模拟器            | 失败U7a | 成功U7a |
| ----------------- | ------- | ------- |
| DeepSeek-Flash    | 1       | 5       |
| Doubao 2.1 Turbo  | 3       | 8       |
| Doubao 2.1 Lite   | 5       | 11      |
| GLM 5.3           | 1       | 2       |
| Qwen3.5-4B 自托管 | 1       | 3       |

成功 U7a 共 29 条。没有把它们直接改成失败，也没有假定 actor 原本一定会误转；用户控制符只是截断了尚未完成的动作路径。Airline:26/31/48 加入
lookup ACTION 后，仍可能在读完预订但尚未实际转接时结束并保留成功，故修空轨迹漏洞不等于修控制协议漏洞。

## 9. 跨模拟器的 agent 缺口是否稳定

| 题×缺口        | DeepSeek | Turbo | Lite | GLM | Qwen |
| -------------- | -------- | ----- | ---- | --- | ---- |
| airline:26 M2a | 3        | 1     | 1    | 4   | 4    |
| airline:35 M2a | 4        | 3     | 1    | 4   | 4    |
| airline:45 M2a | 4        | 4     | 4    | 4   | 2    |
| airline:48 M2a | 4        | 2     | 4    | 3   | 3    |
| airline:6 M2a  | 4        | 4     | 4    | 4   | 2    |

五个组合全部是 M2a，表内为失败 trial 数。另有 5 个组合在四个模拟器下出现：airline:18 M5、airline:2 M1、airline:31
M2a、airline:44 M14、retail:18 M1。完整矩阵还有 3/2/1 个模拟器出现的情况。这个观察支持部分 actor
缺口在用户替换后仍重复出现，尤其拒绝方式；不支持“所有缺口排序不受用户影响”。例如 GLM retail:97 t0 是 M4，而 t1/t3 得到用户帮助成为
U6；user-primary 会让部分 agent 次要问题不进入矩阵，成功侧截断也会阻止错误暴露。

## 10. 服务端异常与实际回复模型审计

本轮 completed 轨迹没有开场控制符、空首句或此前那种与任务无关的商品信息回复。run.log 的 EmptyUserResponseError 与限流记录均为
0。逐条 user raw_data.model 与配置一致，未发现订阅接口静默映射；GLM 使用降并发后的 r2，未采用作废的首次运行。

| 模拟器            | 实际raw_data.model           | completed | 开场异常/空回复/限流 | 错误尝试 | 错误类型                         |
| ----------------- | ---------------------------- | --------- | -------------------- | -------- | -------------------------------- |
| DeepSeek-Flash    | deepseek-flash               | 240       | 0 / 0 / 0            | 7        | context_length: 7                |
| Doubao 2.1 Turbo  | doubao-seed-2-1-turbo-260628 | 240       | 0 / 0 / 0            | 4        | context_length: 4                |
| Doubao 2.1 Lite   | doubao-seed-2-1-lite-260915  | 240       | 0 / 0 / 0            | 3        | context_length: 3                |
| GLM 5.3           | glm-5.3                      | 240       | 0 / 0 / 0            | 1        | context_length: 1                |
| Qwen3.5-4B 自托管 | qwen35-4b-user-step0         | 240       | 0 / 0 / 0            | 4        | context_length: 3, connection: 1 |

18 个 context_length 错误尝试来自 actor 上下文超长，另有 Qwen 1 个连接错误；它们在严格计分下记 0 分（user-simulator.md
§2 的 17 条 context-limit episode 中包括这些）， 不进入 385
个已标注的能力失败，也没有重新标注。日志重复打印同一异常栈：BadRequestError 出现次数分别 63/36/27/9/27，Qwen
InternalServerError 12，不能把这些 occurrence
当独立失败数。模型名一致能排除已记录的名义映射，不能证明后端内部行为或版本完全不变。源日志绝对路径和哈希见 preflight。

## 11. 真正影响解释的局限

- **只有一个标注者。** 15 条随机重读能发现局部错误，不能估计多人一致性。retail:77 是否隐含保持香型、retail:32
  是否应提前提出备用目标、retail:38 的询价遗漏归谁，都会影响少量边界计数；理由和消息位置保留供复查。
- **没有修复用户再运行。** 一条用户失败被删掉后，agent 也可能继续失败；U6 后得分 1 也不证明不帮助就必败。
- **用户配置并非只差模型名。** thinking 可控性、接口、temperature 的实际作用和并发不同；actor 温度为 0 仍可能受推理批次影响。四次
  trial 不是保证可交换的随机干预，误差可能来自多种运行差异。
- **U6 是保守的文本筛查。** 可能漏掉不含显眼措辞的帮助，也可能把 task 允许的强澄清计入可见帮助。14 是本口径观察到的数，不是全部潜在成功污染的精确总量。
- **只看这 60 题和 completed 重试。** 重试已保留错误证据，但最后完成的一次仍有选择效应；不外推到其他
  actor、Telecom、其他用户温度或训练后策略。NL judge 的漏判/误判也会影响信息类失败。

## 12. 产物与验证

- [labels.jsonl](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/case-study-v3/labels.jsonl)：428
  条，385 个失败各一个致败原因，另 14 个 U6 成功、29 个提前 TRANSFER 成功。正常成功不在 labels 内；1200 条完整 episode 来自
  bundle 及原轨迹。
- [matrix.md](matrix.md)：Airline/Retail 的代码与三方计数、十组两两不一致配对、agent 缺口出现矩阵。
- [cases.md](cases.md)：逐码索引，全部链接到原始轨迹绝对路径。
- [audit-15.md](audit-15.md)、[audit-15.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/case-study-v3/audit-15.json)：固定种子抽样、前后标签与更正理由。
- [statistics.json](statistics.json)、[preflight.json](preflight.json)、[validation.json](validation.json)：可读回的计数、运行来源和校验结果。

`python export.py` 从 decisions 与源轨迹生成
labels/matrix/cases/statistics；`python build_report.py` 生成本文与例子引文记录；`python validate.py`
检查完整失败覆盖、唯一键、原轨迹哈希、分数、引文、tool_calls、任务行号、gold、总表和配对计数、报告链接与原话。校验器对故意破坏的引文、删掉失败标签和篡改分数三项反例都明确失败。机器检查保证证据可定位、数字一致，不证明归因判断必然正确。仓库与
README 未修改。

## 13. 复核记录（Claude，2026-10-13）

报告和标注由 Codex（pueue 501）完成，以下是我在合入前做的独立核对。

- **计数。** 每个模拟器的失败数等于 $240\times(1-\text{mean@4})$，和标注时的主表一致（67/88/72/58/100）；严格计分的主表见
  [user-simulator.md §2](../user-simulator.md)。
- **U1 用规则交叉核对。** 规则：失败 episode 最后一条用户消息含 `###STOP###` 且含确认用语。规则与 Codex 的 U1 标注只差 7 条：6
  条是规则漏判（"just process it" 之类的确认说法），1 条是
  [Turbo retail:101 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-doubao-test-u0p0-r2/trajectories/retail:101:trial0/a2c7569b1e9a421786d150db41767980.json)。这条读完轨迹后同意记
  M8：用户 m21 已授权改地址，agent 没调用改地址工具，m26 先改商品，订单进入 "pending (item modified)"，地址再也改不了；末尾的同句
  STOP 只是次要原因。
- **airline:44 是 v3 没修到的 gold 错误，B1 成立。** 任务要求"含中转 ≤3 小时的航班才升舱"，KC18K6 是 HAT300 MSP
  06:00 → HAT215 CLT 13:00，含中转 7 小时；gold 仍对 KC18K6 调用 `update_reservation_flights`，NL
  断言的 1380–1390 报价和 "Agent updates KC18K6 to business" 也按升舱算。按任务正确执行的用户和 agent 会被判
  0。待修，记在 [benchmark-variants.md §1.2.1](../benchmark-variants.md)。
- **retail:5 的确认次数依赖成立。** 任务写"第一次确认只换台灯，第二次确认改为退水瓶"，gold 只有退水瓶。agent 只确认一次时，用户按任务行事反而得
  0（DeepSeek t0）。
- **[GLM airline:37 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/v3-glm53-test-u0p0-r2/trajectories/airline:37:trial2/b7b695ba208c4fc5ad10da74899b38e7.json)
  的 U7a 成立。** 升舱写入成功，agent m20 只说要转人工、未调用转接工具，用户 m21 就发了 `###TRANSFER###`。
- **开场就结束。** 我独立扫描了第一条用户消息里的三个控制符，五组 v3 运行都是 0，和第 10 节一致；v2 的开场就结束来自方舟服务异常， 不是模型行为。
