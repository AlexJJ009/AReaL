# tau2 current train 与 test 的能力覆盖（2026-10-22）

本报告为选择下一批训练题提供逐题统计。Current train 精确恢复 run 3（pueue 680）每 epoch 的 1,083 个任务，test 为实际 eval 使用的 fixed-v3 100 题。统计的是任务要求与 gold 的覆盖，不是模型成功率，也不据此断言某能力已经学会。标签可同时出现，比例均以各 domain、各 set 的任务数为分母，不能相加当作分布。最终图由 Claude 绘制，本目录未生成图。

## 1. 主要缺口

**优先补 Airline 的拒绝/转人工与事实报告、Retail 的 split-payment 拒绝和事实报告，以及 Telecom 的 service/PIN/合约/欠费处理。** Airline 的普通写工具均已覆盖，但 gold 转人工与 route-change refusal 都是 current 0、test 各 2/20。Retail 的 split-payment refusal 是 current 0、test 2/40。Telecom 17 类原子故障全部已覆盖，问题主要是 service 相关任务比例偏低；旧 249 题报告中的“缺充值”已不适用于当前 329 题 pack。

### airline

| 能力标签 | Current train | Test |
| --- | --- | --- |
| Transfer to human (`transfer`) | 0/417（0.0%） | 2/20（10.0%） |
| Route change forbidden (`refuse_route_change`) | 0/417（0.0%） | 2/20（10.0%） |
| No DB write (`no_db_write`) | 17/417（4.1%） | 7/20（35.0%） |
| Required refusal (`refusal`) | 66/417（15.8%） | 14/20（70.0%） |
| Cancellation ineligible (`refuse_cancellation_ineligible`) | 16/417（3.8%） | 7/20（35.0%） |
| Basic economy restriction (`refuse_basic_economy_change`) | 12/417（2.9%） | 4/20（20.0%） |
| Cannot add insurance (`refuse_insurance_after_booking`) | 1/417（0.2%） | 2/20（10.0%） |
| Already flown restriction (`refuse_already_flown`) | 1/417（0.2%） | 2/20（10.0%） |
| Report specific facts (`report_information`) | 10/417（2.4%） | 5/20（25.0%） |
| Several reservations (`multiple_reservations`) | 89/417（21.3%） | 9/20（45.0%） |

### retail

| 能力标签 | Current train | Test |
| --- | --- | --- |
| No split payment (`refuse_split_payment`) | 0/337（0.0%） | 2/40（5.0%） |
| No partial cancellation (`refuse_partial_cancellation`) | 9/337（2.7%） | 5/40（12.5%） |
| No DB write (`no_db_write`) | 5/337（1.5%） | 4/40（10.0%） |
| Report specific facts (`report_information`) | 24/337（7.1%） | 11/40（27.5%） |
| Required refusal (`refusal`) | 86/337（25.5%） | 14/40（35.0%） |
| Modify pending items (`modify_pending_order_items`) | 73/337（21.7%） | 15/40（37.5%） |

### telecom

| 能力标签 | Current train | Test |
| --- | --- | --- |
| No service (`issue_service`) | 23/329（7.0%） | 15/40（37.5%） |
| SIM PIN locked (`fault_sim_pin_locked`) | 10/329（3.0%） | 11/40（27.5%） |
| Main APN broken (`fault_apn_broken`) | 11/329（3.3%） | 9/40（22.5%） |
| Contract ended (`fault_contract_ended`) | 8/329（2.4%） | 6/40（15.0%） |
| Overdue bill suspension (`fault_overdue_bill_suspension`) | 8/329（2.4%） | 6/40（15.0%） |
| Transfer to human (`agent_transfer_to_human_agents`) | 16/329（4.9%） | 12/40（30.0%） |
| Request bill payment (`agent_send_payment_request`) | 4/329（1.2%） | 3/40（7.5%） |
| Resume line (`agent_resume_line`) | 4/329（1.2%） | 3/40（7.5%） |
| VPN broken (`fault_vpn_broken`) | 24/329（7.3%） | 5/40（12.5%） |
| Data saver on (`fault_data_saver_on`) | 28/329（8.5%） | 7/40（17.5%） |
| Refuel data (`agent_refuel_data`) | 80/329（24.3%） | 12/40（30.0%） |

Airline 的 66/417 refusal 与 17/417 no-write 是不同口径：拒绝后取消重订、拒绝减行李但完成改签，都属于含 refusal 的写任务。Retail 同理，86/337 refusal 不代表 86 道纯拒绝题。`report_information` 特指 communicate_info 或经逐条阅读确认的 factual NL 评价要求；synthetic Airline/Retail packs 两者均为空，因此低占比同时暴露了评分信号缺口，不能解读为用户场景里完全没有信息需求。

Telecom 的 MMS 占 current 256/329（77.8%），test 16/40（40.0%）；mobile data 占 50/329（15.2%），test 9/40（22.5%）。Refuel 已有 80/329（24.3%），接近 test 的 12/40（30.0%），应避免继续沿用旧 pack 的 0 充值结论。Persona 分布接近 test。完整故障组合与 persona 的一一对应过严，D1 去重后不能要求复制 test 的完整状态来补覆盖。

## 2. 哪些候选池能补，能用多少题

下表的“剩余”已扣除 current 与 test 重复，并在池内去重；`counts.json` 的 pool `n` 和 counts 就是这部分候选，**其中包含需修复或已被后续审查拒绝的题**。“已有审查支持可用”另要求现有逐题 keep/fix 证据适用于当前导出的 task 对象；它不是本轮重跑 gold replay 的结论。`tasks.csv.quality_status` 和 [pool_support.csv](out/pool_support.csv) 将两者分开，不能把剩余数直接当作可投入训练数。

| 候选池 / domain | 原始数 | 剩余 | 已有审查支持可用 | 质量边界 |
| --- | --- | --- | --- | --- |
| `pool_areal_v1` / airline | 490 | 84 | 0 | Airline/Retail 旧 v1；后续 v2 淘汰/替代的旧题，不直接恢复。 |
| `pool_areal_v1` / retail | 323 | 55 | 0 | Airline/Retail 旧 v1；后续 v2 淘汰/替代的旧题，不直接恢复。 |
| `pool_critic_mix_v1` / airline | 150 | 0 | 0 | v2 的子集，全部已在 current，无新增任务。 |
| `pool_critic_mix_v1` / retail | 71 | 0 | 0 | v2 的子集，全部已在 current，无新增任务。 |
| `pool_areal_telecom_v0` / telecom | 261 | 207 | 170 | 旧指令版本；37 道剩余题须用 v1 的 exactly 2.0 GB 修正。 |
| `pool_areal_telecom_v1` / telecom | 261 | 207 | 207 | §6 已完成第二轮审查；保留 ENV_ASSERTION 和指令修复。 |
| `pool_official_telecom_v1` / telecom | 249 | 38 | 38 | §7 已逐题 keep 的旧 249 pack，取与 current 不同的剩余。 |
| `pool_telecom_full` / telecom | 2285 | 1884 | 38 | D1-safe 官方原始 full 剩余；含需精确 2GB 修复及未逐题审查任务。 |

### 2.1 按缺口选择

**Airline / Retail：本次列出的池里没有可直接新增且能填这些主要缺口的已审查任务。** Airline v1 剩余 84 题中的转人工仅 1 题、route-change refusal 0 题、no-write 0 题、显式 factual report 0 题；该 1 道转人工仍属后续拒绝/淘汰的旧题。Retail v1 剩余 55 题中 split-payment refusal、partial cancellation、no-write 和显式 factual report 均为 0。Critic-mix 的新增数为 0。应在不同 reservation/order 上补新的 policy/NL 任务，而不是把 Retail 37 加回：它正是被 fixed 过滤、与 test 36 同用户且同 gold writes 的重复题。

v1 的质量问题有现成证据：[training-data.md §2–5](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/src-41dde039/docs/research/tau2/training-data.md) 记录了支付 gold 与指令矛盾、24h 取消违规、航线 DB 不一致、HKEG34 乘客错配、商品名称错配和模板套用。v1 的 gold 可达、甚至评测 4/4 都不能证明题面正确；v2 decisions 的逐题拒绝理由保存在 `tasks.csv.quality_note`。国家写法和乘客列表顺序也经过指令修复，不应把旧版本未经修复的文本重新引入。

Telecom 可以补，但稀有 service 故障的总池本来就小。下表每格为“已有审查支持可用 / 剩余候选”，均已扣除重复；同一任务可以覆盖多行，各 pool 也可能相互重合。

| Telecom 缺口 | official v1 剩余38 | AReaL v1 剩余207 | full 剩余1884 |
| --- | --- | --- | --- |
| `issue_service` | 3 / 3 | 2 / 2 | 3 / 9 |
| `fault_sim_pin_locked` | 1 / 1 | 0 / 0 | 1 / 3 |
| `fault_apn_broken` | 2 / 2 | 1 / 1 | 2 / 4 |
| `fault_contract_ended` | 2 / 2 | 0 / 0 | 2 / 2 |
| `fault_overdue_bill_suspension` | 0 / 0 | 0 / 0 | 0 / 2 |
| `agent_send_payment_request` | 0 / 0 | 0 / 0 | 0 / 1 |
| `agent_resume_line` | 0 / 0 | 0 / 0 | 0 / 1 |
| `agent_transfer_to_human_agents` | 2 / 2 | 13 / 13 | 2 / 4 |
| `fault_vpn_broken` | 6 / 6 | 32 / 32 | 6 / 99 |
| `fault_data_saver_on` | 6 / 6 | 23 / 23 | 6 / 93 |
| `issue_mobile_data` | 14 / 14 | 69 / 69 | 14 / 190 |

优先从旧 official pack 的剩余 38 题中挑 service 相关的 3 题：它们合计包含 PIN 1、主 APN 2、合约到期 2（有交集），且没有 current/test 完整状态重复。再从 AReaL Telecom v1 补 VPN、data saver、mobile data；该池还能提供 1 个主 APN 故障。欠费、payment request 与 resume 在这两个已审查池中都无新增；full 剩余只有 2 道欠费题，其中 1 道 gold 要求 payment request + resume，需要先逐题复核，不能承诺已有可用样本。具体 ID 按 pool、能力保存在 [reviewed_candidate_ids.json](out/reviewed_candidate_ids.json)。

AReaL v1 剩余的 13 个 transfer 全都在 initial_state 中已经 refuel 2 GB，随后又超过额度，因此面对“不换 plan”的要求只能升级处理；它们不能替代 test 中 PIN、合约等其他转人工原因。旧 official 剩余 2 个 transfer 更接近这类 service 分支。Full 中已有审查支持的 38 题与旧 official 剩余 38 题逐对象相同，不是另外 38 题。v0 的 170 个可用对象与 v1 对应对象相同，另外 37 个应直接取修正后的 v1；这些池不能相加。

§6 的原始 AReaL Telecom 数据还存在 reward_basis 问题：205 题未明确设置，136 题的原始 gold 无法得满分。现有 v0/v1 packs 已将195题改为 ENV_ASSERTION，并丢弃10道 gold/空轨迹无法区分的题；不能把这项已修问题继续算作 v1 缺陷。§7 旧 official pack 逐题审查全 keep，但有2题含可被另一次 reboot 覆盖的冗余 reboot；gold 调用次数不等于最小必要操作数。

### 2.2 去重口径与复算

Airline/Retail 按实际 task DB 解析用户和目标，去掉与 test 共用 reservation/order 的候选，或同用户且相同 gold DB writes 的候选；gold writes 忽略 action_id/info，按 `(name, arguments)` 的排序多重集合比较，保留重复动作的次数。用户身份来自 gold user lookup、显式 user id 以及目标对象 owner；不因整个共享 DB 里存在某条记录就算 task 引用它。Test 25 的隐含旧预订 G72NSF、35 的 M20IZO 经路线/日期/用户 DB 人工确认后也参与排除。另扣除 current 相同 task ID、同用户同 writes；同池重复只留首个 source 顺序对象。

Telecom 按 D1 的 complete `initial_state` canonical JSON SHA256：对象键排序、数组顺序保留，不把 persona 或去掉 persona 的 task ID 当作状态身份。官方所有任务共享 C1001/L1002，按用户排除会误删整个 domain，因此用 D1 状态规则。Full 2,285 题中，test ID 是 40，但同 test 状态有 41；current ID 是 329，但同 current 状态有 336。两类状态互不相交，再扣除剩余池内重复 24，得到 $2{,}285 - 41 - 336 - 24 = 1{,}884$ 个候选。

AReaL Airline/Retail v1 共 813 题，按 current ID/同用户同 writes 去掉 652，剩余池内同用户同 writes 重复 22，得到 139（Airline 84、Retail 55）。这些池的 test 对象重合在此前筛选时已去除，本轮仍重新检查，结果未命中。AReaL Telecom 每个 v0/v1 原始 261 中 49 个同 current 状态，另 5 个池内重复，剩 207。旧 official 249 中 211 同 current，剩 38。所有排除记录见 [pool_exclusions.json](out/pool_exclusions.json)；一条记录可命中多个排除原因，不能把原因计数相加。

## 3. 标签定义、赋值规则与 test 例子

为满足 brief 要求的每种写操作、拒绝原因、每个 Telecom 故障及其复杂度，最后使用 Airline 30、Retail 26、Telecom 32 个标签，超出建议的 10–20；强行合并会隐藏需要补题的差异。每个 domain 内已按 English group、test share 降序排列。下表给出一句定义及实际赋值规则；[taxonomy_rules.json](out/taxonomy_rules.json) 保存同样的机器可读规则。

**Test 未出现的标签无法提供真实 test example。** Airline 的 `send_certificate`、`refuse_cabin_consistency`、`refuse_payment_restriction`、`refuse_unsupported_fee_waiver`，Retail 的 `refuse_invalid_cancel_reason`，Telecom 的 `faults_1` 共六项 test count 为 0；表中明确写“test 无”，并提供非 test 例子。

`n_writes`：Airline 为六类 gold DB mutator 调用数，Retail 为七类；transfer、read、calculate 不计。Telecom 统一计 agent 的 refuel_data、enable_roaming、resume_line，send_payment_request 和 transfer 单独打 capability tag，用户 make_payment 与设备操作不计 agent DB writes。它衡量 gold 调用次数，不代表每次调用都不可删或最终 DB 发生不同变化。

Telecom 的 n_faults 按 17 个原子故障计数：SMS 与 storage permission 分开、设备和账户 roaming 分开；同一个 generator 的 both-permissions 或双侧 roaming 可以贡献两项。它与按 ID 模板计 generator fault 的数量不同；setup 的 set_user_info、simulate_network_search、普通 abroad 情境不计故障。

### airline

| Tag / group | 定义 | 准确赋值规则 | Test example |
| --- | --- | --- | --- |
| `multiple_reservations` (Complexity) | gold 或 scenario 涉及两个以上既有或新建预订。 | 在初始 DB 存在的 distinct reservation_id（gold 读/写 + scenario 显式 ID + 人工辨认的 test 隐含引用）加 book_reservation 调用次数 >= 2；新建预订的 HATHAT 更新别名不另计；参考旧预订也计入，衡量接触广度。 | `2` |
| `writes_1` (Complexity) | gold 的 DB 写操作次数为 1。 | n_writes == 1；重复调用也逐次计数，transfer 和 read 不计。 | `8` |
| `writes_3plus` (Complexity) | gold 的 DB 写操作次数为 至少 3。 | n_writes >= 3；重复调用也逐次计数，transfer 和 read 不计。 | `18` |
| `writes_2` (Complexity) | gold 的 DB 写操作次数为 2。 | n_writes == 2；重复调用也逐次计数，transfer 和 read 不计。 | `29` |
| `report_information` (Information) | 评价标准明确要求报告具体事实或数值。 | communicate_info 非空，或人工确认 nl_assertions 规定需告知价格、金额、时长等具体事实；仅断言完成某写操作不计。此标签衡量显式评价要求。 | `2` |
| `refusal` (Outcome) | 任务中至少一个实际请求必须被拒绝，包括先拒绝再执行合法替代的任务。 | 仅使用人工逐题审查映射；阅读 scenario、实际 DB 状态和 policy；不把假设条件或用户预算不满足当作 policy refusal。 | `2` |
| `no_db_write` (Outcome) | gold 不包含该领域的 DB 写操作。 | n_writes == 0；transfer 不计 DB write；不自动推断 refusal。 | `2` |
| `transfer` (Outcome) | gold 要求转人工。 | gold actions 含 transfer_to_human_agents；不从用户要求转人工推断。 | `13` |
| `gift_card_payment` (Payment) | gold 实际支付或退款涉及 gift card。 | 上述实际 payment_id 集合中存在以 gift_card_ 开头的 ID；不把仅在 profile 或情境中出现的支付工具计入。 | `16` |
| `multiple_payment_methods` (Payment) | 同任务实际 gold 写入或取消退款涉及至少两种不同 payment_id。 | 汇总写参数 payment_id/payment_methods 及 cancel 目标初始 payment_history 的 payment_id，去重后 >=2。 | `18` |
| `certificate_payment` (Payment) | gold 实际支付或退款涉及 travel certificate。 | 上述实际 payment_id 集合中存在以 certificate_ 开头的 ID；不把仅在 profile 或情境中出现的支付工具计入。 | `8` |
| `refuse_cancellation_ineligible` (Refusal) | 请求取消，但不满足 24h、商务舱、航司取消或保险承保理由之一。 | 人工映射明确列出该 task 的 cancellation_ineligible；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `24` |
| `refuse_basic_economy_change` (Refusal) | 请求直接修改 basic economy 航班，须先拒绝该修改路径。 | 人工映射明确列出该 task 的 basic_economy_change；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `13` |
| `refuse_already_flown` (Refusal) | 请求取消已有已飞航段的预订或修改其 cabin，须说明限制或转人工。 | 人工映射明确列出该 task 的 already_flown；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `13` |
| `refuse_insurance_after_booking` (Refusal) | 请求给已存在的预订追加保险，policy 禁止。 | 人工映射明确列出该 task 的 insurance_after_booking；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `6` |
| `refuse_route_change` (Refusal) | 请求直接改变 origin、destination 或 trip type，须拒绝直接修改。 | 人工映射明确列出该 task 的 route_change；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `13` |
| `refuse_baggage_removal` (Refusal) | 请求减少已订托运行李，policy 只允许增加。 | 人工映射明确列出该 task 的 baggage_removal；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `30` |
| `refuse_compensation_ineligible` (Refusal) | 请求或试探补偿，但不符合 policy 的补偿场景或资格。 | 人工映射明确列出该 task 的 compensation_ineligible；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `2` |
| `refuse_passenger_count` (Refusal) | 请求增删已有预订乘客人数，policy 禁止。 | 人工映射明确列出该 task 的 passenger_count；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `24` |
| `refuse_cabin_consistency` (Refusal) | 请求只给部分航段或乘客改变 cabin，但同预订 cabin 必须一致。 | 人工映射明确列出该 task 的 cabin_consistency；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | test 无；current_train: `10` |
| `refuse_payment_restriction` (Refusal) | 实际要求的支付方法或组合违反预订或修改支付规则。 | 人工映射明确列出该 task 的 payment_restriction；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | test 无；current_train: `20` |
| `refuse_unsupported_fee_waiver` (Refusal) | 实际要求减免、免差价或豁免收费，但 policy 没有授权。 | 人工映射明确列出该 task 的 unsupported_fee_waiver；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | test 无；current_train: `airline_449` |
| `update_reservation_flights` (Write) | gold 包含 update_reservation_flights 写操作。 | evaluation_criteria.actions 中 name == update_reservation_flights；按实际 gold 工具，不按用户口语描述。 | `16` |
| `book_reservation` (Write) | gold 包含 book_reservation 写操作。 | evaluation_criteria.actions 中 name == book_reservation；按实际 gold 工具，不按用户口语描述。 | `8` |
| `cancel_reservation` (Write) | gold 包含 cancel_reservation 写操作。 | evaluation_criteria.actions 中 name == cancel_reservation；按实际 gold 工具，不按用户口语描述。 | `19` |
| `update_reservation_baggages` (Write) | gold 包含 update_reservation_baggages 写操作。 | evaluation_criteria.actions 中 name == update_reservation_baggages；按实际 gold 工具，不按用户口语描述。 | `22` |
| `update_reservation_passengers` (Write) | gold 包含 update_reservation_passengers 写操作。 | evaluation_criteria.actions 中 name == update_reservation_passengers；按实际 gold 工具，不按用户口语描述。 | `22` |
| `send_certificate` (Write) | gold 包含 send_certificate 写操作。 | evaluation_criteria.actions 中 name == send_certificate；按实际 gold 工具，不按用户口语描述。 | test 无；current_train: `airline_800` |
| `cabin_change` (Write subtype) | 至少一次 update_reservation_flights 改变 cabin。 | 按 gold 顺序将 cabin 与该调用前 reservation.cabin 比较；可与 flight_change 同时出现。 | `16` |
| `flight_change` (Write subtype) | 至少一次 update_reservation_flights 改变航班号/日期序列。 | 按 gold 顺序将参数 flights 的 (flight_number,date) 与该调用前 reservation 状态比较；先应用之前的 gold 更新，不比较 price 字段。 | `16` |

### retail

| Tag / group | 定义 | 准确赋值规则 | Test example |
| --- | --- | --- | --- |
| `writes_1` (Complexity) | gold 的 DB 写操作次数为 1。 | n_writes == 1；重复调用也逐次计数，transfer 和 read 不计。 | `5` |
| `multiple_orders` (Complexity) | 用户实际请求或比较涉及多个订单。 | 至少两个 distinct gold write order_id，或人工逐题确认多个实际请求/被拒目标/变体或地址参考订单；test、current、pool 统一映射，排除仅定位目标的 lookup、known_info 背景列表和 profile 伪订单。 | `12` |
| `writes_3plus` (Complexity) | gold 的 DB 写操作次数为 至少 3。 | n_writes >= 3；重复调用也逐次计数，transfer 和 read 不计。 | `32` |
| `writes_2` (Complexity) | gold 的 DB 写操作次数为 2。 | n_writes == 2；重复调用也逐次计数，transfer 和 read 不计。 | `71` |
| `report_information` (Information) | 评价标准明确要求报告具体事实或数值。 | communicate_info 非空，或人工确认 nl_assertions 规定需告知价格、金额、时长等具体事实；仅断言完成某写操作不计。此标签衡量显式评价要求。 | `32` |
| `refusal` (Outcome) | 任务中至少一个实际请求必须被拒绝，包括先拒绝再执行合法替代的任务。 | 仅使用人工逐题审查映射；阅读 scenario、实际 DB 状态和 policy；不把假设条件或用户预算不满足当作 policy refusal。 | `12` |
| `no_db_write` (Outcome) | gold 不包含该领域的 DB 写操作。 | n_writes == 0；transfer 不计 DB write；不自动推断 refusal。 | `12` |
| `transfer` (Outcome) | gold 要求转人工。 | gold actions 含 transfer_to_human_agents；不从用户要求转人工推断。 | `12` |
| `payment_method_choice` (Payment) | 任务要求选择支付/退款方式，或拒绝不合法的支付选择。 | 有 payment_method_id 写参数且所属用户 profile 存在多个支付方法，或人工记录实际付款/退款选择；不把 profile 方法多但无相关请求自动计入。 | `12` |
| `refuse_out_of_scope` (Refusal) | 实际请求新增订单、投诉、丢件退款、闲聊等，超出 policy 和工具范围。 | 人工映射明确列出该 task 的 out_of_scope；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `32` |
| `refuse_partial_cancellation` (Refusal) | 请求从 pending 订单移除部分商品，但工具只支持整单取消。 | 人工映射明确列出该 task 的 partial_cancellation；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `33` |
| `refuse_split_payment` (Refusal) | 请求将一个订单总额拆给多个支付方法，policy 只允许单一支付方法。 | 人工映射明确列出该 task 的 split_payment；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `36` |
| `refuse_cross_product` (Refusal) | 实际请求的换货或改商品跨越 product type，而非同一 product 的不同 options。 | 人工映射明确列出该 task 的 cross_product；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `65` |
| `refuse_gift_card_balance` (Refusal) | 实际拟用 gift card 余额不足所需订单总额或差价。 | 人工映射明确列出该 task 的 gift_card_balance；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `40` |
| `refuse_operation_conflict` (Refusal) | 请求同一订单上互斥的操作或再次修改/换货，违反一次性操作规则。 | 人工映射明确列出该 task 的 operation_conflict；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `27` |
| `refuse_order_status` (Refusal) | 实际请求的订单操作不适用于 DB 中该订单的状态。 | 人工映射明确列出该 task 的 order_status；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `65` |
| `refuse_refund_destination` (Refusal) | 实际要求退款至非原支付方法、非已有 gift card 的目的地。 | 人工映射明确列出该 task 的 refund_destination；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `12` |
| `refuse_same_variant_exchange` (Refusal) | 请求换为相同 options，而 policy 只允许不同 options 的同 product 变体。 | 人工映射明确列出该 task 的 same_variant_exchange；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | `18` |
| `refuse_invalid_cancel_reason` (Refusal) | 实际给出的取消理由不属于 no longer needed 或 ordered by mistake。 | 人工映射明确列出该 task 的 invalid_cancel_reason；依据原文、初始 DB、policy，允许与其他原因及成功写操作同时出现。 | test 无；current_train: `retail_113` |
| `modify_pending_order_items` (Write) | gold 包含 modify_pending_order_items 写操作。 | evaluation_criteria.actions 中 name == modify_pending_order_items；按实际 gold 工具，不按用户口语描述。 | `36` |
| `exchange_delivered_order_items` (Write) | gold 包含 exchange_delivered_order_items 写操作。 | evaluation_criteria.actions 中 name == exchange_delivered_order_items；按实际 gold 工具，不按用户口语描述。 | `9` |
| `return_delivered_order_items` (Write) | gold 包含 return_delivered_order_items 写操作。 | evaluation_criteria.actions 中 name == return_delivered_order_items；按实际 gold 工具，不按用户口语描述。 | `5` |
| `modify_pending_order_address` (Write) | gold 包含 modify_pending_order_address 写操作。 | evaluation_criteria.actions 中 name == modify_pending_order_address；按实际 gold 工具，不按用户口语描述。 | `17` |
| `cancel_pending_order` (Write) | gold 包含 cancel_pending_order 写操作。 | evaluation_criteria.actions 中 name == cancel_pending_order；按实际 gold 工具，不按用户口语描述。 | `32` |
| `modify_user_address` (Write) | gold 包含 modify_user_address 写操作。 | evaluation_criteria.actions 中 name == modify_user_address；按实际 gold 工具，不按用户口语描述。 | `33` |
| `modify_pending_order_payment` (Write) | gold 包含 modify_pending_order_payment 写操作。 | evaluation_criteria.actions 中 name == modify_pending_order_payment；按实际 gold 工具，不按用户口语描述。 | `40` |

### telecom

| Tag / group | 定义 | 准确赋值规则 | Test example |
| --- | --- | --- | --- |
| `agent_refuel_data` (Account action) | gold 要求 agent 执行 refuel_data。 | gold actions 中 requestor == assistant 且 name == refuel_data；initialization 同名工具及用户工具不计。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card\|user_abroad_roaming_disabled_on[PERSONA:None]` |
| `agent_transfer_to_human_agents` (Account action) | gold 要求 agent 执行 transfer_to_human_agents。 | gold actions 中 requestor == assistant 且 name == transfer_to_human_agents；initialization 同名工具及用户工具不计。 | `[service_issue]airplane_mode_on\|break_apn_settings\|contract_end_suspension\|lock_sim_card_pin[PERSONA:None]` |
| `agent_enable_roaming` (Account action) | gold 要求 agent 执行 enable_roaming。 | gold actions 中 requestor == assistant 且 name == enable_roaming；initialization 同名工具及用户工具不计。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card\|user_abroad_roaming_disabled_on[PERSONA:None]` |
| `agent_resume_line` (Account action) | gold 要求 agent 执行 resume_line。 | gold actions 中 requestor == assistant 且 name == resume_line；initialization 同名工具及用户工具不计。 | `[service_issue]airplane_mode_on\|break_apn_settings\|overdue_bill_suspension[PERSONA:None]` |
| `agent_send_payment_request` (Account action) | gold 要求 agent 执行 send_payment_request。 | gold actions 中 requestor == assistant 且 name == send_payment_request；initialization 同名工具及用户工具不计。 | `[service_issue]airplane_mode_on\|break_apn_settings\|overdue_bill_suspension[PERSONA:None]` |
| `faults_2_3` (Complexity) | 初始状态的原子故障数属于 2-3 faults 桶。 | 2 <= n_faults <= 3；n_faults 为本表 17 个 fault_* 标签数量；两种 permission、两侧 roaming 分别计数，setup/sync 不计。 | `[mms_issue]airplane_mode_on\|bad_wifi_calling\|user_abroad_roaming_enabled_off[PERSONA:Easy]` |
| `faults_4_6` (Complexity) | 初始状态的原子故障数属于 4-6 faults 桶。 | 4 <= n_faults <= 6；n_faults 为本表 17 个 fault_* 标签数量；两种 permission、两侧 roaming 分别计数，setup/sync 不计。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|break_app_storage_permission\|data_mode_off\|user_abroad_roaming_enabled_off[PERSONA:Easy]` |
| `faults_7plus` (Complexity) | 初始状态的原子故障数属于 7 or more faults 桶。 | n_faults >= 7；n_faults 为本表 17 个 fault_* 标签数量；两种 permission、两侧 roaming 分别计数，setup/sync 不计。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `faults_1` (Complexity) | 初始状态的原子故障数属于 One fault 桶。 | n_faults == 1；n_faults 为本表 17 个 fault_* 标签数量；两种 permission、两侧 roaming 分别计数，setup/sync 不计。 | test 无；current_train: `[service_issue]airplane_mode_on[PERSONA:None]` |
| `fault_airplane_mode_on` (Fault) | 初始状态注入了 Airplane mode on 故障。 | 从 initial_state.initialization_actions 工具及参数提取：turn_airplane_mode_on；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_device_roaming_off` (Fault) | 初始状态注入了 Device roaming off 故障。 | 从 initial_state.initialization_actions 工具及参数提取：set_user_location(abroad=true) 且最后设备设置为 turn_roaming_off；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_network_2g_only` (Fault) | 初始状态注入了 2G-only network 故障。 | 从 initial_state.initialization_actions 工具及参数提取：set_network_mode_preference(mode=2g_only)；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_mobile_data_off` (Fault) | 初始状态注入了 Mobile data off 故障。 | 从 initial_state.initialization_actions 工具及参数提取：turn_data_off；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card\|user_abroad_roaming_disabled_on[PERSONA:None]` |
| `fault_data_cap_exceeded` (Fault) | 初始状态注入了 Data cap exceeded 故障。 | 从 initial_state.initialization_actions 工具及参数提取：set_data_usage 的 data_used_gb 大于 plan.data_limit_gb + 初始化后 data_refueling_gb；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card\|user_abroad_roaming_disabled_on[PERSONA:None]` |
| `fault_sim_unseated` (Fault) | 初始状态注入了 SIM unseated 故障。 | 从 initial_state.initialization_actions 工具及参数提取：unseat_sim_card；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_carrier_roaming_disabled` (Fault) | 初始状态注入了 Account roaming disabled 故障。 | 从 initial_state.initialization_actions 工具及参数提取：set_user_location(abroad=true) 且最后账户设置为 disable_roaming；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card\|user_abroad_roaming_disabled_on[PERSONA:None]` |
| `fault_messaging_sms_denied` (Fault) | 初始状态注入了 SMS permission denied 故障。 | 从 initial_state.initialization_actions 工具及参数提取：remove_app_permission(app_name=messaging,permission=sms)；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_sim_pin_locked` (Fault) | 初始状态注入了 SIM PIN locked 故障。 | 从 initial_state.initialization_actions 工具及参数提取：lock_sim_card(mode=pin)；不从 gold repair 或 task_id 推断。 | `[service_issue]airplane_mode_on\|break_apn_settings\|contract_end_suspension\|lock_sim_card_pin[PERSONA:None]` |
| `fault_apn_broken` (Fault) | 初始状态注入了 Main APN broken 故障。 | 从 initial_state.initialization_actions 工具及参数提取：break_apn_settings；不从 gold repair 或 task_id 推断。 | `[service_issue]airplane_mode_on\|break_apn_settings\|contract_end_suspension\|lock_sim_card_pin[PERSONA:None]` |
| `fault_messaging_storage_denied` (Fault) | 初始状态注入了 Storage permission denied 故障。 | 从 initial_state.initialization_actions 工具及参数提取：remove_app_permission(app_name=messaging,permission=storage)；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_data_saver_on` (Fault) | 初始状态注入了 Data saver on 故障。 | 从 initial_state.initialization_actions 工具及参数提取：turn_data_saver_mode_on；不从 gold repair 或 task_id 推断。 | `[mobile_data_issue]airplane_mode_on\|bad_network_preference\|bad_vpn\|data_mode_off\|data_saver_mode_on\|data_usage_exceeded\|user_abroad_roaming_enabled_off[PERSONA:Easy]` |
| `fault_wifi_calling_mms_on` (Fault) | 初始状态注入了 MMS over Wi-Fi fault 故障。 | 从 initial_state.initialization_actions 工具及参数提取：set_wifi_calling(enabled=true,mms_over_wifi=true)；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_contract_ended` (Fault) | 初始状态注入了 Contract ended 故障。 | 从 initial_state.initialization_actions 工具及参数提取：suspend_line_for_overdue_bill(contract_ended=true)；不从 gold repair 或 task_id 推断。 | `[service_issue]airplane_mode_on\|break_apn_settings\|contract_end_suspension\|lock_sim_card_pin[PERSONA:None]` |
| `fault_overdue_bill_suspension` (Fault) | 初始状态注入了 Overdue bill suspension 故障。 | 从 initial_state.initialization_actions 工具及参数提取：suspend_line_for_overdue_bill(contract_ended=false)；不从 gold repair 或 task_id 推断。 | `[service_issue]airplane_mode_on\|break_apn_settings\|lock_sim_card_pin\|overdue_bill_suspension[PERSONA:Hard]` |
| `fault_mms_apn_broken` (Fault) | 初始状态注入了 MMS APN broken 故障。 | 从 initial_state.initialization_actions 工具及参数提取：break_apn_mms_setting；不从 gold repair 或 task_id 推断。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `fault_vpn_broken` (Fault) | 初始状态注入了 VPN broken 故障。 | 从 initial_state.initialization_actions 工具及参数提取：break_vpn；不从 gold repair 或 task_id 推断。 | `[mobile_data_issue]airplane_mode_on\|bad_network_preference\|bad_vpn\|data_mode_off\|data_saver_mode_on\|data_usage_exceeded\|user_abroad_roaming_enabled_off[PERSONA:Easy]` |
| `issue_mms` (Issue) | 任务的排障目标为 MMS messaging。 | description.purpose 的完整三个官方模板分别映射 service/mobile_data/mms，以 env_assertions 和 scenario 交叉核查；不读 ticket 的泛化 No Service。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `issue_service` (Issue) | 任务的排障目标为 No service。 | description.purpose 的完整三个官方模板分别映射 service/mobile_data/mms，以 env_assertions 和 scenario 交叉核查；不读 ticket 的泛化 No Service。 | `[service_issue]airplane_mode_on\|break_apn_settings\|contract_end_suspension\|lock_sim_card_pin[PERSONA:None]` |
| `issue_mobile_data` (Issue) | 任务的排障目标为 Mobile data / speed。 | description.purpose 的完整三个官方模板分别映射 service/mobile_data/mms，以 env_assertions 和 scenario 交叉核查；不读 ticket 的泛化 No Service。 | `[mobile_data_issue]airplane_mode_on\|bad_network_preference\|bad_vpn\|data_mode_off\|data_saver_mode_on\|data_usage_exceeded\|user_abroad_roaming_enabled_off[PERSONA:Easy]` |
| `persona_easy` (Persona) | 任务采用官方 easy persona 模板。 | user_scenario.persona 为 null / 以 As a 41-year-old office administrator / At 64 years old, you're a retired librarian 开头的完整模板，分别对应 none/easy/hard；同时校验模板正文一致。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_storage_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card\|user_abroad_roaming_disabled_on[PERSONA:Easy]` |
| `persona_hard` (Persona) | 任务采用官方 hard persona 模板。 | user_scenario.persona 为 null / 以 As a 41-year-old office administrator / At 64 years old, you're a retired librarian 开头的完整模板，分别对应 none/easy/hard；同时校验模板正文一致。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|unseat_sim_card\|user_abroad_roaming_enabled_off[PERSONA:Hard]` |
| `persona_none` (Persona) | 任务采用官方 none persona 模板。 | user_scenario.persona 为 null / 以 As a 41-year-old office administrator / At 64 years old, you're a retired librarian 开头的完整模板，分别对应 none/easy/hard；同时校验模板正文一致。 | `[mms_issue]airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|data_usage_exceeded\|unseat_sim_card\|user_abroad_roaming_disabled_on[PERSONA:None]` |

## 4. 验证与修正记录

### 4.1 实际阅读范围

主 agent 阅读全部 20 个 Airline test 的 scenario、gold 和相关初始 reservation，并详细检查 Retail 12/17/33/36/38/40/62/65/68/77 共 10 题及 policy，满足至少 30 题的要求。三个 sub-agent 只做允许的独立标签核查，均请求 gpt-6.1-sol、medium；没有派发报告审稿或实施分支。Airline 独立读全部 50 个 official tasks，并筛查 v2 全 380、recovered 7、初版候选 85；Retail 读全部 114 个 official tasks，筛查 v2 268 并详细复核相关拒绝、所有 current/pool 多订单判断；Telecom 从原始初始化和 gold 检查 current/test，并独立重建最终 2,705 行 Telecom 标签。

阅读证据：[Airline manual](out/airline_manual_review.json)、[test targets](out/airline_test_target_review.json)、[Retail manual](out/retail_manual_review.json)、[追加 refusal 核查](out/retail_refusal_corrections.json)、[多订单核查](out/retail_multi_order_review.json)、[Telecom manual](out/telecom_manual_review.json)。原始候选 85 中有 1 题在将 gold comparison 从有序列表收紧为排序多重集合后去重，最终为 84；阅读范围不冒充最终成员计数。

### 4.2 确实纠正过的标签

| 纠正任务 | 原因与最终处理 |
| --- | --- |
| Airline test 16 / 32 | 16 同时改 flight/cabin；32 先 cabin、再 flight。比较 action 前状态，避免把更新工具全当改航班或忽略先前更新。 |
| Airline test 2 / 6 / 13 / 26 / 31 / 45 / 48 | 7 个 no-write 任务逐条分开补偿、保险、路线、取消资格与 basic economy 原因；不按关键词统一归类。 |
| Airline test 19 / 24 / 29 / 30 / 35 / 37 / 44 | 合法写入不排除拒绝；分别保留 basic economy、人数、路线、减行李、取消及已飞航段等实际限制。 |
| Airline train 3 / 10 / 34 | 3 是成功 baggage 信息服务；10/34 还涉及预算/选择限制，不把所有 no-write 当相同 policy refusal。 |
| Airline v2 507/560/437/576/710/602/239 | 排除 stale refusal 文本、假设减乘客、四张GC咨询、已明确选择取消重订等误标；完整负例列表在 manual JSON。 |
| Airline 26 个 create-then-update rows | 19 个 current、7 个初版 v1 候选把新 booking 与 HATHAT 别名误算成两份预订，统一折叠；current multiple_reservations=89，test=9。 |
| Retail test 5 / 9 / 17 / 49 / 68 | 5/9 是用户改变主意；17/49 的多次 discovery lookup 不算多个请求订单；68 是正常 order-total 信息服务。 |
| Retail test 65 / 64 / 111 | 65 同时有 processed-status 和 cross-product 限制；64 实际 gold 是 pending modification；111 的 action_id 前缀不改变 task_id。 |
| Retail v2 198/299；official22/103/104 | 198 的 pending 取消有效；299 按最终修正可退；22 的 all possible 只要求合法修改；103/104 有真实 tracking，删除假拒绝。 |
| Retail v2 12/37/136/160/179/216/396/457/487 | 二次全文阅读补实际 cross-product refusal；不是仅出现 rejected 字样便直接赋值。 |
| Retail v2 113/414/446/487/497/514；165/209/305/388/554 | 前组补实际无效取消理由，414 补重复取消已取消订单；后组补已执行后又要求二次 exchange 的一次性限制。 |
| Retail pool v1 55 / 162 | 补实际无效取消理由及 delivered 订单不可取消；旧版本单独读，不复制 v2 结论。 |
| Retail current/pool 多订单 | 移除50个背景/相同订单/profile-only误标，补5个实际请求/被拒/参考多订单任务；current由268改223，pool由51改47。 |
| Telecom suspension / roaming / permissions | contract_ended=true 不同时标 overdue；initial enable_roaming 不标 gold action；双permission/双roaming按原子口径计数。 |

逐任务 before/after 与原因见 [tag_corrections.json](out/tag_corrections.json)，完整规则判断与负例另在各 domain manual JSON。上述 corrections 说明实际阅读如何改变了赋值，不构成对主观判断的零误差保证。Refusal 均有 source-scoped 人工理由，未用关键词直接赋值。同一 task_id 在 v1/v2 的语义可以不同。

### 4.3 机器检查

[verification.json](out/verification.json)：3,658 行；current/test 分别严格等于 417/337/329 和 20/40/40；逐行唯一键、CSV↔JSON 全量 counts、标签顺序、label 长度、精确 JSON schema、每个 source_file 与 task DB 绑定均通过；96 个输入文件 SHA256 与路径解析通过。Pack manifest hash 与实际 bytes 一致，fixed-v3 tasks hash 与 eval snapshot 的强制 hash 一致；Airline flight/cabin 比较无初始预订缺失。

六个故意破坏检查均按预期失败：篡改 tag count、添加 schema 键、把三个 domain 的 test 对象分别塞进候选池、池内重复完整 Telecom state。独立 Telecom 检查直接重建全部 2,705 行的 fault/action/persona/issue/bucket 和 n_writes，与输出零差异；Airline 独立逐行 subtype 复算亦零差异，并发现了已修正的新建预订别名重复计数。未运行模型评测、GPU、pueue、LLM 或 network API，也没有重跑全部 gold replay。

复算命令（已有 venv 提供 PyYAML，`-B` 禁止向源目录写 bytecode）：

```bash
cd /data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-train-test-coverage-20261022
/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/.venv/bin/python -B analyze.py
/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/.venv/bin/python -B verify.py
/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/.venv/bin/python -B write_report.py
```

## 5. 精确 train/test 组成与读取依据

| Domain | 官方 train | Pack | 最终 current | Test |
| --- | --- | --- | --- | --- |
| airline | 30，无 fixed 排除 | AReaL v2 380 + recovered v1 7 | 417 | 20 |
| retail | 74 − 5 = 69 | AReaL v2 268 | 337 | 40 |
| telecom | 不加载官方 train split | official 20261021-v2 329 | 329 | 40 |

直接以 YAML safe_load 读取 [run 3 config](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/run3-std-beta1-v4/logs/root/tau2-ppo/run3-std-beta1-v4/config.yaml)，未调用 load_expr_config，也没有导入训练入口。依据 [get_tau2_dataset](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/src-41dde039/examples/tau2/train.py:39)、[build_ppo_train_dataset](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/src-41dde039/examples/tau2/ppo_data.py:16)、[task_pack](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/src-41dde039/examples/tau2/task_pack.py:46) 和 [fixed exclusions](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/src-41dde039/examples/tau2/benchmark_fixes.json) 重建成员；只执行从 AST 抽出的纯 D1 canonical/fault_key 函数来校验状态身份。

配置的 `_cache/tau2-areal-pack-latest` 本次解析到 `dataset/tau2/tau2-areal-pack-20261011-v2`，另外两个 pack 是 `tau2-official-telecom-pack-20261021-v2`、`tau2-areal-airline-recovered-pack-20261021-v1`。固定 Retail 排除 `37,41,67,72,96`，分别与 test `36,42,68,71,97` 同用户同 gold writes；Airline 无官方 train 排除。samples_per_task_per_epoch=1，gconfig.n_samples=1，所以每个 distinct task 每 epoch 恰好贡献一条 scheduled episode，domain_effective_episodes 不是额外抽样子集。

[build_ppo_eval_dataset](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/src-41dde039/examples/tau2/ppo_data.py:93) 使用 fixed-v3 Airline/Retail 的官方 test 和未修改的官方 Telecom test；各域 task definitions 与 split 均取 `_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data/tau2/domains/`。Airline test 26/31 的 read action、37 的 transfer、Retail 65 的 lookup 均来自实际 fixed-v3 文件，没有用原始任务替换。

## 6. 限制与交付边界

**恢复 Current train 不等于用本次更严格规则重新筛运行中的数据。** 当前 official train 中还有 9 个 Airline task、33 个 Retail task 触及 test 也引用的 reservation/order（包含 read/reference，不都代表相同 gold writes）；详见 [current_test_target_overlap.json](out/current_test_target_overlap.json)。本报告保留它们，以准确描述 run 3；不能把 current/test 当作对象完全隔离的对照。新候选池应用了更严格的共享目标过滤。Airline official train 39 的 gold 取消 MSJ4OA，而场景未给保险承保理由，是独立审查发现的既有 policy/gold 风险，已在 CSV 标为 current_known_policy_issue，未改任务。

“已有审查支持可用”引用历史数据审查，不是本次新的模型质量或 reward replay 保证。Candidate pools 相互重合、同 task 多标签，不能把所有 pool 求和。Share 比较是描述性统计，小 test 规模不支持从一两个 task 推导统计显著性。

本次只在 brief 所在目录写分析脚本、review 记录和交付物；未改运行配置、任务 pack、symlink、训练源码或运行文件。最小消费入口为 [tasks.csv](out/tasks.csv) 与 [counts.json](out/counts.json)。后者顶层严格只有 sets/domains，tag 对象只有 tag/label/group/definition，set statistics 只有 n/counts；质量和规则扩展字段保存在其他文件，Claude 的 plotting script 无需改 schema。

下一步选题依据：Airline/Retail 需要新建不同对象的拒绝及 factual-report 任务，并配置能约束该行为的 ACTION/NL/COMMUNICATE 评价；Telecom 可先挑已审查的 service 稀有分支，再审查 full 的欠费与 payment/resume 新题。这里提供选择依据和 ID，没有新增任务、变更 run 3 或提交训练。
