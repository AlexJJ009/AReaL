# D1：Telecom full 的通过率、题目质量与训练题筛选

## 1. 结论与推荐

**推荐先加 249 题 Telecom，与原 747 题合为 996 题，Telecom 占 25%。** 推荐集合全部是无官方 test 同状态重合、无已知充值评分歧义、且
1–3/4 通过的题；1/4、2/4、3/4 各 83 题，完整初始状态每个 key 只留一题。task id 清单为
[recommended_task_ids.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/recommended_task_ids.json)；全部候选方案和排除
ID 在
[candidate_sets.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/candidate_sets.json)。

D1 的主要问题是多故障排障做不完，并非用户在确认时频繁 STOP。但有一类必须先排除的 benchmark 歧义：**1,120 题只写“愿意充 2 GB”，reward
却要求恰好充 2 GB**。真实轨迹中有只充 1 GB 后 MMS/网速已恢复、仍拿 0 分的例子；对这 1,120 题逐题 CPU 重放也确认，充 1 GB
能满足其余环境目标，仅金额断言不通过。这是 training-data §6 已确认过的同类问题，本轮不修题，直接从候选集合排除。

| 核心结果                                               |                                 实测值 |
| ------------------------------------------------------ | -------------------------------------: |
| 有效任务 / 完成 episode                                |                          2,285 / 9,140 |
| 通过 episode / 通过率                                  |                         1,793 / 19.62% |
| 每题 0 / 1 / 2 / 3 / 4 次通过                          |          1,440 / 378 / 159 / 135 / 173 |
| 机械检查失败：gold 不可达、空轨迹误得分、gold 工具报错 |                              0 / 0 / 0 |
| 判为题目评分歧义、建议排除                             |                               1,120 题 |
| 与 test 40 共享完整初始状态 key                        |            41 题，包含 test 本身 40 题 |
| 去重合、歧义后 / 其中 1–3/4                            |                         1,136 / 483 题 |
| 失败 episode 随机样本的主因                            | agent 59/60，benchmark 1/60，用户 0/60 |
| 用户致败占失败的估计 / 95% Wilson 区间                 |                     0% / \[0%, 6.02%\] |

“1,120 题有歧义”是**任务层面**的筛题结论；“样本中 1/60 由 benchmark 致败”是**实际失败轨迹**的归因。前者大多数轨迹连 MMS
都未修好，不能把这些失败全部归给金额断言。1,120 题中 950 题四次全错、170 题有对有错、没有四次全对题。

## 2. 数据口径与抽样方法

输入为 brief 指定的 `instruct-telecom-full`，不用 `instruct-telecom-full.killed-558` 或
smoke。模型是未训练 Qwen3-4B-Instruct-2507，non-thinking，policy T=0，4 次评测。9,140 行 episode 覆盖每题
trial 0–3，无重复或缺项，score 均为 0/1，`reward == official_score`。按 episode 重新求和，与 summary 的
mean@4 一致。

运行使用**官方原题 + fixed harness**：Telecom 任务和 DB 未打补丁，但 fixed 会对不符合 gold 转接要求的轨迹清零。不能因 JSON
字段叫 `official_score` 就把它当纯 official scorer 分数。D1 全部可见用户响应的 fingerprint 为
`e72d3c7f663d3e1383b00b3884cb4501`（含重试轨迹共 279,986 条），与 10-17 换版文档一致；任务没有 NL 断言，本轮没有用 NL
judge 重判。

通过率是每题四次 reward 的平均，再在题目间平均；每题正好四次，因此也等于总成功 episode / 总 episode。“四次全对”是
4/4，不是通常定义的“至少一次成功 pass@4”。已有 summary 的 $\\mathrm{pass}^4$ 为 173/2285=7.57%。T=0
仍受用户模拟器和推理批次差异影响，四次不能证明单题真实成功概率是 0 或 1。

**故障 key 沿用 manifest 的完整 initial_state canonical JSON SHA256**：对象 key
排序，数组顺序保留，包含初始化数据、动作和消息历史。2,285 题对应 2,253 个 key；2,221 个 key 仅一题，32 个 key 各两题，重合来自不同
issue 描述的相同 setup。去掉 persona 的任务 ID 仍有 2,285 个不同值，不能据此假设每个 key 都有 None/Easy/Hard 三份。故障数按
ID 内 `|` 分隔的故障 token 数统计，某个 roaming token 可同时涉及设备和运营商两侧，故障数不等于 gold 写操作数。

先完成全量统计与终止符/异常筛查，再人工读样本的工具调用序列、对话摘录、reward，展开决定归因的消息。[labels.jsonl](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/labels.jsonl)
逐条保存轨迹绝对路径、SHA256、0-based 消息编号、原文短引、真实 tool_calls、任务说明、gold 和 reward。以下样本互不混进同一个估计分母：

| 读取层                | 抽样方式与种子                                                                  | 数量 / 用途                                 |
| --------------------- | ------------------------------------------------------------------------------- | ------------------------------------------- |
| 失败 episode 随机样本 | 按 source_id 排序，从全部7,347条失败中不放回等概率抽样；Python Random(20261017) | 60条；估计失败来源                          |
| 四次全错题样本        | 从1,440题中等概率抽30题；Random(20261018)；每题独立按字符串种子选一个trial      | 30条；观察全错题机制，不外推全部失败        |
| 定向终段读取          | STOP关键词、非2GB充值、无失败环境断言、扩展STOP筛查                             | 另130条；发现反例和消除误报，不进入比例估计 |
| 初始结构检查          | 读取首个JSON的结构和开场                                                        | 1条；不作因果归因                           |

共 **221 条唯一轨迹**留有阅读记录，其中90条有主因标签。定向记录明确标为
`terminal_messages_and_reward_only`，没有把局部读取写成通读整段。样本 ID 与抽法见
[sampling.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/sampling.json)。失败总体抽样单位是
episode，允许抽中同题的不同 trial；它估计本批已完成失败的组成，不是跨模型/跨后端的用户错误率。

沿用 v3 报告的边界：用户违反场景或模拟规则才归 U；agent 的工具、推理和子目标遗漏归 M/T；双方遵循说明却仍被 gold/reward 拒绝归
B。每条失败只计一个主因，其他观察不叠加。agent 已错误宣布无法解决后，用户按其指引结束，不自动归“用户提前 STOP”。本轮 T1 为排障/必要转接遗漏，T2
为实体参数或工具语义错误，T3 为账户侧修复遗漏，B1 为任务与评分要求不一致。

## 3. 通过率画像

### 3.1 全量分布与故障数

| 通过次数 |  题数 | 占2,285题 |
| -------: | ----: | --------: |
|        0 | 1,440 |    63.02% |
|        1 |   378 |    16.54% |
|        2 |   159 |     6.96% |
|        3 |   135 |     5.91% |
|        4 |   173 |     7.57% |

| 分组 | 题数 | 通过 episode / 总数 | 通过率 | 0/1/2/3/4 次通过的题数 |
| ---- | ---: | ------------------: | -----: | ---------------------- |
| 1    |   20 |               65/80 | 81.25% | 1/1/2/4/12             |
| 2    |  101 |             220/404 | 54.46% | 22/17/16/13/33         |
| 3    |  283 |            441/1132 | 38.96% | 108/52/29/45/49        |
| 4    |  505 |            497/2020 | 24.60% | 281/89/48/36/51        |
| 5    |  596 |            369/2384 | 15.48% | 386/118/45/27/20       |
| 6    |  467 |            155/1868 |  8.30% | 368/68/14/9/8          |
| 7    |  235 |              37/940 |  3.94% | 204/26/4/1/0           |
| 8    |   69 |               8/276 |  2.90% | 62/6/1/0/0             |
| 9    |    9 |                1/36 |  2.78% | 8/1/0/0/0              |

一故障题通过率81.25%，六故障降到8.30%，七至九故障约3%–4%。这是当前题池下的条件均值：MMS
更常包含多故障，充值歧义也集中在复杂组合里，不能把全部下降解释为单纯的步数因果效应。全池直接加入训练会让63%的题在本次T=0采样中没有正样本。

### 3.2 issue、persona 与完整故障组合

| 分组              | 题数 | 通过 episode / 总数 | 通过率 | 0/1/2/3/4 次通过的题数 |
| ----------------- | ---: | ------------------: | -----: | ---------------------- |
| mms_issue         | 1984 |           1296/7936 | 16.33% | 1343/307/122/103/109   |
| mobile_data_issue |  254 |            363/1016 | 35.73% | 94/67/31/14/48         |
| service_issue     |   47 |             134/188 | 71.28% | 3/4/6/18/16            |

| 分组 | 题数 | 通过 episode / 总数 | 通过率 | 0/1/2/3/4 次通过的题数 |
| ---- | ---: | ------------------: | -----: | ---------------------- |
| Easy |  760 |            629/3040 | 20.69% | 469/124/55/53/59       |
| Hard |  756 |            522/3024 | 17.26% | 500/120/52/38/46       |
| None |  769 |            642/3076 | 20.87% | 471/134/52/44/68       |

MMS 占1,984/2,285题（86.83%），且通过率只有16.33%，是全池低分的主体。Hard 为17.26%，Easy/None约20.7%–20.9%；这些
persona 搭配的故障不同，没有同一 setup 的三种 persona 配对，故这里只描述分组差异，不声称 Hard 本身使分数下降约3.5pp。

完整 **2,253 个故障 key** 的题数、成功数、分母、通过率、0–4分布及 ID 全表见
[fault_combinations.csv](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/fault_combinations.csv)；去掉
issue/persona 只保留故障 token 的表见
[fault_signatures.csv](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/fault_signatures.csv)（同样2,253行）。绝大多数
key 只有四次观测，按 key 排名不稳定；用聚合故障数与 issue 描述总体差异更可靠。不是只列出少数高低分组合来代替全量分组。

### 3.3 test 40 与 AReaL Telecom v1 对照

| 范围                          | 题数 / episode | 通过率 | 0/1/2/3/4 次通过          |
| ----------------------------- | -------------: | -----: | ------------------------- |
| D1 full                       |  2,285 / 9,140 | 19.62% | 1,440/378/159/135/173     |
| D1 中真正的 test 40           |       40 / 160 | 45.63% | 10/10/4/9/7               |
| D1 中与 test 同完整状态       |       41 / 164 | 44.51% | 11/10/4/9/7               |
| D1 非 test-key 题             |  2,244 / 8,976 | 19.16% | 1,429/368/155/126/166     |
| D1 与 pack v1 同状态题        |    260 / 1,040 | 34.13% | 121/47/20/20/52           |
| 历史 pack v1 全量             |    261 / 1,044 | 38.03% | 109全错 / 94混合 / 58全对 |
| 历史 pack v1 中匹配 D1 的部分 |      247 / 988 | 36.34% | 见逐key明细               |

D1 的 test 40 比 full 高26.01pp；test 40 不是 full 的同难度缩小版。仅排除 test ID
会漏掉1题：`[mobile_data_issue]bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Easy]`，必须按
key 排除41题。

两侧匹配不是一对一：D1 的260题、历史pack的247题覆盖243个共同key。对这243个key分别先在各侧取均值、再让key等权，D1为
**33.64%**，历史pack为 **36.01%**（−2.37pp）。全部配对与不同分母见
[comparison_v1.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/comparison_v1.json)；`task_stats.csv`
还保留每题匹配的历史分数。D1与历史运行的用户后端不同、persona文本不同，pack的部分指令已修过“恰好2GB”，因此这不是控制变量的模型退化实验。full与pack全量差18.41pp也不能直接当作数据质量下降。

## 4. 题目质量、失败主因与用户异常

### 4.1 机械检查与1,120题评分歧义

已有全量 manifest 逐题 gold=1、empty=0，gold工具报错0；源失败清单为 `[]`，本轮完整复制到
[mechanical_failures.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/mechanical_failures.json)。没有把“某一条断言初始已成立”当成“空轨迹整体得1分”。按
training-data §5–§6，No Service ticket模板、persona年龄、`get_status_bar` 名称、个别初始成立的断言均不单独判错。

本轮另做CPU检查：从原始 gold 复制动作，仅将 `refuel_data.gb_amount`
从2改为1；保持原任务、DB、scorer不变，正常执行用户/agent双侧工具及sync。**1,120/1,120题均无工具错误、其余环境断言全成立，仅
`assert_data_refueling_amount(expected_amount=2.0)` 不成立，reward仍为0。**
同脚本对3题保留gold=1、空轨迹=0作正反对照。结果见
[topup_counterfactual.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/topup_counterfactual.json)，脚本
[check_topup.py](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/check_topup.py)。

题面共同原话为
`You are willing to refuel 2.0 GB of data if necessary, but you do not want to change your mobile data plan.`。“愿意”不等于“必须恰好”，1GB已解决故障且不换套餐，是同规范下应接受的结果。故列
**1,120道有歧义的原题**；不是声称1,120道gold不可达。所有ID及规则见
[benchmark_tasks.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/benchmark_tasks.json)。其中12题也命中test-key，按并集排除1,149题，剩1,136题。

实际失败证据（随机样本第52条，B1）：

[轨迹](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/instruct-telecom-full/trajectories/telecom:%5Bmms_issue%5Dairplane_mode_on%7Cbad_network_preference%7Cbreak_app_both_permissions%7Cdata_mode_off%7Cdata_usage_exceeded%7Cunseat_sim_card%7Cuser_abroad_roaming_disabled_on%5BPERSONA:Hard%5D:trial3/d8a8173bab544e93bd6244155b4d2a09.json)（telecom:\[mms_issue\]airplane_mode_on|bad_network_preference|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on\[PERSONA:Hard\]:trial3）；SHA256
见 labels。

- m98（assistant）：`Perfect! I’ll refuel **1 GB** of data for your line (555-123-2002) to bring you back within your plan limit. This will restore full data access, including MMS, without changing your plan. [{"name": "refuel_data", "arguments": {"customer_id": "C1001", "line_id": "L1002", "gb_amount": 1}}]`
- m102（tool）：`Your messaging app can send MMS messages.`
- m103（user）：`It works! The check now says my messaging app can send MMS messages. Oh, I'm so relieved—thank you!  One small thing though—you mentioned earlier it would be $0.10 per GB, but then the 1 GB came to $2.00. Could you clarify that? I just want to make sure I'm not being overcharged. And is there anything I need to do to pay for it?`

m98给目标线路L1002充1GB，m102工具明确返回可发送MMS，m103用户确认恢复；评分只剩金额断言失败。agent此前也有错误报价，后来澄清，不把这些可恢复错误当作这次0分的主因。

### 4.2 全错题主要暴露哪些能力缺口

| 归因                                              | 失败随机样本60条 | 全错题随机样本30条 |
| ------------------------------------------------- | ---------------: | -----------------: |
| T1：设备侧子目标/必要转接遗漏、循环耗尽步数       |                7 |                  4 |
| T2：号码/线路等实体 grounding、工具与状态语义错误 |               20 |                 15 |
| T3：账户侧充值或开通漫游遗漏                      |               32 |                 11 |
| B1：连接已恢复，仅恰好2GB断言拒绝                 |                1 |                  0 |
| U：用户独立致败                                   |                0 |                  0 |

30条全错题样本里的任务有些同时属于充值歧义类别，但这30条实际轨迹都还有未解决的服务目标，未观察到“已解决却只因金额拿0”的致败路径。不能据此说其任务完全无问题，也不能把1,440道全错题整批当坏题。

- **漏账户侧动作**：agent修完设备设置就说没有其他工具，尽管用户主动提出充值。以下m88甚至声称数据额度与MMS无关，随后结束，属于T3：

[轨迹](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/instruct-telecom-full/trajectories/telecom:%5Bmms_issue%5Dairplane_mode_on%7Cbad_network_preference%7Cbreak_app_sms_permission%7Cdata_mode_off%7Cdata_usage_exceeded%7Cunseat_sim_card%7Cuser_abroad_roaming_enabled_off%5BPERSONA:None%5D:trial2/d6e2313c64d64e1396145690db6a3d3a.json)（telecom:\[mms_issue\]airplane_mode_on|bad_network_preference|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off\[PERSONA:None\]:trial2）；SHA256
见 labels。

- m87（user）：`I checked again, and it still cannot send MMS messages after the reboot. Is there anything else we can check, like Data Saver or my data balance? I’m willing to refuel 2.0 GB of data if that might help, but I don’t want to change my mobile data plan.`

- m88（assistant）：`I appreciate your patience and willingness to try additional steps. Let's clarify a few important points:    - **Data refueling** (adding extra data) does **not** enable MMS functionality. MMS depends on correct network settings, APN configuration, and app permissions—not on data volume.   - Your **data balance** or **data usage`

- m89（user）：`###TRANSFER###`

- **修错线路**：目标号码为555-123-2002，正确线路L1002；agent给L1001开漫游，最后仍未修复。以下为T2：

[轨迹](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/instruct-telecom-full/trajectories/telecom:%5Bmms_issue%5Dbad_network_preference%7Cbreak_apn_mms_setting%7Cbreak_app_both_permissions%7Cuser_abroad_roaming_disabled_off%5BPERSONA:Hard%5D:trial0/93f6377dd8f54949b0c5dca6a236a539.json)（telecom:\[mms_issue\]bad_network_preference|break_apn_mms_setting|break_app_both_permissions|user_abroad_roaming_disabled_off\[PERSONA:Hard\]:trial0）；SHA256
见 labels。

- m18（assistant）：` [{"name": "enable_roaming", "arguments": {"customer_id": "C1001", "line_id": "L1001"}}]`

- m76（assistant）：`I completely understand how frustrating this must be—sending a photo of your grandkids should be simple, not a multi-hour ordeal. I'm truly sorry we haven't resolved this yet.    Since we've exhausted all standard troubleshooting steps—roaming enabled, correct APN settings, proper permissions, network mode, and data restrictions`

- **把修复后的状态当成仍故障**：4条失败的pure official
  reward实际为1，fixed因后续真实转接清零。逐条看都存在agent错误声称仍未修复/读错正常状态，未据此新增坏题。例如以下m38已关闭Wi-Fi
  Calling，m44却把正常MMSC URL说成无效并转接，归T2：

[轨迹](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/instruct-telecom-full/trajectories/telecom:%5Bmms_issue%5Dbad_wifi_calling%7Cbreak_apn_mms_setting%7Cdata_mode_off%7Cuser_abroad_roaming_enabled_off%5BPERSONA:Hard%5D:trial2/8415be2765eb44959fdeeb2712962f97.json)（telecom:\[mms_issue\]bad_wifi_calling|break_apn_mms_setting|data_mode_off|user_abroad_roaming_enabled_off\[PERSONA:Hard\]:trial2）；SHA256
见 labels。

- m38（tool）：`Wi-Fi Calling is now OFF. Status Bar: 📶⁴ Excellent | 5G | 📱 Data Enabled | 🔋 80%`
- m44（assistant）：`Thank you for sharing that. The MMSC URL you provided — `http://mms.carrier.com/mms/wapenc`— **does not appear to be correct or valid** for your carrier.    A valid MMSC URL is typically specific to your carrier and is usually something like:   -`http://mms.cingular.com`  -`http://mms.att.net`  -`http://mms.t-mobile.com`  `
- m46（assistant）：`I understand how frustrating this is, especially while traveling. Let me clarify what’s possible with the tools I have access to.    Unfortunately, **I cannot update or modify APN settings remotely**, including the MMSC URL, even if you provide information. The tools available to me do not support changing or validating APN sett`

脚本最初找到33条“没有失败环境断言的0分”，进一步拆开是：**4条official
reward=1但fixed清零、22条ACTION型合法转接任务漏掉真实转接、7条没有正常完成评分的预算/错误终止（6 max_steps、1
too_many_errors）**。不能把空的断言数组当“全部目标做对”。未新增不可达gold或另一类已证实的任务矛盾。

### 4.3 用户致败估计与U1筛查

在60条等概率失败样本中，用户独立致败 **0/60=0%**，Wilson 95%区间 **\[0%,
6.02%\]**。它说明当前样本中agent缺口占主导；没有证明7,347条失败完全没有用户问题，也没有证明全部成功轨迹都遵守模拟规则。该区间针对本批有限失败轨迹的抽样比例，不是把有相关性的同题trial当成独立实验来估计后端长期错误率。

全量“STOP同句出现yes/confirm/proceed/go
ahead等授权词”命中31条：**24条成功、7条失败，人工读终段均不是U1**。成功例已经通过can_send_mms/run_speed_test；7条失败是agent先误诊、拒绝继续或误称套餐不能发MMS，用户随其建议结束。扩展sure/okay/add/refuel/enable等词后，另查26条失败STOP，也没有看到“正确待执行授权被同句STOP截断”。这些定向样本不用于抬高或降低0/60的估计。

另有
**1,533条失败在没有真实transfer工具调用时出现用户TRANSFER**，以及4条相同形态的成功。这个计数是协议筛查，不能全算用户致败。样本中常见agent已经文本宣告“You
are being
transferred”或输出未被解析为工具的JSON，用户才按宣告回应；此时缺少真实工具/错误结束排障应归agent。只有agent尚未宣告转接、用户自行用控制符截断正确待执行步骤，才符合这里的U7a致败边界。原始计数不应冒充人工归因数量。

### 4.4 14条用户服务异常是什么

| 记录层                         |                     数量 | 解释                                                                 |
| ------------------------------ | -----------------------: | -------------------------------------------------------------------- |
| trajectory JSON文件            |                    9,173 | 9,140条最终完成 + 33条丢弃的错误尝试                                 |
| 重试次数分布                   | 9,108一次；31两次；1三次 | 额外33次尝试，最终无infra排除                                        |
| 空用户响应                     |                       14 | content为空、无tool_calls，reasoning_content非空，finish_reason=stop |
| 其余错误尝试                   |       18连接错误 + 1超时 | 仅凭异常类型不能断言来自DeepSeek还是actor/proxy服务                  |
| 最终完成消息中的真正空用户响应 |                        0 | 空content但带tool_calls是合法工具调用，不算空回复                    |

14份 `tau2-user-failures/*.json` 的 `error`
字段都为null，因为是取得响应后才发现正文与工具均为空；失败原因不是“没有异常”。内容往往被写进reasoning字段，例如
`I checked, and Wi-Fi Calling is currently off. What should I check next?`，但正文为空。记录中的一个长reasoning还出现角色/工具归属困惑；它没有作为正文进入完成对话，不能另算一条已完成episode的U5。

RecordingUserSimulator会抛
`EmptyUserResponseError`，结束整个attempt；episode级重试从头重放，不是只补这一句。14次错误尝试对应的最终episode有
**2成功、12失败**；失败不能倒推为那次空回复造成，因为最终保存的是重新生成的轨迹。逐条原始文件、失败attempt路径、后续完成轨迹、响应短文与hash见
[service_incidents.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/service_incidents.json)；33次错误的统计见
[retry_summary.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/retry_summary.json)。

279,986条已记录用户消息中，机器筛查未命中思维链标记、AI身份拒演等异常正文模式；这仅是字面筛查。没有逐条人工审完所有用户正文，因此不报告“语义异常内容为0”的全量结论。

## 5. 训练题候选与混合比例

| 集合                      | 题数 | 0/1/2/3/4 次通过    | D1 通过率 | 并入747后 Telecom占比 | 合计题数 / batch数 |
| ------------------------- | ---: | ------------------- | --------: | --------------------: | ------------------ |
| eligible_all              | 1136 | 487/223/138/122/166 |    33.65% |                60.33% | 1883 / 15          |
| contrast_only             |  483 | 0/223/138/122/0     |    44.77% |                39.27% | 1230 / 10          |
| contrast_one_per_key      |  479 | 0/222/137/120/0     |    44.68% |                39.07% | 1226 / 10          |
| recommended_249           |  249 | 0/83/83/83/0        |    50.00% |                25.00% | 996 / 8            |
| curriculum_easy_and_mixed |  649 | 0/223/138/122/166   |    58.90% |                46.49% | 1396 / 11          |

以上通过率均是**用来筛题的同一批D1观测**，不是新评测结果。推荐集合精确构造为1/2/3次通过各83题，所以50%是筛选设计的结果，不是训练效果预测。完整ID都在
[candidate_sets.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/candidate_sets.json)。

- `eligible_all`覆盖更广，但487道全错与166道全对会使大量组缺少观测到的组内对比；加进来后Telecom占60.33%，且每epoch15个batch，会明显改变原Airline/Retail训练预算。
- `contrast_only`有483题，全部有正负样本，适合作为后续扩展池；Telecom占39.27%，起步偏重。
- `contrast_one_per_key`只比483少4题，因为官方full本来几乎是一个setup一题。按key“各留若干persona”在这里基本无法缩减规模；没有可直接选出的三种persona对照。
- `curriculum_easy_and_mixed`加入166道全对题，可作warm-up或保留简单操作覆盖，但在本次采样中这些题没有组内对比；它的46.49%领域占比仍偏高。
- **推荐
  `recommended_249`**：从479个去重混合题中，用固定种子20261019，按1/2/3通过各83题、persona轮转抽取，None/Easy/Hard为84/84/81题，MMS/mobile_data/service为212/25/12题。它保留不同难度与用户表现，避免少数极难组合占据大多数训练步；仍以MMS为主，符合源池的主要覆盖。

原训练池实际是 **AReaL pack v2的648题 + official fixed train的99题=747题**；只统计
`_cache/tau2-areal-pack-latest` 会得到648，不能把差额补成99道Telecom。现行 `config_ppo.yaml`
为每epoch每题1条、batch128、8epochs。与249道Telecom合并后，每epoch996条，约8个batch（原来ceil(747/128)=6）；8epochs是7,968条、约64步，对比原5,976条、48步，episode预算增加33.33%，更新步数增加16步。

这里建议的是**题数/episode比例25%**。Telecom对话长，token、API调用和墙钟占比可能高于25%；本轮没有训练测量，不能把“多249题”解读成仅多25%费用。如果要固定原更新步数，就必须减少epoch数或重配域采样权重；那会改变Airline/Retail曝光量，不能同时宣称保持原训练设置。当前推荐保留原747题的每epoch曝光，并清楚接受较大的总预算；本任务只给清单，不启动训练或改配置。

已有task错误的修复路径可沿用Telecom
v1，将“愿意2GB”改成“恰好2GB”，但这属于后续改题和验证。本轮清单不含这些题，未修改官方任务或pack。其余推荐题只通过已有机械检查与本轮抽查，未完成逐题完整语义审查；按training-data
§5的生产筛题标准，正式发布pack前仍应逐题审题，不能把本次候选报告当成249题全部人工验收通过。

## 6. 重读、验证与交付

初始标签保存后，用Random(20261020)从90条有主因的标注中随机抽15条，重读关键实际写入、未满足目标和结束前原文，**主因改动0条、记录改动0条**。[audit15.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/audit15.json)
保存抽中ID、before/after与路径。是同一标注者重读，不是独立一致性检验；本轮遵守“独立完成、不启sub-agent”的要求。

关键产物：

| 文件                                                                                                                                                                                                                                                                                                                              | 内容                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [task_stats.csv](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/task_stats.csv)                                                                                                                                                                                   | 2,285题，每题一行；4次通过数、issue、persona、故障key、test重合、机械结果、B1标记、推荐标记、历史同key分数 |
| [labels.jsonl](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/labels.jsonl)                                                                                                                                                                                       | 221条实际读取轨迹的定位、原文、阅读范围与标签；只有随机样本进入比例                                        |
| [candidate_sets.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/candidate_sets.json) / [recommended_task_ids.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/recommended_task_ids.json) | 全部候选规模、分布、排除ID、推荐249题ID                                                                    |
| [fault_combinations.csv](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/fault_combinations.csv) / [fault_signatures.csv](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/fault_signatures.csv)     | 全量故障组合分组表                                                                                         |
| [benchmark_tasks.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/benchmark_tasks.json) / [mechanical_failures.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/mechanical_failures.json) | 1,120道评分歧义任务与0道机械失败任务的完整清单                                                             |
| [topup_counterfactual.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/topup_counterfactual.json)                                                                                                                                                             | 1,120题1GB重放、3题gold/empty对照                                                                          |
| [comparison_v1.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/comparison_v1.json)                                                                                                                                                                           | 243个共同key的D1与历史pack分数                                                                             |
| [service_incidents.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/service_incidents.json) / [retry_summary.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/retry_summary.json)         | 14条空回复及33条丢弃错误尝试                                                                               |
| [provenance.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/provenance.json)                                                                                                                                                                                 | 输入文件、规则文档、运行源码版本与SHA256                                                                   |
| [verification.json](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/verification.json)                                                                                                                                                                             | 覆盖、重算、锚点、候选约束、反例检查的机器结果                                                             |

脚本均在本目录：`analyze.py`负责统计和抽样，`view.py`/`compact.py`/`targets.py`供阅读，`build_outputs.py`保存本标注者的显式人工决定并生成候选，`check_topup.py`做CPU重放，`finalize_analysis.py`保存异常链路与输入快照，`write_report.py`生成本文，`verify.py`验证。它们不是自动语义标注器；重跑不等于新的人类/模型复核。

可重算命令见
[REPRODUCE.md](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-telecom-full-20261015/case-study-d1/REPRODUCE.md)。所有新产物仅写入case-study-d1，仓库只读，未重新采样、训练、使用GPU、调用付费API或启动sub-agent。任务质量结论的确定范围是已证实的充值歧义、0道机械失败与实际抽读的失败机制；未发现其他坏题不等于其余1,165题已逐题证明无歧义。
