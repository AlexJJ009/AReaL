# 三种用户模拟器的失败归因（fixed-v2 test 60）

**零售掉分主要来自用户模拟器；航空仍主要暴露 actor 的拒绝与工具使用问题。** 三轮各有240个完成的test
episode，无缺题、重复或最终infra排除。逐条标注272个失败：133个agent首因、124个user首因、15个benchmark首因。另检查205个discordant成功侧，确认5个用户引导恢复的U6样本。

## 口径与结果

只取官方test的Airline 20题、Retail 40题，每题4次；两份base运行的train部分不计入。读取fixed-v2的user_scenario、gold
actions与policy后按任务—trial配对。每条失败只给一个首因，取有明确证据、实际阻断目标或导致评分失败的机制；已经恢复的口误不自动成为首因。未将正常拒绝后的结束、任务刻意要求的错误信息，以及条件触发的改主意统统算成用户错误。消息编号从0开始，包含agent开场问候。

resolved_config与运行记录一致：actor为同一Qwen3.5-4B基础权重快照，non-thinking、policy temperature
0、无LoRA；benchmark_variant=fixed；用户temperature 0；NL judge均为DeepSeek-Flash、temperature
0。API两轮各656个总记录，Qwen轮240个；各自test子集恰好240个。只选status=completed且分数对应episodes.jsonl的轨迹，另有DeepSeek
4、Doubao 7、Qwen 7个test失败尝试被重试替换，保留在manifest中，没有把它们重复计为失败episode。

| 用户模拟器             | Airline成功 / 80 | Retail成功 / 160 | 合计mean@4 | Agent首因 | User首因 | Benchmark首因 |
| ---------------------- | ---------------: | ---------------: | ---------: | --------: | -------: | ------------: |
| DeepSeek-Flash         |               41 |              132 |     0.7208 |        46 |       13 |             8 |
| Doubao 2.1 Turbo       |               35 |               97 |     0.5500 |        49 |       55 |             4 |
| Qwen3.5-4B self-hosted |               36 |              107 |     0.5958 |        38 |       56 |             3 |

这是BRIEF指定的新DeepSeek运行，不是README旧轮的0.688。领域细分和pass^4见[matrix.md](matrix.md)。三者不一致的唯一(task,
trial)有128个：Airline 37、Retail 91。DeepSeek与Doubao有65个discordant，DeepSeek与Qwen
92个，Doubao与Qwen 99个；同一单元会进入两个模型对，不能把256次两两discordant当256个独立样本。

## 每个模拟器的主要问题

**DeepSeek：较能守住目标，仍会把“确认”误当“已完成”。**
其13个user首因全部是Retail的U1。另有一例姓名认证信息欠明确，最终归B1。Airline的39个失败主要是agent：21个M2a、4个M14、2个M5等；其余包含欠指定gold与合法handoff误伤。DeepSeek不是无噪声基准：确认退货、换货、地址修改时都可能带STOP，直接让写前确认的agent拿到0分。

**Doubao：最明显的是提前结束。** Retail
63个失败中49个user首因：31个U1、9个首轮结束U4a、3个中途提前结束U4b、6个偏离目标U3。Airline还有1个首轮结束、2个提前TRANSFER、1个角色/元话语混淆。部分首句出现中文售后话术或完全无关内容，随后又恢复原任务；这类恢复过的污染没有一律算成失败首因。

**Qwen：更突出的是跟随agent的错误前提，改变隐藏任务要求。** Retail 53个失败中40个user首因，37个为U3，U1只有1个。典型是编造123
Broadway、1234
Maple等地址；在明确预算或确认分支下仍同意错误操作；把“不记得email”变成“没有email，所以拒绝换货”。Airline的13个U3包括编造原预订之外的路线、乘客或日期，以及忘记中途切换投诉目标。另有2个episode跑到max_steps，分别是airline:2
t1、25 t3，已计入其U3首因。

按首因计，Retail的agent失败数几乎相同：DeepSeek 13、Doubao 14、Qwen
12。相对DeepSeek，Doubao多35个Retail失败，同时多36个user首因；Qwen多25个失败，同时多27个user首因。这是对观察到的失败的分解，不能据此计算“修好用户后的actor分数”。

## 编码与代表轨迹

沿用README
14.2/15.5的M1–M17与T码，本次没有Telecom，所以没有T样本。用户沿用16.15的U1–U7，只作三个细分：U4a为未开始任务就结束，U4b为未完成即认定结束或缩小目标，U7a为转人工工具执行前发TRANSFER；它们已有上位机制，未另造U8。U1严格指授权实际操作的同句STOP；拒绝修改并改提取消时直接STOP，归U4b。

benchmark用B1映射既有“任务有歧义、未指定支付方式、gold固定唯一选择”，B2对应已知fixed转人工规则误伤。BRIEF指向的README第7节主要讲实现差异；具体旧benchmark分类在10.2，合法handoff误伤在14.3。B3只作次要评分敏感性记录：retail:12两个Qwen样本DB正确且已转接，仍因ACTION缺get_order_details(#W7387996)失败。它们首因是用户提前透露任务禁止明说的保留物品，不能据此断言ACTION
evaluator有新bug。

U6是“用户引导agent从错误死路恢复”的观察标签，**不必然是用户违反指令**。对没有明确禁止的补救，只锚定任务分工和前后行为，不伪造一条“被违反”的指令；qwen
retail:5 t1直接引用隐藏scenario，另有全局guideline第9行的逐字泄露证据。五例不等于无该句必败：未做删句重放。

### U1：授权后同句STOP（45个失败）

- [deepseek retail:49 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:49:trial0/dedf5721279141a082455c5db9f348d3.json)，消息20–21：agent问“confirm
  … proceed”，用户答“Yes, please proceed … ###STOP###”；换耳机工具一次也没执行。
- [doubao retail:17 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:17:trial0/9853b3065bf84c878f54fb47400afa90.json)，消息12–13：“Please
  update … Suite 641. ###STOP###”；明确地址目标见reason_for_call第1行，modify尚未发生。
- [qwen retail:70 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:70:trial3/3d21f45b2862430e9b2909268481c8d9.json)，消息24–25：“Just
  process it. I’m done. ###STOP###”；已经知道差价22.55，仍在执行前结束。

### U3：目标、事实和条件被改变（57个失败）

- [qwen retail:33 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:33:trial0/5a58e53c9c3740e2b18d0f6b46690b8d.json)，消息15、18：用户编造“1234
  Maple Avenue … 98101”，违反从订单取Seattle地址且不透露地址的reason_for_call第1行；gold是517 Lakeview
  Drive。
- [qwen retail:90 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:90:trial1/e55014b6f9e74b329ea322ba28707f1c.json)，消息20–22：agent报10x相机3066.23，用户仍“Yes
  … modification”；reason_for_call第1行要求超过3000时以ordered by mistake取消。
- [doubao retail:45 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:45:trial0/a1c32491b3a84f38b94e2b24652d9572.json)，消息20–22：agent列了三个canister选项，用户明确选bagged
  2872451762；reason_for_call第1行规定有多个选项时选bagless。

### U4 / U7a / U5：未开始、缩小目标和角色混淆

- [doubao retail:9 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:9:trial0/71f2b6e3b65e4db189d47dfa37bebbb5.json)，消息0–1：开场问候后直接OUT-OF-SCOPE；完整身份和换货目标均已给定，未出现缺信息的实际请求。U4a共10个。
- [doubao retail:108 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:108:trial3/9711cd545660412aae8f0b00fe3e6a5f.json)，消息16–17：把“实际退除tablet外物品”改为“主要询退款金额”，得知346.93即STOP；U4b共3个。
- [doubao airline:13 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:13:trial0/32959f497918421cb1a5d41c5ed7c727.json)，消息12–13：用户一边请求handoff一边发TRANSFER，agent还没调用工具，fixed记录missing_transfer；U7a共5个。

角色混淆的独立例子：[doubao airline:29 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:29:trial1/5aafbb377e6b448ca883dce25a1c73b3.json)，消息2–3，被问user
ID后输出“之前的scenario是啥…scenario确实缺失了关键信息…OUT-OF-SCOPE”，而known_info第3行已给出raj_brown_5782。

### U6：成功侧也有用户引导恢复（5例）

- [qwen retail:5 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:5:trial1/55cc445caea44044b379f82be2dccdca.json)，消息19–24：agent无视“只换台灯”仍要求确认两件；用户引用scenario的第二次确认分支，改为退水瓶，agent随后正确return。成功掩盖了M8，且用户泄露隐藏指令。
- [deepseek retail:64 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:64:trial3/98cca3afb69346588b1ab2973fe09302.json)，消息16–22：agent只强调pending不能exchange；用户问“if
  … still pending, can we switch … instead?”，才转入可执行modify并成功。
- [qwen retail:97 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:97:trial3/98cba302b66a487d9e352c661a6d0ce8.json)，消息14–18：agent称“system
  doesn’t have a way to look up … addresses”；用户反问从另一订单取地址，下一轮agent读回476 Maple Drive并执行。

另外两例是[doubao retail:64 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:64:trial3/fb96c60c9af944819bd2ae06f3028344.json)，消息14–20与[deepseek retail:97 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:97:trial1/30f9effe8e9744a98a2b570f3f4e390d.json)，消息14–20，均由用户提出pending
modification替代exchange。正常确认、复述已正确排序的航班，以及误以为watch未执行的追问，没有计入U6。

### M2a：资格判断正确，仍用转人工代替拒绝（59个失败）

- [deepseek airline:45 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:45:trial2/8c63a311d7a44de3987f4a1ca615deee.json)，消息6–9：知道basic
  economy、无保险、超过24小时，仍transfer；用户任务明确禁止handoff。
- [doubao airline:35 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:35:trial0/987d8f5a98f54793b52d82523ad70c4f.json)，消息18、28–32：先为不能取消的订单transfer，再正确完成新订票，fixed仍记录unexpected_transfer。
- [qwen airline:48 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/airline:48:trial0/e12befa52af34900b7f93e07ae5c0693.json)，消息6–9：据数据库创建日期拒绝取消是正确的；失败发生在转人工。用户坚持“10小时前”是任务要求，不是U2。

### M14：误认工具能力或订单状态（26个失败）

- [deepseek airline:44 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:44:trial2/d6d0357a878f4fe2a8d758a6da962e38.json)，消息18–20：称没有查时刻工具并转人工；search_direct_flight实际返回scheduled_departure/arrival字段，gold也明确列出这些查询。
- [deepseek retail:56 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:56:trial0/66d819d12eea4d7fbd4d1efaaf21d97a.json)，消息16–17：声称不能只改air
  purifier，实际modify_pending_order_items支持子集；用户按“否则no action”结束，不能反过来怪用户。
- [doubao retail:102 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:102:trial1/de6d4c720b0a4a86892a88350b385cfa.json)，消息32–35：exchange被拆成先return再exchange，第一步改变状态，第二步报Non-delivered
  order cannot be exchanged。

### M15：错误后静默改成更贵的variant（8个失败）

- [deepseek retail:36 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:36:trial1/9af990dd77b943618723c508251613ae.json)，消息30–35：old==new报错后，把原无需改变的umbrella/backpack也换了；总价从可达1130.85变成1146.89，超过1131。
- [doubao retail:36 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:36:trial1/a91e4afedb66427f85d8d0b223902a56.json)，消息30–35：同样擅自改变variant；用户追问“you
  said … 1130.85”，此后订单已锁定。

### M5：退款路由、M8：子目标未执行

- [deepseek airline:18 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:18:trial2/f967d5f941cf44cb91f5fa5b8e90b66f.json)，消息20–28：五单降舱退款全部用credit_card_2929732，gift
  card和另一信用卡的原支付路由被丢失；M5共10个。
- [qwen airline:37 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/airline:37:trial0/68e93f2c61a448f98a67e6c87b04e1a6.json)，消息10、14：用户要求信用卡尾7334，actor却用credit_card_4959530；付款失败与合法handoff评分问题同时存在。
- [doubao retail:101 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/retail:101:trial2/d23d70f25bd64942be2c835ad3b844f2.json)，消息20–34：watch已被授权但没有modify工具调用，后续只完成purifier；M8共9个。
- [deepseek retail:97 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:97:trial3/dca3c628c6ac4888938dd4472df17d57.json)，消息18–20：先改speaker把订单锁定，未先执行已要求的NYC配送地址修改。

### M1：policy需要在调用前执行（10个失败）

- [deepseek retail:18 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:18:trial0/80b9c0eafdca45cfb945fc8c1948889e.json)，消息17–20：同一个item_id
  8069050545换自己，tool允许但policy要求different product option；应提示后选任务给定灰色椅子。
- [doubao airline:44 t1](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:44:trial1/81a84bad00db4d81b5c3460bce998acd.json)，消息37–38：健康用户按change
  of plans取消S61CZX；保险仅覆盖health/weather，不能因为工具执行成功就判合规。

### B1：指令欠明确或有效分支被gold判错（7个失败）

- [deepseek airline:29 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:29:trial2/bc256c70e5b64726965157f7774cd8bf.json)，消息21–24：新票用另一有效信用卡；任务未指定新票付款方式，gold却只接受credit_card_8003957。
- [qwen retail:55 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/step0-selfhost-user/full-qwen35user-v2/trajectories/retail:55:trial3/ba46462c091e44dbba93f14a43cbc6ff.json)，消息23–26：任务未指定取消原因，用户选择合法ordered
  by mistake，gold固定no longer needed。
- [deepseek retail:45 t2](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:45:trial2/889e2f1959874b2ea460b65e7925ae27.json)，消息16–18：agent只给一个bagged选项，用户按条件允许接受；gold/NL却固定bagless及9.89，而实际此分支退款30.49。

身份欠明确的边界例：[deepseek retail:5 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/retail:5:trial3/9cba62ca68d0403eb1b63d1cd09c38b5.json)，消息4–9，known_info只给mei_kovacs_8020与28236，gold认证却要求Mei/Kovacs。用户guideline第8行要求把未提供信息视为未知，Retail
policy第10行又禁止凭ID跳过认证；因此按字面给定信息归B1。若把ID组成部分视为已明确给出姓名，才会归U2；没有宣称gold在技术上不可执行。

### B2：合法handoff被fixed扣零（8个失败）

- [deepseek airline:37 t3](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/deepseek-base-u0p0/trajectories/airline:37:trial3/00930d9db8074044ae6300939647cfdc.json)，消息18–20：正确升舱后对已飞NQNU5R转人工：policy第141行明确要求transfer，gold没有transfer，fixed把official_reward=1改成0。
- [doubao airline:37 t0](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/api-user/doubao-base-u0p0/trajectories/airline:37:trial0/e0663e9a105941d98702d9fd59f25375.json)，消息18–20：同一冲突：正确升级，合法已飞航段handoff触发unexpected_transfer。handoff也包含不可取消的IFOYYZ，不能称整段都无瑕疵。

## 哪些问题值得研究，哪些agent能力仍需补

**用户侧优先研究两条明确机制。**
第一是把authorization当completion：U1发生在actor遵守写前确认的时候，序列级负奖励会惩罚正确的确认动作。可研究如何用真实工具状态区分“用户授权”与“任务完成”，以及带执行前STOP的样本如何进入优势估计。不能简单删掉所有STOP：拒绝类任务本来就可能无需写操作。

第二是隐藏任务目标在对话中漂移：编造地址、越过预算、忽略“第几次确认改主意”等条件。这类用户行为让actor遵从当前用户消息也得到0分。应该分别度量忠于scenario、忠于当前对话与任务终态；需要检查错误前提最先由谁提出，不能只看最后一个错误参数。

**U6值得作为成功率的伴随指标。**
同一个actor的缺口可能在某个用户下被绕过；本次既看到API用户提出可执行modify替代方案，也看到Qwen泄露scenario把agent拉回正确分支。把成功都当actor独立完成，会高估工具语义和目标跟踪能力。五个恢复是可追溯候选，不是因果效应估计。

**agent侧优先补M2a、M14/M15、M5与M8。**
M2a集中在Airline，资格推理往往已正确，动作却变成handoff；应训练“拒绝后继续处理允许的子任务”，并让转人工规则识别policy要求的合法转接。M14/M15需要读懂查询返回字段、子集修改与状态变化，错误恢复时重新核对预算及用户授权。M5需要把每个预订映射回原支付方式；M8需要在写操作前保留全部目标，并处理地址先于items的顺序。M1还提醒我们：API不强制检查policy，tool成功不能代替资格与同variant限制的判断。未声称某一种RL方法已经能解决这些缺口。

## 限制与复核

- 单人逐条归因，没有独立标注者。用seed=20261012从272个失败中随机抽10条重读：首因改变0/10，1/10补充U5次因（qwen airline:44
  t0，消息15以客服口吻回应）。细节见[spotcheck.json](spotcheck.json)。这不是标注者间一致性。其后对抽样之外的deepseek
  retail:5
  t3补做身份边界复核，首因由U2改为B1、删除M10次因，见[final_review_corrections.json](final_review_corrections.json)；未将这次额外更正混入0/10。
- 配置核对的是同checkpoint、greedy temperature
  0。三轮enable_deterministic_inference都为false，trial也没有共同用户采样seed；相同可见前缀跨模拟器的检查没有找到匹配组。因此不能把配对差异提升为“严格证明只由用户造成”。这些配对用于定位分歧机制，未做干预、用户删句或反事实重放。
- 用户侧推理与输出设置不同：Doubao使用reasoning_effort=medium，Qwen关闭thinking并设4096
  max_tokens。比较对象是这三套完整模拟器配置，不能把差异完全归到模型尺寸或强弱上。重试选择也可能偏向能完成的轨迹。
- 124个user首因是有具体观察证据的标签数，不是移除模拟器噪声后能追回124个成功。U3有时是agent错误提议与用户目标漂移共同作用；secondary保留这些边界。没有发现需要另立gold不可执行或环境bug首因的新类。
- 15个discordant成功侧在没有transfer工具调用时已发TRANSFER（DeepSeek 8、Doubao 3、Qwen
  4）。这会保留部分本应暴露M2a的分数；较高分并不证明模拟器更忠实。完整成功侧检查见[discordant_success_review.jsonl](discordant_success_review.jsonl)。

交付：[labels.jsonl](labels.jsonl)、[matrix.md](matrix.md)、本报告。labels每条含消息索引/短引文、用户任务字段行号、gold
actions、评分结果和轨迹/数据哈希；U类另锚定全局用户guidelines，U6无明确违规的例外已注明。[manifest.json](manifest.json)保留配置、来源与重试选择；[validation.json](validation.json)记录覆盖、分数、引文、哈希及配对计数校验。故意删标签、重复标签、改消息索引、改指令引文、改分数的五个负控制均被拒绝。

没有启动新采样、训练、GPU任务或付费API；没有编辑repo代码或README。分析脚本与中间材料均在本case-study目录内。
