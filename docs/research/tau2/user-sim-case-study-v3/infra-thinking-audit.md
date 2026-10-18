# 用户模拟器 infra / thinking / U1 归因 case study（Claude subagent，2026-10-18）

`B=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012`，`W=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/infra-case-study-v3data`；`C=/data_storage/yl_test/lgx/data-1/code/_cache`。只读分析，CPU only，无 API/LLM/pueue 操作。

## 0. 结论

| 假设 | 判定 | 证据 |
| --- | --- | --- |
| (e) infra | 不是 U1 原因；Turbo 首跑本身被服务异常污染 | 除 Turbo 首跑外 user 调用 finish_reason 100% stop，无空消息/畸形 token/重复；U1 在干净的 r2（46）和首跑干净子集（17/281）照常出现 [verified] |
| (d) thinking | 既不充分也不必要，模型相关 | DeepSeek 22/849 vs 1/903（p=1.5e-6）；Turbo U1 集中在无 reasoning 调用（42/188 vs 4/284）；GLM 始终 thinking 0/521 [verified] |
| (c) guideline 歧义 | 机制有直接文本证据，未做反事实 | DeepSeek reasoning 引用"if the instruction goal is satisfied"并推出"confirm 后任务完成"；不 STOP 时写"No stop yet" [verified 文本]；语音版句子能否消除 [hypothesis] |
| (b) persona/场景 | 次要 | 催促词 p=0.15–1；"confirm"场景指令 Lite 17.7% vs 3.1%（p=6e-5）[verified 相关性] |
| (a) 真实行为 | 不能作为合理行为 | GLM 0/521、Qwen 0/507、DS no-think 1/903 在同 prompt 同任务不发生；真实性本身无法判定 [hypothesis] |

U1 的合理表述：文本版 guideline 对"何时算目标达成"有歧义（暴露面），叠加 tau2 的"消息含 `###STOP###` 即终止、agent 不再有回合"语义，使 DeepSeek thinking、Doubao Turbo/Lite 把"用户已确认"当"任务完成"。是否触发由模型倾向决定，thinking 开关只对 DeepSeek 是放大器。

## 1. U1 判定规则与验证

规则 R（`W/lib.py:is_u1_rule`）：(1) `user_stop` 结束，最后一条 user 消息含 `###STOP###` 且不含 TRANSFER/OOS；(2) 前一条是 agent 纯文本（无 tool call）；(3) user 消息去 token 后含授权词（yes/confirm/go ahead/proceed/please do/approve/that's correct/sounds good/please+动作…）；(4) agent 消息末 500 字符含明确确认请求（do you confirm/please confirm/shall I/would you like me to/reply yes…），且不是纯"anything else?"收尾。

验证（`W/rule_eval.py` → `out/rule_eval.txt`），对象是 `case-study-v3/labels.jsonl` 中 primary=U1 的 77 条（DS 13、Turbo 41、Lite 23、GLM 0、Qwen 0），对照五个 run 全部 1,200 个 episode：

| run | TP | FP | FN | 命中但 reward=1 |
| --- | --- | --- | --- | --- |
| v3-deepseek-test-u0p0 | 13 | 0 | 0 | 0 |
| v3-doubao-test-u0p0-r2 | 41 | 2 | 0 | 3 |
| v3-doubao-lite-test-u0p0 | 23 | 0 | 0 | 0 |
| v3-glm53-test-u0p0-r2 | 0 | 0 | 0 | 0 |
| qwen-v3 | 0 | 0 | 0 | 0 |

recall 77/77；约 310 个标注为非 U1 的失败上 FP=2：`r2 retail:101:trial0`（标 M8）和 `r2 retail:108:trial3`（标 U4b），两者用户确实授权后同句 STOP，标注者按先发生的致败原因记了别的码。3 个 reward=1 的命中是 `r2 retail:27:trial1/2/3`（agent 提议的 return 不在 gold，STOP 无害）。规则修订了两次：第一版 FP 6、reward=1 命中 17（把 okay/sure/correct 当授权词、把 "Let me confirm the total" 当确认请求），第二版 FP 2 / 5，第三版 FP 2 / 3。修订在同一批标注上完成，**in-sample**；规则在 GLM（521 回合）和 Qwen（507）上为 0，不是恒真。

Turbo r2 规则命中 46 = 41（标注 U1）+ 2（他码）+ 3（reward=1）。

**确认回合（opportunity）**：agent 纯文本消息含确认请求且下一条是 user 消息。U1 率 = 其中 user 含授权词且含 `###STOP###` 的比例。

## 2. Q1 infra 审计

抽取 `extract.py`，审计 `audit1.py audit_errors.py audit2.py usage8.py doubao1-6.py leak.py leak2.py`。

### 2.1 逐 run 汇总

| run | 数据 | 完成/计划 | 失败 attempt（user侧 / actor侧） | user finish≠stop | 空消息 | token畸形/同条多token | 异常 episode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v3-deepseek-test-u0p0 | v3 | 240/240 | 0 / 7 ctx | 0/1384 | 0 | 0/0 | 0 |
| **v3-doubao-test-u0p0（Turbo 首跑，exit 1）** | v3 | **238/240**（`airline:13:trial0`、`airline:44:trial2` 未完成）| 37 / 2（32 EmptyUserResponse、4 overdue balance、1 Timeout）| **8/1600** | 0 | 0/3 | **86/238（36%）** |
| v3-doubao-test-u0p0-r2 | v3 | 240/240 | 0 / 4 | 0/1375 | 0 | 0/0 | 0 |
| v3-doubao-lite-test-u0p0 | v3 | 240/240 | 0 / 3 | 0/1288 | 0 | 0/0 | 0 |
| **v3-glm53-test-u0p0（exit 1）** | v3 | **37/240** | **886 RateLimitError** / 0 | 0/218 | 0 | 0/0 | 0 |
| v3-glm53-test-u0p0-r2（并发32，retries 6）| v3 | 240/240 | 0 / 1 | 0/1741 | 0 | 0/0 | 0 |
| qwen-v3（自托管，thinking off）| v3 | 240/240 | 1 Connection error / 3 | 0/1680 | 0 | 0/0 | `airline:37:trial2` max_steps（201 条，reward 0）|
| deepseek-nothink-test-u0p0 | v2 | 240/240 | 0 / 4 | 0 | 0 | 0/0 | 0 |
| deepseek-nothink-test-u0p1 | v2 | 240/240 | 1 Connection error / 0 | 0 | 0 | 0/0 | 0 |
| deepseek-test-u0p1、-u1p0（thinking）| v2 | 240/240 | 0 / 1、4 | 0 | 0 | 0/0 | 0 |
| deepseek-base-u0p0 | v2 base集 | 656/656 | 2 Connection error / 26 | 0 | 0 | 0/0 | 0 |
| **doubao-base-u0p0** | v2 base集 | 656/656 | 3 EmptyUserResponse / 26 | 1 | 0 | 0/2 | **54/656（8%）** |

- **数据版本**：`manifest.json` 的 `task_file_sha256` 对照 `$C/tau2-bench-…-fixed-v2|v3/data/tau2/domains/*/tasks.json`。四个 `deepseek-*-test-*` 为 v2（airline `ec04aec0…`、retail `03a79539…`），`v3-*`/qwen-v3 为 v3（`25086da1…`、`5b7a8fa3…`）。thinking vs no-think 配对只能在 v2 内。
- **actor 侧 context_length_exceeded**：错误文本 `areal_context_limit: prompt_tokens=32905 exceeds max_total_tokens=32768`，是 actor 的 32768 上限。每个 source_id 恰有 1 个 completed attempt（`out/audit1.txt`），无重复/缺失；但存在 selection：特别长的 actor 轨迹首轮被剔除后重跑。
- **一致性**：所有 completed run `sum(user_llm_calls)` = trajectory user 消息数；`user_llm_requests` 全 completed；response id 无重复（`out/usage8.txt`）。
- **model/fingerprint**：DeepSeek 所有 run `model=deepseek-flash`、`system_fingerprint=aeb56401ca74e127821c4f9126dcb669` 恒定；Doubao/GLM fingerprint 为 null，model 字符串与 launch.sh 一致，无后端切换证据。
- **reasoning 存放**：`messages[i].raw_data.choices[0].message.reasoning_content`（复制在 `provider_specific_fields.reasoning_content`，Doubao 另有 `encrypted_content`），token 数在 `raw_data.usage.completion_tokens_details.reasoning_tokens`。Qwen 和 DS no-think 无该字段。
- **终止判定**：`$C/tau2-bench-b7ea…/src/tau2/user/user_simulator.py:is_stop` 做子串判断 `STOP in content or TRANSFER in content or OUT_OF_SCOPE in content` [verified: 源码]。`launch.sh` 未设置 `TAU2_GATE_TERM`，授权同句 STOP 不会补执行。
- **reasoning 泄漏**：13 个 run 里 `<think>`/`reasoning_content` 字样为 0；唯一泄漏在 Turbo 首跑：`airline:31:trial0/1` 的 user 消息回显 system prompt（含 `<scenario>\nInstructions:\n\tDomain:\n</scenario>`），`airline:18:trial2` 引用 guideline 并称没有客户背景。

### 2.2 Turbo 首跑：服务异常形态

- 开局空响应突发：32 次 EmptyUserResponse 集中在 UTC 05:05–05:11，其中 25 次在 `policy_turns=0`；05:23 的 4 次 `account has an overdue balance` 使 `airline:13:trial0`、`airline:44:trial2` 耗尽重试，`summary.json: completed 238, infra_excluded 2`，`exit-code.txt=1`。
- 238 个 completed 中 86 个含异常 user 消息（`out/doubao2.txt doubao3.txt`）：236 条含 CJK 字符且与任务无关（"中药+多西环素…"在 `airline:13:trial1`、`airline:24:trial0`、`airline:26:trial2` 等逐字重复；"哈哈，好的好的！我们就这么说定啦…"在 32 个 episode 重复）；9 个 episode 开场白即 `###OUT-OF-SCOPE###`（`airline:19:trial0/1`、`retail:111:trial2/3`、`retail:5:trial2`、`retail:32:trial3`、`retail:71:trial2`、`retail:79:trial1`、`retail:108:trial2`），其 reasoning 称 scenario 只有 Instructions 和 Domain 标签；8 条 finish_reason=length（completion_tokens≈4097，退化循环，`airline:44:trial0`、`retail:100:trial0`、`retail:102:trial3`、`retail:40:trial1`、`retail:64:trial3` 等；4097 提示平台默认上限 4096 [hypothesis]）；11 条超 3000 字符；1 个 max_steps（`retail:111:trial0`）。异常占比在 14 分钟窗口内随时间上升，不集中于某 domain/trial/task（airline 31、retail 55；trial 0–3 为 24/21/21/20；只有 `airline:48` 四个 trial 全异常）。
- 服务端而非 harness：异常 episode 的首个 user 调用 prompt_tokens 与 r2 同 (domain,task,trial) 逐个相等（`out/doubao6.txt`：`537/537, 435/435, 491/491`，`ptok<r2-100` 占比 0）。同代码同 endpoint（`ark…/api/v3`）同配置的 r2 零异常。具体机制（prompt cache、网关路由、欠费降级）无法从文件判定 [hypothesis]。
- 旧 `doubao-base-u0p0` 同样污染（54/656，20 个开场即 token，48 条 CJK）。Plan endpoint（Lite、GLM r2）无这些症状。
- 能否产生 U1：不能。首跑干净子集（152 episode）U1 17/281 回合=6.0%，异常 86 episode 仅 2/97；base 集干净 100/1215=8.2%、异常 3/65；r2 46/472=9.7%；Lite 23/444=5.2%。首跑 U1 数 19 偏低只因 36% 被污染，**不应进入跨 simulator 对比**，case study 用 r2 是对的。

### 2.3 GLM 首跑

886 次 `RateLimitError: Requests are too frequent` 全在 UTC 08:42–08:43 一分钟内（并发 256、retries 2），240 个只完成 37，run.log 16.6 MB。r2 并发 32、retries 6 零此类错误。首跑 37 个 completed 的 U1 为 0/59，与 r2 一致但样本太小。

### 2.4 其他

DeepSeek v3/Lite/GLM r2/Qwen 的 user 侧 finish_reason 全 stop，最大 completion tokens：DS 3,863、Lite 5,130、GLM r2 7,219，无触顶；Qwen 设 `max_tokens=4096`，最大 408。除 Turbo 首跑和 GLM 首跑外无错误突发。

## 3. Q2 thinking vs non-thinking

### 3.1 DeepSeek 开关（v2 数据）

`args.txt` 参数顺序为 `NAME USER TASK_SET POLICY_T USER_T`，故 `deepseek-test-u0p1` 是 policy T=1/user T=0，`deepseek-test-u1p0` 是 policy T=0/user T=1.0（thinking 下 user T 不生效，属 API 文档行为，文件内无法验证 [hypothesis]）。

| 配对 | thinking | no-think | Fisher p |
| --- | --- | --- | --- |
| actor T=0：`deepseek-test-u1p0` vs `deepseek-nothink-test-u0p0` | 10/416（2.4%）| 1/451（0.2%）| 0.0045 |
| actor T=1：`deepseek-test-u0p1` vs `deepseek-nothink-test-u0p1` | 12/433（2.8%）| 0/452 | 0.00017 |
| 合并 | 22/849（2.6%）| 1/903（0.1%）| 1.5e-6 |

v3 thinking `v3-deepseek-test-u0p0` 13/401（3.2%）与 v2 同量级，版本影响小。no-think 唯一命中 `deepseek-nothink-test-u0p0 retail:108:trial0`。

| run | reward | user消息/ep | 字符数 | reasoning tok/条 | early STOP | TRANSFER结尾 |
| --- | --- | --- | --- | --- | --- | --- |
| think u1p0 | 0.721 | 5.84 | 212 | 285 | 14 | 59 |
| no-think u0p0 | 0.742 | 6.45 | 184 | 0 | 4 | 44 |
| think u0p1 | 0.713 | 5.91 | 219 | 284 | 13 | 57 |
| no-think u0p1 | 0.775 | 6.33 | 189 | 0 | 2 | 37 |

early STOP = STOP 时已执行写操作数 < gold 写操作数。thinking 对话更短、消息更长、更早 STOP、更常转接；reward 低 2–6pp，未做显著性检验。

### 3.2 跨 simulator（v3 test，`out/stats3.txt opp.txt`）

| simulator | reasoning 情况 | 确认回合 | U1 | 率 | reward | user消息/ep | early STOP | TRANSFER结尾 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-Flash | 99%调用有 | 401 | 13 | 3.2% | 0.721 | 5.77 | 12 | 54 |
| Turbo r2 | 74%有，26% rtok=1 | 472 | 46 | 9.7% | 0.633 | 5.73 | 44 | 48 |
| Lite | 99%有 | 444 | 23 | 5.2% | 0.700 | 5.37 | 29 | 45 |
| GLM 5.3 r2 | 100% | 521 | 0 | 0% | 0.758 | 7.25 | 5 | 40 |
| Qwen3.5-4B | 0%（off）| 507 | 0 | 0% | 0.583 | 7.00 | 6 | 42 |

### 3.3 Turbo 的 reasoning 是自适应的，U1 集中在无 reasoning 调用

r2 的 reasoning_tokens 双峰（=1 的 364 条，≥50 的 1,010 条，2–49 仅 1 条；`out/think_dist.txt`），无 reasoning 的比例随 user 消息序号上升：序号1 8%，序号2 18%，序号3 24%，序号4 47%，序号5 64%（0 起），而确认回合多在后半段。

| | 有 reasoning（rtok>1）| 无（rtok≤1）|
| --- | --- | --- |
| Turbo r2 | 4/284=1.4% | **42/188=22.3%**（p=2.4e-14）|
| Turbo 首跑 | 0/214 | 19/164=11.6% |
| doubao-base | 13/818=1.6% | 90/462=19.5% |

Turbo 46 个 U1 中 42 个无 reasoning 记录（例如 `r2 retail:61:trial0/2/3` 字面相同的 `Yes, I confirm I want to proceed with this modification. ###STOP###`，`retail:9`、`retail:5` 四个 trial 同类）。这是观察性分解：模型在简单确认时跳过思考，所以不能说"关掉 thinking 会增加 U1"，但否定了"Turbo 的 U1 是 thinking 产物"。Lite（99% 有 reasoning）U1 5.2%。总体：DeepSeek on→多，Turbo off→多，GLM on→零，Qwen off→零，无统一方向。

## 4. Q3 U1 为什么发生

### 4.1 reasoning 里的理由

DeepSeek reasoning 是原始 CoT；Doubao `reasoning_content` 是摘要体（几乎都以 "Got it,"/"Okay," 开头）；Turbo 46 条里仅 4 条有 reasoning。关键词归类（`reas6.py`，多选，正则粗糙，DeepSeek 13 条逐条读过）：

| 类别 | DS v3 (13) | DS v2 u1p0 (10) | DS v2 u0p1 (12) | Lite (23) | Turbo (有reasoning 4) |
| --- | --- | --- | --- | --- | --- |
| A 认为"我确认后任务就完成" | 11 | 7 | 8 | 12 | 3 |
| B 决定把 STOP 附在这条回复 | 10 | 10 | 11 | 14 | 4 |
| E 复述 guideline "if the instruction goal is satisfied" | 7 | 8 | 4 | 0 | 0 |
| D 想到 agent 可能还要处理仍 STOP | 5 | 6 | 6 | 1 | 0 |
| C 提到 persona | 4 | 1 | 3 | 9 | 3 |
| 无 reasoning 记录 | 0 | 0 | 0 | 0 | 42 |

DeepSeek 原文（`v3-deepseek-test-u0p0`）：
- `retail:108:trial0`："Then likely stop… But maybe too abrupt? The agent may need process. The directive says if instruction goal satisfied, generate STOP."
- `retail:60:trial2`："The task is complete once confirmed. But the agent may need to perform action. In these simulations, if goal satisfied, generate stop."
- `retail:51:trial1`："The instruction says if instruction goal satisfied, generate ###STOP###. Confirm and maybe STOP."
- `retail:77:trial2`："After confirm, maybe agent will process and we can end with ###STOP###? … Since instruction says if goal satisfied, generate ###STOP### token."
- `retail:102:trial3`："Once confirm, likely done. We should generate ###STOP### after confirmation if task satisfied."

同模型其他确认回合不 STOP：`airline:16:trial2`"No stop yet, because task not complete."，`airline:16:trial3`"we should wait for the agent to process and confirm completion."（`out/glm9.txt`）。即 DeepSeek 有两种读法，U1 是其中一种占上风。Lite 摘要体同类：`retail:18:trial3`"the task will be completed once the user sends back the confirmation reply"。

GLM/Qwen 为零（`out/glm10.txt`）：有授权词的确认回合里 GLM reasoning 只在 11/353（3%）提到 STOP，从不发出；DeepSeek 在 191/343（56%）提到，13 次发出（6.8%）。GLM 没有把"现在结束"当候选；我未逐条验证全部 521 条 [hypothesis]。Qwen 无 reasoning。

### 4.2 与场景、persona、agent 措辞、任务的关系（`out/corr4.txt stat11.txt`）

- 任务集中：命中任务数 Turbo 24、Lite 15、DS 9，并集 31（29 retail，仅 `airline:8`、`airline:29`）。top：`retail:5`（Turbo 4、Lite 4）、`retail:9`（Turbo 4、Lite 4、DS 3）、`retail:108`（DS 3、Turbo 2）、`retail:100/101/111/27/61/64`（Turbo 各 3）。DS 的 9 个任务中 6 个也在 Turbo、6 个在 Lite 的命中集。任务级 Spearman：DS–Turbo 0.32，DS–Lite 0.41，Turbo–Lite 0.26。
- persona 词（`impatient|hurry|rush|busy|quickly|as soon as|urgent|asap|get things done…`，扫描 task_instructions+reason_for_call+known_info+unknown_info，persona 字段为 null）：DS 3/95 vs 10/306（p=1.0），Turbo 15/112 vs 31/360（p=0.15），Lite 7/103 vs 16/341（p=0.45）。不支持 persona 为主因。正则粗糙，召回未验证。
- 场景含 "confirm"（如 `retail:5/9`"When the agent asks for confirmation, suddenly change your mind…"，`retail:18`）：DS 4/54 vs 9/347（p=0.083），Turbo 9/78 vs 37/394（p=0.53），Lite 11/62（17.7%）vs 12/382（3.1%）（p=5.9e-5）。
- agent 措辞（Turbo+Lite+DS 合并，1,317 回合）：明确要求回复 yes 54/529=10.2%；一般确认请求 28/510=5.5%（两者 p=0.0055）；提问式 "Shall I…/Would you like me to…" 0/41（对其余 p=0.067）；其他 0/237。分 simulator 方向一致（Turbo 17.0/7.7/0，DS 5.1/2.9/0，Lite 8.1/5.1/0）。仅相关性，与题目难度、对话阶段混杂。
- 已有写操作后的确认：Turbo 17.1% vs 之前 8.2%，DS 3.2% vs 3.2%，Lite 4.6% vs 5.3%，仅 Turbo 有差，不下结论。

### 4.3 STOP 与授权同句

77 条标注 U1 都是最后一条 user 消息同时含授权词和 `###STOP###`，其后无 assistant 回合。

### 4.4 若 agent 被允许执行（`pending5.py`，`out/pending5.txt`）

对规则命中且 reward=0 的 episode，取 gold 中 `tool_type=write` 且 `action_match=False` 的动作，检查其全部参数叶子值是否都出现在 STOP 前最近 5 条 assistant 消息文本。文本级上界，不保证 tool call 与文本一致，也不保证 NL assertions/DB 终态满足 [hypothesis]。

| simulator | U1 且 reward=0 | 全覆盖 | 部分覆盖 | 未覆盖 | reward=1 无害命中 |
| --- | --- | --- | --- | --- | --- |
| DeepSeek v3 | 13 | 13 | 0 | 0 | 0 |
| Turbo r2 | 43 | 40 | 2（`retail:100:t1`、`retail:101:t0`）| 1（`retail:33:t1`）| 3 |
| Lite | 23 | 20 | 2（`retail:102:t0`、`retail:71:t2`）| 1（`retail:39:t3`）| 0 |

未/部分覆盖多为地址字段、`user_id` 不在确认文本（`retail:33:t1`、`retail:39:t3` 只缺 `modify_user_address` 的 `user_id`）或多订单任务漏子目标。保守结论：73/79（约 93%）的待执行动作文本上与 gold 吻合。

## 5. 归因判定

**(e) infra：不是 U1 原因** [verified]。依据见 §2。污染的 Turbo 首跑与 doubao-base 反而 U1 更少；原因（服务端退化/串话）[hypothesis]，但意味着这两个 run 不能做 Doubao 基线。

**(d) thinking：不是通用解释** [verified 数据；机制 hypothesis]。DeepSeek thinking 把 U1 从约 0.1% 抬到约 2.6%；Turbo 反向（无 reasoning 22.3% vs 1.4%）；GLM thinking 为零；Qwen 非 thinking 为零。`reasoning_effort=medium` 如何映射到 ARK thinking 开关，文件内无文档，只观察到 reasoning_tokens 双峰 [hypothesis]。

**(c) guideline 歧义：机制支持，因果未验证**。文本版 `simulation_guidelines.md` 第 13 行"The goal is to continue the conversation until the task is complete"和第 14 行"If the instruction goal is satisified, generate the '###STOP###' token"；语音版 `simulation_guidelines_voice.md` 多一句"Agreeing to an action is not the same as the action being completed"。user 模拟器读文本版（user tools 为 None 时读 `simulation_guidelines.md`）。DeepSeek reasoning 明文引用第 14 行并把"goal satisfied"解释成"用户已确认"，5/13 写到 agent 可能还要处理仍 STOP。同一 guideline 下 GLM/Qwen/DS no-think 几乎为零，所以歧义是必要暴露面，不是充分条件。没有任何 run 用过修补后的 prompt；需要同一 prefix、仅差该句、在 DS think/Turbo/Lite 上多次采样的对照实验 [hypothesis]。

**(b) persona/场景：次要** [verified 相关性]。催促词无显著效应；"confirm"场景指令 Lite 显著、DS 边缘、Turbo 不显著；31 个任务命中，top5 占比 DS 69%、Lite 57%、Turbo 37%。

**(a) 真实用户行为：不能认定为合理** [verified 对比；hypothesis 真实性]。同场景下其他 simulator 会等 agent 执行；真实用户是否在授权句离开无法判定。即使有，tau2 终止语义使 agent 无法执行，对 benchmark 是缺陷，对 RL 是错误信号。

## 6. 限制

- v2/v3 混用：thinking/no-think 配对只在 v2 内，两个 thinking v2 run 的 user T / policy T 设置不同；v3 无 DS no-think。
- 每 run 仅一次采样；Fisher 把回合当独立，实际 p 偏小，效应量远大于偏差，方向可信。
- 规则 R 为 in-sample 校准；persona/催促/"confirm" 正则粗糙；reasoning 分类为关键词计数，仅 DeepSeek 35 条逐条读过。
- Doubao reasoning 是摘要；Turbo 的 U1 多数没有 reasoning，其"为什么 STOP"读不出。
- 未读取 `tau2-user-failures/*.json` 内容（只用 error attempt 的 `error` 文本）。
- guideline 的三个 `.md` 均为指向共享 base cache 的 symlink，全程只读。

## 7. 脚本索引（均在 `W`）

`extract.py`（抽取到 `data/`）、`lib.py`（加载器+U1 规则）、`cfg.py audit1.py audit_errors.py audit2.py usage8.py leak.py leak2.py`（Q1）、`doubao1-6.py`（Turbo 首跑）、`rule_eval.py`（规则验证）、`stats3.py opp.py think_dist.py stat11.py`（Q2 统计）、`reas1.py reas6.py snip7.py glm9.py glm10.py`（reasoning 阅读/归类）、`corr4.py fin12.py pending5.py`（任务/persona/措辞相关、可救回性）。输出均在 `out/*.txt`。