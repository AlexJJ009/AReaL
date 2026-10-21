# 训练与评测基础设施问题

> 来源：原 README 第 13.2（挂起）、15.6、16.14.8 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

## 1 proxy 的 session 容量泄漏：两条 episode 挂起 1 小时（已复现，已修复：`cefb0772`）

- **现象**：controller 的 task 611、643（`retail:59:trial3`、`retail:99:trial3`）都分到 engine rank
  3。submit 之后，它们一直没有打出 `STARTING SIMULATION`，也没有写尝试记录，直到 `request_timeout`（3600
  s）后才被拒绝。评测因此多花了约 45 分钟（总耗时 77 分钟，v1 是 31 分钟）。这两条按 infra 排除，不影响其他 episode 的分数。之前 5
  次运行都没有出现过。

- **机制**（读代码确认，`areal/experimental/openai/proxy/`）：

  1. 每个 episode 先 `grant_capacity`（容量 +1），再 `start_session`（容量 −1，新建 session
     `<task>-<idx>`）。
  1. 客户端 `_start_session` 每次请求的超时是 10 s；遇到 HTTP 429、超时等错误时无限重试。
  1. 修复前，服务端的 `start_session` 不是幂等的。如果它在客户端超时之后才执行完，第一次请求照样建出 `<task>-0`、消耗 1 个容量，但这个
     session 没人使用；客户端的重试又建出 `<task>-1`，再消耗 1 个容量。
  1. 每发生一次，这个 proxy 就永久少 1 个容量。`_cleanup_stale_sessions` 在 3600 s 后会删掉孤儿
     session，但不会归还容量。这个 proxy 上最后到达的 episode 因此一直拿到 429，只能等 controller 的
     `request_timeout`。

- **与本次运行的日志吻合**：rank 3 的 task 267（`retail:1`）和 275（`retail:93`）都正常完成，但 proxy 在 11:11:23
  把 `267-0`、`275-0` 当作 stale session 删除。正常完成却留下 `-0`，说明它们实际用的是 `-1`，也就是各发生了一次泄漏。泄漏 2
  次，挂起也正好 2 条。11:11:23 往前推 3600 s 是 10:11 左右，那时 rank 3 正有 34–35 个 episode 并发。

- **修复**：

  1. **`start_session` 幂等**（根治）：客户端每次 start 生成一个 `request_id`，所有重试都带同一个 id。服务端见到已处理过的
     id，就返回当时建的 session，不再消耗容量；如果那个 session 已经结束，返回 409。记录的 id 随 stale 清理一起过期。
  1. **重试打日志**：start 的每次重试都记 WARNING，拿不到容量时不再无声地等到超时。
  1. **`export_trajectories` 移到线程里执行**：构造和序列化 token 张量是 CPU 计算，原来直接在 async handler
     里跑，会阻塞事件循环，让同时到达的 start 超过 10 s 的客户端超时。这是诱因的候选之一，移走后它不再能阻塞别的请求。

  - **没有采用**"删除 stale session 时归还容量"：episode 在 start 之后崩溃时，这样会多发容量。幂等修好之后，grant 与 start
    已经一一对应，不需要这个兜底。

- **验证**：

  - 复现脚本 `scripts/official-runner/repro_proxy_capacity_leak.py` 用真实的 proxy server
    和客户端代码，在 episode A 的 `start_session` 执行期间持有 session 锁 12 s。
    - 修复前的提交 `49dadd88`：A 用的是 `A-1`，`A-0` 成了孤儿，后到的 episode B 等 40 s 仍收到 429，输出
      REPRODUCED；
    - 修复后：客户端超时重试一次，拿回 `A-0`，B 立即启动，容量回到 0，输出 NOT REPRODUCED。
  - `tests/experimental/openai/test_proxy_rollout_server.py` 新增 6 个测试。分别去掉服务端去重、客户端
    `request_id`、`to_thread`，都有对应测试失败。
  - proxy server 与 gateway 测试 50 个、tau2 测试 176 个全部通过。`test_proxy_integration.py` 需要 GPU
    和 SGLang，没有运行。

- **没有确认的**：实际运行中是什么让 `start_session` 超过 10 s 才执行完，没有访问日志能证明。候选是同步 handler 在线程池里排队或等
  `_lock`，以及上面的 export 阻塞事件循环。幂等修复不依赖诱因是哪一个。

- **与 `2d7d342a` 无关**：涉及的文件在那次提交里没有改动，这条路径也不经过组内成员丢失的处理。

## 2 Qwen3 的加速配置核查

Qwen3.5-4B 是 24 层 Gated DeltaNet 加 8 层 full attention；Qwen3-4B 是 36 层全部 full attention。

**推理（SGLang）：**

| 配置                                         | Qwen3 上                                 | 说明                                         |
| -------------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| `attention_backend: flashinfer` + CUDA graph | 可用                                     | 本次评测就在用；FA3 只支持 Hopper            |
| FLA / causal-conv1d                          | 不使用                                   | 只服务于线性注意力层，装着也不影响           |
| KV cache 容量                                | 每卡 37.4 万 token，Qwen3.5 是 155 万    | `mem_fraction_static=0.75` 时实测，小 4.2 倍 |
| `disable_radix_cache: true`                  | 训练 rollout 保持关闭；eval 卡可单独打开 | 见下文 radix 一节                            |

- 评测时每个 SGLang 实例只有 4–10 条请求在跑，`num_retracted_reqs = 0`，瓶颈在 DeepSeek 用户的响应延迟。
- 训练 rollout 时每个实例 64 条并发、上下文也更长，要关注 `num_retracted_reqs`，必要时调高 `mem_fraction_static`。

**Actor update（4 卡 FSDP，用 `actor_probe.py` 跑 3 步）：** 合成数据每卡 8 条序列、长 6K–16K token，每 4
条共享一段 3K token 的前缀。

| 变体                                           | 能否启动                 | token/s | 峰值显存 | 说明                                                                            |
| ---------------------------------------------- | ------------------------ | ------- | -------- | ------------------------------------------------------------------------------- |
| **现有配置**：FSDP + FA2 + grad ckpt + packing | 是                       | 13,435  | 39.7 GB  | "每个 microbatch 只放一条序列"的规则只对 `qwen3_5` 生效，Qwen3 自动恢复 packing |
| Qwen3.5 现有配置（对照）                       | 是                       | 16,106  | 36.3 GB  |                                                                                 |
| Ulysses SP=2                                   | 是                       | 12,258  | 28.2 GB  | 32K 以内用不上                                                                  |
| Tree training                                  | 需 `pad_to_maximum=True` | 7,249   | 54.5 GB  | padding 的开销大于共享前缀省下的计算                                            |
| Archon，默认 selective AC                      | 否，OOM                  | —       | —        |                                                                                 |
| Archon，`ac_mode=full`                         | 是                       | 12,980  | 37.6 GB  |                                                                                 |
| Archon，`ac_mode=full` + compile               | 是                       | 13,368  | 39.5 GB  |                                                                                 |

- **结论：沿用现有 FSDP 配置，不再加其他优化。**
- 在这个长度区间，Qwen3 每 token 比 Qwen3.5 慢约 17%，原因是 36 层 full attention 的计算量。
- **没有测的：**
  - FP8：A100 不支持。
  - Qwen3 的 critic 路径：`Qwen35TokenCriticForCausalBackbone` 只适用于 Qwen3.5，Qwen3 会走通用路径。用
    SAO 之前要单独验证。

## 3 SAO 推权重崩溃：原因与修复（2026-10-14）

**结论**：SAO 训练时把新权重传给推理服务会崩溃，原因是推理服务意外继承了训练端的显存管理工具。修复后 SAO smoke（526）2
步都正常推送了权重，可以开始正式训练。

### 现象和影响

SAO 每训练一步 actor，就要把新的权重从训练用的 GPU（0–3）传给负责生成 rollout 的推理服务 SGLang（GPU 4–7），否则下一步采样用的还是旧
policy。传的方式叫 xccl：训练端和推理端组成一个通信组，用 NCCL 在 GPU 之间直接广播。

在 Qwen3.5-4B 上，每次传权重都会让 SGLang 崩溃，报 `illegal memory access`（访问了非法的显存地址）。9 月 27 日的 SAO
运行和 10 月 14 日 critic 的 resume（506）都是这样。不修，SAO 一步都跑不完。

### 原因

训练端为了省显存，用了一个叫 torch_memory_saver（TMS）的工具：它在进程启动时被"预加载"进去，接管这个进程里所有的显存申请，方便训练间隙把显存腾出来。

AReaL 启动 SGLang 时，把训练端的环境变量原样传了过去，其中就有"预加载 TMS"这一项。结果是 SGLang 的配置里写着不用这个工具，实际上它的显存申请全被
TMS 接管了。NCCL 往这种被接管的显存里写数据时出错。

### 怎么确认的

Codex 在 8 张卡上照生产环境搭了一个只推权重、不跑对话的复现，每次只改一个条件：

| 布局                                     | SGLang 是否继承了 TMS | 结果                                                      |
| ---------------------------------------- | --------------------- | --------------------------------------------------------- |
| 1 个训练进程 + 1 个 SGLang               | 否                    | 正常                                                      |
| 4 个训练进程 + 4 个 SGLang（和生产一样） | 否                    | 正常                                                      |
| 4 + 4                                    | 是                    | 崩溃，报错和生产环境完全一样                              |
| 4 + 4，加上修复                          | 是，但启动前被清掉    | 正常；4 台 SGLang 上 627 个参数逐个核对，和训练端完全一致 |

只改"是否继承 TMS"这一个条件，就能让崩溃出现和消失，所以它就是原因。

### 修复

启动 SGLang 之前，把环境里 TMS 相关的变量去掉，其他不动（`d8f72c51`）。Codex 写的初版把这些变量名在 SGLang 的代码里又写了一遍；review
时挪到了注入它们的同一个文件（`areal/utils/offload.py`），以后改名只需改一处。新增的测试在旧代码上失败、修复后通过。

### 实际训练里的验证

SAO smoke 526 用修复后的代码跑了 2 步，两次推权重都成功，各用 1.5–2.3 秒；第 2 步评测 pass^1：总体 0.725，Airline
0.525，Retail 0.825。

### 顺带的两件事

- **`A_log` 精度**（`6bf1d96e`）：Qwen3.5 有一组参数 `A_log`，SGLang 用高精度（fp32）存，而 AReaL
  传之前统一转成了低精度（bf16），训练端对它的微小更新传过去就被舍掉了。复现实验里推完权重核对，正好是这 24 个参数对不上。现在这组参数按 fp32 传。
- **500 的 OOM**：当时 SGLang 报错的位置也在 TMS 里，很可能是同一个原因，但没有复现，未证实。

## 3 Radix（前缀）缓存：只给 dedicated eval 打开（2026-10-20，pueue 656）

- 到 649 为止，所有 PPO run 的训练 rollout 和 eval 都关着 radix（AReaL 默认
  `disable_radix_cache: true`）。多轮对话每轮都要重算整段历史（约 9k token）的 prefill；rollout 在并发 24
  左右就饱和，可能与此有关，尚未测量。
- 换权重时缓存会被清掉：SGLang 的 `update_weights_from_disk/distributed/tensor` 请求里 `flush_cache`
  默认为 `True`，AReaL 不覆盖这个默认值。
- 新开关 `eval_enable_radix_cache=true`（`examples/tau2/utils.py`，默认关闭）只对 dedicated eval
  server 打开 radix，训练 rollout 保持关闭（commit 2e8a2228，Codex 实现、Claude review；测试
  `tests/test_tau2_eval_radix.py` 先红后绿）。独立的 checkpoint 评测直接传
  `sglang.disable_radix_cache=false`。
- Qwen 用户 server（`sglang.launch_server` 直接启动）本来就开着 radix；它的权重不更新，没有风险。
- 收益未测：回放
  benchmark（`_artifacts/AReaL/codex-sao-math/tau2-eval-radix-20261020/run-bench.sh`，60 个
  episode、738 轮，不调用 API） 等有空闲 GPU 时跑。2×2 里四个 run 的 radix 设置要一致，run 3 不打开。
- 事故：656 检查独立评测的 override 时调用了 `load_expr_config`，它把解析后的配置写回了 649 run 目录的
  `config.yaml`（把 radix 改成 false， 多出一个新字段）。Claude 已恢复原值，被改写的版本留在 656 任务目录
  `overwritten-649-config.yaml`。以后的 brief 要求先把配置复制到任务目录再加载。

## 4 Qwen 用户服务器与 eval 共卡时崩溃；训练改为离线评测（2026-10-21，pueue 670–672）

- **现象**：run 3 第一次启动（670）17 分钟后，GPU 7 上的 Qwen3.5-4B 用户 SGLang 服务报
  `CUDA error: an illegal memory access`（`process_batch_result_decode`），进程退出，之后所有
  episode 连不上用户而被 rejected。670 已停，没有完成任何一步。
- **可能原因（未确证）**：用户服务器与 dedicated eval 共用 GPU 7，只有 `--mem-fraction-static 0.15`，KV 池
  54,736 token，崩溃前占用 0.86。10-18 的自部署用户重跑在独占卡上用 0.75，跑完 3,360 条没有出错。
- **处理（用户决定）**：用户服务器独占 GPU 7（`USER_MEM` 默认 0.75，KV 池 868,601 token），并在
  `tau2-ppo-qwenuser-20261021/launch_ppo_qwenuser.sh` 里加了崩溃后 5 秒自动重启；rollout 保持 GPU
  4–6。训练期不再做在线评测： Codex 671 加了默认关闭的 `offline_eval_only`（`a7dd717d`），打开后不创建 dedicated
  eval，照常每 10 步和最后一步保存 actor 与 critic 的 HF checkpoint；离线评测入口
  `scripts/tau2/evaluate.py --task-set ppo`（`e77c6350`）用同一 100 题、4 次、T=0。 测试先红（13
  failed）后绿（130 passed）。Claude 复核了 A 的代码改动。
- **run 3（672，`run3-std-beta1-v2`）**：代码快照 `src-a7dd717d`；训练题 Airline 417（含 668 救回的 7
  题，pack `tau2-areal-airline-recovered-pack-20261021-v1`）、Retail 337、Telecom
  329（`tau2-official-telecom-pack-20261021-v2`）， 每 epoch 9 步，14 epoch 共 126 步；参数在
  `run3-formal-overrides-v3.txt`。run 2 用同一快照和参数，加 `+critic_user_scenario=true`。 上面第 3 节的
  eval radix 开关在离线评测模式下不起作用。
