# AReaL standard GRPO 调研：同步、异步与 Qwen3.5-4B 配方

## 1. 结论与证据边界

目标是判断哪些代码可以直接复用、哪些参数不能照搬，以及 PPO 失败为什么既不能证明 GRPO 必败，也不能证明换成 GRPO 就会好。

- AReaL 已有异步 GRPO 和同步运行模式，共用 `PPOTrainer`；不需要重新实现 trainer。
- 官方 GSM8K GRPO 是工程化 GRPO 配方，不是逐项复刻 DeepSeekMath 原论文：默认异步、decoupled loss、额外 batch
  advantage normalization、token 平均 loss，默认没有 reference KL。
- 同步模式可以用现有代码；严格原论文 objective 不能只靠把 staleness 改成 0 得到。
- 本轮检索没有找到 **AReaL + Qwen3.5-4B（尤其 Base）+ 纯文本数学 + 全参训练 + 完整配置和收敛证据**
  同时满足的公开配方。这是有边界的检索结果，不是宣称全网不存在。
- 已找到 Qwen2.5-1.5B 官方全程成功配方，以及 Qwen3.5 其他尺寸的官方配置/社区曲线。它们能分别提供算法控制组和模型架构适配参考，不能拼起来冒充 4B
  的已验证结果。

调研日期：2026-09-22。只读调研，没有启动训练或改训练代码。

| 来源                  | 固定边界                                                                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 当前工作区            | `/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math`，HEAD `d6521ab99242a58eceeb3e1351a7065720e2a89b`；存在既有未提交内容，未改动 |
| 当前分支基础          | `v2.1.0` 对应 `ecc8b0e4dfb4e3965f67121a5c87866a07c390ae`，之后有本地 PPO 修改                                                                        |
| 本次 GitHub main 快照 | `c74b343c53c785eb620943fcabbd6b8263e28051`；读取官方 math YAML、README、actor、loss 和同步说明                                                       |
| 外部证据              | GitHub issue/PR 正文、评论、合并状态；社区数字为作者报告，未在本机重跑                                                                               |
| 检索边界              | AReaL 源码、Qwen3.5/4B/GRPO-sync issues/PR、相关框架公开配方；GitHub 公共 REST 后续达到限流，不能声称穷尽全部 forks                                  |

## 2. 先把“standard”分成两个问题

**算法维度**决定 advantage 和 loss 怎么算；**调度维度**决定生成与训练是否重叠。没有 critic 不等于同步，使用 `PPOTrainer`
也不等于用了 learned-value PPO。

原始 outcome GRPO 的核心是同一道题采样 G 个回答：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

每个回答的有效生成 token 使用同一 outcome advantage。原论文 policy objective 对回答内 token 先平均，再对回答平均，并有
reference KL 项。现代 RLVR 经常关掉 KL 或改变 reduction；这些仍常被叫 GRPO，但参数对齐时必须说清楚。

```mermaid
flowchart LR
    P[同一道题] --> G[生成 G 个回答]
    G --> R[逐回答验证奖励]
    R --> A[组内相对优势]
    A --> U[Actor clipped loss 更新]
    U --> W[发布新权重和版本]
    W --> G
```

| 模式            | 生成/训练关系                                  | 需要关注                                                     |
| --------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| 同步 GRPO       | 当前版本生成一批，训练并发布后才开始下一批     | 仍有训练/推理数值差异；同步不等于 logprob 一致               |
| AReaL 异步 GRPO | 推理 GPU 预取后续数据，同时训练 GPU 更新当前批 | 版本滞后、partial rollout、behavior/proximal 修正            |
| PPO + critic    | 使用学习到的 value 与 GAE                      | 多出 value head、critic target、bootstrap 和 critic 优化风险 |

## 3. AReaL 实际如何实现异步 GRPO

### 3.1 现成入口和真实调用链

- `examples/math/gsm8k_rl.py`：加载配置、数据和 MathAgent，交给 `PPOTrainer`。
- `examples/math/gsm8k_grpo.yaml`：选择无 critic 的 group reward normalization。
- `areal/api/cli_args.py`：`GRPOConfig` 是 `PPOConfig` 的兼容别名类，不是另一套训练器。
- `areal/trainer/rl_trainer.py`：取 batch、重算 logprob、算 advantage、actor update、发布权重并推进版本。
- `areal/trainer/ppo/actor.py`：reward/advantage 处理和 loss 组装。
- `areal/utils/functional/functional.py`：`ppo_actor_loss_fn` 实现 ratio、clip、rejection 和
  token reduction。
- `areal/infra/workflow_executor.py` 与 `staleness_manager.py`：后台 producer/consumer
  及版本容量控制。

当前工作区定位：`rl_trainer.py:699` 的 prepare batch、`:784` 的 proximal logprob/advantage
路径、`:826` 后 actor update 与版本发布；`staleness_manager.py:92` 的容量计算。上游文件可能增行，应按函数查找。

### 3.2 三个 policy 的用途

- behavior：实际生成 token 时使用的 policy；保存真实 rollout logprob。
- proximal：当前训练 batch 开始时，用 actor 重算得到、在本轮更新期间冻结的参照 policy。
- current：正在被优化的 actor。

忽略 rejection 等附加细节，decoupled surrogate 是：

```text
w = pi_proximal / pi_behavior
r = pi_current / pi_proximal
J = E_behavior[w * min(r * A, clip(r, 1-epsilon, 1+epsilon) * A)]
```

外面的比例处理陈旧数据，里面的比例限制 actor 相对 proximal 的更新幅度。`recompute_logprob=true` 不是把 behavior
记录丢掉，而是为这个修正增加 proximal logprob。partial rollout 可以跨权重版本，不能简单假定每条长轨迹来自单一 frozen policy。

官方默认 math YAML 的组合是：

```yaml
rollout:
  max_head_offpolicyness: 2
actor:
  recompute_logprob: true
  use_decoupled_loss: true
  rejection_sampling:
    metric: ratio
    upper: 5.0
```

**字段拼写注意**：`docs/en/algorithms/async.md` 的示例写过 `recompute_logprobs`（复数），源码 dataclass 和实际
YAML 使用 **`recompute_logprob`**（单数）。照搬配置应以 consumer 为准。

## 4. 同步 GRPO 是否已有代码？有

官方
[训练性能排查文档](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/docs/en/best_practices/algo_perf.md)
给出了现成组合：

```yaml
rollout:
  max_head_offpolicyness: 0
actor:
  recompute_logprob: false
  use_decoupled_loss: false
```

`max_head_offpolicyness=0` 控制调度；另两个开关把 loss 改回普通 clipped 路径。这三个参数的职责不同。只改第一个也能同步调度，但仍保留
decoupled loss。

staleness manager 的可继续生成容量是：

```text
(max_staleness + current_version + 1) * consumer_batch_size
- (accepted + running)
```

零 staleness 下，一个版本只放行一个 consumer batch 的预算；下一批要等新版本发布。底层仍使用异步线程和队列，“同步”指训练与采样的版本关系，不是消灭
Python async。

从官方 GSM8K 配方切到同步排查模式的命令（**本次未执行**）：

```bash
python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  rollout.max_head_offpolicyness=0 \
  actor.recompute_logprob=false \
  actor.use_decoupled_loss=false \
  actor.rejection_sampling=null
```

这仍是原 YAML 的 Qwen2.5-1.5B 配方，不是 Qwen3.5-4B 启动命令，也不是完整原论文 GRPO 复刻。

## 5. 为什么官方默认不等于严格 standard GRPO

| 项目              | 原论文/最小组相对目标         | 当前 AReaL GSM8K 配方                                     |
| ----------------- | ----------------------------- | --------------------------------------------------------- |
| critic            | 无                            | 无                                                        |
| outcome advantage | 组内 reward 均值/标准差       | `reward_norm` 组内，然后再 `adv_norm` batch               |
| loss aggregation  | 每回答 token 平均，再回答平均 | 有效 token 总体平均                                       |
| reference KL      | 直接加进 objective            | 默认 `kl_ctl=0`；打开后主要走 token reward/advantage 路径 |
| ratio             | 普通 token ratio              | 默认 decoupled token ratio + correction                   |
| rollout policy    | 固定 old policy 的 batch      | 默认允许陈旧和 partial rollout                            |

组内 reward 标准化会抵消正线性 reward scaling 和平移（除 epsilon、裁剪等影响）；不要把 `reward_scaling=10`
简化理解成“GRPO 梯度必然放大十倍”。额外的 batch advantage normalization 和 loss reduction 才是需要单独核对的差异。

原论文的 sequence mean 与 token mean 在长短回答不同时不等价。当前本地和本次读取的上游 loss 都是 masked token sum 除
token count。AReaL [PR #1546](https://github.com/areal-project/AReaL/pull/1546) 提供
`loss_aggregation=seq_mean` 等选项，但**本次核实仍未合并**，且 PR 明说未跑 native GPU/multi-node
training。不能把它作为当前版本的可用参数或 4B 收敛配方。

若先要“组相对、同步、无额外 whitening”的诊断基线，可在同步配置上明确 `adv_norm: null`，保留 `reward_norm`
group，`discount=1`、`gae_lambda=1`，禁用额外过滤；这仍然保留 token-mean reduction，且默认无 KL。必须准确命名，不标成完整
DeepSeekMath reproduction。

## 6. 找到了哪些真正能抄的社区证据

### 6.1 最扎实的 AReaL 数学控制组：Qwen2.5-1.5B-Instruct

[官方 math README](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/examples/math/README.md)、[YAML](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/examples/math/gsm8k_grpo.yaml)、[已合并 PR #1634](https://github.com/areal-project/AReaL/pull/1634)。

这是最接近“现成、配置明确、有完整 schedule 证据”的 AReaL math GRPO 作业：8×A800、GSM8K、FSDP、group=4、10
epoch/290 step、seed=1。官方默认 rollout 使用 SGLang。

| 学习率      | 最佳 eval reward | 最后 eval reward | 社区报告                |
| ----------- | ---------------: | ---------------: | ----------------------- |
| 旧 `1.7e-5` |          0.53563 |          0.01782 | 第 3 epoch 起崩溃       |
| 新 `6e-6`   |          0.78412 |          0.77767 | 完成 10 epoch，保持稳定 |

PR #1369 引入 FP32 master 参数和 optimizer state 后，BF16 以前被量化吞掉的小更新重新生效，旧 LR 不再稳定；PR #1634
重新调低 LR。旧 README 的 H800/BF16 `0.79570` 不能与新 FP32 表混用。上述是作者的单 seed 配方验证，不是跨模型 universal
LR rule。本地 YAML 已经是 `6e-6`；不能因此断言本地 PPO 失败已被这个 PR 解释。

### 6.2 AReaL 的 Qwen3.5 成功证据：其他尺寸、其他数据

[已合并 PR #1384](https://github.com/areal-project/AReaL/pull/1384) 和
[Qwen3.5-2B YAML](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/examples/vlm/qwen3_5_2b_megatron_geometry3k_grpo.yaml)：

- 官方 recipe：Qwen/Qwen3.5-2B、Geometry3K、8 GPU、Megatron + vLLM、异步 GRPO。
- PR 有 2B、27B、35B-A3B 训练 reward 曲线图片及部分 engine 测试证据，属于社区报告的正向证据；本文不从图片推算最终分数，也不等同
  held-out improvement。
- 关键适配：`bridge_type=megatron-bridge`、`use_padded_seq=true`、`use_bridge_for_update_weights=true`、vLLM
  `gdn_prefill_backend=triton`。
- 这些开关针对 Qwen3.5 混合 attention/GatedDeltaNet 的布局和权重转换，不能平移为 FSDP 配置；也不能把 2B/27B/MoE 的超参当作
  dense 4B 的验证结果。

### 6.3 精确 4B：找到了失败报告，未找到对应成功闭环

[AReaL #1319](https://github.com/areal-project/AReaL/issues/1319) 给出 Qwen3.5-4B + FSDP +
SGLang 的 TIR/数学配置，但内容是 actor/ref 显存异常、H100 OOM；后续也有 4B 用户报告高显存。此 issue 被 stale bot
关闭，**不是修好了**。

[AReaL #1261](https://github.com/areal-project/AReaL/issues/1261) 中维护者说合并 #1280 后 FSDP +
SGLang 应可运行；这是依赖兼容说明，没有指定 4B 的长程收敛曲线，不能升级为已验证 recipe。旧评论中的升级依赖建议也不能直接用于本地锁定环境。

[AReaL #1290](https://github.com/areal-project/AReaL/issues/1290) 则报告 Qwen2 GSM8K 从
SGLang 换到 vLLM 后崩溃；维护者在 H200 复现过，根因未确定，最终也只是 stale 关闭。它说明推理后端不能视作无关变量，但**不证明当前
vLLM、Qwen3.5 或所有配置都会失败**。

## 7. 如果要同步 standard GRPO，外部实现如何选

Hugging Face [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) 是现成同步训练器，并在
[GRPOConfig 源码](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)
明确区分：

- `loss_type="grpo"`：每回答长度归一化；适合拿来对照原始 GRPO reduction。
- 本次读取 main 的默认是 `loss_type="dapo"`，不能只看类名叫 GRPOTrainer。
- `scale_rewards="group"` 是组内标准差归一化；`beta` 控制 KL，默认是 0。

这段是 2026-09-22 raw main 源码事实；GitHub API 限流导致本轮未取得 TRL HEAD SHA，不当作固定 release
contract。TRL 是算法实现参考，不凭此宣称任意 Qwen3.5-4B 环境都能直接跑通。

### 7.1 精确 Qwen3.5-4B-Base 数学配方：值得参考，但属于 LoRA

本轮找到
[praysimanjuntak/qwen3.5-4b-grpo-gsm8k](https://github.com/praysimanjuntak/qwen3.5-4b-grpo-gsm8k/tree/00f716bfdaaf9e3fede514911b87511e5f091de1)。codeload
archive commit 为 `00f716bfdaaf9e3fede514911b87511e5f091de1`。已读取
`scripts/run_train.sh`、`src/train_grpo.py`、`pyproject.toml` 和四个
`results/eval_*.json`，仓库还包含逐题 rollout 文件。以下是仓库公开材料，不是本机复现。

| 维度         | 内容                                                                      |
| ------------ | ------------------------------------------------------------------------- |
| 初始化       | `Qwen/Qwen3.5-4B-Base`                                                    |
| 框架         | TRL GRPOTrainer + colocated vLLM 0.19.1                                   |
| 训练方式     | LoRA rank 32，单 A100/H100 80GB 的脚本                                    |
| 训练数据     | GSM8K                                                                     |
| LR / KL beta | `1e-6` / `0.04`                                                           |
| 采样 / 长度  | N=8，temperature=0.9，completion=1024                                     |
| batch        | per-device=8，gradient accumulation=4；不可直接当成 AReaL 的 prompt batch |
| schedule     | 1 epoch，warmup ratio=0.03                                                |
| GSM8K test   | n=1319，greedy pass@1：85.44% → 92.42%                                    |
| MATH-500     | n=500，greedy pass@1：62.2% → 61.6%                                       |

这份最接近用户要的 exact-model 作业，但仅支持“GSM8K 提升”的作者报告，不能宣称广泛数学泛化提升。训练命令没有显式指定 `loss_type`，而依赖给的是
`trl>=1.3,<2.0` 范围；因此不能保证它是严格原论文 GRPO，也不能只抄算法名字忽略实际安装版本。它没有提供本机全参 AReaL 的 LR 结论。

## 8. 对当前情况的建议

**推荐先保留 AReaL，使用官方同步模式做控制；如果“严格原论文 GRPO”是不可妥协的要求，则优先选择显式支持 sequence-mean objective
的成熟实现作算法参照。** 不建议为了寻找一个名字更标准的 YAML，先把 FSDP、SGLang、数据和训练器全部更换。

PPO 失败和 GRPO 的关系应按故障层划分：

| 如果失败来自                                               | 换成 GRPO 是否能绕过                            |
| ---------------------------------------------------------- | ----------------------------------------------- |
| critic/value head、critic 优化、GAE/终止 bootstrap         | 可以去掉相关训练路径，是有价值的隔离实验        |
| actor 前后向/GatedDeltaNet、padding、训练推理 logprob 差异 | 通常绕不过，GRPO 共用这些组件                   |
| 权重发布错误、异步陈旧                                     | 同步有助缩小范围，但仍需核对发布与版本          |
| 数据难度、奖励解析、组内全对/全错                          | 不自动解决；全同奖励组没有 outcome 相对学习信号 |
| LR、optimizer precision、loss reduction 对齐错误           | 不自动解决；社区已有 GRPO 真实崩溃反例          |

建议未来实验按以下顺序决定，而不是本次直接启动：

1. **核对已经跑过的 GSM8K 那一版**：若确实是当前官方 Qwen2.5 配方、跑完整且有 held-out 曲线，就复用证据，不重复跑；只有
   smoke/几步不够当收敛控制组。
1. **固定 Qwen3.5 checkpoint 身份**：`Qwen/Qwen3.5-4B` 与 `Qwen/Qwen3.5-4B-Base` 分开；本地 PPO PRD
   指向 Base。社区 instruct/post-trained 配方不自动适用于 Base、nonthinking 和 DAPO 8K。
1. **做最小同步 GRPO 控制**：同模型、同数据/奖励、同 backend，去掉 critic，固定 group 大小和明确 loss 语义。保持已经必要的
   Qwen3.5 padded/独立序列路径。不要为“照抄上游”撤销模型边界适配。
1. **先看有效更新再看收敛**：确认每题 G 个回答归组正确、非零组内方差比例、同版本 inference/trainer logprob、更新后权重发布、有效 token
   mask、截断比例。几步完成只证明链路，不证明长程稳定。
1. **同步控制稳定后再加异步**：保持其余条件不变，对比 staleness 与 decoupled correction；不同时改
   LR、N、thinking、数据和后端。

这次没有审计既往 PPO 全部日志，因此没有判定其具体根因，也没有判定既往 GRPO 成功或失败。记忆只用于定位已有 qualification/termination
工作，机制结论来自本次源码与公开材料。

### 8.1 后续问题：5e-6、6e-6 和正在挂起的 GRPO

`5e-6` 是常见候选量级之一，不是 GRPO 的统一标准，更不是不同模型/框架/LoRA 与全参之间的可比单位。`6e-6` 比 `5e-6` 高
20%；不能说一定无差别，也不能仅凭它判定会崩。社区明确的 AReaL 对照是 `1.7e-5` 与 `6e-6`，相差约 2.83 倍，而且只验证了 Qwen2.5-1.5B
的特定配置。

当前正式 GRPO run 的已保存 runtime 配置已读回：

- run：`grpo-dapo-official-n4-20260921-v1`，Pueue task 88。
- 模型：Qwen3.5-4B-Base snapshot `1001bb4d826a52d1f399e183466143f4da7b741b`。
- actor 配置 LR：**6e-6**，显式 warmup_steps=5，constant scheduler；不是旧 1.7e-5。训练各步真正 LR 仍应结合
  scheduler/step 指标判读。
- DAPO，prompt batch=128，N=4，max_new_tokens=8192，SGLang/FSDP 各 4 卡，staleness=2，decoupled
  loss 和 batch advantage normalization 开启。
- 本轮 Pueue 读回：task 88 已于 **2026-09-22 02:28:16 UTC**
  退出，`Failed(1)`，不是仍在训练。退出状态本身不证明发生了学习率导致的数值/奖励崩溃。

证据：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-grpo-math-official/runs/grpo-dapo-official-n4-20260921-v1/evidence/resolved-config.json`
与 Pueue 当前状态。读取后未修改、取消或重启任务。

进一步读取 `evidence/steps/1.json`：首步有
`update_successful=1`、`optimizer_steps_since_init=1`、`published_version=1`，grad_norm=0.1672，但该步
LR 指标为 `0.0`。这证明 optimizer 调用/发布链路走完，不证明发生有效非零学习更新或已稳定收敛。actor rank 1 的
`perf_tracer/actor/traces-r1.jsonl:93210` 在 `global_step=1` 的 `compute_logp`/FSDP
forward 链路记录 `AttributeError`；controller 末尾是 SIGTERM，没有完整 traceback。当前证据不支持把这次退出归因于高 LR
导致的数值或 reward collapse；具体 AttributeError 对象和传播根因仍需另行诊断。

### 8.2 后续根因已定位：包装器 queue.py 遮蔽标准库

后续定位到 Pueue 父进程的完整 traceback：我们新增的 `launch-20260921/queue.py` 遮蔽标准库 `queue`，正式首步后的
snapshot hook 导入 `datasets -> pyarrow` 时抛出 `ImportError: cannot import name Queue`。父进程
finally cleanup 随后终止训练。真实虚拟环境的 CPU import A/B 已确认这个首要根因，责任属于我们新增的启动包装层。

actor AttributeError 则有原函数 CPU 模拟支持：worker 清理等待线程 5 秒后仍销毁 model，未结束的 forward 在下一
microbatch 访问已删除的 `self.model`。这些 shutdown 函数与上游基线相同；但原现场没有异常属性名，不能将模拟消息当作原始
traceback。详见
[根因与复现报告](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-grpo-math-official/audits/attributeerror-20260922/root-cause.md)。这更新了上一节“待定位”的状态，未启动训练或改运行代码。

## 9. 四个快速判断练习

**概念：把 `max_head_offpolicyness` 设成 0，就完全复现原论文 GRPO 了吗？**

<details>
<summary>参考答案</summary>

没有。它只处理版本调度。还要核对 advantage normalization、loss reduction、KL 放置、behavior/proximal 分母、额外
rejection 等。

</details>

**源码：在哪两处判断“无 critic”与“sequence mean”是否成立？**

<details>
<summary>参考答案</summary>

actor 的 compute_advantages 中，没有 values 时使用零值；functional.py 的 ppo_actor_loss_fn 末尾是有效
token 总和除 token 数，因此当前是 token mean，而非每回答先平均。

</details>

**证据：Qwen3.5-2B Geometry3K 的曲线，加上 4B 的模型支持声明，能作为 4B Base DAPO 成功证据吗？**

<details>
<summary>参考答案</summary>

不能。模型尺寸、初始化、模态、数据、后端和优化预算都不同。只能分别说明架构适配和相邻配方可参考。

</details>

**设计：同步 GRPO 能训练、异步崩了，下一步该换模型还是改 reward？**

<details>
<summary>参考答案</summary>

先保持模型和 reward 不变，比较行为/训练版本、logprob ratio、权重发布和陈旧程度；只有其他证据支持时才换模型或
reward。这样才能把异步因素与其他变化区分开。

</details>

## 10. 主要来源索引

- [AReaL 异步机制](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/docs/en/algorithms/async.md)

- [AReaL 算法实现说明](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/docs/en/algorithms/grpo_series.md)

- [同步诊断配置](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/docs/en/best_practices/algo_perf.md)

- [当前 math GRPO 配置](https://github.com/areal-project/AReaL/blob/c74b343c53c785eb620943fcabbd6b8263e28051/examples/math/gsm8k_grpo.yaml)

- [FP32 master 后的 LR 修复与完整对照 #1634](https://github.com/areal-project/AReaL/pull/1634)

- [Qwen3.5 Megatron 支持、recipe 与 reward 曲线 #1384](https://github.com/areal-project/AReaL/pull/1384)

- [Qwen3.5-4B 显存问题 #1319](https://github.com/areal-project/AReaL/issues/1319)

- [Qwen3.5 环境兼容答复 #1261](https://github.com/areal-project/AReaL/issues/1261)

- [GSM8K vLLM 崩溃报告 #1290](https://github.com/areal-project/AReaL/issues/1290)

- [未合并的可选 loss aggregation #1546](https://github.com/areal-project/AReaL/pull/1546)

- [DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300)

- [AReaL 论文](https://arxiv.org/abs/2505.24298)

- [Qwen3.5-4B-Base LoRA recipe 固定版本](https://github.com/praysimanjuntak/qwen3.5-4b-grpo-gsm8k/tree/00f716bfdaaf9e3fede514911b87511e5f091de1)

- [TRL GRPOConfig，2026-09-22 main 读取](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)

- [本地正式 GRPO runtime 配置](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-grpo-math-official/runs/grpo-dapo-official-n4-20260921-v1/evidence/resolved-config.json)
