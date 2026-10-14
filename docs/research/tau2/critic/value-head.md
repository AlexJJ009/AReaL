# 优化器与 value head

> 来源：原 README 第 16.14.3 节。拆分前的全文见 git 提交 b8d95027 中的
> docs/research/2026-10-09-tau2-high-baseline-audit/README.md。

- **优化器**：配置写的是 `optimizer.type: adam`，`fsdp_engine.py` 实际构造的是
  `torch.optim.AdamW`；critic 和 actor 配置都把 `weight_decay` 设成 0（AReaL 默认 0.01），实际行为等于
  Adam。critic 从预训练权重出发做回归，衰减把权重拉向 0 而不是初始值，没有明确好处，设 0 与 PPO 实现的常见做法一致。Adam
  对梯度的整体缩放基本不敏感，所以梯度裁剪（grad norm 20–400 裁到 1）不决定步长；每个参数每步大约移动 lr 那么多。

- **value head**：Qwen3.5 的 `token_critic.py` 是 `nn.Linear(hidden, 1, bias=False)`，权重按
  std 0.02 正态初始化，输出不加约束。这是 LLM PPO 的常规结构：OpenRLHF、verl、TRL 的 critic 都是"LM 骨干 + 输出 1
  维的线性层"。线性无界的理由在 RLHF：return 里有 reward model 的分数（无界的 logit）和逐 token 的 KL 惩罚，value
  的取值范围事先不知道。我们这里 reward 只有 0/1，`kl_ctl: 0`、 `discount: 1`、critic 的 GAE λ 为 1，目标必然在
  $[0,1]$ 内，无界线性 head 是沿用的默认，不是为这个任务选的。第 0 步输出约 $\pm 3$ 来自这个初始化：std 0.02 的权重乘以 2560
  维、范数一百多的隐状态。

- **修 A 的做法**：只加一个 bias（$V = w^\top h + b$）不够：$b$ 设成均值、$w$ 仍按 std 0.02 随机初始化，$b$ 会被
  $\pm 3$ 的噪声盖住。所以两处一起改：

  | 改动                   | 作用                                                                                                                          |
  | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
  | $w$ 零初始化           | 第 0 步对所有 token 输出 $b$，MSE 从约 10 降到目标方差；最后一层只有 1 个输出，零初始化没有对称性问题，梯度 $(V - y)h$ 不为 0 |
  | $b$ 初始化为训练集均值 | 起点在正确的水平上（监督学习的先验 bias 初始化，例如 RetinaNet）                                                              |

  head 和骨干用同一个学习率 5e-6，不单独设。Adam 让每个参数每步大约移动 lr，与梯度大小无关：$b$ 只有 1 个参数，即使 lr 为 1e-4，80
  步也只移动约 0.008，整体水平修正不了；$w$ 有 2560 个分量，乘上范数一百多的隐状态，lr 1e-4 时一步就能让输出整体移动零点几。smoke 498 用
  1e-4 时，第 1 步后预测从全部 0.78 变为均值 0.63、范围 0.43–0.81，正好朝那一个 batch 的目标均值 0.62 靠，整体水平会逐 batch
  跳动。所以起点靠初始化修好，之后不靠 head 的大学习率。`critic_head_lr` 字段保留，正式实验不设。

  训练时参数以 fp32 存储（`optimizer_dtype: float32`，前向和反向由 FSDP2 转成 bf16 计算），lr 量级的小更新不会被 bf16
  舍入掉：498 的 recover checkpoint 里 `score.bias` 是 fp32，两步后从 0.78 变为 0.779909，Adam 状态也是
  fp32。导出给 SAO 的 HF checkpoint 转成 bf16（`score.bias` 读出为 0.78125），0.78 附近的精度约 0.004。

  另一条路是 sigmoid + BCE（PRM/verifier 对 0/1 成功率的常用做法；Farebrother et al. 2024 显示 value
  回归改为分类在深度 RL 中更稳），留作上面三处改完后仍漂时的备选。这些都只针对 A。

- **实现**（`7da3a4f3`、`275548de`）：`areal/api/cli_args.py` 和原训练路径不变，另起入口
  `scripts/tau2/run.sh data critic-head`（`scripts/tau2/train_critic_head.py`）。critic
  引擎在远端 worker 里构造，所以新入口用 `PPOTrainer` 子类返回 `examples/tau2/critic_head.py` 里的引擎子类，head
  设置放在一个 `PPOCriticConfig` 子类里随 RPC 传给 worker。head 是否带 bias 由模型配置里的
  `areal_value_head_bias` 决定：引擎子类把它写进自己的 `model_config`，保存 checkpoint 时随 config.json
  落盘；Qwen3.5 adapter、`Qwen35ScalarValueModel` 和 SAO 的 head 预检
  `scripts/sao/check_critic_head.py` 都按它构建或检查 head，所以 SAO 用普通的 `FSDPEngine` 就能加载带 bias 的
  critic，不需要这个 模块；有 bias 没标记、有标记没 bias，都会报错。从已训练的带 bias critic 重启时，引擎子类保留原
  head、不重新初始化。新增字段 `critic_head_bias_init`（必填，$[0,1]$）和 `critic_head_lr`（不填则与骨干同为
  5e-6）。核心代码的改动：adapter 和 scalar value 模型读这个标记，`FSDPEngine` 在 `from_pretrained` 时把自己的
  `model_config` 传给 adapter（与 adapter 自己从同一路径重读的配置相同），新增 `_optimizer_param_groups`
  挂钩，`train_critic.main` 接受 `trainer_cls`。

- **测试**：`tests/test_tau2_critic_head.py`（16 个）和
  `tests/test_sao_critic_head_guard.py`（新增 7 个），CPU 上跑。 覆盖：第 0 步每个 token 都输出
  bias；零权重仍有梯度；参数组只把 `score.weight`/`score.bias` 分给 head；配置子类经 RPC 序列化后保持类型；按引擎的保存方式写出的
  checkpoint，用 SAO 的普通引擎加载后 bias 和输出完全一致，并通过 head 预检；从这个 checkpoint 重启时 head 不变；标记与 bias
  不一致时加载和预检都失败。故意改坏 adapter 传配置、scalar value 读标记、预检的一致性 检查、重启时跳过初始化这四处，对应测试都会失败。critic 与
  SAO 相关的 130 个测试通过；check-config 通过（747 行、94 步，不填 bias 时报错）。还没在 GPU 上跑过。

- **bias 的取值**：训练目标的期望，按 479 的 temperature 1 成功率计算：pack v2 加官方 train（648:99）约
  0.91，[data-mix.md](data-mix.md) 的新配比为 0.78。
