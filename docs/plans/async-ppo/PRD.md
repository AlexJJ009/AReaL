# AReaL 单机异步 PPO：全量 DAPO 一轮与前五步验收 PRD

状态：执行协议 v3；用户已要求 PRD/checklist 定稿后直接实施，全部验收通过才完成。独立环境、限定范围 Docker 清理、数据清洗和分组件真实 GPU
预检已完成，联合预跑与正式实验验收仍在执行。 日期：2026-09-20。原始需求见 [request.txt](request.txt)。本阶段称为“异步 PPO
qualification”，后续再实现 SAO。

2026-09-21 修订：首个正式运行在第33次联合更新后作废并停止。发现上游按当前 batch padding 宽度推断截断，导致同一终止轨迹的 GAE return
随同批序列长度变化。原始产物保留用于诊断，不能作为合格 PPO 实验或与修复后的步骤拼接。修正后必须从同一 Base、新 run root、新 candidate SHA
重新通过前五步监督并完成135步。此前硬件/依赖/数据清洗事实保留；loss 及正式实验验收重新检查。

## 1. 目标与范围

在本机八张 GPU 上，以 Qwen3.5-4B-Base、FSDP2、SGLang、uv 独立环境运行纯文本数学异步 PPO。使用全量清洗后的 DAPO、8,192
token 回答上限，固定种子完整训练 1 epoch；前五个有效 actor+critic 更新逐步监督，之后每 20 step 保存并以 N=4 评测五个测试集，epoch
结束保存并评测。

用户已明确选择：“跑完整 1 epoch；前 5 step 逐步监督，之后每 20 step 保存并评测
N=4”。第五步通过是继续训练的门槛；完整任务必须等本轮训练、末尾结果和资源收尾完成。

优先 PPO：它直接覆盖后续 SAO 所需的 scalar critic、GAE、actor/critic 共置和异步版本同步。GRPO 可减少 critic
适配，但不能验收这些链路，所以不作为本阶段的替代成功结果。4B dense 可以训练 PPO，容量与速度需实测；Qwen3.5 critic 需要薄适配，不能只换模型路径。

用户已授权建立仓库/分支/worktree、清理 Docker 盘，并明确要求 PRD/checklist 定稿后立即执行。最终参数、执行候选 SHA、数据与 lock
摘要在启动完整一轮运行前冻结并展示；不重复索要相同授权。关键决策持续记录，全部验收完成后交结果和决策报告。

非目标：本轮实现 SAO、额外 epoch/种子矩阵、刷榜/宣称 AIME 提升、MATH 训练、额外 GRPO 对照、Megatron/Archon 后端迁移、Agentic
工具环境、全局 CUDA/driver 改造。用户关于 DAPO 优于 MATH 的判断作为本次数据选择依据，本轮结果不能证明“只有 DAPO 才能学会 AIME”。

## 2. 已核实的起点

本节保留最初现场快照；后续安装和清理结果以第 10 节及实际验收记录为准。

| 对象      | 事实                                                                                           | 边界                                                          |
| --------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 本机      | 8×A100-SXM4-80GB，SM80；535.129.03 kernel driver + cuda-compat-13-0 580.126.20                 | 旧运行曾用 A800；本次须写 A100，不能把旧显存/吞吐结论直接套用 |
| 工作区    | 独立 AReaL 仓库、分支 codex/sao-math、受管理 worktree 已创建；doctor 无 error/warning          | 未安装 AReaL 运行环境                                         |
| 上游基线  | v2.1.0，ecc8b0e4dfb4e3965f67121a5c87866a07c390ae                                               | 固定 release，不跟随 main                                     |
| Base 权重 | 本地 Qwen3.5-4B-Base snapshot 1001bb4d826a52d1f399e183466143f4da7b741b；两片权重，共约 8.8 GiB | 文件存在已检查；在新 AReaL 环境加载未检查                     |
| 旧 PRD    | batch=128 prompts、microbatch=1/GPU、K4 为 4+4 条轨迹、eval n=16                               | 本次 PPO 无 privileged teacher；batch/轨迹数不能混称          |
| Docker    | /ebs/docker 约余 36 GiB；141 个镜像、1 个已退出容器；数据盘余约 33 TiB                         | 大量镜像属于 Tri-EvoBench；unused 不自动等于可删              |
| 五集      | AIME24=30、AIME25=30、AMC23=40、BeyondAIME=100、MATH500=500，共 700                            | HMMT 不纳入本次指标                                           |
| 数据坏例  | AIME24 有 7 个纯整数前导零答案；AIME25/BeyondAIME 未发现同类问题                               | 尚未修正文件                                                  |

本地路径（作为现场配置，不写死进通用 Python 代码）：

- 仓库：/data_storage/yl_test/lgx/data-1/code/AReaL
- worktree：/data_storage/yl_test/lgx/data-1/code/\_worktrees/AReaL/codex-sao-math
- 运行/数据/证据根：/data_storage/yl_test/lgx/data-1/code/\_artifacts/AReaL/codex-sao-math
- Base
  模型：/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b

## 3. 安装协议：直接用 uv，不拉完整 AReaL 镜像

可行。参考 Dockerfile 提取 FSDP2 + SGLang + Qwen3.5 必需依赖，放在数据盘上的独立 .venv；uv cache、编译
cache、checkpoint 同样位于数据盘。节省的是 Docker 盘和镜像冗余层，依赖本身仍占磁盘。

| 层           | 目标                                                                                                 | 处理                                                                    |
| ------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Python       | 3.12                                                                                                 | uv 管理，独立 .venv                                                     |
| 基础计算     | Torch 2.9.1+cu129，TorchAO 0.15.0                                                                    | 沿用 v2.1.0 SGLang 锁定；不复用 Verl 的 Torch 2.11/cu130 venv           |
| 推理         | SGLang 0.5.10.post1                                                                                  | 使用上游 sglang extra 与 cu129 index                                    |
| FSDP/Qwen3.5 | torch_memory_saver 0.0.9、kernels 0.12.2、FLA 0.4.2                                                  | 按实际 import/forward 路径验证                                          |
| 编译扩展     | FA2 2.8.3；causal-conv1d 1.6.0 为 Dockerfile 起点                                                    | 选择 cp312/Torch2.9/CUDA12/SM80 匹配 wheel；锁定 URL、版本、SHA256      |
| 不主动构建   | Megatron 专用 Apex/TransformerEngine/grouped_gemm，Hopper FA3/FlashMLA/DeepGEMM，Agentic CLI/sandbox | 本次 FSDP2 dense 无需这些功能；传递依赖若不可避免须记录，不启动对应后端 |

源码核查纠正了上一份调研的一处描述：v2.1.0 的 cuda extra 并未直接列出 FA2；Dockerfile 在 uv sync
之外手动安装它。causal-conv1d 又被 pyproject 的 override 禁用，而 Dockerfile 手动安装。因此“一条原始 uv sync 等于完整
Docker 环境”不成立。

实现要求：

1. 复用根 pyproject.toml/uv.lock，在本分支加入薄的项目 extra（建议名 sao-math），锁定所需扩展并处理 causal-conv1d
   的排除规则；不另建第二套手写解析器或全量依赖清单。
1. 从 sglang/tms/kernels 功能集合起步，补 FLA/必要扩展；避免为不用的 Megatron 链编译依赖。直接 uv pip 临时补包装好后，必须回写
   pyproject/uv.lock 或可验证的 wheel 构建记录，不能遗留未锁定依赖。
1. 最终以 uv lock --check、uv sync --locked --extra sao-math、uv pip check 原始结果和真实
   import/kernel 读回证明可复建。sao-math extra 已实现；安装采用 scripts/sao/bootstrap_env.sh 的先装
   Torch、再编译扩展两阶段流程。运行使用显式激活环境的 scripts/sao/run_ppo.sh；普通 uv run 默认不选择
   extra，可能同步掉加速依赖，不作为本实验入口。
1. 本机 CUDA13 compat 作为项目启动环境保留；Torch cu129 的扩展不得误用 nvcc13 编译。优先匹配 wheel；确需编译时使用项目隔离的
   CUDA12 构建工具并锁定产物，不修改机器默认 CUDA 或原有 Verl 环境。
1. 验证导入 torch、sglang、fla、causal_conv1d、flash_attn 及实际被选择的 kernel；用正反向与 SM80 读回证明路径，不以
   import 成功代替训练可用。

## 4. 算法和参数协议

标为“建议”的数值是首版方案，不冒充用户指定或官方 Qwen3.5 成功配方。最终启动使用一份 resolved config；定义→覆盖→consumer 读回必须一致。

| 项目                  | 首版值                                                                                                      | 依据                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 算法                  | 异步 clipped PPO + scalar critic + token GAE                                                                | 用户最终选择 PPO；标准 clipping 建议                                       |
| 拓扑                  | actor/critic：fsdp:d4p1t1，共置；rollout：sglang:d4p1t1，另 4 卡                                            | 用户 4+4；实际 GPU UUID/rank 映射记录                                      |
| 模型                  | Qwen3.5-4B-Base，本地 snapshot；全量文本骨干训练，无 LoRA/teacher                                           | 用户模型；首版建议                                                         |
| 训练长度 / 种子       | total_train_epochs=1，total_train_steps=null；seed=42、shuffle=true、drop_last=false                        | 一轮为用户明确选择；42 为固定种子建议；前 5 step 逐步监督                  |
| prompt / response     | 1,024 / 8,192；总长至少 9,216                                                                               | prompt 沿用历史建议，response 用户要求；不静默截断题目                     |
| batch                 | 128 个不同 prompt/step × n_samples=4 = 512 条轨迹/step                                                      | prompt batch 参考旧 PRD，n=4 沿用官方 PPO 示例；无组内 reward baseline     |
| 更新次数              | actor/critic 的 ppo_n_minibatches 都为 1，每 step 各一次 optimizer.step                                     | 前五步各累计 5 次；整个 epoch 按实际 batch 数累计；critic-only warmup=0    |
| microbatch            | 从 1 条序列/GPU 起；actor 与 critic 均启用梯度检查点、BF16                                                  | 建议；AReaL mb_spec 的真实拆分须读回，token cap 不等于固定条数             |
| 学习率                | actor=1e-6，critic=5e-6，constant，grad clip=1                                                              | 4B 小规模建议；不照搬 1.5B 示例 1.7e-5；无额外预热阶段                     |
| PPO clip / value clip | 0.2 对称 / 0.5                                                                                              | dataclass 默认；eps_clip_higher=null、c_clip=null                          |
| GAE                   | discount=1，gae_lambda=1，gae_timestep_unit=token                                                           | 首版沿用默认；不加 SAO 动态 lambda/双 lambda                               |
| reward                | 正确=1、错误/格式错误=0；scale=1、bias=0                                                                    | 可解释二元数学奖励；超时/异常为基础设施错误                                |
| 标准 loss 开关        | use_decoupled_loss=false，recompute_logprob=false，rejection_sampling=null，importance_sampling_level=token | 分母保留真实 behavior logprob                                              |
| 排除 DAPO/其他增强    | dynamic_bs=false，无动态筛题回调，无组 reward normalization，无 overlong penalty，无 SAPO/CISPO/M2PO        | DAPO 只作为数据源；不因全对/全错丢组                                       |
| KL / teacher          | kl_ctl=0、ref=null、teacher=null；不加蒸馏                                                                  | 官方数学示例无 KL，减少不必要模型副本                                      |
| 优势归一化            | batch mean/std；actor.reward_norm=null、gconfig.reward_normalization=false                                  | 沿用 PPO 示例 batch advantage 归一化                                       |
| 生成                  | temperature=1、top_p=1、明确 EOS；max_head_offpolicyness=2                                                  | 首版建议；不要求异步数值严格可重现                                         |
| 模板 / thinking       | 建议沿用此前 Base 非 thinking 约定；固定实际 tokenizer/template 与渲染样例                                  | 不推断 8K 必须开启 thinking；train/eval 一致                               |
| 评测 / 保存           | 五集全量 N=4；每 20 个完成 step + epoch 尾步；建议另做更新前 step0 基线                                     | N=4/frequency=20 为用户要求；temperature=1、模板、8K 固定，独立评测 RNG    |
| 频次字段              | saver.freq_steps=20、evaluator.freq_steps=20，freq_epochs=1、freq_secs=null；recover 同步记录可恢复状态     | 明确更新后 step20/40… 与尾步；不止保存 HF actor 权重；次数和阶段由日志读回 |

全局 batch 128 不等于每卡 128。512 条轨迹分到 4 个训练 rank 后，约为 128 条/rank，再做 microbatch 梯度累积。增加 global
batch 主要提高积累与生成工作量；常驻 actor+critic 的容量取决于优化器状态、激活、序列长度和 offload。若预检 OOM，优先减 rollout
并发/调整 microbatch 与 offload；必要时在正式五步之前把 batch 改为 64 并重新冻结协议，不能在五步中途更换配置继续累计。

PPO loss 明确定义为有效 response token 上的 masked mean：
$L\_{actor}=-\\mathrm{mean}_{mask}\\min(r_t A_t,\\mathrm{clip}(r_t,0.8,1.2)A_t)$，其中
$r_t=\\exp(\\log\\pi_\\theta-\\log\\pi\_{behavior})$。critic 使用上游 clipped value
MSE。Prompt/padding 不入 loss；EOS 与被截断尾部的 value/bootstrap 处理须用小张量测试绑定上游实际行为。

终止边界明确约定：使用生成端实际 stop reason 传递逐轨迹 `terminated`/`truncated`
布尔标记，二者恰有一个为真。本协议将8K上限视为人为截断（time limit），沿用上游预期的 continuation bootstrap 语义。正常 stop/EOS 的
bootstrap 为0；response 长度截断使用该轨迹真实末尾状态 `V(s_T)`。不得用 padded tensor 宽度、同批最长序列或仅 token
总长推断。相同轨迹单独运行、追加 padding、与不同长度轨迹混批，其有效 token 的 return 必须一致。当前 gamma=lambda=1 且无 KL，独立
oracle 为有效响应 token 上 `return = reward + truncated * V(s_T)`；除 CPU
测试外，原生联合预跑和正式前五步都直接读回真实 critic values/returns/flags 校验这个等式，并核对 consumer 的截断计数等于真实 stop
reason 计数。

“标准”限定为上述 PPO 目标。异步引入行为策略滞后，不宣称等同于严格 on-policy PPO，也不把 decoupled PPO、DAPO、DIS
混称为同一方法。AReaL 官方 gsm8k_ppo.yaml 开启了 decoupled loss、重算 logprob、ratio
rejection，需要显式覆盖。GRPOConfig 只是 PPOConfig 的兼容别名，不能单凭类名判断算法；以 critic、优势和 loss 实际 consumer
为准。

## 5. 数据协议与已确认 case

训练输入从既有 DAPO 原始修订派生，复用已有清洗逻辑与 manifest：

- 修订：BytedTsinghua-SIA/DAPO-Math-17k@65877096c24ffa7abc4e4fa5edb95cf3413a5674。
- 本地入口：/data_storage/yl_test/lgx/datasets/math-benchmarks-20260916/derived/dapo-math-17917-unique/train.parquet。
- 输入 17,917 行，文件
  SHA256=134671de0cd455477e3bbca80f125ea32094084ef1ae62e2fa3e1b066414ba4c。“unique”指
  ID，历史 manifest 仍有 730 条相同题面重复。
- 历史 PRD 清洗后 full pool=17,156，训练子集=8,192；本次使用全量合格题，不套 difficulty quota、语言筛选或 teacher
  prompt 长度过滤。按 PPO 的实际 bare prompt 重新去重、检查冲突答案、prompt 长度、评测交叉；新行数以 manifest 为准。固定
  seed=42 打乱后完整一轮，不只取前 640 题或旧 8,192 子集。
- 题目只进入 student prompt，ground truth 只进入 reward。清除 PRD privileged-answer/teacher
  入口；不把答案或解题过程追加给 actor/rollout。
- 对完整五集做题面重复和答案格式检查；去重采用 NFKC/空白标准化匹配并记录边界，不能声称完全语义去污染。已知冲突答案隔离，不按模型表现删题。

五集复用已审计 test/\*.parquet（位于既有 dapo-prd-answer-only-qwen35-4b-base-hard1of8-8192-v2
数据包），建立新的 ppo-math-v1 数据版本。AIME24 已确认：

| 零基行号 / source raw id | 原 ground truth | 修正 |
| ------------------------ | --------------- | ---- |
| 7 / 67                   | 025             | 25   |
| 15 / 75                  | 073             | 73   |
| 18 / 78                  | 023             | 23   |
| 23 / 83                  | 045             | 45   |
| 24 / 84                  | 033             | 33   |
| 25 / 85                  | 080             | 80   |
| 26 / 86                  | 055             | 55   |

用户所举 043→43 作为规则回归例；没有在当前已审计五集中找到 043，不能编造为实际改动。仅对确认属于十进制数值答案、完整匹配 ^0+\[0-9\]+$ 的 ground
truth 用 str(int(x)) 规范化；不改题面/ID/LaTeX/小数/分数/有语义的进制或编码。AIME25、BeyondAIME 的同类改动数当前为
0，其他格式问题仍需完整逐行校对。

交付 source/new SHA、逐行 correction map、原答案留存、验证报告，行数保持 30/30/40/100/500。AReaL loader 复用
load_from_disk 入口，薄转换为 messages + answer + source_id/metadata；避免重写
trainer/数据框架。格式测试涵盖空答案、末尾 boxed、重复 Problem:、截断、元数据回连和 train/eval gold round-trip。

数据修正与评分器测试都要做。保留此前末个完整 boxed/fboxed 与保守数学等价的语义；移植其小型 scorer/测试时固定源 SHA，避免引入整个 Verl
依赖。基础设施异常不得记为错题 0。训练和评测使用同一 scorer 版本；不自动认可上游 math-verify 0.8.0 与旧环境 0.9.0
等价，必要依赖变更一并锁定并回归。

## 6. 必要实现与执行顺序

1. **工作区**：已建立；补现场配置与证据目录，保持源码、可写环境、缓存、运行产物分离。
1. **Docker 清理**：先保存镜像/容器/volume/build cache 与引用关系清单，再按精确 ID
   删除可归属、无活动引用且有重建或归档依据的历史对象。用户授权已提供；不做全局 prune。Tri-Evo pinned/current 镜像及有恢复用途的 volume
   不因 unused 标记被删除。记录实际删除与 df 前后差额；若大对象无法确认保留需求，明确留下并说明清理范围，不把“已审计”标为“已清理”。
1. **uv 依赖**：实现最小、可锁定安装流程；在新环境完成 kernel 正反向和多卡通信检查；保留原有 CUDA13 与旧实验环境。
1. **数据与 scorer**：产出修正版五集、DAPO bare 数据、manifest；复用审计过的奖励语义。
1. **Qwen3.5 critic 薄适配**：保留预训练文本骨干，加可训练标量 value head；验证 \[B,L\]/\[B,L,1\]、next-token
   mask 对齐、有限 loss/梯度、骨干与 head 均能更新、checkpoint 保存恢复。actor 仍为生成模型；不随机初始化整个 critic，不以
   logits/常量 value 代替 critic。对两个模型分别审计 missing/unexpected keys：只有新 value head/有意排除的视觉或
   LM head 可列明确白名单，禁止 strict=false 掩盖骨干错载。
1. **预检**：评分器正则/boxed 提取与 LaTeX 语义匹配；PPO/GAE/value loss 数值与梯度 oracle；8K actor/critic
   正反向；8 GPU NCCL 与实际 4+4 子组；SGLang 输出 token logprob；同版本 trainer/inference logprob
   对齐；至少一次 actor→rollout 权重更新与读回。另见第 6A 节，推理、actor
   update、FLA/卷积加速全部要在正式启动前有证据。每项失败先修复，不消费五步成功计数。
1. **正式运行**：从固定 Base 起点运行同一个 candidate/config 下 1 epoch；预检更新不污染正式初始模型。前五步逐步检查；通过后继续该
   run。故障保留现场，不能拼接不同候选的成功步；若改源码/loss/数据，生成新候选并重做受影响门槛。
1. **结果**：更新前 baseline，以及 step20/40… 和 epoch 尾步五集 N=4；每个保存点的 actor、critic 与可恢复训练状态，末尾
   checkpoint 加载一致性探针、资源收尾。第五步不因达到监督门槛而自动终止或额外全量评测。

优先复用原生 PPOTrainer/FSDPEngine、local scheduler、RLVRWorkflow/数学 workflow、scorer 测试、saver 与
tracer；只补缺失的 critic、数据/入口绑定和证据导出。Qwen3.5 的 delta attention/padded 路径必须保留。旧 PRD 的
teacher/后验逻辑不移入本基线。

## 6A. 启动前的正确性与加速门槛

本地 config.json 明确有 32 层：24 个 linear_attention（Gated DeltaNet），8 个
full_attention。两个执行端分别验证，不能从“包已安装”或训练端有 FLA 推断 SGLang 同样生效。

| 范围               | 预期与证据                                                                                                                                                                                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| scorer             | 复用 PRD scorer 的末个完整 boxed/fboxed、正则和规范化，再做 math-verify/LaTeX 语义匹配。正例包括 033/33、1/2/0.5、等价 LaTeX；反例包括大小写符号、根的重数、错进制、缺框/坏框/最后框错误、超时不能计0。                                                     |
| actor/critic 加载  | 二者均从同一 Base snapshot 的文本骨干初始化；actor 生成 logits，critic 逐 token scalar。检查 key 映射、dtype、24+8 层结构、可训练参数和骨干抽样校验；只有 critic 新 value head 允许新初始化。                                                               |
| loss               | 独立张量 oracle 检查 PPO clip 两侧与正负 advantage、GAE/return/value loss、shift/response mask、truncation、microbatch/FSDP 缩放；对照 loss 数值和梯度，禁止用实现自身输出作唯一真值。                                                                      |
| SGLang 推理        | 上游 SGLangConfig.attention_backend 默认 fa3，A100 必须覆盖为通过实测的 SM80 后端，首选 flashinfer；检查 prefill/decode 的全注意力和 GDN/conv 算子、CUDA graph replay、调度重叠、实际 server args。分块 prefill/混合缓存仅在 Qwen3.5 数值对齐后启用并记录。 |
| FSDP2 Actor update | actor/critic.attn_impl 首选 flash_attention_2；线性层走 FLA chunk/fused GatedDeltaNet、causal-conv1d 或经核实的等价 fused 实现。真实 8K batch 的 loss→backward→optimizer 路径必须在 profiler 中出现，不能只做 inference forward。                           |
| 实测方法           | 冷启动/JIT/warmup 与稳态耗时分离；对固定 token/mask 的小输入比较参考路径的 value/logprob/loss/梯度，使用明确 BF16 容差；再测 8K 真实路径的算子、tokens/s、update 耗时和显存。若仅有路径证据，没有完整同输入对照，不宣称具体倍数加速。                       |

优化开关不是越多越好；FA3/FlashMLA 等 Hopper 专用实现不加入 SM80 验收。Qwen3.5 padded/GDN 路径不能为追求 packing 或
chunked LM head 而绕过。可用 fallback 仅用于诊断；正式启动前必须解释并解决任何违反上述加速契约的慢路径。

## 7. 验收 checklist

运行项当前均未验收；机器记录独立保留 agent_status、human_status 和 evidence.level，不把写完文档标为实验通过。

| ID  | 必须成立的结果                                                            | 必须交付的证据                                                                                                            |
| --- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| C01 | 独立 repo、v2.1.0 基线、codex/sao-math worktree 正确                      | Git SHA/ref/remote、agent-wt doctor                                                                                       |
| C02 | uv 锁定环境可重建且必要 CUDA kernel 正反向通过                            | lock SHA、installed inventory、uv check、实际 kernel/ABI/SM80/CUDA 库读回                                                 |
| C03 | 精确范围 Docker 清理已执行，受保护对象保留                                | 删除 ID/用途/恢复依据，before/after inventory 与实际回收字节                                                              |
| C04 | 本地 Base actor 和 SGLang 都加载同一模型                                  | 模型 shard/index hash、snapshot、加载日志、真实生成；没有下载替代模型                                                     |
| C05 | critic 是可训练逐 token scalar value，加载/反向/保存恢复正确              | shape/mask、梯度、参数差异、checkpoint reload 探针                                                                        |
| C06 | DAPO 转换完成，五集 700 题完整，已修正 7 条已知 gold，train/eval 格式正确 | source/new manifest、correction map、排除/冲突清单、scorer 正反例；gold=043 规则回归                                      |
| C07 | 启动前配置/consumer绑定通过，启动后实际读回仍是本协议 PPO/8K              | 两阶段证据：prelaunch resolved config 与启动后首batch读回；candidate/config/command/lock/data/scorer 摘要                 |
| C08 | 所有八卡按 4+4 分配且通信、权重同步可用                                   | GPU UUID/PID/rank 对照、NCCL、版本与权重数值抽样读回                                                                      |
| C09 | 真实异步：生成下一批与当前 actor/critic 更新有时间重叠                    | 同一 run 的请求起止、train span、样本 behavior version、publish/consume version 与滞后分布；不能只给 async 字段/HTTP 调用 |
| C10 | 完成第 1 个有效联合更新                                                   | steps/1.json：有效样本/token、reward/异常数、actor/critic loss、梯度/更新计数、发布版本                                   |
| C11 | 完成第 2 个有效联合更新                                                   | steps/2.json，同上                                                                                                        |
| C12 | 完成第 3 个有效联合更新                                                   | steps/3.json，同上                                                                                                        |
| C13 | 完成第 4 个有效联合更新                                                   | steps/4.json，同上                                                                                                        |
| C14 | 完成第 5 个有效联合更新                                                   | steps/5.json，同上；此时 actor/critic optimizer 更新各 5，接着运行完整 epoch                                              |
| C15 | 每20步及尾步保存/五集 N=4，checkpoint 可读、退出后资源正常                | 每个点 2,800 条响应、五集指标；检查原始 step19 对应第20次更新；恢复状态与收尾                                             |
| C16 | 启动前评分器格式和语义正确                                                | 旧 PRD scorer 对齐、正则/boxed/LaTeX 正负例、超时分类、规范化后 gold round-trip                                           |
| C17 | 启动前 loss、GAE、value target、FSDP 累积缩放正确                         | 独立小张量 loss/梯度对照、padding/prompt mask、clipping 两侧、截断与 rank 切分一致性                                      |
| C18 | 启动前 SGLang 推理加速已实际生效                                          | prefill/decode backend 读回、稳态 kernel trace、同输入正确性对照、tokens/s 与显存                                         |
| C19 | 启动前 Actor update 加速已实际生效                                        | 真实 loss→backward→optimizer 的 profiler，attention/GatedDeltaNet/conv 路径、warmup 与稳态耗时分开                        |
| C20 | 24 线性层/8 全注意力层的对应高效算子生效，训练与推理分别验证              | 两端独立 kernel 证据；FSDP 的 FLA/causal-conv1d 前后向、SGLang 对应实现；无未披露慢 fallback                              |
| C21 | 完整遍历清洗后 DAPO 一轮                                                  | shuffled ID digest、每个题目恰好消费一次且 n=4、尾 batch 覆盖、epoch_done；不靠 drop_last 丢末尾                          |

步骤定义：上游 global_step 从 0 开始，所以原始 0..4 对应验收 step 1..5。一个 step 必须包含真实带 reward 的 batch、非空正确
mask、有限 loss/梯度、实际 actor 和 critic 参数更新、权重发布成功。预热/日志行/PID 活着/只 backward 不 optimizer.step
不计数。梯度累积的多次 backward 也不能冒充多个 step。

C09 要求捕获可归属的执行时间重叠及版本链；权重发布时短暂屏障允许。max_head_offpolicyness=2 不能直接等同所有 token 实际滞后不超过 2：需核对
admission/in-flight 语义，输出真实滞后和任何越界原因，未解释异常不得通过。

每步保留正确/错误/不可解析/截断/基础设施失败数、token 数、actor/critic loss、value/return/advantage 范围、clip
fraction、ratio、梯度范数、版本、耗时与显存峰值。所有 NaN/Inf、样本丢失、伪造 reward
或未解释的零更新判失败。真实全错必须如实报告；不通过筛题制造上升曲线。EV 在 return 方差为零时标为 undefined，不伪造有效指标。

预算：清洗后题数记为 D，完整一轮包含 4D 条训练轨迹，更新数以 ceil(D/128) 为设计预期，再与实际 dataloader 长度和尾 batch 读回核对。若 D
约 17,156，则约 135 step；此为估算，新清洗行数未冻结。五步监督仅覆盖前 2,560 条轨迹。每次评测 700×4=2,800
条；若总135步，baseline+20/40/60/80/100/120+尾步共8次，即22,400条评测。明确记录全部生成 token/GPU
小时和被丢弃的预取成本；限制预取，不跨 epoch 消费第二轮。连续无进度30分钟触发诊断是建议阈值，不将长生成自动记为模型失败。

启动前必须通过 C01–C06、C07 的配置冻结部分、C08、C16–C20；启动后读回补齐 C07，前五步另通过 C09–C14，才继续 epoch。所有 C01–C21
通过才可宣布本阶段“异步 PPO 全量 DAPO 一轮完成”。参数展示与冻结为执行记录，不重复索要已给出的执行授权。Docker
按可确认对象完成限定范围清理并披露保留项，不为提高回收数字删除归属不明的第三方镜像。加速门槛需要实际所选高效实现的证据，训练与推理实现名可以不同。数据审计须分别保存
AIME24/AIME25/BeyondAIME 的扫描和改动数量。五步内不要求 reward/accuracy 提升；完整一轮有提升也不等于 SAO 已实现或已证明因果增益。

## 8. 结果与后续 SAO 边界

评测按每题4条响应输出 mean@4（4条中平均正确率）和 empirical pass@4（至少1条正确），分别计算五集不加权 macro；700题加权总体另命名。不能把
pass@4 写成 pass@1，也不能与旧 mean@16/pass@16 冒充同一预算。每题的4次采样使用可区分的随机种子，固定全局 seed 不等于4次重复同一输出。

交付物为：可重建 uv 环境、最小适配代码及测试、数据版本与 correction map、Docker 清理收据、启动前
loss/模型/加速证据、前五步监督记录及完整一轮真实异步 PPO 的结果。SAO 单独进入下一阶段，才加入 n=1、DIS、双 lambda、critic
更新比例等机制，重新制定验收，不在 PPO 验收时提前声称实现。

## 9. 源码与历史证据入口

- 固定 release：examples/math/gsm8k_ppo.yaml（4+4、n=4、decoupled
  默认）；areal/api/cli_args.py（PPO 默认、版本滞后定义）。
- loss
  与版本：areal/trainer/ppo/actor.py:229、areal/utils/functional/functional.py:452、areal/trainer/rl_trainer.py:677。
- critic 风险：areal/engine/core/model.py:7、areal/engine/fsdp_engine.py:1004/1065（Qwen3.5
  vision 路径未建 scalar head）。
- uv/编译差异：Dockerfile:114/120/160、pyproject.toml:141/235、uv.lock。
- 旧 PRD
  配置：/data_storage/yl_test/lgx/data-1/code/\_worktrees/verl/codex-math-prd-opsd/examples/on_policy_distillation_trainer/run_qwen35_4b_math_prd_epoch.sh。
- 旧数据逻辑：同仓
  examples/on_policy_distillation_trainer/prepare_dapo_prd_data.py（去重、冲突、teacher prompt
  过滤与难度抽样必须逐项区分）。
- 已确认 scorer case：同仓 docs/algo/math_prd_reward_audit.md；原证据
  /data_storage/yl_test/lgx/data-1/code/\_artifacts/verl/codex-math-reward-audit-fix/audit/。

## 10. 执行中确认的适配与冻结值

- 全量合格训练集为 17,157 题，五集 700 题；128 prompts × N=4，完整一轮 135 次联合更新，末批为 5 题/20 条轨迹。清洗逐行记录、输出
  SHA、17,857 条 gold roundtrip 均保留在数据产物目录。未使用原 8,192 题难度子集。
- 本地 CUDA13 compat 已在实际八卡通信和 kernel 正反向中读回；PyTorch 2.9.1+cu129 与项目隔离的 CUDA12.9 编译器配套。系统
  CUDA13 和旧 Verl 环境保留。FlashInfer JIT 使用本机编译器对应的 C++ runtime 和锁定 cuRAND wheel
  的头文件，均由本项目启动脚本绑定。
- 上游 AReaL 有四项有意覆盖的包元数据约束：SGLang 的 OpenAI 2.6.1→2.30.0、soundfile 0.13.1→0.12.1、TorchAO
  0.9.0→0.15.0，以及 Torch 的 cuDNN 9.10.2.21→9.16.0.29。因此原生 uv pip check
  非零；不得称为零冲突安装。check_dependencies.py 只认可这四个精确覆盖，另以本机实际训练/推理预检完成运行资格验证。
- Qwen3.5 scalar critic 复用预训练文本骨干，新增 score head；修复 FLA norm 按 config.dtype 单独创建 BF16
  参数造成的 FSDP 混合 master dtype。真实 8K critic 更新、score head 梯度/参数差值及 DCP 保存恢复已通过。
- Transformers 5.3 的 Qwen3.5 GDN/conv 路径不消费 packed cu_seqlens，必须每 microbatch 一条独立序列，避免跨题
  recurrent state 污染。FA2/FLA/conv 高效实现继续使用；批量与单条 logprob 一致性已通过。末批仅为满足四个 DP rank
  均分而展平题组，保留全部样本 ID、mask 与奖励。
- Docker 已按归档清单回收 10 个旧 Terminal-Bench 镜像/11 个标签，释放约 12.1 GiB；未删除容器或 volume。约 12.1 GiB
  的可恢复归档在数据盘，完整 SHA/清理前后读回见 docker-cleanup 目录。
- 协议按“启动前检查→前五步监督→完整一轮结束检查”执行。全局 agent-workflow 的 formal-run
  模板把所有高风险项（包括实验后结果）设为启动前要求，并要求新的目标绑定人工反馈，不能表达这个已授权分阶段流程。保留其真实未确认状态，不伪造 human_status
  或声称机械 formal-run gate 已通过；用户本任务明确执行授权是启动依据，所有业务检查仍必须在相应阶段通过。
