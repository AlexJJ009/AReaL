# τ² critic EV：代码口径、可恢复证据与离线诊断

审查对象是 `/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math`，读取时 HEAD 为
`890e5ee45d47d4dd52ce43814a8819270fdad91e`。下文 `文件:行号` 均相对此目录。先读 issue 与 README 16.14
的所有子节，再读代码；没有读取另一组 case study。没有 GPU、API、pueue 或训练目录写操作。只新增本目录的分析与离线脚本。

## 1. 最关键的勘误：常量 bias 不会降低这里报告的 EV

验证 EV 从一开始就是减去残差均值后的方差比，并非 $1-\mathrm{MSE}/\operatorname{Var}(R)$。当前实现见
`scripts/qualification/critic_pretrain.py:138`；同时读取
`git show 04be68ab:scripts/qualification/critic_pretrain.py`，历史 479/480 的聚合公式相同。因此
README 16.14.4 把 Airline 低 EV 主要归结为整体水平校准，不能从该 EV 指标本身推出；16.14.5 并非首次改为 bias-corrected
EV。

对轨迹 $i$，设有效 action-prefix token 数为 $n_i$，每个 token 的 target 都为 $r_i$，预测为
$v_{it}$，$N=\sum_i n_i$。JSON 聚合的是：

$$
b=\frac{1}{N}\sum_{i,t}(v_{it}-r_i),\qquad
\mathrm{MSE}=\frac{1}{N}\sum_{i,t}(v_{it}-r_i)^2,
$$

$$
\mu_R=\frac{\sum_i n_i r_i}{N},\qquad
\operatorname{Var}_{token}(R)=\frac{\sum_i n_i r_i^2}{N}-\mu_R^2,
$$

$$
\mathrm{EV}_{token}=1-\frac{\max(0,\mathrm{MSE}-b^2)}{\operatorname{Var}_{token}(R)}.
$$

所有 token 预测加同一个常数，只改变 bias 和 MSE，不改变 EV。一个任意水平的常数 predictor 在标签方差非零时 EV 都是 0；其 MSE
仍可能很坏，所以 level drift 对 SAO advantage 依然有害。正 affine 校准还会调整预测尺度，其提升可能说明尺度不对，不能全部归因于平移。某领域内
trajectory-mean 的 $R^2$ 较高，也不保证逐 token EV 高：trajectory 内散布会损耗 token EV。

令
$\bar v_i=\operatorname{mean}_t v_{it}$、$s_i^2=\operatorname{Var}_t v_{it}$，$w_i=n_i/N$，则：

$$
\operatorname{Var}_{token}(V-R)=\sum_i w_i s_i^2+
\operatorname{Var}_{w}(\bar v_i-r_i).
$$

第一项是同一轨迹内预测波动，第二项是 trajectory-mean 的残差方差。这个分解完全可从既有 JSON 得到，能直接区分“排名有信号、逐 token
噪声大”和“轨迹均值本身没有排序能力”。

此外，报告的 `reward_mean` 是等轨迹成功率，`target_mean` 是 token 加权成功率，两者一般不同。`macro_mse` 是等轨迹的 token
MSE 平均，既不是等轨迹均值预测的 MSE，也不是 macro EV（`scripts/qualification/critic_pretrain.py:156`）。拿等轨迹
$p(1-p)$ 比较 token MSE 会改变比较口径，必须用 JSON 的 `target_var` 或明确重算 macro 指标。

## 2. Target 与 mask：是在每个 agent action 之前预测 terminal reward

`scripts/tau2/train_critic.py:545` 将 rollout `loss_mask` 左移一位，最后一位清零。若原 mask 在 token $j$
处标记该 token 为 agent 输出，则 critic 监督位置为 $j-1$，即看到 prefix、尚未看到该 action token。这一位移与在线
advantage 路径一致（`areal/trainer/ppo/actor.py:226`）。它不是对 user/tool token 直接计算 loss；下一个
agent token 的 prefix 可以已经包含刚发生的 user/tool event。

验证 target row 直接把整个序列填为 terminal reward，再用 shifted mask
选有效位置（`scripts/tau2/train_critic.py:554`）。在线训练来自 actor 的 GAE return：gamma、critic lambda
均为 1，KL 为 0，terminal reward 不归一化，τ² 明确
`bootstrap_mask=False`（`examples/tau2/agent.py:851`），所以 action token return 退化为相同的
terminal reward，包括以 0 评分的截断 episode。GAE target 写入见 `areal/trainer/ppo/actor.py:274` 和
`areal/trainer/ppo/actor.py:354`。未发现“把 user token 当作 supervised action”或 validation
target 比 train target 多一位的证据。

损失是 masked plain MSE、token_mean，无 value clipping。实现 `areal/trainer/ppo/critic.py:165`
显式对 masked squared error 求和除有效 token 数；前向的 score 无 sigmoid。因而一条很长的失败/成功轨迹在训练目标与 token EV
上都更重。这不是指标 bug，但长轨迹常有更多排查、沟通和重复，长度相关的成功率与域差异可能同时影响结果。【猜测，需从 rollout 分层验证】

把同一 terminal label 重复给很多 prefix 不等于拥有同样多独立样本；早期 prefix 的 conditional expected reward
难估，后期 prefix 有更多 outcome 线索。bootstrap 应按轨迹/task cluster 抽样，不能按 token 独立抽样。

## 3. JSON 足以恢复什么，PT 不足以恢复什么

`scripts/tau2/train_critic.py:590` 的 `_metric_row`
保留每条轨迹：`n_tokens`、`target`、`prediction_mean/min/max`、`error_sum`、`sse`、`target_sum`、`target2_sum`、task/domain/episode
identity 与 truncation。有效 token 预测向量在计算这些统计后没有保存。

这些统计允许 CPU 精确恢复：

- 每域/整体 token EV、bias、MSE；trajectory cluster bootstrap 的相同统计。
- 以每条 trajectory mean 为预测的等轨迹或 token 加权 EV、相关、affine fit；校准若在同一批 test 上拟合只能作为诊断，不是
  unbiased performance。
- 每条轨迹的预测方差：$s_i^2=\mathrm{sse}_i/n_i-(\mathrm{error\_sum}_i/n_i)^2$。数值舍入的小负数可裁到 0。
- 预测二阶矩：$\sum_t v_{it}^2=\mathrm{sse}_i+2r_i\mathrm{error\_sum}_i+n_i r_i^2$。可恢复 pooled
  mean/variance 和 outcome covariances。

不能恢复：第一个/最后一个 token 的预测、事件前后预测跳变、时间次序、turn 内与 turn 间的方差分解。min/max 也没有位置。

fixed-validation PT 是 `torch.save(rollouts)`（`scripts/tau2/train_critic.py:1027`），存的是
model 输入与 reward 等原始 rollout，没有 checkpoint-dependent values。验证前向 `compute_values` 只把
values 转 CPU，然后写 summary rows（`scripts/tau2/train_critic.py:676`、`:719`）。现有训练 rollout
dump 保留 `input_ids/attention_mask/loss_mask/action_origin_mask/turn_ids` 等 replay
fields，见 `areal/infra/workflow_executor.py:1112`；JSONL dump 是 tensor replay 和可读
segments，不能替代缺失的 value sequence。

事件分析可行：用一个既有 checkpoint 对固定 PT 重做一次前向，保存所有 shifted masked prefix values 与 positions，把
shifted turn IDs 对齐，再人工/规则标注误工具调用、错误信息、policy violation。每次 event 后应观察第一个可见该 event 的
agent prefix 的 value 变化，而不是把事后观测放回 event 前。已写的脚本保存完整 value sequence、first/last values 和
shifted turn IDs；精确 message-span 标签仍需结合 dump segments/原 episode message 文本，不能仅靠 turn ID
给事件定性。

## 4. 真正的优化预算、冻结范围与初始化

`areal/trainer/rl_trainer.py:1010` 每个 critic-only outer step 调一次
`critic.ppo_update`；`areal/trainer/ppo/critic.py:96` 按 `ppo_n_minibatches=1`
组织训练，`areal/engine/fsdp_engine.py:1054` 在 microbatch 的前后向之后只做一次 `optimizer_step`。4 个 DP
rank 与任意 packing microbatch 不会把 94 outer steps 变成 376 次独立 optimizer update。验证虽做 backward
求 grad norm，但不 optimizer step（`scripts/tau2/train_critic.py:697`、`:731`）。非有限 grad 会跳过
update（`areal/engine/fsdp_engine.py:917`），应以 run 里的成功标志/optimizer
counter核对实际成功数；本审查没有自行声称所有 94/80 次均成功。

配置要求一个 PPO minibatch、constant LR、零 warmup、冻结 attention、plain token
MSE（`scripts/tau2/train_critic.py:333`、`:376`），因此现有 `train_critic.py` 加 override 取消冻结会先被
validator 拒绝。现有入口也不是 offline replay CLI：即使 validation 使用缓存，也仍构造 actor、SGLang、user
workflow 并采集训练 rollout。不能把现有在线命令标为“无 API 的便宜复现”。

attention 冻结是把路径包含 `self_attn` 和 `linear_attn` 的整个 module 所有参数设
`requires_grad=False`，Qwen3.5 同时匹配两者，Qwen3 只匹配
`self_attn`（`areal/engine/fsdp_utils/critic_freeze.py:22`、`:81`；`areal/engine/fsdp_engine.py:619`）。这会冻结这些
attention module 内的 projections、Gated DeltaNet 相关参数等，而非单独冻结 attention
matrix；FFN/MLP、module 外的 norm、embeddings 与 score 是否更新由其各自 requires_grad 决定，并非全 backbone
冻结。attention 运算还会处理经过可训练 MLP 改变的输入，所以不等于完全无法学习跨 token 事件。是否限制了所需信息路由，需
ablation，不能据此直接判因果。

Adam 配置实际构造
`torch.optim.AdamW`（`areal/engine/fsdp_engine.py:1543`），weight_decay=0，master weights
float32；不是 bf16 小步全部被舍入的设置。grad norm clip=1 与 LR=5e-6 是不同控制。Adam 对稳定的整体梯度尺度近似不敏感，但
clipping 随 batch 动态改变会影响一二阶动量，不能从“Adam”推断 clipping 永远无影响，也不能把每步每个参数移动严格写成 LR。94 步的
$94\times\mathrm{LR}$ 仅是粗略量级，不是严格位移界。

head variant `examples/tau2/critic_head.py:47` 将 score.weight 清零、score.bias 设置为指定均值，head
第 0 步所有 token 输出 bias。head 与 body 在 `head_lr=None` 时共用 optimizer LR；可选 head 参数组见
`examples/tau2/critic_head.py:70`。从有 bias 标记的 checkpoint 或 sealed artifact
起步时不重复清零（`:62`），而 bias 标记随 model config 传播。零 head 的首个 backward 对 backbone 的梯度为零，对 head
有梯度；之后 head 非零才有 backbone 梯度。这个只损失首步 backbone 更新，不足以单独解释 80–94 步结果。【机制已验证于结构，因果判断未实验】

普通 Qwen3.5 head 的初始化是 std 取 initializer_range（默认 0.02）且无 bias；专门 adapter 用最终 hidden
state 做线性 score，见 `areal/models/transformers/token_critic.py:56`、`:105`、`:111`。mean-init
解决初始常数水平与随机输出散布；它也会移除 trajectory 内随机 head scatter，从而改善 EV，不能把这一效果全部称为 bias 校准。

数据拼接确实是 pack rows 后 official rows（`scripts/tau2/train_critic.py:459`），是否按此顺序训练取决于
sampler shuffle。sampler 构造 `areal/utils/dataloader.py:56` 消费配置 shuffle；当前严格 trainer 会
`set_epoch` 并在 recover 时也设 seed（`scripts/tau2/train_critic.py:1141`、`:1144`）。当前源码已经包含
README 中 resume 的修复：`skip_recovery_weight_update=True`（`:1125`），不应把历史 506
的错误当作当前代码行为。历史实验仍需用 frozen worktree 或提交的实现解释。

## 5. 已备好最便宜的 executable 离线实验

本目录 `offline_probe.py` 新增三个命令：`extract`、`fit`、`smoke`，没有 online trainer、rollout server
或远端模型调用。extract 在本地读取模型和既有 rollout，强制 HF offline；支持 τ² replay JSONL 目录或 trusted fixed
PT。fit/smoke 只用 NumPy，不导入模型或调用 CUDA。

extract 使用 repository 的 Qwen3.5 text backbone adapter，直接取 `last_hidden_state`，避免构造 full
vocabulary logits。每条 trajectory 对所有 shifted action-prefix positions 均匀无放回采 32 个，每个样本权重
$n_i/k_i$，近似原 token objective：期望等于原 token 总损失，但单次 sample 有额外方差。`--samples 0` 保留全部
positions 可做精确 token objective，显著增大 feature 内存与 CPU ridge 成本。hidden size 从模型实际输出取得，不硬编码
2560。提取还保存全部 action-prefix values，事件分析不受 32 点采样限制。

fit 按 source task identity、domain 分层划 train/dev，同一 task 的多个 epoch episode不能跨
train/dev；feature mean/scale 只在 train 计算，ridge alpha 只看 dev MSE，fixed test 只报告。默认全域
ridge 加 intercept，另报告 test 两域 EV；可另做分域 ridge，以免 pooled 正 EV 只是 domain identity。标签置换按
episode、在各 domain 和 train/dev split 内实施；这会保留 token 内 label
一致性。它破坏真实标签相关信号，同时提供数据污染/过拟合对照。

memorization control 取每域每标签至多 4 个完整 episode，并保留其 sampled prefixes，只有两个 outcome
标签都存在才定义；还报告相同输入上 episode-permuted labels 的拟合误差。该控制衡量 head 容量，不能当作 generalization
或“容易学真实信号”的证明。small GPU smoke 只有 2 episodes，可能缺少失败，不能拟合/判断 EV，目的只是确认 loader、mask 和产物形状。

下面命令只作为待排队 proposal，本次没有执行 GPU 部分。路径均使用已有 mix-r2 数据；验证 PT 与训练 JSONL来自同一 run，避免把另一个 run
的固定验证集混进来。

```bash
SRC=/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math
CASE=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-ev-case-study/codex
RUN=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261013-critic-head/critic-head-qwen35-mix-r2
BASE=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
TRAIN_ROWS=$RUN/logs/root/tau2-critic-head/critic-head-qwen35-mix-r2/rollout
TEST_ROWS=$RUN/tau2-critic-fit-evidence/fixed-validation-rollouts.pt
PYTHONPATH=$SRC CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$BASE" --rows "$TRAIN_ROWS" --limit 2 --out "$CASE/base-smoke.npz"
PYTHONPATH=$SRC CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$BASE" --rows "$TRAIN_ROWS" --out "$CASE/base-train.npz"
PYTHONPATH=$SRC CUDA_VISIBLE_DEVICES=0 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" extract --model "$BASE" --rows "$TEST_ROWS" --out "$CASE/base-test.npz"
OPENBLAS_NUM_THREADS=8 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit --train "$CASE/base-train.npz" --test "$CASE/base-test.npz" --out "$CASE/base-ridge.json"
OPENBLAS_NUM_THREADS=8 "$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit --train "$CASE/base-train.npz" --test "$CASE/base-test.npz" --permute --out "$CASE/base-ridge-permuted.json"
```

先从 2 条的实测 GPU 时间与显存外推全量，不承诺分钟数。【成本估计】一张卡、每条只做一次 backbone forward，无 backward、无 SGLang、无
user/judge API；每模型的约 1280 训练 + 60 验证 episode前向。32×2560×4 bytes 约 0.31 MB/episode，1340 条约
0.44 GB feature 浮点数据未压缩，ridge 转 float64 还需额外内存；模型 bf16 text weights约 8 GB，长序列
activation/workspace 额外，80GB 卡一条 32K 预计可容纳但未跑过。抽样减小 feature 存储，不减少 backbone forward。

若只要事件曲线，不需要全量训练抽特征：指定已有最终 critic HF checkpoint，对 60 条 fixed PT 做一次 extract。checkpoint
应读其 `tau2-critic-fit-evidence/summary.json` 的 `final_critic_checkpoint`（mix-r2尚未生成时不能假造
final 路径）；已完成 v2 的已存在 checkpoint 是
`critic-head-qwen35-v2/checkpoints/root/tau2-critic-head/critic-head-qwen35-v2/critic/epoch1epochstep46globalstep93`。其
fixed PT 必须换成 v2 同 run 的缓存。带 bias marker 的 checkpoint会先验证 safetensors keys 实际包含
score.weight/score.bias，防止 silent random head；base checkpoint 的 score 本来随机，base
`.values.json` 不可当 trained value curve。

提取产物记录输入 SHA256、model config/index 等 JSON SHA256、权重文件名及大小；这不是完整权重内容哈希。NPZ保留 episode
cluster、task、domain、prefix positions、action-token count，支持之后 cluster CI。正式报告还应将
checkpoint路径/提交/run 与输入 manifest绑定。需要先核对 rollout dump 中是否有最终 rejected 但仍 dump 的
records：executor 在 acceptance finalization 前
dump（`areal/infra/workflow_executor.py:1537`），不能默认目录里的每一条都参与了训练。当前 script不重建
accepted-step subset，只作为基于现有 recorded trajectory 的离线监督诊断。

GPU extraction 未执行，因此 flash-attention/FLA 环境兼容性、数值重现与 memory/time 未验证。CPU执行
`python offline_probe.py smoke` 通过：known synthetic signal final EV=0.62759；episode-label
permutation=0.03353；将所有 feature 置零 EV≈0，原 positive-EV assertion会失败。这个 smoke验证 ridge与
split/control 逻辑，不验证真实模型读取。第一次随机化“破坏 feature”控制在有限 test上偶然仍有 EV=0.112，因此改用确定的全零
feature失效检验；没有把偶然结果包装成通过。

## 6. 如何读实验，下一步哪项最有信息

1. CPU先从既有 JSON 分解 within-trajectory scatter 与 trajectory-mean residual，并做 trajectory
   cluster bootstrap。成本最低、没有新 API/GPU；能够纠正“只是 bias”的解释。
1. 已训练 critic 只对固定 60 trajectories前向，取 first/last/turn/event values；与 JSON mean/SSE 在导出
   bf16误差内核对。若 late prefix明显能区分而早期不能，才有证据将限制定位到前缀可见信息或 token目标权重。
1. base 与 final backbone各做一次 feature extraction，重复 CPU ridge 正则/置换对照。若 frozen ridge 的
   heldout per-domain EV显著优于原 score，说明既有表示中有线性可读信号、当前训练/head objective优化尚有改进空间；若
   ridge失败，只排除了这个有限样本、线性、指定表示的诊断，不能证明“没有 learnable signal”。任务难度迁移、representation、token
   sampling、ridge选择和少数失败仍可解释失败。
1. 在完全相同 stored train rollouts 上做多 epoch offline MSE、LR与冻结 attention ablation需要另一段
   offline training harness；现有 online CLI不能完成且 validator强制冻结。没有在本次偷偷修改
   repo或提出不存在的可执行训练入口。若要判定优化瓶颈，先做同集 overfit：true labels与 episode-permuted
   labels都能拟合说明容量足够，只有 heldout true labels显著正 EV才证明可泛化。

Retail 首 token 的 task-only ceiling低，限制早期 task identity baseline；它不限定看到事件后的 conditional
value，也不是 token平均 EV的上限。每域只有20/40固定 trajectories，token数再多也不会增加独立 outcome数。应把 per-domain
cluster CI、失败数量、event后预测和训练分布差异并列，而不是把某条零附近 EV曲线直接归因为 user randomness或冻结 attention。

## 7. CPU Adam 补充：同一 cached feature 上精确区分 head LR 与更新次数

另新增 `head_adam.py`，输入上述 train/test NPZ；不导入 transformer、不做 backbone forward、不调用 CUDA。它在
raw hidden features 上训练 fp32 linear head，weight=0、bias默认0.78，torch Adam（也可选
AdamW）、weight_decay=0、betas=(0.9,0.999)、eps=1e-8、clip grad norm=1。它保留一个 minibatch 16 个完整
episode，每个 episode所有 sampled prefix一起训练，以 cached $n_i/k_i$权重计算 token MSE；尾 batch保留。与
ridge共用完全一致的 seed42、按 task分层的 dev划分；限制一个 domain时仍先创建完整 split，以免 Retail单独拟合时悄悄换 dev
tasks。

每个 epoch后在固定 train/dev计算 MSE、EV、bias，epoch0初始化也计入候选。只根据固定 dev MSE选
checkpoint，test直到选择结束才计算一次，并报告各域。JSON保留 epoch曲线、每 epoch累计的实际成功 optimizer updates、preclip
grad norm和 bias参数；`.head.pt` 保存选中的 fp32 head与源 NPZ SHA256。非有限
gradient会报错，不能把跳过更新当作完成。若最优是 epoch0也照实报告，避免强迫挑一个变差的已训练 checkpoint。

以下六个命令在 feature extraction完成后只花 CPU，不需额外 GPU或 API。每个命令从同一个初始化开始、相同 sampler RNG顺序，2/10/20
epoch分别是相同训练路径的不同长度；同一个 LR前两个 epoch应完全一致。新增训练任务/epoch带来的抽样噪声被移除，诊断 head本身的优化是否走得太慢。

```bash
for lr in 5e-6 5e-5; do
  for epochs in 2 10 20; do
    "$SRC/.venv/bin/python" "$CASE/head_adam.py" \
      --train "$CASE/base-train.npz" --test "$CASE/base-test.npz" \
      --lr "$lr" --epochs "$epochs" --bias 0.78 --batch-episodes 16 \
      --out "$CASE/base-head-adam-lr${lr}-ep${epochs}.json"
  done
done
"$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit \
  --train "$CASE/base-train.npz" --test "$CASE/base-test.npz" \
  --domain airline --out "$CASE/base-ridge-airline.json"
"$SRC/.venv/bin/python" "$CASE/offline_probe.py" fit \
  --train "$CASE/base-train.npz" --test "$CASE/base-test.npz" \
  --domain retail --out "$CASE/base-ridge-retail.json"
```

域内 Adam也支持相同的 `--domain airline|retail`；这会让 batch、更新次数与 pooled版本不同，需读 JSON计数。低/高
LR结果比较应先看同预算 train/dev曲线，测试的配置间比较属于探索，不能用 test挑配置再称为独立最终分数。可以只根据 dev选一个配置，再正式引用其 test结果。

该实验精确复现的是 **cached sampled-token目标、raw固定表示上的 head
Adam优化**，不是在线原训练的端到端复现：没有更新FFN/MLP、norm或embedding，没有重新采样用户和actor，也不是全部token目标（除非
extraction设 `--samples 0`）。因此同集head训练仍差、ridge同集拟合好，支持 head优化/尺度瓶颈；head
train拟合好但dev/test差，支持样本量/泛化/分布差异；head未学会不能单独证明冻结attention或user随机性为因果。20
epochs的CPU重复是existing episodes反复训练，不能当作增加独立outcome证据。

CPU synthetic
smoke使用两域、每task四个prefix、成败标签皆有的已知线性信号，而非单个常量episode；`python head_adam.py --smoke`通过，test
EV=0.999925、实际80 updates、dev选择epoch17。把train和test
features都置零后EV=0，原正EV判据会失败。结果保存到`head_adam_smoke.json`。这验证gradient/update计数、dev选择和known-signal拟合；真实
cached features的CPU实验尚未执行，因为GPU extraction被明确留作proposal。
