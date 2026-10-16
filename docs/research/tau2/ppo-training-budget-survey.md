# PPO/GRPO 训练预算调研：related-work 中做 RL 训练的工作 vs. 我们的 run

调研日期：2026-10-15。范围：只包含 `related-work.md` 和 `critic-accuracy-and-ppo.md`（远端
`docs/research/tau2/` 下）里出现、且实际做 RL 训练的工作；另加 2 个 anchor（标明在清单之外）。 数据来源：arXiv HTML
的训练设置/附录，以及官方 GitHub 配置。

验证等级：**V** = verified（给出来源 + 章节/配置键）；**E** = estimated（由已验证数字推算，写明公式）；**U** =
unverified；空白 = 论文未报告或没取到。 **复核（Claude，2026-10-16）**：以下数字已对照原文或配置逐字核对：FACA 120 步 ×
128、lr 1e-6、KL 1e-3、只用 τ Airline+Retail train split（arXiv HTML App. D.1/E）；SAO batch
128、actor 1e-6、critic 5e-6、critic 10 步 warmup、K=2、约 400 步后与 GRPO 分叉（§4.1/4.2）；GiGPO
actor 1e-6、critic 1e-5；AReaL 官方 tau2 配置 500 步、lr 1.7e-5（1.7B/8B）和
5e-6（MoE）、`ppo_n_minibatches: 1`、数据为官方 `tau2/train`（`origin/main` 的
`examples/tau2/config_*_airline.yaml`）。其余标 V 的数字只经过下面说的摘要读取。

方法局限：WebFetch 是小模型对页面的摘要，不是我逐字读原文。标 V
的数字都带有章节或配置键，但用于关键结论的数字（SAO、FACA、GiGPO、T1）引用前建议回原文再核一次。若 WebFetch
提示页面被截断，附录里的数字可能没取到，我一律留空，没有猜。

## 我们的 run（对照基准）

Qwen3-4B-Instruct-2507，PPO（critic，GAE γ=λ=1），actor lr 1e-6，critic lr 1e-5，48 optimizer
steps，每步 128 episodes、1 个 minibatch（每 batch 1 次 actor 更新），8 epochs × 747 tasks，约 6144
episodes，eval 无提升。

- actor 梯度更新总数：48
- lr × updates（Σlr，忽略 scheduler）：48 × 1e-6 = 4.8e-5

## 1. 总表

### 1a. 带 user simulator 的 tau / tau2 类 RL

| 工作                                                                                                | 模型                                                      | 算法                                                                | 环境 / user                                                                           | steps                                                                                                     | batch（prompts × samples）                                                      | 每 batch 更新数                                                    | actor lr / critic lr                                          | 总 episodes                              | 报告的 gain                                                                                         | 验证                                 |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **AutoForge** [2512.22857](https://arxiv.org/html/2512.22857)                                       | Qwen3-Thinking-30B-A3B                                    | ERPO（GRPO 的环境级扩展）+ DAPO dynamic sampling                    | 自合成 10 个环境 1078 tasks，评测 tau2；训练 user = GPT-4.1                           |                                                                                                           | 32 × 8 = 256（§4.1）                                                            |                                                                    |                                                               |                                          | tau2 Retail 74.8 / Airline 62.0 / Telecom 76.3（Table 1；doc 记录 58.8/58.0→74.8/62.0）             | V（§4.1）；steps、lr、更新数均未报告 |
| **FACA** [2608.17499](https://arxiv.org/html/2608.17499)                                            | Qwen3-8B、14B（先在 MUA-RL 的 1,580 条轨迹上 SFT 冷启动） | 组相对优化 + clipped ratio，U2U 过程优势（λ=0.5）加到终局 advantage | tau-bench airline + retail train split；user = 冻结 DeepSeek-V4-Flash                 | **120**（§4.2，所有头条结果用 step-120 checkpoint）                                                       | 8B：16 × 8 = 128（App. D.1，App. E Table 6）；14B 未报告                        | 未报告                                                             | 8B 1e-6，KL 1e-3 / 无 critic                                  | **15,360 = 120 × 128**（App. E Table 6） | 9 个 tau 域平均 8B 34.66→40.57（+5.91），14B 42.51→52.73（+10.22）（§5.1，3 seed 均值）             | V（App. D.1/E）                      |
| **IAPO** [2608.24588](https://arxiv.org/html/2608.24588)                                            | Qwen3-4B                                                  | GRPO 式（只改 trajectory→token 的 advantage 映射），verl/FSDP       | tau2 训练集 178 tasks；训练 user 未报告                                               |                                                                                                           | 每 prompt 16 samples，max 30 turns（prompts 数未报告）                          |                                                                    |                                                               |                                          | tau2 Qwen3-4B：IAPO 38.40 ± 1.63，GRPO 34.79 ± 1.05，GiGPO 37.22 ± 1.04（Table 1）                  | V（Experimental Setup）；其余空白    |
| **AReaL-SEA (Gao et al.)** [2601.22607](https://arxiv.org/html/2601.22607v3)                        | Qwen3-30B-A3B-2507、Qwen3-235B-A22B-2507                  | GRPO（无 critic）+ dynamic filtering                                | tau2 airline/retail/telecom；训练 user = SFT 过的 user model，评测 GPT-4.1            | 未报告                                                                                                    | 30B：8×16~8×64（128~512）；235B：16×16 = 256（App. 1）；Table 4：8×64 优于 8×32 | 未报告                                                             | 30B 5e-6；235B 1e-5；constant（App. 1）/ 无 critic            | 未报告                                   | 30B Airline 60.0→70.5，Retail 69.1→75.0，Telecom 85.4→95.6；235B Airline 64.0→73.0（Table 1）       | V                                    |
| **MUA-RL** [2508.18669](https://arxiv.org/html/2508.18669)                                          | Qwen3-8B/14B/32B (non-thinking)                           | GRPO，KL β=0.001                                                    | 115 retail + 50 airline tasks（tau-bench）；训练 user = GPT-4o-2024-11-20             | 未报告；E：25 epochs × ⌈165/32⌉≈5 ≈ 125                                                                   | batch 32 × 8 rollouts（§4.1，25 epochs）                                        | 未报告                                                             | RL lr 未报告                                                  | E：165 × 25 × 8 = 33,000                 | tau2 Airline 8B 12.5→19.0，14B 14.8→38.0，32B 23.5→45.4（Table 1）                                  | 设定 V（§4.1）；steps/episodes E     |
| **Fission-GRPO** [2601.15625](https://arxiv.org/html/2601.15625v2)                                  | Qwen3-1.7B/4B/8B                                          | GRPO + fission 阶段                                                 | tau/tau2；评测 user = GLM-5，训练 user 未说明                                         | tau 部分未报告（BFCL 消融 234 updates）                                                                   | 8 × 8 = 64（§4.1）                                                              | 未报告                                                             | 1e-6，cosine warmup / 无 critic                               |                                          | Qwen3-4B TAU1 Retail 20.0（GRPO）→37.4（§4.2, Table 1）                                             | V（§4.1）                            |
| **verl-recipe tau2_airline**（PR #121）[链接](https://github.com/verl-project/verl-recipe/pull/121) | Qwen2.5-7B-Instruct + LoRA                                | GRPO                                                                | tau2 airline，30 train / 20 held-out；hosted 70B user                                 | 20                                                                                                        | 未给                                                                            | 未给                                                               | lr=1e-4；PR 称默认 4e-6/2e-5 时曲线是平的                     |                                          | held-out mean@4：0.275→0.5625、0.375→0.55；评测噪声 ±0.05                                           | V（PR 描述，非论文）                 |
| **SAO** [2607.07508](https://arxiv.org/html/2607.07508)                                             | Qwen3-30B-A3B-Thinking-2507（policy 和 value 同源）       | 单 rollout + 学习的 critic，Skip-Observation GAE，双边 IS           | 数学 TIR、SWE-Bench Verified（OpenHands）、写作风格任务；无 tau user，judge = GLM-4.7 | 摘要称可稳定训练 1000 步；出现的节点 ~160（GRPO 崩溃）、~400（与 GRPO 分叉）；主 run 总步数未明确（§4.2） | **128 × 1**（group size 1）；GRPO 对照 16 × 8（§4.1）                           | critic 每 batch **K=2** 次 value update（§3.2/4.1）；policy 未报告 | actor 1e-6 / **critic 5e-6**；critic **10 步 warmup**（§4.1） | 未报告；E：128 × 400~1000 = 51k~128k     | AIME2025 97.3 vs GRPO 84.2；SWE-Bench Verified 29.8 vs 27.0（GRPO w/DIS）vs 23.0 base（Tables 1/2） | V（§4.1）；总步数 E                  |

### 1b. 其他多轮 agent / 环境交互 RL

| 工作                                                                  | 模型                                                          | 算法                                                          | 环境 / user                                                              | steps                                                      | batch                                                          | 每 batch 更新数                                                                                       | actor lr / critic lr                                                                                                      | 总 episodes                       | 报告的 gain                                                                             | 验证                                     |
| --------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------- |
| **GiGPO** [2505.10978](https://arxiv.org/html/2505.10978)             | Qwen2.5-1.5B/7B-Instruct                                      | GiGPO（无 critic）；PPO 为对照                                | ALFWorld、WebShop；无 user                                               | **150 iterations**（App. E.1）                             | 16 groups × 8 = 128 envs（App. E.1）                           | mini-batch 256（ALFWorld）/ 64（WebShop）；epochs 未报告；mini-batch 与 128 envs 的单位关系论文没说清 | 1e-6 / critic 1e-5（仅 PPO 对照）（App. E.1）                                                                             | E：150 × 128 = 19,200（论文未给） | 比 GRPO：1.5B ALFWorld +13.3%、WebShop +10.6%；7B +12.6% / +9.1%（§5.2）                | V（App. E.1）；总 episodes E             |
| **Luck Is Not Skill** [2609.24144](https://arxiv.org/html/2609.24144) | Qwen3.5-2B + LoRA r32                                         | GRPO（TRL），DAPO token-level loss，无 KL                     | 模拟 back-office 工具环境，13 tools，2,000 tasks；无 user                | **100**（§5.3）                                            | 24 × 8 = 192（§5.3）                                           | 每 generation batch 1 次 optimizer step（§5.3/App. C）                                                | 1e-5 / 无 critic                                                                                                          | 19,200 = 24 × 8 × 100（§5.3）     | 工具故障下最终成功率 66.8% vs 61.8%（+5.1pp），但未达到预注册学习曲线判据（Table 3）    | V                                        |
| **P-GRPO** [2603.10009](https://arxiv.org/html/2603.10009)            | Gemma-2B、Qwen3-1.7B、Qwen3-8B                                | P-GRPO（按偏好组 running mean/std 归一化）                    | MovieLens-1M、合成 persona review、Goodreads、KGRec；无 user simulator   | 未报告                                                     | 未报告                                                         | Algorithm 1 有 T epochs，未给值                                                                       | lr 被 sweep，值未报告                                                                                                     |                                   | MovieLens Qwen3-8B top-1：65.77% vs GRPO 63.79%（§4.2.1）                               | V（§4.1.2）；训练预算全空白              |
| **RAGEN** [2504.20073](https://arxiv.org/html/2504.20073)             | Qwen2.5 0.5B（符号任务）、3B（WebShop）                       | PPO 或 GRPO（GAE γ=λ=1）                                      | Bandit / Sokoban / FrozenLake / WebShop；无 user                         | 100–200（§3.2；WebShop 100，App. C.2）                     | 8 × 16 = 128（§3.2）                                           | update batch 32（App. C.2）；epochs 未取到                                                            | lr 未取到（附录前半缺）                                                                                                   | E：128 × 100~200 = 12.8k~25.6k    | 0.5B Sokoban 20.7%、FrozenLake 21.5%；PPO 在 Bandit/Sokoban 比 GRPO 稳（§4.2, Table 5） | 设定 V；lr 空白                          |
| **EVPO** [2604.19485](https://arxiv.org/html/2604.19485)              | Qwen2.5-3B-Instruct（Sokoban/FrozenLake/WebShop），7B（MATH） | PPO + EV 门控（EV ≤ 0 退回 batch 均值 baseline），RAGEN2/verl | Sokoban、FrozenLake、WebShop、MATH；无 user                              | Sokoban 500；FrozenLake 200；WebShop 100（App. C Table 2） | 未报告                                                         | 未报告                                                                                                | actor 4e-6 / 1e-6 / 5e-7，critic 1e-5 / 1e-5 / 5e-6（Sokoban / FrozenLake / WebShop；App. C Table 2，每任务单 seed 调过） |                                   | 未报告具体数字                                                                          | V（App. C Table 2）；MATH 行被截断，留空 |
| **ArCHer** [2402.19446](https://arxiv.org/html/2402.19446)            | GPT-2 actor（~100M）；另试 Mistral 7B                         | 层级 off-policy actor-critic                                  | Detective / 20Q / Guess My City / WebShop；LLM oracle                    | iterations 未指定（App. Table 3）                          | 每 iteration 128 轨迹（WebShop 512），batch 256，replay 10,000 | actor 3 次 + critic 50 次 / iteration                                                                 | actor 3e-4（20Q 3e-5）/ critic 6e-4                                                                                       |                                   | 样本效率约 100×：PPO >100k samples vs ArCHer \<1000（正文）                             | V；GPT-2 的 lr 不能与 4B 全参数比        |
| **SWEET-RL** [2503.15478](https://arxiv.org/html/2503.15478)          | Llama-3.1-8B-Instruct（70B 另测）                             | 离线：非对称 critic + DPO                                     | ColBench（backend / frontend）；LLM 模拟人类协作者                       | 按 epoch：actor 1 epoch，critic 4/8（Table 5）             | actor / critic batch 8；15k / 6k 离线轨迹                      | –                                                                                                     | 2e-7 / critic 2e-7                                                                                                        | 离线，不适用                      | backend 成功率 40.4 vs 34.4（multi-turn DPO，Table 2）                                  | V；不是 on-policy PPO                    |
| **SPA-RL** [2505.20732](https://arxiv.org/html/2505.20732)            | Llama-3.2-3B-Instruct                                         | PPO + GAE（用 fused 过程 reward）；GRPO/RLOO 为对照           | WebShop（1,938 train）、ALFWorld（2,851）、VirtualHome（4,920）；无 user | RL 只训 1 epoch（App. D），步数未报告                      | 未报告                                                         | 未报告                                                                                                | RL lr 1e-5（actor/critic 不区分）（App. D）；LoRA                                                                         |                                   | 论文 Table 1/3 未取到具体数                                                             | V（App. D）；多数空白                    |

### 1c. 带 critic 的 PPO 在单轮推理上的预算（critic 文档中引用，用于 lr / warmup 参照）

| 工作                                                        | 模型                          | 环境                                                                                 | steps                                                                        | batch                                      | 每 batch 更新数                                                | actor lr / critic lr                                                        | critic warmup                                                     | gain                                                                | 验证                     |
| ----------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------ |
| **T1** [2609.11042](https://arxiv.org/html/2609.11042)      | Qwen3.5-122B-A10B（MoE）      | 终端 agent（Terminus-2 / Harbor），无 user；PPO + 独立全尺寸 critic，γ=λ=1，clip 0.2 | 最优 checkpoint step 110（§6.2）；总步数未报告；3 epochs on T1-15k（§1/2.4） | 512 trajectories（一任务一条）（§4.4/6.2） | 未报告                                                         | actor 1.0e-6 / **critic 1.5e-5**（§4.5，15×；Intro 写 30×，论文内部不一致） | 在 TMax-15k 上 critic-only **1 epoch**，之后才更新 policy（§4.5） | Terminal-Bench 2.1：43.8（base）→49.4（SFT）→64.0（RL）（§6.2/6.3） | V                        |
| **VC-PPO** [2503.01491](https://arxiv.org/html/2503.01491)  | Qwen2.5-32B                   | AIME 等（长 CoT）                                                                    | 总步数未报告                                                                 | 未报告                                     | 未报告（Alg.1 只有 E、M 符号）                                 | 1e-6 / 2e-6（§4.1）                                                         | value 预训练 50/100/150 步，最优 **100**（Table 3）               | AIME 8k：5.6→41.9；16k：GRPO 38.9→VC-PPO 48.8（Tables 1/2）         | V                        |
| **VAPO** [2504.05118](https://arxiv.org/html/2504.05118)    | Qwen2.5-32B base              | AIME24                                                                               | ~5,000（摘要）                                                               | 512 prompts × 16（§5.1 item 7）            | mini-batch 512；数量/epochs 未报告                             | 1e-6 / 2e-6（§5.1，基础 PPO 设置）                                          | value warmup **50 步**（§5.1 item 1）                             | AIME24：vanilla PPO 5；无 value 预训练 ~11；完整 VAPO 60（Table 1） | V                        |
| **VinePPO** [2410.01679](https://arxiv.org/html/2410.01679) | DeepSeekMath 7B、RhoMath 1.1B | MATH / GSM8K                                                                         | MATH 1000（~8 dataset epochs）；GSM8K 650（Table 1）                         | 64 prompts × 8 = 512 episodes（Table 1）   | **PPO epochs 2**，mini-batch 64（Table 1）→ E：512/64 × 2 = 16 | actor 1e-6 / critic 未报告（Table 1）                                       |                                                                   | VinePPO 比 PPO 少 2.8×~9× 梯度步（§6.2）                            | V（Table 1）；C.6 被截断 |
| **SPPO** [2604.08865](https://arxiv.org/html/2604.08865)    | R1-Distill-Qwen 1.5B / 7B     | 数学（verl，Reasoning360 reward）                                                    | 未报告                                                                       | 全局 batch 256（1.5B）/ 512（7B）（§4.1）  | 未报告                                                         | 1e-6 / **5e-6**（§4.1）                                                     |                                                                   | 平均 48.06 vs GRPO 47.08（1.5B）；58.11 vs 57.44（7B）（Table 1）   | V                        |

### 1d. anchor（不在两份文档里，仅用来定位我们的预算）

| 工作                                                                                                                                                                                                                                                                           | 模型            | 算法                                                    | 环境                                            | steps                                                              | batch                                                            | 每 batch 更新数                                                                                                                                                 | lr                                                                         | 总 episodes                               | gain                                                        | 验证           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- | -------------- |
| **Search-R1** PPO [2503.09516](https://arxiv.org/html/2503.09516)、[train_ppo.sh](https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/main/train_ppo.sh)                                                                                                               | Qwen2.5-3B/7B   | PPO（GAE λ=γ=1，KL 0.001，clip 0.2）                    | NQ/HotpotQA 搜索，无 user                       | 论文 App. B.2：500；脚本 `total_training_steps=1005`（二者不一致） | `train_batch_size=512`，PPO `n_agent=1`；GRPO `n_agent=5`        | `ppo_mini_batch_size=256` → **512/256 = 2 个 minibatch**；`ppo_epochs` 脚本未设，verl main 默认 1（`actor.yaml` 第 119 行；Search-R1 当时 pin 的版本未核实，U） | actor 1e-6 / **critic 1e-5**；warmup ratio 0.285 / 0.015                   | 500 × 512 = 256k；1005 步口径 ≈ 515k（E） | 平均相对提升 3B +20%、7B +24%（§4.4；Intro 写 41%，不一致） | V              |
| **AReaL `examples/tau2`** [config_8b_airline.yaml](https://raw.githubusercontent.com/inclusionAI/AReaL/main/examples/tau2/config_8b_airline.yaml)、[config_1.7b_airline.yaml](https://raw.githubusercontent.com/inclusionAI/AReaL/main/examples/tau2/config_1.7b_airline.yaml) | Qwen3-8B / 1.7B | group-style（`n_samples: 8`，`adv_norm`，无 critic 段） | tau2 airline；user = 自托管 Qwen2.5-72B，temp 0 | `total_train_steps: 500`（上限，不是实际跑的步数）                 | 8B：`train_dataset.batch_size: 30` × 8 = 240；1.7B：16 × 8 = 128 | **`ppo_n_minibatches: 1`**                                                                                                                                      | `lr: 1.7e-5`（constant，两个 config 相同），`eps_clip: 0.4`，`kl_ctl: 0.0` | 上限 E：500 × 240 = 120k                  | README 只有 reward 曲线                                     | V（config 键） |

### 1e 清单中出现但未入表的

- RGPO（2607.20515）：核对后不是 RL，是 DPO/SimPO/IPO 的离线加权损失，已剔除。
- RLHI（2509.25137）、Noisy RLVR（2510.00915）、UARM（2606.19818）、Off-policy DAE、Back to
  Basics、ReMax、GRPO 原文、Secrets of RLHF、Ilyas 2020、SPPO 之外的 Le Critique / GenAC / Best
  Practice Critic Optimization、SENTINEL、ToolVerse、CLaaS：本次没有打开训练细节（多数是离线、单轮
  RLHF、MuJoCo，或只在 doc 里有摘要），不填数。
- Coverage, Not Targeting（2609.02417）：related-work 已决定不引用，不查。

## 2. 分析

### 2.1 我们的预算位置

| 指标            | 我们  | FACA 8B | GiGPO   | SAO             | Luck         | AReaL tau2 8B（anchor，config 上限） | Search-R1 PPO（anchor） |
| --------------- | ----- | ------- | ------- | --------------- | ------------ | ------------------------------------ | ----------------------- |
| steps           | 48    | 120     | 150     | ≥400，称到 1000 | 100          | ≤500                                 | 500 / 1005              |
| 每步 episodes   | 128   | 128     | 128     | 128             | 192          | 240                                  | 512                     |
| 每步 actor 更新 | 1     | 未报告  | 未报告  | 未报告          | 1            | 1                                    | 2                       |
| actor lr        | 1e-6  | 1e-6    | 1e-6    | 1e-6            | 1e-5（LoRA） | 1.7e-5                               | 1e-6                    |
| 总 episodes     | ~6.1k | 15.4k   | E 19.2k | E 51k~128k      | 19.2k        | ≤120k                                | 256k+                   |

观察：

- **每步 batch（128）并不小。** FACA、GiGPO、SAO 的 batch 都是 128，所以差别不在 batch，而在 **步数**：同类工作用
  100–150 步，SAO 用 400–1000 步；我们 48 步，是这组里最短的。
- **lr 并不异常低。** FACA、GiGPO、SAO、T1、Search-R1、VAPO 的 actor lr 都是 1e-6，我们与它们一致。真正偏低的是 AReaL
  tau2（1.7e-5）、Gao 的 AReaL-SEA（5e-6~1e-5）和 Luck（1e-5 LoRA），这些是 GRPO 且 tau2
  上有明确提升的工作。所以"lr 1e-6 太小"不是充分解释；"1e-6 且步数仅 48"才是偏小的组合。
- **Σlr**：我们 4.8e-5（48 次更新）。FACA/GiGPO 若每步 1 次更新，Σlr = 1.2e-4 / 1.5e-4（E，2.5–3×）；SAO
  400–1000 步为 4e-4~1e-3（E，8–20×）；Search-R1 PPO 约 500~2000 次更新、Σlr 约 5e-4~2e-3（E，忽略 0.285
  的 warmup；10–40×）；AReaL tau2 8B 若跑满 500 步为 8.5e-3（E，约 180×，上限）。
- **总 episodes** 6.1k 比同类 tau/多轮 RL（FACA 15.4k、Luck 19.2k、GiGPO ~19k、MUA-RL ~33k）少 2.5–5
  倍，比 AReaL/Search-R1 量级少 1–2 个数量级。
- **诚实的限制**：这是预算对比，不证明"flat eval 是预算不足造成的"。同类 FACA 在 120 步取得 +5.9pp 也是用 SFT 冷启动的 8B，且 3
  seed 均值；tau2 评测噪声本身就大（verl-recipe PR 报告 ±0.05；IAPO 4B 的 GRPO↔GiGPO 差只有 1.4pp）。48
  步即使训练有效，也可能在评测噪声以内。

### 2.2 同类工作是否每 batch 多次更新

- **报告了 minibatch/epoch 数的**：VinePPO（PPO epochs 2、mini-batch 64，E 约 16 次更新/步）、Search-R1
  anchor（2 minibatch）、ArCHer（actor 3 + critic 50 次/iteration）、SAO 的 critic（K=2 次 value
  update/batch）、AReaL tau2 config（`ppo_n_minibatches: 1`）、Luck（1 次 optimizer
  step/batch）。
- **没报告的**：FACA、AutoForge、IAPO、Gao、MUA-RL、Fission-GRPO、T1、VC-PPO、VAPO、SPPO、EVPO、SPA-RL、P-GRPO。这些论文里"每
  batch 几次更新"多数没写，不能默认它们是多 minibatch。
- 所以"1 次更新/步"并不是异类：AReaL 官方 tau2 config 和 Luck 都是。但 AReaL 用 1.7e-5 补偿，Luck 用 1e-5。我们是"1
  次更新 + 1e-6 + 48 步"三者叠加。
- 另一个相关点：SAO 对 critic 单独做 K=2 次更新 + 10 步 warmup，actor 与 critic lr 比 1:5；T1 的 critic lr 是
  actor 的 15 倍并做了 1 epoch critic-only warmup；VAPO/VC-PPO 做 50–100 步 value warmup；我们的
  critic lr 比是 10×，**没有 warmup**（critic 文档第 6 节记录）。带 critic 的 PPO 工作几乎都做了
  warmup，这是比预算更值得对照的差异。

### 2.3 可比预算建议

没有找到 tau/tau2 上带 critic 的 PPO（critic 文档的结论一致），最近的是 SAO（128/batch、actor 1e-6、critic
5e-6、10 步 warmup、400+ 步，非 tau）。建议分档，按成本从低到高：

1. **补步数（最小改动）**：保持 128 episodes/步、1 minibatch、actor 1e-6 / critic 1e-5，把 48 步拉到
   **≥120–150 步**（FACA/GiGPO 的量级，约 15–19k episodes，约 20–25 个 epoch over 747
   tasks）。成本约为现在的 2.5–3 倍。这是判断"flat 是不是步数不够"的最低成本实验。
1. **提高每步更新量**：128 拆成 4 个 minibatch（每个 32）× 1 epoch，或 2 个 minibatch × 2 epochs；actor lr
   保持 1e-6。对应 Search-R1 / VinePPO 的结构，总更新数 ×4（约 190 次/48 步，或 ≥480 次/120 步）。风险：4B 单任务 128
   条轨迹，每个 minibatch 只有 32 条，梯度方差更大；需要看 `clipfrac` 和 `approx_kl`。
1. **提高 lr**：actor 到 3e-6~5e-6（对照 AReaL-SEA 30B 的 5e-6；AReaL tau2 官方是 1.7e-5，但那是 GRPO 且
   `eps_clip: 0.4`），critic 按 5–10× 同步放大。没有 4B + critic 的直接先例，需要先做 ~30–50 步的短 sweep，看训练
   reward 曲线而不是 eval。
1. **critic warmup**：参照 SAO（10 步）、VAPO（50 步）、VC-PPO（100 步）、T1（1 epoch）。我们是 γ=λ=1、critic
   只作 baseline，此时 critic 不准只增加方差、不引入偏差，所以 warmup 不是必须；但它是同行里一致做的步骤，若第 2 个 epoch 以后训练
   batch EV 仍 ≤ 0，应该先加。这是用户已决定"不做"的部分（critic 文档第 6 节），仅作对照记录。
1. 评测必须同时报告 3 seed 或更多评测 rollout 的方差：tau2 评测噪声与 3–5pp 的提升同量级，没有这一点，预算加大后仍可能看不出信号。

### 2.4 本调研的限制

- 多数数字来自 WebFetch 摘要，没有逐字对原文。论文内部不一致已标注（Search-R1 500 vs 1005 步、T1 的 critic lr 15× vs
  30×、VC-PPO 的 AIME 提升数、Search-R1 的 +41% vs +24%）。
- FACA、SAO 附录里 minibatch/epoch 没取到；SAO 的 "1000 步" 来自摘要，具体 run 的步数未明确。
- RAGEN 的 lr、EVPO 的 MATH 行、VinePPO 的 critic lr（C.6）因页面截断没取到，留空。
- GiGPO 的 mini-batch 256 与 128 envs 的单位关系论文没说明（verl 的 mini-batch 通常按 prompt 数再乘
  n），所以没据此推更新数。
