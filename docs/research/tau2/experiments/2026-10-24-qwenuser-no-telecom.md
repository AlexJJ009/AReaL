# 实验记录：Qwen 用户、去掉 Telecom 的一组 PPO 训练

| 项                 | 内容                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------- |
| 编号               | EXP-2026-10-24-qwenuser-no-telecom                                                           |
| 状态               | run card 草稿（2026-10-24）；run 3′ 设计已同意，未启动；方法 run 的设计待定（§5）            |
| 设计 / 执行 / 批准 | GongxunLi / Claude / GongxunLi                                                               |
| 代码               | `codex/sao-math`，沿用 run 3 的源码快照 `tau2-ppo-qwenuser-20261021/src-41dde039`            |
| 产物目录           | `_artifacts/AReaL/codex-sao-math/tau2-ppo-qwenuser-20261021/`（与 run 3 同一目录，便于对照） |
| 相关文档           | [../research-direction/reflection-20261024.md](../research-direction/reflection-20261024.md) |

## 1. 背景

run 3（`run3-std-beta1-v4`，pueue 680/686）是纯 baseline：标准 critic、γ=λ=1、β=1、无 M2，用户模拟器为自托管
Qwen3.5-4B（T=0，non-thinking），训练域 airline + retail + telecom。

DeepSeek 用户评测（pueue 689，用户和 judge 都是 `deepseek-flash`，T=0，每题 4 次，mean@4）显示：

- run 3 step 120 比 v3ext step 120（DeepSeek 用户训练、无 Telecom）retail 低 9.3pp、合计低 6.2pp；
- Telecom 在 DeepSeek 用户下 step 0 就有 46%，Qwen 用户评测只有 5–10%；run 3 训练中 Telecom 奖励 0.03 →
  0.13。

但 run 3 和 v3ext 不只差用户：v3ext 没有 Telecom，每步 128 条全是 airline/retail，120 步约 20 epoch；run 3 中
Telecom 占约 28%，120 步约 13 epoch。两个原因现在拆不开。

## 2. run 3′：问题与设计（已同意）

**问题**：去掉 Telecom 之后，Qwen 用户训练出的 checkpoint 在 DeepSeek 用户评测下还比 DeepSeek 用户训练（v3ext）差吗？

**和 run 3 的差别（一个变量：训练数据去掉 Telecom）**，步数按 v3ext 的定法对齐到 120 步：

| 项         | run 3                                              | run 3′                                       |
| ---------- | -------------------------------------------------- | -------------------------------------------- |
| 训练域     | airline 417 + retail 337 + telecom 329             | airline 417 + retail 337                     |
| task packs | areal pack + telecom pack + airline recovered pack | areal pack + airline recovered pack          |
| 环境变量   | `TAU2_TELECOM_PACK=...telecom-pack-20261021-v2`    | 不设                                         |
| epoch      | 14（实际跑到 126 步）                              | 20（754 条 / 128 ≈ 6 步每 epoch，约 120 步） |

其余全部沿用 run 3：同一源码快照和 launcher（`launch_ppo_qwenuser_k20261023.sh`）、Qwen3.5-4B 用户（GPU
7）、NL judge 为火山 plan `deepseek-v4-1-flash-261017`（key
`ark-plan-20261023.env`）、actor+critic GPU 0–3、rollout GPU 4–6、
`++offline_eval_only=true`（训练中不评测）、`rollout.max_concurrent_rollouts=32`、`++train_user_boundary_values=true`。

**和 v3ext 仍然存在的差别**（对照时要记住）：用户与 judge；源码快照（`src-41dde039` vs `src-2a4f25bb`）；airline 多 7
题 recovered pack（417 vs 410）；rollout 并发 32 vs 128。

**预先登记的假设**（跑之前写下，结果出来后不改）：

- H1：若 run 3′ step 120 在 DeepSeek 用户评测下 retail 与 v3ext（63.1）相差在 6pp 以内，run 3 的差距主要来自
  Telecom； 方法 run 继续用 Qwen 用户。
- H2：若仍低 9pp 左右，弱用户训练本身有损害；方法 run 要考虑换强用户（成本见 §4）。
- 两种结果都会改变下一步的决定，这个 run 值得跑。介于 6–9pp 之间的结果按噪声处理，不下结论（retail 只有 40 题，粗估一个标准误约 6pp）。

**评测**：训练结束后另起评测任务，DeepSeek 官方用户 + judge，checkpoint step 0/30/90/120，沿用 pueue 689
的脚本和余额守卫，约 45 分钟、约 ¥27。评测要单独确认后再排。

## 3. 时间估计（推断）

依据 checkpoint 时间戳：

- run 3（Qwen 用户，含 Telecom）：globalstep 9 → 89 用了 21.7 h，约 16 分钟/步；
- v3ext（DeepSeek 用户，无 Telecom）：globalstep 49 → 119 用了 9.3 h，约 8 分钟/步。

Telecom episode 很长（约一半撞到 max_steps），去掉后每步会变快；本地 Qwen 用户比 DeepSeek API 慢多少还不知道。run 3′ 估计
8–16 分钟/步，120 步约 **16–32 小时，最可能 20–25 小时**。

## 4. 成本和外部依赖

- GPU：gpu8 组 8 张卡独占，期间不能跑别的 GPU 任务。
- 用户模拟器本地，不花钱。
- NL judge 用火山 plan（`726679fa`）的周额度，与其他使用者共享。启动前用一次请求探测，确认返回的模型名是
  `deepseek-v4-1-flash-261017`。
- 如果改用 DeepSeek 用户训练：约 15,000 episode，按 pueue 689 的单价（2,000 条 ¥27.4）约
  ¥200，超过目前余额能承受的范围（¥211.88，与 phai-relay 共用）。

## 5. 方法 run（设计待定）

用户的设想：带剧本 critic + SAO 的长度自适应 λ（$\lambda = 1 - 1/(1.5L)$，$L$ 为整条轨迹的 agent token 数）。

待定的两点：

1. "SAO"只换 λ，还是用整套 SAO 配置？本仓库之前的 SAO run（`sao-formal-qwen3-v3-mse`）相对 PPO 改了 6 处：λ、不白化
   advantage、DIS clip 0.3/5.0 代替 PPO clip、critic 冻结 attention、critic lr 5e-6、每步先更新
   critic 2 次 （[2026-10-15-qwen3-ppo-grpo-sao.md](2026-10-15-qwen3-ppo-grpo-sao.md)
   §2、§4.0）。那组对比里 PPO 与 SAO 在 120 步分不出优劣。
1. 对照组。"带剧本 + λ=1"在 Qwen 用户、无 Telecom 的设置下还没有跑过（649 是 DeepSeek 用户）。只跑 run 3′ 和"带剧本 + SAO
   λ"，两者差两个变量。

## 6. 运行记录

- 2026-10-24 08:3x PDT，pueue
  690（gpu8，`-w tau2-ppo-qwenuser-20261021/run3p-std-beta1-notel-c128`），名称
  `run3p-std-beta1-notel-c128`。用户批准：rollout 并发 128（相对 run 3 的第二处改动，与 v3ext 相同），保留 120
  步（不截到 84 步）。
- 预检：火山 judge 探测返回 `deepseek-v4-1-flash-261017`、无额度错误；check-config 显示 Train
  prompts=754、steps=120、domains \[airline, retail\]、`adv_norm`
  batch、`gae_lambda`=1.0；覆盖参数在 `run3p-formal-overrides.txt`。

## 7. 结果

（训练和评测结束后填写。）

## 8. 讨论更新（2026-10-24，未定稿）

**速度（读到的数据）**：run 3 每步 `timeperf/rollout` 650–1610 s，训练相关（actor、critic 更新，取 value、重算
logp）约 250 s； Qwen 用户服务器平均只有 0.74 个在途请求（最多 6）。run 3 的 episode 时长中位数（10-23 08:40–10:30 的
672 条）：retail 69 s、 airline 80 s、telecom 345 s（p90 1331 s，最长 1810 s，撞到 1800 s 超时）。瓶颈是
rollout 并发 32，不是用户服务器。

**提速提案（待批准）**：`rollout.max_concurrent_rollouts` 32 → 128（与 v3ext 相同）；步数 120 → 约 84（14
epoch）。依据：之前 PPO 的提升集中在前 70 步（§4.0 of 2026-10-15 记录：0→70 +8.3pp，70→120 +4.2pp 且 CI 含
0）；actor lr 为常数，截断不改变前 84 步的训练；对照用 v3ext globalstep 79/89 的 checkpoint。估计约 8 分钟/步，84 步约
11 小时（推断）。

**方法 run**：只换 actor 的 λ（`gae_lambda=sao_length_adaptive_gae`，α=1.5），critic 目标保持
λ=1（`critic_gae_lambda=1.0`，与 SAO 论文和之前的 SAO run 相同），advantage 白化、clip、critic 设置都保持
PPO。SAO 论文没有披露 advantage 是否归一化（见
[../../sao/2026-09-23-sao-paper-gap-audit.md](../../sao/2026-09-23-sao-paper-gap-audit.md)），之前
SAO run 的"不白化"是项目选择。

**教训**：Telecom 的问题本可以在正式训练前用 step 0 的分领域奖励（训练用户，约 1 小时）发现；run 3 第 1 步起 Telecom 训练奖励就只有
0.03。以后换用户或换领域的正式 run，run card 里加这一项预检。

## 9. run 2′（2026-10-24 用户批准排队）

- pueue 691（gpu8，排在 690
  之后，`-w tau2-ppo-qwenuser-20261021/run2p-priv-beta1-notel-c128`），名称
  `run2p-priv-beta1-notel-c128`，覆盖参数 `run2p-formal-overrides.txt` = run 3′ 的参数 +
  `++critic_user_scenario=true critic.mb_spec.max_tokens_per_mb=36864`。
- check-config 与 run 3′ 的解析配置逐行对比，只差两处：`critic_user_scenario` false → true（critic
  读剧本），critic microbatch 上限 32768 → 36864（剧本让 critic 输入变长，沿用 649/688 的值；只影响切分，不改
  loss）。`user_td_beta: null`、`user_reliability_weight: false`（不加 M2）、 λ=1、并发 128、120
  步、domains \[airline, retail\]。
- 问题：同一 Qwen 用户、无 Telecom 下，带剧本 critic 是否比标准 critic（run 3′）训练得更好。评测另行确认。
- run 4（带剧本 + SAO λ）属于方法设计，用户设计好后再定。

## 10. 运行记录更新（2026-10-24 22:30 PDT）

- run 3′（pueue 690）08:29:58 → 22:25:57 完成，Success，约 14.0 小时、120 步；无
  Traceback/ERROR/NaN，用户服务器无重启。训练奖励（每 20 步平均） 0.558 → 0.641；critic
  EV（更新前，`critic_explained_variance_pre_update`）0.09 → 约 0.40。
- run 2′ 第一次排队（pueue 691）在 690 结束后 0.3 秒失败（exit 2）：pueue 把
  `bash -c "<等待 90 秒>; exec ..."` 拼成一条 shell 命令，等待命令 没拿到参数，launcher 立刻启动，发现 690
  的用户服务器还占着 40080，拒绝启动。没有开始训练。原因是我写错了排队命令。
- 修复：改用 `run2p-priv-beta1-notel-c128/launch.sh`（cd 到 run 目录后 exec 原 launcher，参数不变），重排为
  pueue 692，22:27 启动时端口已空闲、 8 张卡已释放。

## 11. 评测排队（2026-10-25 00:40 PDT，用户批准）

- checkpoint：run 3′ step 0/30/60/90/120 + run 2′ step 30/60/90/120（step 0
  是同一个初始模型，只测一次），共 9 个；每题 4 次，T=0。
- pueue 693（排在 692 后）：DeepSeek 官方 `deepseek-flash` 作用户和 judge，ppo 题集（airline 20、retail
  40、telecom 40），8 卡； 余额守卫：开跑前低于 ¥80 不启动，每 5 分钟查一次，低于 ¥30 停止。目录
  `_artifacts/AReaL/codex-sao-math/tau2-deepseek-eval-20261025/`。
- pueue 694（排在 693 后）：本地 Qwen3.5-4B 用户（GPU 7）+ 火山 plan judge，`--task-set test`（只测
  airline 20、retail 40），7 卡。 目录 `tau2-qwenuser-eval-20261025/`。
- 代码：`checkpoint_eval.py` 新增 `--task-set {ppo,test}`（commit
  `3843fe41`、`9baa2cf1`），两个评测都用 `src-9baa2cf1` 快照； 已核对 run 3′ 配置下 test 题集与 ppo 题集去掉
  telecom 后的 60 题完全相同。dry-run 两个计划都正确列出 9 个 checkpoint。
- 余额：00:40 PDT 为 ¥110.46（10-24 评测结束时 ¥211.88，共用账户被其他用途花了约 ¥101）。

## 12. 结果（2026-10-25 19:30 PDT）

运行：run 2′（pueue 692）22:27 → 14:31 完成，120 步，无报错；2 条用户失败（Qwen 用户 32768 上下文超限）。DeepSeek 评测
693 14:31 → 15:56 成功，余额 ¥400.05 → ¥350.44（¥49.61，共用账户，差额可能含其他用途；北京时间 05:31–06:56 跑的，单价与
10-24 晚高峰那次相同，未见夜间优惠）。 Qwen 评测 694 失败：240 条 episode 全部跑完，但 runner 仍按 400 条的 ppo 题集校验，把每个
checkpoint 判为失败并重跑了一次（我的 bug， `bd8cadb0` 修复，加了测试并做过变异检查）。因此每个 checkpoint 有两份完整的 240
条结果；下面的 Qwen 分数取自两份 attempt 的 `summary.json`， `eval/summary.md` 里的 failed 标记不代表没有分数。

指标：mean@4（每题 4 次成功率的平均），agent T=0，用户和 judge T=0，infra 剔除 0 条。差值 CI 为按题配对
bootstrap（20,000 次， `tau2-deepseek-eval-20261025/paired_boot.py`）。

DeepSeek 用户评测（ppo 题集，单次 attempt）：

| checkpoint                | airline                   | retail                    | telecom                   |
| ------------------------- | ------------------------- | ------------------------- | ------------------------- |
| step 0                    | 13.8                      | 45.0                      | 49.4                      |
| run 3′ 30 / 60 / 90 / 120 | 21.3 / 16.3 / 17.5 / 18.8 | 51.9 / 50.6 / 58.8 / 54.4 | 46.3 / 49.4 / 45.0 / 55.0 |
| run 2′ 30 / 60 / 90 / 120 | 17.5 / 17.5 / 20.0 / 17.5 | 57.5 / 56.9 / 65.6 / 65.6 | 50.6 / 56.3 / 55.6 / 67.5 |

Qwen 用户评测（airline + retail，两次 attempt 平均）：

| checkpoint                | airline                   | retail                    |
| ------------------------- | ------------------------- | ------------------------- |
| step 0                    | 16.3                      | 36.3                      |
| run 3′ 30 / 60 / 90 / 120 | 20.0 / 19.4 / 27.5 / 21.3 | 35.6 / 39.7 / 42.5 / 48.8 |
| run 2′ 30 / 60 / 90 / 120 | 15.6 / 25.0 / 15.6 / 23.1 | 33.8 / 43.1 / 44.1 / 43.1 |

配对比较（retail 除注明外）：

| 比较                                                                | 差值                 | 95% CI           |
| ------------------------------------------------------------------- | -------------------- | ---------------- |
| DeepSeek：run 2′ 120 − run 3′ 120                                   | +11.3pp              | \[+1.3, +21.2\]  |
| DeepSeek：同上，telecom                                             | +12.5pp              | \[+3.7, +21.9\]  |
| DeepSeek：同上，airline                                             | −1.3pp               | \[−11.3, +10.0\] |
| DeepSeek：run 2′ − run 3′，90 和 120 合并                           | +9.1pp               | \[+1.9, +16.3\]  |
| DeepSeek：run 3′ 120 − step 0                                       | +9.4pp               | \[+0.6, +18.1\]  |
| DeepSeek：run 2′ 120 − step 0                                       | +20.6pp              | \[+11.3, +30.0\] |
| DeepSeek：run 3′ 120 − v3ext 120（10-24 评测）                      | −8.7pp               | \[−18.8, +1.9\]  |
| DeepSeek：run 2′ 120 − v3ext 120                                    | +2.5pp               | \[−7.5, +12.5\]  |
| DeepSeek：step 0（10-25）− step 0（10-24）                          | −1.3pp               | \[−10.0, +8.1\]  |
| Qwen：run 2′ 120 − run 3′ 120                                       | −5.6pp               | \[−15.3, +3.7\]  |
| Qwen：run 3′ 120 − step 0                                           | +12.5pp              | \[+3.7, +21.9\]  |
| Qwen：同一 checkpoint 两次 attempt 之差（step 0 / 3′ 120 / 2′ 120） | −5.0 / −2.5 / +6.2pp | 都含 0           |

对预先登记假设的判断：H1（去掉 Telecom 后 run 3′ 与 v3ext 相差 6pp 以内）不成立，相差 8.7pp，落在登记的"6–9pp
按噪声处理"区间，不下结论。

局限：每个 run 只有一个训练种子，训练种子间的方差没有测过；DeepSeek 评测每个 checkpoint 只跑了一次；上表做了十余次比较，单个 CI 不含 0
不能当作确定结论；T=0 下同一 checkpoint 两次评测仍相差最多 6pp。

## 13. run 4 / run 5 run card（2026-10-25，用户批准排队）

**问题**：在 Qwen 用户、无 Telecom 的设置下，actor 改用 SAO 长度自适应 λ（$\lambda=1-1/(1.5L)$，$L$ 为整条轨迹的
agent token 数；critic 目标保持 λ=1）时，带剧本 critic 的收益是否大于标准 critic 的收益（2×2 的交互作用）。

| run                                 | critic | actor λ | 参照   | 唯一差别 |
| ----------------------------------- | ------ | ------- | ------ | -------- |
| run 4 `run4-priv-saolam-notel-c128` | 带剧本 | SAO     | run 2′ | actor λ  |
| run 5 `run5-std-saolam-notel-c128`  | 标准   | SAO     | run 3′ | actor λ  |

**设置**：覆盖参数 = 参照 run 的参数 +
`actor.gae_lambda=areal.trainer.ppo.lambda_fn.sao_length_adaptive_gae ++actor.gae_lambda_kwargs={alpha:1.5}`（`run4-formal-overrides.txt`、`run5-formal-overrides.txt`）。代码
`src-41dde039-saolam-895bc69`：run 2′/3′ 的快照加上 `895bc69a` 对 PPO 契约检查的修改（只放开 actor λ，仍要求
critic λ=1、clip 0.2、 batch 白化、无 DIS），与原快照逐文件比较只有 `examples/tau2/train.py`
不同。check-config 与参照 run 的解析配置逐行比较，只差 `gae_lambda` 和 `gae_lambda_kwargs` 两行。启动脚本
`<run 目录>/launch.sh`。

**预先登记的判断**（主指标：DeepSeek 用户评测 retail，step 90 与 120 合并，按题配对 bootstrap）：

- 机制成立：(run 4 − run 2′) − (run 5 − run 3′) > 0 且 CI 不含 0。
- SAO λ 是通用改进、与剧本无关：两行差值相近，交互 CI 含 0，且两行各自为正。
- SAO λ 无用或有害：两行差值都不为正。
- 不下结论：交互 CI 含 0 且两行各自 CI 都含 0。按 run 2′/3′ 的经验（单次评测同一 checkpoint 相差最多
  6pp、单种子），这是最可能的结果之一； 它说明需要更多种子，而不是方法无效。

**成本**：GPU 每个约 14–16 小时（gpu8 独占），依次跑约 30 小时；火山 judge 少量；之后 DeepSeek 评测 8 个 checkpoint 约
¥45，Qwen 评测约 2 小时， 另行确认。

**停止条件**：Traceback/NaN、用户服务器反复重启、火山 `AccountQuotaExceeded`（launcher 的 quota breaker
会停）；训练奖励前 30 步比参照 run 低 0.05 以上时暂停并报告。

**已知风险**：run 2′/3′ 的差异来自单一种子，尚未复现（见 §12 局限）。用户决定先跑 run 4/5（2026-10-25）。

## 14. 排队记录（2026-10-25 19:25 PDT，用户批准：先 run 4 后 run 5，两者的评测都挂上）

gpu8 组（并行 1，按编号顺序执行；评测用 `--after` 依赖各自的训练任务，训练失败则其评测不跑，run 5 不依赖 run 4 的评测）：

| pueue     | 内容                                                                                                       | 目录                                                      |
| --------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 695       | run 4 训练                                                                                                 | `tau2-ppo-qwenuser-20261021/run4-priv-saolam-notel-c128/` |
| 696       | run 4 DeepSeek 用户 + judge 评测，ppo 题集（400 episode/checkpoint），step 30/60/90/120                    | `tau2-deepseek-eval-run45-20261025/eval-<trial>/`         |
| 697       | run 4 Qwen 用户 + 火山 judge 评测，test 题集 airline + retail（240 episode/checkpoint），step 30/60/90/120 | `tau2-qwenuser-eval-run45-20261025/eval-<trial>/`         |
| 698       | run 5 训练                                                                                                 | `tau2-ppo-qwenuser-20261021/run5-std-saolam-notel-c128/`  |
| 699 / 700 | run 5 的 DeepSeek / Qwen 评测，同上                                                                        | 同上                                                      |

- step 0 不重评，沿用 2026-10-25 评测（§12）。评测代码快照 `src-895bc69a`（含 `bd8cadb0` 的 test
  题集校验修复）。四个评测都用 check-config 生成的 config.yaml 做过 `--dry-run`：各 4 个 checkpoint，planned
  episodes 400 / 240，checkpoint 路径正确。
- DeepSeek 评测脚本：余额低于 ¥80 不启动，运行中低于 ¥30 停止，余额记录在 `balance.log`。预计每个 run ¥22。
- Codex read-back（只读，无 GPU、无 API）：codex-review 组的 701 / 702 等对应训练写出 "Train step 2/120
  done." 后，用 `scripts/research/dispatch.sh` 派发 `readback-run4-20261025` /
  `readback-run5-20261025`（brief 在各自目录的 `BRIEF.md`，结果写 `REPORT.md`）。检查项：config.yaml 与参照
  run 逐键 diff、启动日志里打印的 λ、实际代码快照、SAO λ 在运行时生效的证据、前两步健康、 695–700 的队列参数。

## 15. 固定种子（2026-10-25 20:20 PDT，用户规则：后续 launch 用同一份种子，第一个种子 42）

- 已固定且所有 run 相同（未改）：训练器 `seed: 42`（数据打乱、各 rank 的 torch/numpy/random），policy SGLang
  `sglang.random_seed: 42`， `vllm.seed: 42`。`gconfig.seed` 保持
  null：每个请求固定种子会让同一题的组内采样完全相同。
- 之前未固定：Qwen 用户服务器没有传 `--random-seed`，SGLang 每次启动随机抽（run 3′ 30334310，run 2′
  472006262，run 4 836178993）； Qwen 评测的用户服务器同理。
- 新训练 launcher `launch_ppo_qwenuser_k20261025.sh`（旧文件不动，保留 run 2′/3′/4
  的出处）：`SEED=${SEED:-42}`，用户服务器加 `--random-seed $SEED`，训练器加
  `seed=$SEED sglang.random_seed=$SEED vllm.seed=$SEED`。run 5 的 check-config
  与改前的配置逐字节相同； `SEED=43` 时 `seed`、`sglang.random_seed`、`vllm.seed` 三行变成 43，说明覆盖生效。
- Qwen 评测：`09a8bbd3` 给 `checkpoint_eval.py` 加 `--user-seed`（默认 42，计入结果指纹），测试会在删掉种子参数后失败。
- 生效范围：run 5（698）和 Qwen 评测 697/700 在启动前切换；run 4 已在跑，用户服务器种子仍是 836178993。DeepSeek
  评测没有本地用户服务器， 不受影响。
- 局限：SGLang 未开 deterministic inference，异步 rollout 的批次组成也随时间变化，所以固定种子不等于可逐位复现。

## 16. run 5 改用 DeepSeek 官方 judge 重启（2026-10-27，用户决定）

- 经过：run 5（698）10-26 20:55 PDT 因火山 plan 5 小时额度耗尽，熔断器干净退出（code 86），状态移到
  `failed-20261026-2100-quota/`；10-27 00:35 从头重启后，用户决定训练 judge 不再用火山（额度不稳定，与他人共享），
  再次停掉，状态移到 `failed-20261027-0035-volcano-judge-killed/`。两次都没有可用的 checkpoint，run 5 从
  step 0 重跑。
- 新 launcher `launch_ppo_qwenuser_k20261027.sh`（复制 k20261025，旧文件不动）：NL judge 改为
  `deepseek-flash` @ `https://api.deepseek.com`，key 取
  `tau2-doubao/deepseek-official.env` 的 `DEEPSEEK_API_KEY` （启动时检查文件权限 600 和变量存在）。旧
  launcher 的 `TAU2_DEEPSEEK_ENV_FILE` 指向别的项目的 key 文件，新文件已改正。
- 新代码快照 `src-41dde039-saolam-895bc69-dsquota`：与 `src-41dde039-saolam-895bc69` 只差
  `examples/tau2/api_reliability.py`（取自合并后的 `codex/sao-math` `7bbe2b0a`：DeepSeek 402 /
  Insufficient Balance 计为额度错误，触发熔断）。
- check-config（`check-config-dsjudge.log`）与 seed 42 的配置逐键比较，只有三项不同：`nl_judge_llm`、
  `nl_judge_llm_base_url`、`nl_judge_llm_api_key_env`。
- 启动前 DeepSeek 官方余额 ¥328.22。pueue 698（训练）和 699（DeepSeek 评测，依赖 698）原地重启；Qwen 评测 也改用
  DeepSeek 官方 judge、step 串行（`run-eval.sh` 加
  `--judge-backend deepseek-official --max-parallel-steps 1`，代码 快照
  `src-7bbe2b0a`，`runtime-dsjudge.json`；dry-run 确认 judge 为 `deepseek-flash` @
  api.deepseek.com，plan 里没有火山 key）：700 = run 5（依赖 698）原地重排，711 = run 4 新排。卡住的 run 4
  旧评测移到 `failed-20261026-quota-hang-eval-run4-priv-saolam-notel-c128/`。
- **对 2×2 的影响**：run 2′、3′、4 的训练 judge 是火山 plan `deepseek-v4-1-flash-261017`，run 5 是官方
  `deepseek-flash`。§13 的交互项 (run 4 − run 2′) − (run 5 − run 3′) 因此混入了 judge 差异；两个 judge
  的一致率未测。 NL judge 只决定少量 episode 的 reward（Qwen 评测中约 17.5%），影响大小未知。

## 17. run 4 / run 5 结果与 2×2（2026-10-28）

全部完成：run 5 训练（698，10-27 15:25 第 120 步，无 traceback、无额度错误），DeepSeek 评测 699、Qwen 评测 700 /
711（Qwen 评测的 judge 为 DeepSeek 官方，run 2′/3′ 的 Qwen 评测为火山 plan），infra 剔除均为 0。

第 120 步 mean@4（%），agent / 用户 / judge T=0；EV 为第 101–120 步
`critic_explained_variance_pre_update` 均值：

| 配置                        | DeepSeek 用户 A / R / T | Qwen 用户 A / R | EV   |
| --------------------------- | ----------------------- | --------------- | ---- |
| step 0                      | 13.8 / 45.0 / 49.4      | 16.3 / 36.3     | —    |
| run 3′（标准，$\lambda=1$） | 18.8 / 54.4 / 55.0      | 21.2 / 48.8     | 0.40 |
| run 2′（剧本，$\lambda=1$） | 17.5 / 65.6 / 67.5      | 23.1 / 43.1     | 0.52 |
| run 5（标准，SAO）          | 13.8 / 56.3 / 54.4      | 12.5 / 44.4     | 0.48 |
| run 4（剧本，SAO）          | 22.5 / 55.6 / 51.9      | 23.8 / 41.9     | 0.52 |

§13 的判据 (run 4 − run 2′) − (run 5 − run 3′)（DeepSeek 用户评测，按题配对 bootstrap 95% CI，20,000
次）：airline +10.0 \[0.0, +22.5\]，retail −11.9 \[−23.1, −0.6\]，telecom −15.0 \[−29.4,
−1.9\]。判据要求 > 0 且 CI 不含 0，结果方向相反： $\lambda=1$ 时剧本 critic 的收益在 SAO 下消失，run 4 ≈ run 5。Qwen
评测（两者同为 DeepSeek judge）run 4 − run 5：airline +11.3 \[0.0, +25.0\]，retail −2.5 \[−15.0,
+9.4\]。局限：每个配置一个训练种子；run 5 训练 judge 与另外三个不同；未做多重比较校正。

run 4 的轨迹 case study（Codex
712，`_artifacts/AReaL/codex-sao-math/case-study-run4-20261027/REPORT.md`）：没有整体训练崩溃； step
120 Telecom 中 actor 工具报错 83/160（run 3′ 50、run 2′ 24），重复 XML 设备请求、占位参数（如
`+1234567890`）也更多。

后续（2026-10-28 用户批准）：§5 C 的 GPU forward 按 Codex 705 的预注册原样运行（pueue 713，critic A = run
2′、B = run 3′，8,671 条固定轨迹）；同时 Codex 714 准备加入 run 4 / run 5 critic
的扩展（`critic-forward-run45-20261028/`，只准备，GPU 运行另出 run card）。

## 18. run 2′ 换种子复现（2026-10-28，用户批准）

- 问题：λ=1 下剧本 critic 的收益（run 2′ − run 3′，DeepSeek 评测 retail +11.3、telecom +12.5）是否只是 run
  2′ 一次的运气；§17 的 2×2 交互项全靠这一个 run。
- trial `run2p-priv-beta1-notel-c128-s43`（pueue 716，排在 715 之后）；评测 717（DeepSeek 用户 +
  judge）、718（Qwen 用户， DeepSeek judge），都依赖 716。
- 与 run 2′ 的差别：`SEED=43`（训练器 `seed`、policy `sglang.random_seed`、`vllm.seed`、Qwen 用户服务器
  `--random-seed`；run 2′ 为 42 / 42 / 42 / 随机 472006262）；NL judge 改为 DeepSeek 官方
  `deepseek-flash`（与 run 5 相同）； 快照 `src-41dde039-saolam-895bc69-dsquota`（比 run 2′ 的
  `src-41dde039` 多一个也接受 SAO 的 actor λ 校验，λ=1 时无影响，以及 402 额度分类）。overrides 文件与 run 2′ 相同。
- check-config：与用 run 2′ 原 launcher / 快照 / overrides 生成的参照逐行比较，只差
  `seed`、`sglang.random_seed`、`vllm.seed` 三行（42 → 43）和三项 judge 设置。日志在 trial 目录
  `check-config-s43.log`、`check-config-run2p-reference.log`。
- 读法：DeepSeek 评测 retail / telecom 接近 65 / 67，说明 run 2′ 的收益稳；接近 55（run 3′ 水平）说明可能是种子运气，
  2×2 交互项不可靠；介于其间则合并两个种子再算。
- 成本：训练约 14.5 小时（8 卡），DeepSeek judge 约 ¥7；评测 API 约 ¥25。启动前余额 ¥479。
