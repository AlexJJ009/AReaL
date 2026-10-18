# critic 在用户边界上的离线前向（2026-10-18）

用 PPO v3（step 119）和 SAO（step 59）的 critic，对各自最后 4 个 epoch 的训练 rollout（各 2,988 条
episode）重新前向，读出每个 agent→用户边界的 $V_t=\text{value}[a-1]$、$U$（五种估计）和 $V_{next}$。目的：看 $U$
能否反映 agent 这一轮做得对不对，用户段意外 $V_{next}-U$ 能否定位用户错误，以及 $U$ 用哪个位置。Codex 写脚本（pueue 614），8 卡 10
分钟（pueue 623）。完整表格：
[reports/user-boundary-forward-codex.md](reports/user-boundary-forward-codex.md)；产物：`_artifacts/AReaL/codex-sao-math/tau2-critic-boundary-20261018/`。

## 结论

1. **$U$ 几乎分不出 agent 的提议对不对。** 以"确认前 agent 文本覆盖了待执行 gold 写操作的全部参数"为正类，PPO 的 value\[b\]
   AUC 0.573 \[0.514, 0.633\]，边界前的 $V_t$ 已有 0.552，增益 0.021；SAO 增益为负。对最终成功的 AUC 是 0.87 /
   0.83，但 $V_t$ 已达 0.865 / 0.829。critic 学到的是"这道题、这个状态大概能不能成"，不是"这一句说得对不对"。
1. **U1 的意外确实落在用户段。** U1 边界上 $U$ 均值 0.675（PPO），正确提议后正常回复的边界为 0.736，差别不大；而 U1 的 $R-U$ 均值
   −0.63，其他边界约 0。critic 在用户 STOP 之前并不知道会失败，失败全部记在用户这一段。
1. **用户段意外最大的边界全是终止边界，而且大多不是用户的错。** PPO 最负 1%（152 个）全部是终止边界（$R-U$），其中 U1 占 31%（基准
   0.6%，富集约 50 倍）；其余是用户正常道谢、STOP，reward 却是 0，多半是 agent 做错了数据库操作、到结算才暴露。SAO 同理（U1 占
   38%）。Claude 用 `boundaries.jsonl` 重算，比例一致。
1. **$U$ 取哪个位置差别很小。** value\[b\]、value\[b−1\]、末 8 / 32 token 均值、整段均值之间 AUC 差不超过
   0.02，没有理由改用均值。

## 对方法的含义

- **β 拆分在终止边界上会削弱 agent 的结果信号。** 当前实现（`areal/trainer/ppo/gae.py`）对内部（类型 1）和终止（类型
  3）边界都拆分， 终止边界的 actor TD 为 $(U-V_t)+\beta(R-U)$。$R-U$ 的大头是 agent 自己的错误（结论 3），β=0.5
  会把这部分也砍掉一半。正在跑的 β=0.5（pueue 618）测的就是这个统一缩放的版本。
- **truncate + bootstrap 在截断处给 agent 的信号很弱。** 截断前的 agent token 拿 $U_k-V_i$；结论 1 说明 $U_k$
  和 $V_t$ 差不多， 所以这部分优势接近 0，效果接近"截断后 mask"。bootstrap 的好处主要是不扔掉截断前的 critic 训练数据，而不是给 agent
  新的信号。
- **critic 意外不能单独当检测器。** 它在终止边界上把 U1 和 agent 的结算失败混在一起，需要规则或 judge 区分。

## 局限

- 训练任务上的 in-sample 检查；PPO 与 SAO 的 checkpoint 训练轮数、策略分布不同，差异不能归因于算法。
- "提议正确"是文本参数覆盖的代理标签，既会漏判也会误判。
- U1 的 reward 已被提前 STOP 影响，正常回复后的成功率不是 U1 的反事实成功率。
