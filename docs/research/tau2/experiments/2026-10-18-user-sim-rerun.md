# 用户模拟器 v3 矩阵重跑（2026-10-18）

| 项   | 内容                                                                            |
| ---- | ------------------------------------------------------------------------------- |
| 状态 | 已入队（pueue 596–604，`gpu8`），2026-10-18 02:05 北京时间 596 开始运行         |
| 批准 | 用户 2026-10-18 批准（API + GPU）                                               |
| 产物 | `_artifacts/AReaL/codex-sao-math/user-sim-rerun-20261018/`                      |
| 对照 | [user-simulator.md §2](../user-simulator.md) 的 v3 五组（`user-sim-20261012/`） |

## 1. 要回答的问题

1. v3 里 U1（同意操作的同一句里 `###STOP###`）的分布能否复现：DeepSeek 13、Turbo 39–41、Lite 23、GLM
   0、Qwen3.5-4B 0。
1. 在用户 guideline 里加一句话，能否消掉 U1；消掉之后，用户致败的总数是否跟着下降，还是换成别的错误。
1. thinking 开或关对用户行为的影响（DeepSeek 开 / 关两组）。

## 2. 设定

与 v3 相同：actor 为训练前的 Qwen3.5-4B（non-thinking，T=0）；fixed-v3 test 60 题（Airline 20、Retail
40）× 4； 源码为 pinned worktree `tau2-v3-e0fdf4a3`；NL judge 为 DeepSeek-Flash 官方 API；用户
temperature 0。启动脚本复用 v3 的 `api-user/launch.sh` 和
`step0-selfhost-user/smoke_then_full.sh`（复制到新目录，未修改）。

| pueue | run                            | 用户                            | 数据                |
| ----- | ------------------------------ | ------------------------------- | ------------------- |
| 596   | r-deepseek-test-u0p0           | DeepSeek 官方（V4.1），thinking | fixed-v3            |
| 597   | r-deepseek-u1patch-test-u0p0   | 同上                            | fixed-v3 + U1 patch |
| 598   | r-deepseek-nothink-test-u0p0   | DeepSeek 官方，thinking 关      | fixed-v3            |
| 599   | r-turbo-plan-test-u0p0         | Doubao 2.1 Turbo（火山订阅）    | fixed-v3            |
| 600   | r-turbo-plan-u1patch-test-u0p0 | 同上                            | fixed-v3 + U1 patch |
| 601   | r-lite-plan-test-u0p0          | Doubao 2.1 Lite（火山订阅）     | fixed-v3            |
| 602   | r-lite-plan-u1patch-test-u0p0  | 同上                            | fixed-v3 + U1 patch |
| 603   | r-glm53-test-u0p0              | GLM 5.3（火山订阅），并发 32    | fixed-v3            |
| 604   | r-qwen35user                   | Qwen3.5-4B 自部署               | fixed-v3            |

与 v3 的差别：

- DeepSeek 后端已是 V4.1（旧版不可用，见
  [issue](../issues/2026-10-17-deepseek-user-backend-switch.md)）；NL judge 也随之变成 V4.1。
- v3 的 Turbo 走火山标准接口（`ark.env`），这次按用户要求走订阅接口（`ark-plan.env`），模型 id 相同。
- 火山侧模型 id 与 v3
  相同（`doubao-seed-2-1-turbo-260628`、`doubao-seed-2-1-lite-260915`、`glm-5.3`）；订阅接口没有可查询的模型列表，无法确认是否有更新版本。

**U1 patch**：在 `simulation_guidelines.md` 和 `simulation_guidelines_tools.md` 的
`###STOP###` 规则前插入一行， 取自官方语音版 guideline：

> - Agreeing to an action is not the same as the action being completed. If the agent
>   offers to do something (e.g., cancel an order, process a refund), wait for the agent
>   to confirm it is done before ending the conversation.

patch 数据目录是 fixed-v3 的副本，这两个文件为实体文件，其余仍是指向共享 cache 的软链接。

**事故记录**：第一次写 patch 时，副本里的这两个文件是软链接，写入落到了共享 cache
`tau2-bench-b7ea9074…/data/tau2/user_simulator/`。约 2 分钟内发现并删除插入行；恢复后的 sha256 与另外三份独立副本
（`phai-relay/recipes/tau2`、`fixed-v2`、`tau2-bench-v1.0.1`）一致。期间 GPU 空闲、pueue 无运行任务，没有
run 读到改动。

## 3. 怎么读结果

- 按 v3 的分类（U/M/B）逐条标注失败，每个失败只记一个致败原因，与 v3 逐题对照。
- patch 后 U1 接近 0 且用户致败总数明显下降：U1 是 prompt 歧义，环境层修掉即可，"用户出错"方向在 τ² 上不再值得投入方法。
- patch 后 U1 下降但换成其他用户错误、总数不降：支持"不可靠用户修不干净"。
- 旧 v3 轨迹可以用当前 judge 重新计分，把 judge 版本变化和用户变化分开。

并行的只读 case study（infra 审计、thinking 开关对比、U1 归因，分析对象是 v3 的旧轨迹）：
[user-sim-case-study-v3/infra-thinking-audit.md](../user-sim-case-study-v3/infra-thinking-audit.md)，脚本和中间结果在
`user-sim-rerun-20261018/infra-case-study-v3data/`。

## 4. 结果（2026-10-18，9 组全部 240/240 完成，infra 排除 0）

U1 用审计里的规则计数（`user-sim-rerun-20261018/analysis/count_u1.py`），只是筛选，逐条标注见 case study v4（pueue 605）。patch 效果为按题配对 bootstrap（60 题，20,000 次，`boot_patch.py`）。

| 运行 | mean@4 | Airline | Retail | 失败 | U1（其中 reward 0） | v3 对照 mean@4 / U1 |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4.1 | 0.750 | 0.525 | 0.863 | 60 | 4（3） | 0.713 / 13（旧版） |
| DeepSeek V4.1 + patch | 0.758 | 0.563 | 0.856 | 58 | 2（0） | — |
| DeepSeek V4.1 关 thinking | 0.738 | 0.500 | 0.856 | 63 | 0 | — |
| Turbo（订阅） | 0.592 | 0.538 | 0.619 | 98 | 50（46） | 0.633 / 46 |
| Turbo + patch | 0.692 | 0.538 | 0.769 | 74 | 17（17） | — |
| Lite | 0.683 | 0.550 | 0.750 | 76 | 24（23） | 0.700 / 23 |
| Lite + patch | 0.758 | 0.588 | 0.844 | 58 | 0 | — |
| GLM 5.3 | 0.813 | 0.575 | 0.931 | 45 | 0 | 0.758 / 0 |
| Qwen3.5-4B 自部署 | 0.571 | 0.425 | 0.644 | 103 | 2（2） | 0.583 / 0 |

patch 效果：Turbo +10.0pp（95% CI [+2.1, +17.9]，p=0.015），Lite +7.5pp（[+0.8, +14.2]，p=0.028），DeepSeek +0.8pp（[−3.8, +5.4]）。DeepSeek 开 vs 关 thinking：+1.3pp（[−5.4, +8.3]）。

初步读法：

- Doubao 的 U1 复现了（Turbo 50 vs 46，Lite 24 vs 23），GLM 仍为 0；DeepSeek 新版的 U1 从 13 降到 4。
- 一句话的 patch 让 Lite 的 U1 归零，Turbo 从 50 降到 17，分数随之上升；失败数的减少略小于 U1 的减少，其余用户错误是否增加要看逐条标注。
- v3 对照的分数差同时包含用户版本和 NL judge 版本（DeepSeek V4.1）的变化。
