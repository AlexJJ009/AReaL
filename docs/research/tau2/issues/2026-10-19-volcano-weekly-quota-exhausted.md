# 火山 plan 周额度用尽，642 停在 step 24（2026-10-19）

## 现象

2026-10-19 14:02 PDT 起，火山 plan 端点（我们自己的 key `tau2-doubao/ark-plan.env`）对所有请求返回
`429 AccountQuotaExceeded`："You have exceeded the weekly usage quota. It will reset at
2026-10-26 00:00:00 +0800 CST"，即 10-25 09:00 PDT。

pueue 642（PPO + 全局 β=0.5，豆包 Turbo 用户）在 14:11 完成 step 24/120 后不再推进，8 张 GPU 利用率为 0。到
18:49：

- `AccountQuotaExceeded` 650 次，`RateLimitError` 4111 次，`Tau2InfrastructureError` 1498 次；
- 每个 episode 换新 session 重试 3 次后被 reject，rejected 从额度用尽前的 4 涨到 40（accepted 1106）。

额度报错走 infra 错误分支，episode 被 reject，没有以 reward 0 进入训练。step 24 的 rollout 跨过 14:02，这个 batch
里是否混入受影响的 episode 没有逐条核对。

## 影响

- 训练用户（`doubao-seed-2-1-turbo-260628`）和 judge / 分析调用（`deepseek-v4-1-flash-261017`）共用这个
  key，全部不可用。
- 放着不停的话，642 会持续 reject task，几天内会跳过成轮的 task，改变训练数据分布。

## 处理

- 642 于 18:59 PDT 由 `pueue kill` 停掉（用户决定），GPU 已全部释放。
- recover checkpoint 在
  `tau2-ppo-doubao-20261019/ppo-formal-qwen3-doubao-beta05/checkpoints/.../recover_info`，
  对应 step 20（`global_step` 19，0 起），写于 11:50。续跑时丢弃 step 21–24。
- 用户决定：找到新的订阅接口之前，暂停所有依赖用户模拟器或 judge 的后续实验（包括带剧本 critic 的 run 1）。不调 API
  的工作（实现、单测、离线诊断代码）可以继续。
- 换接口后，用户模型会和 642 前 24 步不同，续跑前要先决定是从 step 20 续跑，还是用新用户从头跑。

## 后续（2026-10-20）

用户开通了新的火山 plan 订阅，key 存在 `tau2-doubao/ark-plan-20261020.env`（权限 600，变量名同为
`ARK_PLAN_API_KEY`，sha256 前缀 `68b698be`）。旧的 `ark-plan.env` 保持不动。10-20
的探测结果：`doubao-seed-2-1-turbo-260628` 和 `deepseek-v4-1-flash-261017`
直接可用，返回的模型名与请求一致，不需要另外开通；按 260910 请求，返回的仍是 261017。新 run 读新的 key 文件；GPU run 仍需用户批准。

## 修复：额度熔断（pueue 646，2026-10-20 合入 `500bd211`）

- **分类**：响应体 `error.code` 为 `AccountQuotaExceeded` 或 `insufficient_quota`
  时判为额度用完。另有一条保守的兜底规则：message 里写着 usage quota exceeded 时也判为额度用完；LiteLLM 有时会把 body
  弄丢，这条规则就是为这种情况准备的。`AccountRateLimitExceeded` 和其他 429、5xx、超时都判为偶发错误。
- **额度用完**：只请求一次，不在请求级重试，也不换新 session 重放 episode。所有 rollout 进程共用一个状态文件（加 `flock`），5
  分钟内累计 3 次就锁定 fatal。trainer 每 0.1 s 检查一次这个文件，发现后停止派发，以退出码 86 退出，并在日志里写明额度重置时间。还没凑齐的
  batch 不会被当成一步训练保存；之后从 recover checkpoint 续跑。
- **偶发错误**：最多重试 8 次，退避 1、2、4、8、16、30、30、30 s，每个请求超时 120 s，总预算 600 s。仍然失败就照旧 reject 这个
  episode，训练继续。
- **新增统计**：`rollout/quota_errors`、`rollout/transient_api_errors`、`rollout/rejected_episodes`。
- **还没验证过的**：真实 GPU 训练里的熔断和退出、跨节点挂载下 `flock` 是否可用。
