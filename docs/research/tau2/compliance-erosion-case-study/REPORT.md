# RL 训练是否侵蚀程序合规：已有轨迹的 case study（2026-10-22）

问题：outcome reward 只查最终 DB 状态，不惩罚跳过程序。训练是否让 agent 更常跳过"读回详情后取得明确同意再写库"和身份认证， reward
是否偏向这些捷径。Codex 执行（pueue 683），Claude 复核。只用 CPU 和规则，不调用任何 LLM。

- Codex 完整报告：[codex-report.md](codex-report.md)（其中 `out/`、`scripts/` 的相对链接指向服务器产物目录）。
- 服务器产物：`_artifacts/AReaL/codex-sao-math/tau2-compliance-erosion-20261022/`（逐条
  episode/write 数据、规则、审读表、重算脚本）。

## 1 数据

| run          | 说明                                 | 用户                              | 固定用户 eval checkpoint |
| ------------ | ------------------------------------ | --------------------------------- | ------------------------ |
| ppo_standard | 标准 PPO（v3 + ext 续训，共 120 步） | DeepSeek 官方 `deepseek-flash`    | 0–120                    |
| sao_long120  | SAO 120 步                           | DeepSeek 官方                     | 0–120                    |
| sao_mix40    | SAO，Qwen3.5-4B，题池不同            | DeepSeek 官方                     | 0–40                     |
| 649          | PPO + 特权 critic                    | 火山 `deepseek-v4-1-flash-261017` | 0–120                    |
| 631          | 标准 critic，β=0.5                   | 火山 260910                       | 0–120                    |
| 680          | run 3，进行中，只读到已写出的步      | 本地 Qwen 用户                    | 无                       |

共 65,979 条训练 episode、22,384 条 eval 结果。不同 run 的用户、actor、题池不同，只能比较同一 run 内的趋势，不能把 run
之间的差别归因于 critic 或用户。

## 2 结果

**有侵蚀信号，但幅度小、不一致；没有证据说明 reward 偏向"未经同意的写入"。**

1. **严格确认时序变差（固定用户 eval，step 0 → final）。** 严格口径把"一次确认后连做多个写操作"也算违规。PPO retail 32.0% →
   39.2% （+7.2pp \[2.6, 12.3\]），SAO airline 35.3% → 47.6%（+12.3pp \[4.8, 18.5\]），631
   retail +9.8pp \[0.0, 19.4\]；其余 run/domain 的 CI 含 0。完全没有匹配同意的写入（raw unconfirmed）只有 649
   airline 显著上升：5.4% → 13.8%（+8.3pp \[0.6, 16.6\]）。
1. **身份程序。** Airline 在用户给出 ID 前就写库的 episode 比例：PPO 1.2% → 2.6%，649 1.8% → 3.7%，631 1.0%
   → 3.2%（CI 都高于 0）， 绝对值很低。
1. **reward 关联只支持弱命题。** 同题、同 10 步窗口内配对：retail 训练中带严格违规的 episode reward 高
   21–28pp（PPO、649、631、SAO）； 但 raw unconfirmed 的 reward 差接近 0，CI 含 0。airline 上严格违规的
   episode reward 反而更低。
1. **成功率增长主要不是捷径带来的。** PPO retail full-outcome reward step 0 → final
   +18.8pp，其中"成功且含未确认写入"只占 +2.5pp （CI 含 0）；649、SAO 分别为 +23.1 / +3.8、+15.6 / +3.1。
1. **反例。** sao_mix40 airline 训练中 raw unconfirmed 从 20.0% 降到 10.6%；649 retail、680 retail
   都没有上升。680 Telecom 有 55.3% 的 episode 截断（多数撞 `max_steps`），已排除在比率之外。

## 3 Claude 的读法

1. **数字核对。** 我用 `out/episodes.jsonl` 重算了 PPO retail 固定 eval：40 道共同题，step 0 严格违规 55/172 =
   32.0%、raw 25/172 = 14.5%；step 120 为 96/245 = 39.2%、47/245 = 19.2%，与报告一致。案例 6
   逐条读过原文，描述属实。
1. **第 3 条的正关联很可能是混杂。** 一次确认后连做多个写操作，本身就意味着把用户要的几件事都做完了，而 retail 的成功恰恰要求做完全部 操作。所以"严格违规的
   episode reward 更高"不能读成 reward 奖励跳过确认。真正对应"跳过同意"的 raw unconfirmed 没有 reward 优势。
1. **最强的证据是"reward 容忍程序丢失"，不是"reward 奖励程序丢失"。** 案例 6（sao_mix40 retail task 42）：step 0
   先认证、读回新地址、 得到 yes 再写；final 不认证，读回和写库放在同一条消息里，两者 reward 都是 1。这是 outcome reward
   的盲区，但只是"不惩罚"， 梯度不会主动把策略推向跳步。这类案例的用户其实在上一句已经给出了完整新地址并要求修改，属于较轻的违规。
1. **测量精度是主要限制。** 独立 agent 审读 48 个写操作：三分类一致 32/48，违规与否一致 42/48；12 个自动挑出的违规候选里 5 个是误报
   或边界。bootstrap CI 只反映抽题误差，不包含分类误差；几 pp 的变化和分类误差在同一量级。
1. **对研究方向的含义。** 现有轨迹不足以支撑"outcome-based 交互式 RL 为了换取用户配合而系统性侵蚀合规"作为论文主线：变化存在，但 幅度小，而且
   reward 偏好的证据落在可以用"多做完几件事"解释的那一类上。如果继续走这个方向，需要 (a) 一个不被任务完成度混杂的对照： 同题内、写操作集合相同的
   episode 之间比较是否确认；(b) 用户更不可靠的设定（例如 680 的 Qwen 用户、或注入"同意即结束"的用户），
   看侵蚀是否随用户可靠度变强。后者若成立，也正好是和"多 persona 训练"类工作的区分点。
1. **简报的一处错误。** 简报里写 649 使用 DeepSeek 官方用户，Codex 按保存的 route 更正为火山 261017。这一点
   [privileged-critic-case-study](../privileged-critic-case-study/REPORT.md) 已经记录（"v3
   的训练用户后端不同"），不影响已有结论。
