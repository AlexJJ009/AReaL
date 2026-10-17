# issue：DeepSeek user simulator 在 2026-10-17 上午换了后端模型

发现于 2026-10-18（北京时间），通过扫描 trajectory JSON 里每次 user / judge 调用的
`system_fingerprint` 与 `created`。

## 事实

- 配置里的模型名始终是 alias `deepseek-flash`，返回的 `model` 字段也不变。
- `system_fingerprint` 在 2026-10-17 09:30–09:49（北京时间）之间从
  `aeb56401ca74e127821c4f9126dcb669` 变成 `e72d3c7f663d3e1383b00b3884cb4501`。
- 2026-10-18 查询 `GET /models`：`deepseek-flash` 的 `name` 为 `DeepSeek-V4.1-Flash`；
  列表里只有 `deepseek-flash` 和 `deepseek-v4-pro`，没有可固定的旧版本 id。

| run | 后端 | 说明 |
|---|---|---|
| PPO v3 + ext（0–120） | 全部旧 | 训练和全部 eval |
| GRPO-matched | 全部旧 | |
| SAO `sao-formal-qwen3-v3-mse` | 混合 | 训练约 step 66 之前为旧、之后为新；eval 0–50 旧，eval 60 混合，eval 70–120 新 |
| D1 telecom full（579） | 全部新 | |

## 影响

- PPO 与 SAO 在 step ≥70 的 endpoint 比较是跨 user 的比较。
- SAO 自身的 0→70/120 和 70→100/110/120 比较也跨 user。
- Codex 595 的 SAO case study 不知道这次切换；其中跨算法的后期显著差异（如 step 110 PPO−SAO
  +9.75pp）不能归因于算法。
- 粗看 user 行为（SAO eval trajectory，按 fingerprint 分组）：user 消息平均字符数从旧的
  180–182 降到新的 172–177，每 episode user 轮数 13.4–13.7 → 13.5–14.1。这些数字和 policy
  变化混在一起，汇总接近也不能排除影响 reward 的变化（STOP 时机、编造细节等）。
  脚本：`phai:/data_storage/yl_test/lgx/tmp/claude-boot/fpswitch.py`。

## 建议

- 不重跑训练。在当前 user 上重新 eval 固定 checkpoint（base、PPO 120、SAO 70、SAO 120），
  让所有 endpoint 比较落在同一 user 上。需要用户批准（GPU + API）。
- 之后每个 run 记录 `system_fingerprint`，run 中途 fingerprint 变化时报警或中止。
- 长期可考虑自部署开源权重的 user simulator，以保证可复现。
