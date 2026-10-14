# SAO 正式运行每 10 步评测并存档

- 状态：已采纳（用户，2026-10-14；`b8d95027`）

## Context

原配置每 5 步评测一次，会评 9 次、约 2160 条 episode，和训练本身一样多。正式任务 527 因评测每 10 步而存档每 5 步，被
`areal/trainer/async_eval.py` 的 `_validate_save_eval_sync` 拒绝，启动后 1.5
分钟退出（[sao-runs.md](../sao-runs.md)）。

## Decision

formal 模式评测改为每 10 步，同时设 `saver.freq_steps=10`，两者必须一致；训练前和每 10 步各评一次，共 5 次。

## Consequences

- smoke 模式两者都是 2，check-config 也不构建 trainer，所以之前没查出来；读回解析后的配置确认一致后重新排队（534）。
