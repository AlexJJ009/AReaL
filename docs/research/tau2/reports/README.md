# 最终报告

来自 `_artifacts/AReaL/codex-sao-math/tau2-*` 的最终 Markdown 报告（不含日志、checkpoint 和数据）。报告里指向
`/data_storage/...` 的链接是运行时产物，不入库。

- [xccl-qwen35-round1.md](xccl-qwen35-round1.md)：Qwen3.5 xccl 推权重崩溃，第 1 轮（`A_log`
  精度修复）；对应 [issue](../issues/2026-10-14-xccl-qwen35-weight-sync.md)。
- [xccl-qwen35-round2.md](xccl-qwen35-round2.md)：第 2 轮 GPU 复现，报告写作时仍在进行中。
- [critic-recover-fix.md](critic-recover-fix.md)：critic-only recover 的修复报告。
- [sao-mix.md](sao-mix.md)：SAO mix pack 入口的实现报告（[sao-runs.md](../sao-runs.md) 未直接引用）。
- critic EV case study 的报告在
  [../critic/reports/](../critic/reports/ev-case-study-codex.md)。
