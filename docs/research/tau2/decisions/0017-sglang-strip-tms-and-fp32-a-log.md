# 启动 SGLang 前清掉 TMS 预加载变量；`A_log` 按 fp32 传

- 状态：已采纳（`d8f72c51`、`6bf1d96e`，2026-10-14）

## Context

Qwen3.5-4B 上每次传权重都会让 SGLang 报 `illegal memory access`，SAO 一步都跑不完。原因：AReaL 启动 SGLang
时把训练端的环境变量原样传过去，其中有预加载 torch_memory_saver（TMS）的一项，SGLang 的显存申请被 TMS 接管。8 张卡上只改"是否继承
TMS"这一个条件，崩溃出现和消失（[infra-issues.md §3](../infra-issues.md)）。

## Decision

启动 SGLang 之前，把环境里 TMS 相关的变量去掉，其他不动，改动放在注入这些变量的同一个文件 `areal/utils/offload.py`。另外 Qwen3.5
的 `A_log` 参数按 fp32 传，因为 SGLang 用 fp32 存，bf16 传输会舍掉训练端的微小更新。

## Consequences

- SAO smoke 526 用修复后的代码跑了 2 步，两次推权重都成功。
- 627 个参数在 4 台 SGLang 上逐个核对，与训练端一致。
