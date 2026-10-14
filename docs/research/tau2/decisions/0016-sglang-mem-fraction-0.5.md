# critic mix 重跑把 SGLang `mem_fraction_static` 降到 0.5

- 状态：已采纳（用户确认，2026-10-14）

## Context

500 在 `mem_fraction_static=0.65` 下，一台 SGLang 在动态申请显存时 OOM 退出，训练端没有发现，之后约四分之一的 episode
仍发往它，连接失败 464 次，任务退出（[critic/data-mix.md](../critic/data-mix.md)）。

## Decision

重跑 510 只改这一项：`mem_fraction_static` 降到 0.5，每张推理卡多留约 12 GB 给动态显存，其余设置与 500 相同，从头训练。

## Consequences

- OOM 很可能与 TMS 有关（[ADR 0017](0017-sglang-strip-tms-and-fp32-a-log.md)），但没有复现，未证实。
- 待修：SGLang 服务器退出后训练端不会察觉。
