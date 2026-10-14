# fixed-v3 不采纳两项补丁：9 题 TRANSFER 说明和 retail:45 追问

- 状态：已采纳（Claude review，2026-10-12）

## Context

Codex 的 fixed-v3
初版（`fab5e7b9`）提出两项修改（[benchmark-variants.md §1.2.1](../benchmark-variants.md)）：给 9
道题（airline 13/26/31/32/48，retail 27/32/65/94）追加"请求转人工后要等真正转接再发 ###TRANSFER###"；让
retail:45 的用户在只有一个选项时追问其他选项。

## Decision

两项都回退。tau2 全局用户规则 `simulation_guidelines.md` 第 15 行已规定被转接后才发
TRANSFER，用户提前发是模拟器违反协议（U7a），不是题目问题，只在 9 道题里重复等于针对一批题打补丁。retail:45 数据库里有 3 款可用的 canister
吸尘器，agent 只列一款属于枚举不全（M7），这道题本来就在考这一点，case study 里的 B1 标注应改为 agent 失败。

## Consequences

- 提前 TRANSFER 的影响记在 [user-simulator.md §2](../user-simulator.md) 的已知偏差里。
- airline:44 的 gold 问题已发现，尚未修补（见
  [benchmark-variants.md §1.2.1](../benchmark-variants.md)）。
