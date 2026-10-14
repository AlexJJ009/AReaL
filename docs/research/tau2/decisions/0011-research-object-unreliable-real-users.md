# 研究对象定为真实用户的不可靠反馈，模拟器是可控代理

- 状态：已采纳（2026-10-13）

## Context

motivation review
指出：问题陈述说的是真实用户在线训练，证据却全部来自模拟器违反剧本，必须先定研究对象是哪一种（[research-direction/motivation-review-20261013.md](../research-direction/motivation-review-20261013.md)）。

## Decision

研究对象是真实用户的不可靠反馈。真人同样会确认后离开、改主意、不给信息，不能假定用户质量高。真人的 prompt 改不了、不能重来，所以改 prompt
和换模型只用来构造可靠度不同的用户（persona），按概率注入固定错误用来做受控对照；重采样不是可比 baseline。

## Consequences

- 训练用不可靠用户，评测用可靠用户并检查 policy 合规。
- 方法必须区分"用户错"和"用户难"（[research-direction/problem-and-plan.md §1](../research-direction/problem-and-plan.md)
  决策 15）。
