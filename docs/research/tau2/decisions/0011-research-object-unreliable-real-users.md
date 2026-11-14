# 研究对象定为真实用户的不可靠反馈，模拟器是可控代理

- 状态：原动机于 2026-10-13 采纳；10-19 起实际实验收窄为模拟器训练，不再作为已验证的真人学习方法。

## Context

motivation review
指出：问题陈述说的是真实用户在线训练，证据却全部来自模拟器违反剧本，必须先定研究对象是哪一种（[research-direction/motivation-review-20261013.md](../research-direction/motivation-review-20261013.md)）。

## Decision

研究对象是真实用户的不可靠反馈。真人同样会确认后离开、改主意、不给信息，不能假定用户质量高。真人的 prompt 改不了、不能重来，所以改 prompt
和换模型只用来构造可靠度不同的用户（persona），按概率注入固定错误用来做受控对照；重采样不是可比 baseline。

## 后续范围变化

[10-19 问题与计划 §1](../research-direction/problem-and-plan.md)明确，用户剧本只在模拟器中可得，
因此方法针对“用模拟器训练”，不覆盖“从真人部署中学习”。[10-24 反思 §2–3](../research-direction/reflection-20261024.md)
进一步指出用户错误伤害训练尚未证实、归因不可辨识；[10-30 简报 §5、§7](../research-direction/brief-20261030/BRIEF.md)
记录了后续实验未稳定改善评测，并重新提出研究问题。原 Decision 保留为动机来源；真人反馈仍是愿景， 新的研究对象或方法需由用户选择，不能由这些实验结果视为已采纳。

## Consequences

- 训练用不可靠用户，评测用可靠用户并检查 policy 合规。
- 原要求是区分“用户错”和“用户难”；当时的决策 15 见 `5f66f66` 中的
  `research-direction/problem-and-plan.md`，不是现版 §1。后续的可辨识性限制见上述反思。
