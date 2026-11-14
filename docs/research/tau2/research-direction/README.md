# 研究方向：RL 训练中如何对待用户模拟器的质量

本目录按日期保留问题、假设和结果的变化。10-30 简报汇总了完成的实验；旧计划不代表当前获批的方法，11-09 injection 设计仍是待讨论草案。

- [problem-and-plan.md](problem-and-plan.md)：10-19 至 10-20 的问题定义、PPO 推导和剧本 critic
  计划；persona 加权假设已有后续反证（旧版见提交 5f66f66）。
- [paper-roadmap.md](paper-roadmap.md)：10-14 至 10-16 的历史论文计划，动机链的证据缺口见 10-24 反思。
- [analysis.md](analysis.md)：用户噪声问题的分析：随机性与系统性偏离、候选信号、实验设计。
- [related-work.md](related-work.md)：相关工作与撞车检查。
- [motivation-review-20261013.md](motivation-review-20261013.md)：动机审查（2026-10-13）。
- [appendix-method-draft-16.11.md](appendix-method-draft-16.11.md)：方法设计草案，已暂停（[ADR 0010](../decisions/0010-pause-method-draft.md)）。
- [appendix-math-review-codex.md](appendix-math-review-codex.md)：对方法草案的数学审稿。
- [reflection-20261024.md](reflection-20261024.md)：切入点反思：为什么放弃"把用户的作用从 agent 的 credit
  里分出来"，转向带剧本 critic + bootstrap（2026-10-24）。
- [discussion-20261026-qwenuser-runs.md](discussion-20261026-qwenuser-runs.md)：Qwen
  用户训练的 run 3 / 3′ / 2′ / 4 的结果、advantage 与 loss 的完整式子、待做的 case study（2026-10-26）。
- [brief-20261030/](brief-20261030/README.md)：10-30 英文简报与评测、训练指标 CSV，汇总 2×2 和 seed-43
  结果；corruption-robust bandit 是待讨论方向。
- [injection/DESIGN.md](../injection/DESIGN.md)：11-09 用户错误注入设计，草案、待讨论，未实现或运行。
