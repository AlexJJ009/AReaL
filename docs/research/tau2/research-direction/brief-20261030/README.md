# 给 ChatGPT 网页端的上下文包（2026-10-30）

用途：让 ChatGPT 网页端了解项目现状，帮助重新澄清研究问题、构思下一步的方法。

## 文件

| 文件                 | 内容                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| `BRIEF.md`           | 英文自包含简报：设定、PPO 推导、试过的方法与审稿意见、当前结果、导师建议的两篇 bandit 论文、待讨论的问题 |
| `eval_mean4.csv`     | 所有 run 各 checkpoint（step 0/30/60/90/120）、两种用户评测的 mean@4                                     |
| `train_per_step.csv` | 5 个 run 每步的训练 reward 和 critic EV                                                                  |

CSV 由 `../../scripts/collect_brief_tables.py` 从备份的 `summary.json` 和 `main.log` 生成。

## 可以一起给的材料

`BRIEF.md` 是入口，其余按需要给。简报 §8 列了它引用的仓库文档；问到推导细节时再补
`problem-and-plan.md`、`discussion-20261026-qwenuser-runs.md`、`appendix-math-review-codex.md`
这类原文更有效。两篇论文的 PDF 在本地 `.scratch/papers/`。GitHub connector 只能读到已推送到 fork 的文件。
