# 给 ChatGPT 网页端的上下文包（2026-10-30）

用途：让 ChatGPT 网页端了解项目现状，并就"借鉴 corruption-robust bandit 改 critic loss / advantage"做分析和调研。

## 文件

| 文件 | 内容 |
| --- | --- |
| `BRIEF.md` | 英文自包含简报：设定、PPO 推导、试过的方法与审稿意见、当前结果、两篇论文与映射、要问的问题 |
| `PROMPT.md` | 第一条消息的提示词 |
| `eval_mean4.csv` | 所有 run 各 checkpoint（step 0/30/60/90/120）、两种用户评测的 mean@4 |
| `train_per_step.csv` | 5 个 run 每步的训练 reward 和 critic EV |

CSV 由 `../../scripts/collect_brief_tables.py` 从备份的 `summary.json` 和 `main.log` 生成。

## 用法

1. 在 ChatGPT 建一个 Project，上传：`BRIEF.md`、两个 CSV、两篇论文 PDF，以及下面几份仓库文档（简报 §8 列的）：
   - `docs/research/tau2/research-direction/problem-and-plan.md`
   - `docs/research/tau2/research-direction/discussion-20261026-qwenuser-runs.md`
   - `docs/research/tau2/research-direction/appendix-math-review-codex.md`
   - `docs/research/tau2/experiments/2026-10-24-qwenuser-no-telecom.md`
   - `docs/research/tau2/research-direction/related-work.md`
2. 把 `PROMPT.md` 的内容作为第一条消息发出。
3. 若用 GitHub connector：只有推到 fork 之后它才能读到这些新文件。
