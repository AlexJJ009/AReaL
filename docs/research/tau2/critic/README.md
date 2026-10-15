# critic 实验

critic-only 训练了好几轮（479、499、510），验证集上的 EV 一直在 0 附近，最好也只有 0.1–0.2。这里记录扩数据、value
head、数据配比、EV 上限和 case study。

- [critic0-expanded-data.md](critic0-expanded-data.md)：critic₀ 扩数据实验（478–480）。
- [value-head.md](value-head.md)：优化器与 value head（决策见
  [ADR 0012](../decisions/0012-critic-head-init.md)）。
- [ev-ceiling.md](ev-ceiling.md)：只按题目身份预测时的 EV 上限。
- [ev-ceiling-explainer.md](ev-ceiling-explainer.md)：解释方差（EV）与"按题目身份预测的上限"。
- [data-mix.md](data-mix.md)：critic 训练数据配比（决策见
  [ADR 0013](../decisions/0013-critic-data-mix.md)）。
- [ev-case-study.md](ev-case-study.md)：critic EV case study；对应
  [issue](../issues/2026-10-14-critic-ev-case-study.md)。
- SAO 使用的 critic：[sao-runs.md](../sao-runs.md)。
