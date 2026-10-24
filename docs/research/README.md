# 研究文档索引

- [tau2/](tau2/README.md)：τ² 基线审计、训练数据、critic、SAO 运行、研究方向和决策记录。
  - [tau2/README.md](tau2/README.md)：问题、状态、结论和目录。
  - [tau2/decisions/](tau2/decisions/README.md)：τ² 决策记录（ADR）。
  - [tau2/experiments/](tau2/experiments/README.md)：实验记录，每个实验一份。
  - [tau2/critic/](tau2/critic/README.md)：critic 实验。
  - [tau2/research-direction/](tau2/research-direction/README.md)：研究方向讨论。
  - [tau2/scripts/](tau2/scripts/)：核查脚本。
- sao/：SAO 相关核查，2026-09。
  - [sao/2026-09-22-sao-miles-values-audit.md](sao/2026-09-22-sao-miles-values-audit.md)：yyht/miles-values
    SAO 实现核查。
  - [sao/2026-09-23-sao-paper-gap-audit.md](sao/2026-09-23-sao-paper-gap-audit.md)：SAO
    paper gap audit。
  - [sao/2026-09-23-sao-single-rollout-async-is.md](sao/2026-09-23-sao-single-rollout-async-is.md)：SAO
    的 single rollout、异步偏差与 PPO/GRPO 动机核对。
  - [sao/sao-critic-attention-freeze.md](sao/sao-critic-attention-freeze.md)：SAO hybrid
    critic attention
    freezing（[ADR 0004](../decisions/0004-sao-implementation-semantics.md) 引用）。
- areal/：AReaL 架构与 GRPO 调研。
  - [areal/areal-verl-architecture-guide.md](areal/areal-verl-architecture-guide.md)：从
    verl 到 AReaL，读懂配置、Rollout、Loss 与 Infra。
  - [areal/2026-09-22-areal-standard-grpo-survey.md](areal/2026-09-22-areal-standard-grpo-survey.md)：AReaL
    standard GRPO 调研。
