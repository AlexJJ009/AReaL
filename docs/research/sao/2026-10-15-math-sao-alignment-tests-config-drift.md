# 数学 SAO 对齐测试因配置字段漂移失败（2026-10-15）

**状态**：未修，不影响 τ²。

**现象**：`tests/test_sao_official_alignment.py` 有 31 个测试失败，错误都是

```
omegaconf.errors.ConfigKeyError: Key 'evaluation_rollout' not in 'PPOConfig'
```

代表用例：`test_sao_yaml_resolves_current_async_ppo_contract`。在 `codex/tau2-ppo` 的基线
`4141393b` 上同样失败，与 τ² PPO 的改动无关；该测试文件、对应的数学 YAML 和入口在 `4141393b` 与 `17650347` 之间逐字节相同。

**原因**：数学 SAO 的 YAML 里有 `evaluation_rollout` 段，测试把它转换成 `PPOConfig`，而 `PPOConfig`
没有这个字段。τ² 的配置类型带有 `evaluation_rollout`，数学这边没有，是哪次提交引入的尚未查。

**怎么修**：二选一，需要先查 `evaluation_rollout` 何时进入数学 YAML。

- 数学 YAML 不需要独立评测 rollout：从 YAML 里去掉这一段。
- 需要：测试改用带 `evaluation_rollout` 字段的配置类型。

不改 `areal/api/cli_args.py`。

**发现经过**：τ² PPO 第二轮（pueue 557）补跑相关测试时发现。证据在
`_artifacts/AReaL/codex-sao-math/tau2-ppo-20261015/` 的
`tests-r2-supplement-green.log`（31
failed）、`tests-r2-baseline-sao.log`（基线复现）、`sao-alignment-baseline-comparison.json`（文件比较）。
