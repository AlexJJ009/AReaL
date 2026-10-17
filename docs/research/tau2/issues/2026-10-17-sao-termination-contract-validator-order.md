# issue：`tests/test_sao_termination_contract.py` 3 项测试因 validator 顺序失败

发现于 2026-10-17（review `codex/user-td-split` 时；在 base `9626bc6b` 上同样失败，与该分支无关）。

## 现象

`test_entry_rejects_overrides_inconsistent_with_return_oracle` 的 3 个参数（分别改
`discount`、`gae_lambda`、`gae_timestep_unit`）失败。

- 测试用通用 PPOConfig 构造配置，预期报 `actor.<field>` 相关的错误。
- 实际先触发 dedicated async eval 的 guard：`saver.mode != sync`，报错信息与测试的 regex 不匹配。

证据：`_artifacts/AReaL/codex-sao-math/tau2-user-td-split-20261017/` 下的
`existing-regressions.log`、`preexisting-sao-base.log`、`preexisting-failures.md`。

## 影响

被测的 return-oracle 检查本身没有被验证到：目前这 3 项一直是红的，会掩盖这段检查真正的回归。

## 建议

二选一：

- 测试构造配置时先满足 async eval 的要求（`saver.mode: sync`），让它走到真正要测的 `actor.<field>` 检查；
- 如果 guard 的先后顺序本来就是要测的内容，就改测试的预期。

不要为了让测试通过而改 validator 的行为。
