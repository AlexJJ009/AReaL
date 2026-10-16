# issue：`tests/test_tau2_critic_training.py` 11 项测试依赖官方原始 tau2 数据

发现于 2026-10-16（合并 critic 预实验分支时；合并前后失败相同，与合并无关）。

## 现象

`codex/sao-math`（`d8d040db` 及之后）上 11 项测试失败：

- 不设
  `TAU2_DATA_DIR`：`FileNotFoundError ... .venv/lib/python3.12/data/tau2/domains/airline/tasks.json`。
- 设为服务器上的缓存
  `_cache/tau2-bench-b7ea9074...-fixed-v3/data`：`ValueError: ... does not hold the official airline tasks (sha256 25086da1..., expected ccd8ba73...)`。

测试要求 `TAU2_DATA_DIR` 指向未修补的官方数据，服务器上只有 fixed-v3 版本。

## 影响

代码本身没有问题的证据；但这 11 项在服务器上一直是红的，掩盖了真实回归。

## 建议

二选一：

- 在服务器上准备一份官方原始数据（按 sha256 校验），测试通过环境变量或 fixture 指向它；
- 数据不可用时让这些测试 `skip` 并写明原因，而不是失败。
