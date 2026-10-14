---
task_id: 20261013T161101Z-sao-task-pack-policy-wrapper-f0be4efe
report_id: 20261013T161101Z-final-bda98c
kind: final
workspace: /data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math
generated_at: '2026-10-13T16:17:59Z'
window_start: '2026-10-13T16:17:59Z'
window_end: '2026-10-13T16:17:59Z'
---

# SAO task-pack policy wrapper

## 目标与结果 {#goal}

已实现 pack + 官方 fixed train 的 SAO policy 训练入口：
`scripts/tau2/run.sh policy-pack mixed sao train_task_pack=/absolute/pack ...`。 CPU
验证得到规格预期的 1280 prompts/epoch、60 validation prompts、40 steps、2560 episodes。 未提交代码，未启动 GPU
或 pueue 任务。

## 文件变更 {#progress}

共 8 个任务文件：

- `scripts/tau2/train_sao_pack.py`：新 wrapper，合并训练行、展开 schedule、检查采样次数并记录 pool 数量。
- `scripts/tau2/run.sh`：新增并说明 `policy-pack`，沿用 `config_sao.yaml`、domain pruning 和
  check-config 分支。
- `examples/tau2/train.py`：增加可选 `main(args, train_rows_hook=None)`，普通入口拒绝设置 pack；wrapper
  路径增加三字段 split 检查。
- `examples/tau2/utils.py`：新增默认 `None` 的 `Tau2PPOConfig.train_task_pack`。
- `examples/tau2/config_grpo.yaml`：声明 `train_task_pack: null`，让继承它的 SAO YAML 接受规格中的
  Hydra override。
- `examples/tau2/task_pack.py`：承接原 `_task_pack_rows`，保留字段、绝对路径转换和 shuffle 行为。
- `scripts/tau2/train_critic.py`：按原名导入共享 helper，移除原实现和不再使用的 import。
- `tests/test_tau2_sao_pack_training.py`：20 个 CPU 测试，覆盖组合行、预算、入口 guards、验证集、disjointness
  和 shell routing。

## Hook 与兼容性 {#decisions}

wrapper 调用原 `main()`，仅替换训练 dataset 构建。它用原 `get_tau2_dataset()` 取得未展开预算的官方行， 在前面加入共享
helper 生成的 pack 行；pack 行携带绝对 `task_pack` 路径与 provenance，官方行携带 `None`。 随后原
`build_sampling_schedule()` 处理合并 pool，再执行原 seeded shuffle。

用 `(domain, task_id, task_pack)` 对实际 schedule 计数，原 pool 中未采样的任务也参与检查。
每条任务必须有相同的正整数重复次数，跨域倍数不同、非整倍数预算和截断 schedule 的 `task_limit` 都报错。 没有 domain budget
时，每条任务排一次。验证集的原构建调用保持不变。

新增 disjointness guard 只作用于 wrapper，保留普通 qualification 路径原有的 train/dev overlap 行为。
[7 种原 policy selector 对比](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/legacy-policy-parity.json)的
argv、stdout、stderr 与 HEAD 逐字节一致。 独立代码审查发现的普通入口兼容问题已修复，复查无剩余具体问题。

## 测试与实测 {#evidence}

最终 focused 命令（设置官方 `TAU2_DATA_DIR`，`CUDA_VISIBLE_DEVICES=''`）：

```bash
.venv/bin/python -m pytest -q tests/test_tau2_sao_pack_training.py tests/test_tau2_critic_training.py tests/test_tau2_task_pack.py tests/test_tau2_sao_contract.py
```

[95 passed](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/pytest-focused.log)。这包括已有
critic / task-pack 测试，及验证集 rows、调用参数不变和 check-config 不创建 worker／解析 snapshot 的测试。

运行
[mutation-checks.py](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/mutation-checks.py)：依次禁用普通入口
pack 拒绝、pack 必填、SAO-only、fixed-only、domain coverage、 相同正重复次数、禁止截断、disjointness、shell
SAO-only 共 9 个 guards。
[每次对应 negative test 都失败](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/mutation-results.json)（exit
1），随后恢复源文件；最后再次运行 focused suite 得到上述 95 passed。

`source .venv/bin/activate; pre-commit run --files <上述 8 文件>` 的适用 hooks
[全部通过](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/pre-commit.log)；
`bash -n scripts/tau2/run.sh` 和 `git diff --check` 通过。未提交，因此未对其他 agents 的文件运行全仓格式修改。

真实 CPU 命令及完整环境见
[check-config.sh](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/check-config.sh)，使用指定
fixed-v3 data、pinned actor、critic 目录和全部指定 overrides。
[实际输出](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/check-config.log)：pack
airline 150 + official 30 = 180；pack retail 71 + official 69 = 140；每条 4 次， 得到 airline
720、retail 560，共 1280 prompts/epoch。batch 64 对应 20 steps/epoch；2 epochs 共 40 steps、2560
episodes；validation 60。 结果与规格一致，无需调整预期数字。

## 限制与已有失败 {#scope}

更广的 SAO/tau2 CPU
回归：[命令](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/pytest-cpu-regression-command.log)与[输出](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/pytest-cpu-regression.log)记录
**155 passed、31 failed、1 skipped**。31 项失败全部来自未修改的 `test_sao_official_alignment.py`： 30
项仍用 `PPOConfig` 读取含 `evaluation_rollout` 的 YAML（当前入口用 `SaoPPOConfig`）；1 项仍 patch
`PPOTrainer`（当前入口导入 `AsyncEvalPPOTrainer`）。 失败测试与 8
个相关源码/配置文件已[确认与 HEAD 字节一致](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-sao-mix/pytest-cpu-regression-baseline.log)，属于既有测试失配。
CUDA 参数因禁用 CUDA 而跳过。未修复这些范围外测试，也未验证 GPU 训练质量。

首轮的 Hydra key 缺失和官方数据目录未设置问题已修复，最终证据使用正确环境。 其他 agents 的 docs 和既有 `tau2-user-failures/`
保持原状；未修改 `areal/api/cli_args.py`、launcher/scheduler 或依赖。 除了上述既有回归失败，没有未解决的实现疑问。

## 交付状态 {#next_steps}

指定实现与 CPU 检查已完成，代码保留为未提交变更。本任务不需要额外授权动作；本报告不代表 GPU 训练已经运行。
