# 旧机器与工作区配置归档

归档日期：2026-11-13。用户确认 phai 开发机和 ailab workspace 当前均不可用，工作暂时在 WSL 本地进行。
以下保留旧环境的路径与工具用法。配置新机器时，按实际目录、账号、凭据、资源和权限重新核对。

## phai

- SSH 别名：`phai-lgx-dev`。
- 研究 worktree：`/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math`，分支
  `codex/sao-math`。
- worktree 工具：`scripts/research/new-worktree.sh codex/<topic>`。
- Codex 调度：`scripts/research/dispatch.sh <branch> <task> <brief>`，通过 pueue 的
  `codex-review` group 运行 `gpt-6.1-sol`。
- pueue 配置：`/data_storage/yl_test/lgx/runtime/verl/pueue/config/pueue.yml`。
- 运行产物：`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/<worktree>/<task>/`；Codex
  日志为 `logs/codex.log`。
- backlog：`/data_storage/yl_test/lgx/data-1/code/_notes/AReaL/BACKLOG.md`，配套入口为
  `scripts/research/backlog`。
- τ² 凭据目录：`/data_storage/yl_test/lgx/home/.config/tau2-doubao/`。

WSL 与 phai 原先使用以下同步方式：

1. WSL 本地提交后，执行 `git push phai codex/sao-math`。
1. 当时 `remote.phai.push = refs/heads/*:refs/heads/sync/*`，因此远端接收到 `sync/codex/sao-math`。
1. 在 phai 上执行 `git merge --ff-only sync/codex/sao-math`，对改动文件运行 pre-commit，再删除 sync 分支。
1. WSL 执行 `git pull --ff-only`，取回远端新增的提交。

`pueue add` 当时使用任务自己的 artifact 目录作为工作目录，避免临时文件遮蔽 Python 标准库。 工具源码仍保存在
`scripts/research/`。

## ailab workspace

- 开发机：`science-agent`；工作区：`/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev`。
- 工具说明：[scripts/ailab/README.md](../../../../scripts/ailab/README.md)。
- 机器规则与资源边界：[AGENTS-workspace.md](../../../../scripts/ailab/AGENTS-workspace.md)。
- 该环境通过 detached process 和 `check.sh` 管理作业，GPU 由平台 RJob 提供。
- WSL 当时通过 `areal-codex` / `areal-wait`，经 `phai-ssh --on ailab` 调用远端工具。

两套工具保留为迁移材料。当前 WSL 入口和仓库规则不加载旧机器指令。
