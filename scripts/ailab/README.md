# ailab 开发机的工作区工具

phai 的 GPU 机器被回收后，研究工作搬到 ailab 开发机（`science-agent`），GPU 通过平台的 `rjob` 申请。本目录是那台机器上 工作区
`/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev` 的工具源，从 phai-relay 的
`/root/phai-dev/bin` 迁移改写而来（去掉 pueue，换成分离进程加 `check.sh` 轮询）。

**为什么存在仓库里**：那块 GPFS 是共享的，别的作业也可写。2026-11-09 工作区里的 AReaL checkout 消失，我们自己和两个 phai-relay
会话的日志里都查不到对应操作（见 `../../docs/research/tau2/infra-issues.md`）。所以脚本在这里维护，机器上的 那份是可丢弃的副本，用
`install.sh` 装回去。

## 安装与自检

```bash
# 在开发机上的 checkout 里
cd <workspace>/AReaL && scripts/ailab/install.sh    # 覆盖安装 + 跑守卫测试 + doctor
<workspace>/bin/doctor.sh                            # 长任务开跑前先跑一次
<workspace>/bin/rebuild-checkout.sh                  # checkout 丢了用它重建（fork + bundle）
```

`doctor.sh` 会检查 checkout、脚本、工具、模型文件数和剩余空间，任何一项失败就返回非 0。改它的时候要确认它还能变红：把 `AGENTS.md`
临时挪开，它必须报 FAIL。

## 文件

| 文件                   | 作用                                                                       |
| ---------------------- | -------------------------------------------------------------------------- |
| `lib.sh`               | 工作区环境（HOME、缓存、TMPDIR、代理）和 `resolve <branch>` → `WT` / `ART` |
| `dispatch.sh`          | 在分支的 worktree 里跑 Codex（`gpt-6.1-sol`），连接错误自动续跑            |
| `check.sh`             | 非阻塞状态查询：`codex` / `job` / `rjob`                                   |
| `new-worktree.sh`      | 唯一创建 worktree 的方式（走 `agent-wt`）                                  |
| `doctor.sh`            | 工作区完整性检查                                                           |
| `rebuild-checkout.sh`  | 从公开 fork 加 `_notes/AReaL/sao-math-ahead.bundle` 重建 checkout          |
| `fetch_model.py`       | 按清单下载并逐文件校验（LFS 文件用 sha256，其余用 git blob sha1）          |
| `codex-hooks/guard.py` | Codex 的 PreToolUse 守卫：rjob 要授权、任务名必须 `lgx-`、不许 push        |
| `manifests/*.json`     | 固定的模型与数据集版本                                                     |

## 本机（WSL）上的入口

`~/.local/bin/areal-codex` 和 `~/.local/bin/areal-wait`，都经 `phai-ssh --on ailab` 到开发机。CPA
key 只在 WSL 上，经 SSH stdin 传入，不落盘。

## 下载源

- 模型权重走 **ModelScope**（代理下 20 MB/s）；HuggingFace 只有 0.3 MB/s。校验用的仍是 HF 的哈希，所以镜像差一个字节
  就会失败，而不是被当成对的。
- Python 包走 **pypi.org + 平台代理**；清华镜像在这台机器上 403，内部镜像不可达。
- `/usr/local/bin/curl` 是坏的，用 `/usr/bin/curl`。
