# areal-dev — AReaL τ² research workspace on the ailab dev machine

Rules for every agent (Codex, Claude) working in
`/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev`. Code conventions and
the research rules are in the repository's `AGENTS.md` and `AReaL/AGENTS.md`, which
Codex reads inside the worktree; this file holds what only the machine decides.
`bin/dispatch.sh` prepends it to every Codex prompt.

This machine (`science-agent`) runs the phai-relay production controller and is shared
with a colleague on the same root account. It is small: 6 CPU, 13 GB RAM, no GPU of its
own. GPUs come from RJobs.

Never stop, restart or reconfigure platform services. Never write to `/var/lib/phai`,
`/root/phai-deploy`, `/root/.codex` (the colleague's Codex login), `/root/phai-dev` (the
phai-relay workspace) or `.../gpfs2-agent-deepresearch/{zzx,nzy,zjy,shangjunlin}`. Never
write to `/tmp`: temporary files go to `tmp/` (`TMPDIR`, set by `bin/lib.sh`), files
worth keeping to the task's artifact directory. The sibling `.../lgx/sglang-qwen35-4b/`
belongs to another task; read only.

## Layout

| Path                              | Contents                                                                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `AReaL/`                          | Main checkout on `codex/sao-math`. Shallow (depth 460); `git fetch --unshallow` only if history is really needed                             |
| `_worktrees/AReaL/<slug>/`        | One worktree per branch; `<slug>` is the branch name with `/` → `-`                                                                          |
| `_artifacts/AReaL/<slug>/<task>/` | Everything a task produces: `BRIEF.md`, `logs/`, `REPORT.md`, evidence, run outputs. Work on the main checkout uses `_artifacts/AReaL/main/` |
| `bin/`                            | Workspace tools; each script's header says how to use it                                                                                     |
| `_notes/AReaL/`                   | `BACKLOG.md` and `grants/` (RJob grants, written only by Claude)                                                                             |
| `models/`                         | Pinned model revisions with their manifests; `bin/fetch_model.py <manifest> <dir>` verifies every file                                       |
| `home/`, `tmp/`                   | `HOME` of the dev tools, and `TMPDIR`                                                                                                        |

`uv`, `codex` and `agent-wt` in `home/.local/bin` are symlinks to the installs under
`/root/phai-dev/home/.local`; use them, do not reinstall or modify them.

## Branches, worktrees and jobs

- Create worktrees only with `bin/new-worktree.sh <branch> [<base>] [<task-id>]`; never
  `git worktree add` or a copied checkout. Branches are `codex/<topic>`, merged back
  into `codex/sao-math`.
- A worktree holds source changes only. Run outputs, data copies and logs go to the
  artifact directory.
- There is no pueue on this machine. Claude starts Codex with `areal-codex` (which runs
  `bin/dispatch.sh` detached) and waits with `areal-wait codex <branch> <task>`; long
  jobs start detached (`setsid nohup … < /dev/null &`) and report through
  `bin/check.sh`.
- Codex leaves its changes uncommitted and runs focused tests; Claude reviews and
  commits.
- Codex never pushes, never opens PRs and never uses `gh`. The GitHub remote `alex` is a
  public fork: pushing needs the user's approval, after a secret scan of the diff.

## RJobs (the only GPUs)

- Submission needs Claude's grant for the worktree: `_notes/AReaL/grants/<slug>.json`,
  e.g.
  `{"rjob_submit": {"max_gpu": 1}, "expires": <unix time>, "by": "claude", "why": "<what this run answers>"}`.
  `bin/codex-hooks/guard.py` refuses a submit without a grant, above its GPU count, or
  with a job name not starting with `lgx-`. Codex states in its REPORT what it would
  submit; Claude decides.
- Job names start with `lgx-`; `rjob delete|stop|patch` is limited to those, because the
  namespace (`ailab-llmagent`) is shared.
- Submission parameters that match this platform: `--namespace ailab-llmagent`,
  `--charged-group llmagent_gpu`, `--private-machine group`,
  `--image registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab`,
  `--mount gpfs://gpfs2/gpfs2-agent-deepresearch:/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch`,
  `--custom-resources brainpp.cn/fuse=1`, `--store-host-nvme=True`.
- An RJob pod sees only the GPFS mount, not `/root`. Everything a run needs (code, data,
  weights, venv) must be under `.../lgx/`. Scratch that does not need to survive goes to
  the pod's own NVMe, not to GPFS.
- RJob nodes reach the internet only through this machine's platform egress; no other
  tunnel or proxy.

## Disk

The GPFS store is shared and nearly full (10 TB, ~98% used). Check `df` before writing
anything large, keep only the newest checkpoints, and delete data copies when the
investigation ends. `bin/fetch_model.py` stops below 40 GB free; any script that writes
GB-scale output must do the same.

## Keys and network

- Model and notification keys live in `/root/phai-dev/home/.config/phai-keys/` (mode
  600, read only for us) and the τ² keys in `/root/phai-dev/home/.config/tau2-doubao/`.
  Read that directory's `README.md` before choosing one. Never print a key value and
  never write one to disk on a worker host; load with `set -a; . <file>; set +a`.
- τ² users and judges use DeepSeek only: `deepseek-flash` on the official API, or
  `deepseek-v4-1-flash-261017` on the Volcano plan. Never `deepseek-v4-pro`, never the
  Volcano pay-as-you-go endpoint, and never switch model or endpoint without the user's
  approval.
- Outbound traffic goes through `httpproxy-headless.kubebrain.svc.pjlab.local:3128`
  (`bin/lib.sh` exports it); internal mirrors are direct. `/usr/local/bin/curl` is
  broken — use `/usr/bin/curl`.
