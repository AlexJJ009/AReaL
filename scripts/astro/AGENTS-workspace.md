# /root/lgx — AReaL workspace on astro-a100

Rules for every agent (Codex, Claude) working in `/root/lgx` on `astro-a100`. Code
conventions and the research rules are in the repository's `AGENTS.md`, which Codex
reads inside the worktree; this file holds what only the machine decides.
`bin/dispatch.sh` prepends it to every Codex prompt.

This machine is a rented matpool container (Guangzhou, China Mobile egress): 24 CPU, 180
GB RAM, two A100-PCIE-40GB, 400 GB overlay root. It is used mostly by us, but `/root`
also holds other tools (Jupyter, VS Code/Trae servers, `/root/miniconda3`); leave them
alone. There is no init system: background processes die with the container.

**Codex runs without a filesystem sandbox here**: the container denies user namespaces,
so Codex's sandbox cannot start, and jobs run as root with full access. The guard hook
(`bin/codex-hooks/guard.py`) blocks the worst mistakes; everything else depends on you
staying inside the task's worktree and artifact directory.

Never read, modify or stop anything under `cpa/` or its processes (`cli-proxy-api`,
`mihomo`, `cpa-usage-keeper`): it is the model gateway every agent here depends on, and
it holds OAuth credentials. Never touch `/root/.ssh`, `/root/.codex`, or other people's
files under `/root`. Temporary files go to `tmp/` (`TMPDIR`, set by `bin/lib.sh`), not
`/tmp`; files worth keeping go to the task's artifact directory.

## Layout

| Path                              | Contents                                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `Areal/`                          | Main checkout on `codex/sao-math` (origin: the public GitHub fork)                                                  |
| `_worktrees/AReaL/<slug>/`        | One worktree per branch; `<slug>` is the branch name with `/` → `-`                                                 |
| `_artifacts/AReaL/<slug>/<task>/` | Everything a task produces: `BRIEF.md`, `logs/`, `REPORT.md`, evidence, run outputs (`main/` for the main checkout) |
| `bin/`                            | Workspace tools, installed from `Areal/scripts/astro/`; each script's header says how to use it                     |
| `_notes/AReaL/`                   | Backlog and notes outside every worktree                                                                            |
| `home/`, `tmp/`                   | `HOME` of the dev tools (uv cache, `CODEX_HOME`), and `TMPDIR`                                                      |
| `cpa/`                            | CPA stack (mode 700). Off limits                                                                                    |

## Branches, worktrees and jobs

- Create worktrees only with `bin/new-worktree.sh <branch> [<base>]`. Branches are
  `codex/<topic>`, merged back into `codex/sao-math`.
- A worktree holds source changes only. Run outputs, data copies and logs go to the
  artifact directory.
- Claude starts Codex with WSL `astro-codex` (runs `bin/dispatch.sh` detached) and waits
  with `astro-wait codex <branch> <task>`. Long jobs start with
  `bin/run-job.sh <branch> <task> -- <command>` and report through `bin/check.sh job`.
- Codex leaves its changes uncommitted and runs focused tests; Claude reviews and
  commits. Codex never pushes, never opens PRs and never uses `gh`.

## GPUs

The two A100s are local. A GPU run, GPU evaluation or paid API job needs the user's
approval first (see the repository `AGENTS.md`); Codex states in its REPORT what it
would run, and Claude decides. Check `nvidia-smi` before starting anything on a GPU.

## Disk

The root overlay has ~140 GB free (2026-11-14); `/public` is a large shared mount and
`/mnt` a 5 GB network disk. Check `df -h /root` before writing anything large; any
script writing GB-scale output must stop below 40 GB free.

## Network and keys

- Direct: GitHub, PyPI (and the `pypi.tuna` mirror), `api.deepseek.com`. Blocked:
  HuggingFace, OpenAI, Google (connection reset). For blocked hosts use ModelScope or a
  per-command proxy through the CPA egress (`https_proxy=http://127.0.0.1:17893`).
  Docker Hub is blocked and Docker cannot run here at all.
- Models are reached through the CPA at `http://127.0.0.1:8317/v1`. τ² users and judges
  use DeepSeek only: `deepseek-flash` on the official API, or
  `deepseek-v4-1-flash-261017` on the Volcano plan; never `deepseek-v4-pro`, never the
  Volcano pay-as-you-go endpoint, and never switch model or endpoint without the user's
  approval.
- Never print a key value and never write one into the repository or an artifact.
