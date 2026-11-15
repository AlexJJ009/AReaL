# astro-a100 workspace tools

Tools for the temporary AReaL workspace on `astro-a100` (a rented matpool container: 24
CPU, 180 GB RAM, two A100-PCIE-40GB, Guangzhou / China Mobile egress), set up 2026-11-14
after the phai and ailab machines became unavailable. Adapted from `../ailab/`. The
machine-side copy under `/root/lgx/bin` is disposable; this directory is the source and
`install.sh` puts it back.

## What the machine can and cannot do (measured 2026-11-14)

| Item          | Result                                                                                                                                                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Privileges    | Unprivileged Docker container: no `CAP_SYS_ADMIN`/`NET_ADMIN`, seccomp + AppArmor `docker-default`, cgroups read-only, `mount` and `unshare` (incl. user namespaces) denied, no `/dev/kvm`, no `/dev/net/tun`, no init system |
| Docker        | Static dockerd 27.3.1 (`vfs`, no bridge/iptables) starts, but `docker import` already fails with `unshare: operation not permitted`; no container can run. Docker Hub is also blocked                                         |
| Codex sandbox | Not available (bubblewrap needs user namespaces; the legacy landlock path panics). Codex runs with `danger-full-access`; `codex-hooks/guard.py` is the only seatbelt                                                          |
| Network       | Direct: GitHub, PyPI/tuna, `api.deepseek.com`, the CPA trojan exits. Blocked: HuggingFace, OpenAI, Google, Docker Hub                                                                                                         |

## Layout on the machine (`/root/lgx`)

`Areal/` checkout (the primary working directory), `_worktrees/`, `_artifacts/`,
`_notes/`, `bin/` (this directory), `home/` (HOME, uv cache,
`CODEX_HOME=home/.codex-cpa`), `tmp/` (TMPDIR), `cpa/` (CPA stack, mode 700). The
machine-level agent rules are `AGENTS-workspace.md`, installed as `/root/lgx/AGENTS.md`.
SSH and Codex land in `Areal/` (the repo checkout), not the parent directory.

## Install and check

```bash
astro-ssh 'cd /root/lgx/Areal && scripts/astro/install.sh'   # copies bin/, runs guard tests + doctor
astro-ssh doctor.sh                                           # before any long job
```

`install.sh` never writes `cpa/keys` or an existing `home/.codex-cpa/config.toml`. The
config was placed once from the AiLab backup
(`~/phai-backup/secrets-20261113/credentials.tar`,
`root/phai-dev/home/.codex-cpa/config.toml`, project path changed to `/root/lgx/Areal`).
`uv` (0.11.1) is the WSL binary copied to `home/.local/bin`. `codex` (0.169.0) must be
the whole standalone release directory (`~/.codex/packages/standalone/releases/<ver>/`
on WSL, copied to `home/.local/lib/codex/` and symlinked): a lone `codex` binary starts,
but every tool call fails because `codex-code-mode-host` is missing. `doctor.sh` checks
for it.

## CPA (`cpactl.sh`, installed as `/root/lgx/bin/cpactl`)

The phai-cpa compose project rebuilt as native processes, because Docker cannot run
here:

| Service | Binary                                                        | Listens                                                            | Notes                                                                                                                                |
| ------- | ------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| egress  | mihomo v1.19.32                                               | 127.0.0.1:17893 (US exit), 127.0.0.1:17894 (Gemini fallback group) | the original cpu1 `mihomo/config.yaml`, dialling the same trojan exits directly; only `allow-lan`/`bind-address` changed to loopback |
| cpa     | CLIProxyAPI v8.0.42 (official release, checksum verified)     | 127.0.0.1:8317                                                     | the original `keys/config.yaml`; only `server.host` → 127.0.0.1 and `oauth.auth-dir` → `/root/lgx/cpa/keys/auths`                    |
| keeper  | CPA Usage Keeper v2.0.1 (official release, checksum verified) | \*:18319 (login required)                                          | env adapted to loopback addresses; DB restored from the 2026-11-12 AiLab backup (`keys/keeper/app.db`)                               |

```bash
astro-ssh cpactl status        # services, 401 from the API without a key, egress 204
astro-ssh cpactl start         # after a container restart (idempotent)
ssh -N -L 8317:127.0.0.1:8317 -L 18319:127.0.0.1:18319 astro-a100   # use it / see the dashboard from WSL
```

The OAuth credential is shared with any other CPA that has a copy (the AiLab one is down
as of 2026-11-14). Run only one CPA that refreshes it; two refreshers invalidate each
other's refresh token.

## WSL entry points (`wsl/`, installed in `~/.local/bin`)

| Tool                                              | Use                                                                                                                           |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `astro-ssh [--deadline S] [--raw] 'cmd'`          | run a command with the workspace environment (`bin/lib.sh`); retries only connection-setup failures, ends within the deadline |
| `astro-codex <branch> <task> <brief> [session]`   | upload the brief and start `bin/dispatch.sh` detached; the CPA key goes over SSH stdin                                        |
| `astro-wait [--every S] [--max S] codex\|job ...` | poll `bin/check.sh` until the job ends; run it in the background                                                              |

```bash
astro-ssh 'new-worktree.sh codex/<topic>'
astro-codex codex/<topic> <task> ./brief.md
astro-wait codex codex/<topic> <task>          # run_in_background
astro-ssh 'run-job.sh codex/<topic> <task> -- uv run pytest -q tests/test_x.py'
astro-wait job codex/<topic> <task>
```

## Syncing code

The checkout's `origin` is the public GitHub fork. Without a GitHub push, sync from WSL
over SSH: `git push astro <branch>:refs/heads/sync/<branch>` (WSL remote `astro` =
`astro-a100:/root/lgx/Areal`), then on astro `git merge --ff-only sync/<branch>` and
delete the sync branch.
