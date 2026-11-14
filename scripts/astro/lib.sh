# SPDX-License-Identifier: Apache-2.0
# Sourced by bin/*. Environment of the AReaL workspace on astro-a100 (a matpool container with two A100s,
# Guangzhou, China Mobile egress), and resolve <branch> -> WT / ART.
#
# Layout under D=/root/lgx: Areal/ (primary checkout), _worktrees/AReaL/<slug>, _artifacts/AReaL/<slug>,
# _notes/AReaL, bin/ (this tooling, installed by scripts/astro/install.sh), home/ (HOME for our jobs: caches,
# CODEX_HOME), tmp/, cpa/ (the CPA stack, mode 700, see bin/cpactl).
export D=/root/lgx
export HOME=$D/home
export XDG_CACHE_HOME=$HOME/.cache XDG_STATE_HOME=$HOME/.local/state XDG_CONFIG_HOME=$HOME/.config
export UV_CACHE_DIR=$HOME/.cache/uv
REPO=$D/Areal
NOTES=$D/_notes/AReaL
export TMPDIR=$D/tmp
export PATH=$HOME/.local/bin:$D/bin:$PATH
# Network (2026-11-14): GitHub, PyPI and api.deepseek.com are reachable directly; HuggingFace, OpenAI and Google
# are not (connection reset). Model and API traffic that needs the outside goes through CPA (127.0.0.1:8317) or
# its egress proxy 127.0.0.1:17893; set it per command (`https_proxy=http://127.0.0.1:17893 ...`), not here,
# so domestic mirrors (pip tuna, ModelScope) stay direct.
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
# Our Codex config: the CPA provider and the guard hook. /root/.codex is never used.
export CODEX_HOME=$HOME/.codex-cpa

# resolve <branch>  ->  sets WT (its worktree) and ART (_artifacts/AReaL/<slug>, created if missing)
resolve() {
  WT=$(git -C "$REPO" worktree list --porcelain | awk -v b="refs/heads/$1" '/^worktree /{w=substr($0,10)} $0=="branch "b{print w}')
  if [ -z "$WT" ]; then echo "no worktree for branch $1; create it with bin/new-worktree.sh" >&2; return 1; fi
  if [ "$WT" = "$REPO" ]; then ART=$D/_artifacts/AReaL/main; else ART=$D/_artifacts/AReaL/$(basename "$WT"); fi
  mkdir -p "$ART"
}
