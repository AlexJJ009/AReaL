# SPDX-License-Identifier: Apache-2.0
# Sourced by bin/*. Environment of the AReaL workspace, and resolve <branch> -> WT / ART.
#
# Layout under D=$AREAL_WORKSPACE_ROOT: Areal/ (primary checkout), _worktrees/<slug>,
# _artifacts/<slug>, _notes/, bin/ (this tooling, installed by scripts/astro/install.sh),
# home/ (HOME for our jobs: caches, CODEX_HOME), tmp/, cpa/ (CPA stack if present, mode 700).
_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -r "$_DIR/workspace.env" ] && . "$_DIR/workspace.env"
: "${AREAL_WORKSPACE_ROOT:?Set AREAL_WORKSPACE_ROOT in workspace.env or environment}"
export D="$AREAL_WORKSPACE_ROOT"
export HOME=$D/home
export XDG_CACHE_HOME=$HOME/.cache XDG_STATE_HOME=$HOME/.local/state XDG_CONFIG_HOME=$HOME/.config
export UV_CACHE_DIR=$HOME/.cache/uv
REPO=$D/Areal
NOTES=$D/_notes
export TMPDIR=$D/tmp
export PATH=$HOME/.local/bin:$D/bin:$PATH
# Network (2026-11-14): GitHub, PyPI and api.deepseek.com are reachable directly; HuggingFace, OpenAI and Google
# are not (connection reset). Model and API traffic that needs the outside goes through CPA (127.0.0.1:8317) or
# its egress proxy 127.0.0.1:17893; set it per command (`https_proxy=http://127.0.0.1:17893 ...`), not here,
# so domestic mirrors (pip tuna, ModelScope) stay direct.
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
# Our Codex config: the CPA provider and the guard hook. /root/.codex is never used.
export CODEX_HOME=$HOME/.codex-cpa
# Default working directory: the Areal checkout, not the workspace parent.
cd "$REPO"

# resolve <branch>  ->  sets WT (its worktree) and ART (_artifacts/<slug>, created if missing)
# The slug is always the branch name with / -> -, including for the primary checkout (Areal/).
resolve() {
  WT=$(git -C "$REPO" worktree list --porcelain | awk -v b="refs/heads/$1" '/^worktree /{w=substr($0,10)} $0=="branch "b{print w}')
  if [ -z "$WT" ]; then echo "no worktree for branch $1; create it with bin/new-worktree.sh" >&2; return 1; fi
  local slug=${1//\//-}
  ART=$D/_artifacts/$slug
  mkdir -p "$ART"
}
