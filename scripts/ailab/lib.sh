# SPDX-License-Identifier: Apache-2.0
# Sourced by bin/*. Environment of the AReaL research workspace on the ailab dev machine, and
# resolve <branch> -> WT / ART.
#
# This machine is shared with a colleague and runs the phai-relay controller. HOME, caches and TMPDIR all point
# inside this workspace so nothing is written to /root (shared) or /tmp (forbidden by the dev-root AGENTS.md).
export HOME=/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev/home
export XDG_CACHE_HOME=$HOME/.cache XDG_STATE_HOME=$HOME/.local/state XDG_CONFIG_HOME=$HOME/.config
export UV_CACHE_DIR=$HOME/.cache/uv
D=/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev
REPO=$D/AReaL                # primary checkout; agent-wt derives _worktrees/AReaL and _artifacts/AReaL from it
NOTES=$D/_notes/AReaL        # BACKLOG.md and the rjob grants; outside every worktree
export TMPDIR=$D/tmp
# uv, codex and agent-wt are symlinks in our bin to the installs under /root/phai-dev/home/.local; /usr/bin comes
# before /usr/local/bin because /usr/local/bin/curl is broken on this machine.
export PATH=$HOME/.local/bin:/usr/bin:$PATH
# Outbound traffic goes through the platform egress proxy; internal mirrors stay direct.
export http_proxy=http://httpproxy-headless.kubebrain.svc.pjlab.local:3128 https_proxy=$http_proxy
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=localhost,127.0.0.1,::1,.pjlab.local,.pjlab.org.cn,100.64.0.0/10,10.0.0.0/8
export NO_PROXY=$no_proxy
# /root/.codex is the colleague's login: never used or edited. Ours holds the CPA provider and the Codex guard hook.
export CODEX_HOME=$HOME/.codex-cpa

# resolve <branch>  ->  sets WT (its worktree) and ART (_artifacts/AReaL/<slug>, created if missing)
resolve() {
  WT=$(git -C "$REPO" worktree list --porcelain | awk -v b="refs/heads/$1" '/^worktree /{w=substr($0,10)} $0=="branch "b{print w}')
  if [ -z "$WT" ]; then echo "no worktree for branch $1; create it with bin/new-worktree.sh" >&2; return 1; fi
  ART=${WT/\/_worktrees\//\/_artifacts\/}
  [ "$WT" = "$REPO" ] && ART=$D/_artifacts/AReaL/main
  mkdir -p "$ART"
}
