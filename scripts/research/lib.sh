# SPDX-License-Identifier: Apache-2.0
# Sourced by scripts/research/*. Paths of the AReaL research workspace on phai.
C=/data_storage/yl_test/lgx/data-1/code
REPO=$C/AReaL                # primary clone: agent-wt derives _worktrees/ and _artifacts/ paths from it
NOTES=$C/_notes/AReaL        # BACKLOG.md, BACKLOG-done.md; outside every worktree
PUEUE_CONFIG=/data_storage/yl_test/lgx/runtime/verl/pueue/config/pueue.yml
export PATH=/data_storage/yl_test/lgx/home/.local/bin:$PATH

# resolve <branch>  ->  sets WT (its worktree) and ART (_artifacts/AReaL/<slug>, created if missing)
resolve() {
  WT=$(git -C "$REPO" worktree list --porcelain | awk -v b="refs/heads/$1" '/^worktree /{w=substr($0,10)} $0=="branch "b{print w}')
  if [ -z "$WT" ]; then echo "no worktree for branch $1; create it with scripts/research/new-worktree.sh" >&2; return 1; fi
  ART=${WT/\/_worktrees\//\/_artifacts\/}
  mkdir -p "$ART"
}
