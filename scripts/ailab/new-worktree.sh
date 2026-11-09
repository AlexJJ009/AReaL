#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: new-worktree.sh <branch> [<base-ref>] [<task-id>]
# The only supported way to create a worktree: agent-wt places it at _worktrees/AReaL/<slug> and its artifact
# directory at _artifacts/AReaL/<slug> (<slug> is the branch name with / -> -). Defaults: base codex/sao-math,
# task id = the slug.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
branch=$1 base=${2:-codex/sao-math} task=${3:-}
[ -n "$task" ] || task=${branch//\//-}
cd "$REPO"
agent-wt create "$branch" --base "$base" --task "$task" --json > /dev/null
resolve "$branch"
agent-wt doctor "$WT" --json |
  python3 -c 'import json,sys; d=json.load(sys.stdin); e=d.get("errors") or []; [print("doctor:", x) for x in e + (d.get("warnings") or [])]; sys.exit(1 if e else 0)'
echo "worktree $WT"; echo "artifacts $ART"
