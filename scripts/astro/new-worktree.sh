#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: new-worktree.sh <branch> [<base-ref>]
# Creates the branch's worktree at _worktrees/AReaL/<slug> and its artifact directory at _artifacts/AReaL/<slug>
# (<slug> = branch with / -> -). Default base: codex/sao-math. An existing branch is checked out as is.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
branch=$1 base=${2:-codex/sao-math}
slug=${branch//\//-}
wt=$D/_worktrees/AReaL/$slug
mkdir -p "$D/_worktrees/AReaL"
if git -C "$REPO" show-ref --verify --quiet "refs/heads/$branch"; then
  git -C "$REPO" worktree add "$wt" "$branch"
else
  git -C "$REPO" worktree add -b "$branch" "$wt" "$base"
fi
resolve "$branch"
echo "worktree $WT"; echo "artifacts $ART"
