#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: rebuild-checkout.sh   Recreates AReaL/ from the public fork plus the commits only WSL had.
# Needed because the GitHub fork is one commit behind our work: _notes/AReaL/sao-math-ahead.bundle carries the
# rest, so no transfer from WSL is required. A full clone of the fork is killed on this machine (13 GB RAM, the
# fork has the whole upstream history), hence --depth.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
BUNDLE=$NOTES/sao-math-ahead.bundle
[ -s "$BUNDLE" ] || { echo "missing $BUNDLE" >&2; exit 1; }
[ -e "$REPO" ] && { echo "$REPO already exists; move it aside first" >&2; exit 1; }
cd "$D"
nice -n 10 git clone --depth 460 --single-branch --branch codex/sao-math --origin alex \
  https://github.com/AlexJJ009/AReaL.git "$(basename "$REPO")"
cd "$REPO"
git bundle verify "$BUNDLE"
git fetch "$BUNDLE" 'refs/heads/codex/sao-math:refs/remotes/wsl/codex-sao-math'
git merge --ff-only refs/remotes/wsl/codex-sao-math
echo "HEAD $(git rev-parse HEAD)  tree $(git rev-parse 'HEAD^{tree}')"
