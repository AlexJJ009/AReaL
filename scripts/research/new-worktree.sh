#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: new-worktree.sh <branch> [base-ref (default codex/sao-math)]
# The only supported way to create a worktree: agent-wt places branch codex/<topic> at
# _worktrees/AReaL/codex-<topic> and its artifact directory at _artifacts/AReaL/codex-<topic>.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
agent-wt -C "$REPO" create "$1" --base "${2:-codex/sao-math}" --task "${1//\//-}" --json > /dev/null
resolve "$1"
agent-wt doctor "$WT" --json | python3 -c 'import json,sys; d=json.load(sys.stdin); e=d.get("errors") or []; [print("doctor:", x) for x in e + (d.get("warnings") or [])]; sys.exit(1 if e else 0)'
echo "worktree $WT"; echo "artifacts $ART"
