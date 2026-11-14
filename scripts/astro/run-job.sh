#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: run-job.sh <branch> <task> -- <command...>
# Runs a command detached in the branch's worktree; output goes to <artifacts>/<task>/logs/job.txt and the last
# line is EXIT=<rc>, which `check.sh job <branch> <task>` (and WSL `astro-wait job ...`) waits for.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
[ $# -ge 4 ] && [ "$3" = "--" ] || { sed -n 3p "$0" >&2; exit 2; }
resolve "$1"; TD=$ART/$2; shift 3
mkdir -p "$TD/logs"; out=$TD/logs/job.txt
[ -f "$out" ] && ! grep -q '^EXIT=' "$out" && { echo "task $TD already running" >&2; exit 1; }
cmd=$(printf '%q ' "$@")
cd "$WT"
setsid nohup bash -c "echo \"\$(date -Is) start: $cmd\"; $cmd; echo EXIT=\$?" > "$out" 2>&1 < /dev/null &
echo "started $! -> $out"
