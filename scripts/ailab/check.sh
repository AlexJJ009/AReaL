#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: check.sh <kind> <args...>   Non-blocking: exit 0 = finished (prints the result), 1 = still running
# (prints one status line), 2 = cannot tell (prints why). WSL `areal-wait` polls it; nothing here sleeps.
#   codex <branch> <task>       Codex task finished: process gone and logs/codex-last.txt present
#   job <branch> <task>         a detached job wrote EXIT= to <task>/logs/job.txt (see bin/run-job.sh)
#   rjob <job-name>...          every named rjob is terminal (Succeeded, Failed or Stopped)
. "$(dirname "$0")/lib.sh"
kind=${1:-}; shift || true
case $kind in
  codex)
    resolve "$1" 2>/dev/null || { echo "no worktree for $1"; exit 2; }
    log=$ART/$2/logs/codex
    if pgrep -f -- "-o $log-last.txt" > /dev/null || pgrep -f -- "dispatch.sh $1 $2 " > /dev/null; then
      echo "running; events $(( $(date +%s) - $(stat -c %Y "$log-events.jsonl" 2>/dev/null || echo 0) ))s ago"; exit 1
    fi
    [ -s "$log-last.txt" ] || { echo "exited without last.txt"; tail -n 5 "$log-stderr.txt" 2>/dev/null; exit 2; }
    cat "$log-last.txt"; exit 0 ;;
  job)
    resolve "$1" 2>/dev/null || { echo "no worktree for $1"; exit 2; }
    out=$ART/$2/logs/job.txt
    [ -f "$out" ] || { echo "no $out"; exit 2; }
    grep -q '^EXIT=' "$out" 2>/dev/null || { echo "running; last: $(tail -n 1 "$out" | cut -c1-160)"; exit 1; }
    tail -n 20 "$out"; exit 0 ;;
  rjob)
    [ $# -ge 1 ] || { sed -n 2,8p "$0"; exit 2; }
    pending=0
    for j in "$@"; do
      # rjob prints one "rjob <name> (showname=...): <State>" line per job; anything else means we cannot tell.
      s=$(. /etc/profile.d/ssh-init.sh 2>/dev/null; rjob get "$j" 2>/dev/null |
          sed -n "s/.*rjob $j (showname=[^)]*): \([A-Za-z]*\).*/\1/p" | head -n 1)
      case $s in
        Succeeded|Failed|Stopped) echo "$j $s" ;;
        "") echo "$j unknown"; exit 2 ;;
        *) echo "$j $s"; pending=1 ;;
      esac
    done
    exit $pending ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
