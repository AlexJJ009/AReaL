#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: dispatch.sh <branch> <task> <brief-file> [resume-session-id]
# Runs Codex (gpt-6.1-sol, through the CPA on 127.0.0.1:8317) in the branch's worktree, so it reads the
# repository AGENTS.md. The task directory <artifacts>/<task>/ holds BRIEF.md and logs/. WSL `astro-codex` starts
# this script detached and `astro-wait codex` polls bin/check.sh, which is why nothing here backgrounds itself.
# Logs: <task>/logs/codex-{events.jsonl,last.txt,stderr.txt}.
#
# NO SANDBOX: this machine denies user namespaces, so Codex's bubblewrap sandbox cannot start (2026-11-14), and
# jobs run with danger-full-access as root. The guard hook ($CODEX_HOME/hooks.json -> bin/codex-hooks/guard.py)
# is the only seatbelt; keep briefs scoped and review the diff.
# A turn that fails on a transport/provider error (no last.txt) is resumed in the same session, at most
# MAX_AUTO_RESUME times (default 8), backing off 1, 2, 4, then 10 minutes.
# CPA_API_KEY must be in the environment (astro-codex passes it over SSH stdin; it is never written here), and
# $CODEX_HOME/config.toml keeps it out of the commands Codex runs.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
resolve "$1"
TD=$ART/$2
mkdir -p "$TD/logs"
[ "$(realpath "$3")" = "$TD/BRIEF.md" ] || cp "$3" "$TD/BRIEF.md"
PROMPT=$TD/BRIEF.md
LOG=$TD/logs/codex
cd "$WT"
[ -n "${CPA_API_KEY:-}" ] || { echo "missing credentials" >&2; exit 1; }
SANDBOX=(--dangerously-bypass-hook-trust -c sandbox_mode=danger-full-access -c approval_policy=never)
CONTINUE="The previous turn stopped on a model connection error; nothing is wrong with your work. Continue the same task from where you stopped and finish it as the original prompt says."

resume() {  # resume <session-id> <prompt-file>
  rm -f "$LOG-last.txt"
  codex exec resume -m gpt-6.1-sol "${SANDBOX[@]}" --json -o "$LOG-last.txt" "$1" - < "$2" \
    >> "$LOG-events.jsonl" 2>> "$LOG-stderr.txt"
}

transient_failure() {  # the last turn failed for a reason outside the task
  [ ! -s "$LOG-last.txt" ] && tail -n 5 "$LOG-events.jsonl" | grep -q '"turn.failed"' &&
    tail -n 5 "$LOG-events.jsonl" | grep -qiE 'stream disconnected|network error|unknown provider|high demand|overloaded|capacity|rate limit|timed out|status (429|5[0-9][0-9])|connection (reset|refused)'
}

status=0
if [ -n "${4:-}" ]; then
  SESSION=$4; resume "$SESSION" "$PROMPT" || status=$?
else
  # The workspace AGENTS.md is prepended because Codex only discovers AGENTS.md files inside the worktree.
  { cat "$D/AGENTS.md"; printf '\n---\nArtifact directory for this task: %s\n---\n\n' "$TD"; cat "$PROMPT"; } |
    codex exec -m gpt-6.1-sol "${SANDBOX[@]}" --json -o "$LOG-last.txt" -C "$WT" - \
      > "$LOG-events.jsonl" 2> "$LOG-stderr.txt" || status=$?
  SESSION=$(head -n 1 "$LOG-events.jsonl" | sed -n 's/.*"thread_id":"\([^"]*\)".*/\1/p')
fi
printf '%s\n' "$CONTINUE" > "$LOG-continue.txt"
for attempt in $(seq "${MAX_AUTO_RESUME:-8}"); do
  { [ -n "$SESSION" ] && transient_failure; } || break
  echo "$(date -Is) transient failure; auto-resume $attempt of session $SESSION"
  pause=$(( 60 << (attempt - 1) )); sleep $(( pause > 600 ? 600 : pause ))
  status=0; resume "$SESSION" "$LOG-continue.txt" || status=$?
done
exit "$status"
