#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: dispatch.sh <branch> <task> <brief-file>  ->  prints the pueue task id
# Queues Codex (gpt-6.1-sol) in pueue group codex-review, working in the branch's worktree so it reads the
# repository AGENTS.md. Task directory <artifacts>/<task>/ holds BRIEF.md, run-codex.sh, logs/codex.log,
# logs/last.txt and everything the task writes. CODEX_TIMEOUT (seconds, default 14400) bounds the run.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
resolve "$1"
TD=$ART/$2
mkdir -p "$TD/logs"
[ "$(realpath "$3")" = "$TD/BRIEF.md" ] || cp "$3" "$TD/BRIEF.md"
cat > "$TD/run-codex.sh" <<RUN
#!/usr/bin/env bash
cd "$TD"
exec timeout ${CODEX_TIMEOUT:-14400} codex exec -m gpt-6.1-sol --skip-git-repo-check -s danger-full-access \\
  -c approval_policy=never -c model_reasoning_effort=high -C "$WT" -o "$TD/logs/last.txt" \\
  "Read $TD/BRIEF.md and complete it. Write everything the task produces under $TD." < /dev/null > "$TD/logs/codex.log" 2>&1
RUN
pueue -c "$PUEUE_CONFIG" add -g codex-review -w "$TD" -l "$2" -p -- bash "$TD/run-codex.sh"
