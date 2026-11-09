#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: doctor.sh   Checks the workspace is intact before any long job: prints one line per item and exits
# non-zero if any FAIL. On 2026-11-09 the AReaL checkout disappeared from this GPFS directory with no entry in
# any of our logs, so the state this prints is checked, not assumed. Rebuild the checkout with
# bin/rebuild-checkout.sh.
set -u
. "$(dirname "$0")/lib.sh"
rc=0
say() { printf '%-6s %s\n' "$1" "$2"; [ "$1" = FAIL ] && rc=1; return 0; }

[ -d "$REPO/.git" ] && say OK "checkout $REPO" || say FAIL "checkout missing: $REPO"
if [ -d "$REPO/.git" ]; then
  head=$(git -C "$REPO" rev-parse --short=7 HEAD 2>/dev/null)
  branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)
  dirty=$(git -C "$REPO" status --porcelain | wc -l)
  say OK "HEAD $head on $branch, $dirty uncommitted files"
fi
for f in bin/lib.sh bin/dispatch.sh bin/check.sh bin/new-worktree.sh bin/fetch_model.py bin/codex-hooks/guard.py AGENTS.md; do
  [ -s "$D/$f" ] && say OK "$f" || say FAIL "missing $f"
done
for t in uv codex agent-wt git python3; do
  command -v "$t" > /dev/null && say OK "tool $t -> $(command -v "$t")" || say FAIL "tool $t not on PATH"
done
# Models: every manifest in models/ must have a directory whose files all match (hash check, so this is slow;
# --quick compares file count and size only).
for m in "$D"/models/*.json; do
  [ -e "$m" ] || continue
  dir=$D/models/$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"].split("/")[-1])' "$m")
  if [ ! -d "$dir" ]; then say FAIL "model dir missing for $(basename "$m")"; continue; fi
  n=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["files"]))' "$m")
  have=$(find "$dir" -maxdepth 1 -type f ! -name '*.part' ! -name '*.done' | wc -l)
  [ "$have" -ge "$n" ] && say OK "model $(basename "$dir"): $have/$n files" ||
    say FAIL "model $(basename "$dir"): only $have of $n files (bin/fetch_model.py resumes)"
done
free=$(df -BG --output=avail "$D" | tail -1 | tr -dc 0-9)
[ "$free" -ge 40 ] && say OK "free space ${free}G" || say FAIL "free space ${free}G below the 40G floor"
exit $rc
