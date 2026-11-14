#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: doctor.sh   Checks the astro workspace before a long job: one line per item, non-zero exit if any FAIL.
# The CPA stack does not survive a container restart (no init system); when doctor reports it DOWN, run
# bin/cpactl start.
set -u
. "$(dirname "$0")/lib.sh"
rc=0
say() { printf '%-6s %s\n' "$1" "$2"; [ "$1" = FAIL ] && rc=1; return 0; }

[ -d "$REPO/.git" ] && say OK "checkout $REPO" || say FAIL "checkout missing: $REPO"
if [ -d "$REPO/.git" ]; then
  say OK "HEAD $(git -C "$REPO" rev-parse --short=8 HEAD) on $(git -C "$REPO" rev-parse --abbrev-ref HEAD), $(git -C "$REPO" status --porcelain | wc -l) uncommitted files"
fi
for f in bin/lib.sh bin/dispatch.sh bin/check.sh bin/run-job.sh bin/new-worktree.sh bin/cpactl bin/codex-hooks/guard.py \
         home/.codex-cpa/config.toml home/.codex-cpa/hooks.json AGENTS.md; do
  [ -s "$D/$f" ] && say OK "$f" || say FAIL "missing $f"
done
for t in uv codex git python3 nvidia-smi; do
  command -v "$t" > /dev/null && say OK "tool $t -> $(command -v "$t")" || say FAIL "tool $t not on PATH"
done
if out=$("$D/bin/cpactl" status 2>&1); then say OK "cpa: $(echo "$out" | grep -E 'api|egress via' | tr '\n' ';')"
else say FAIL "cpa down (bin/cpactl start): $(echo "$out" | tr '\n' ';')"; fi
code=$(curl -s -o /dev/null -m 5 --noproxy '*' -w '%{http_code}' http://127.0.0.1:8317/v1/models || true)
[ "$code" = 401 ] && say OK "cpa api enforces auth (401 without key)" || say FAIL "cpa api without key returned $code (expected 401)"
gpus=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';')
[ -n "$gpus" ] && say OK "gpus: $gpus" || say FAIL "nvidia-smi shows no GPU"
free=$(df -BG --output=avail "$D" | tail -1 | tr -dc 0-9)
[ "$free" -ge 40 ] && say OK "free space ${free}G" || say FAIL "free space ${free}G below the 40G floor"
exit $rc
