#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: install.sh   Installs this directory's tooling into the AReaL research workspace on the ailab dev
# machine, then runs doctor.sh. Idempotent: it overwrites the workspace copies and creates nothing else.
#
# The workspace lives on a shared GPFS store that other jobs can write (2026-11-09: the AReaL checkout
# disappeared from it with no entry in any log of ours or of the two phai-relay sessions). So the scripts are
# maintained here, in the repository, and the copy on the store is disposable.
#
# Run it from a checkout on the dev machine itself:
#   cd <workspace>/AReaL && scripts/ailab/install.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
D=${AREAL_WORKSPACE:-/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx/areal-dev}
[ -d "$D" ] || { echo "no workspace at $D (set AREAL_WORKSPACE)" >&2; exit 1; }

mkdir -p "$D/bin/codex-hooks" "$D/models" "$D/_datasets" "$D/_notes/AReaL/grants" "$D/tmp" \
         "$D/home/.local/bin" "$D/home/.cache/uv" "$D/_artifacts/AReaL/main" "$D/_worktrees/AReaL"
install -m 644 "$HERE/lib.sh" "$D/bin/lib.sh"
install -m 644 "$HERE/fetch_model.py" "$D/bin/fetch_model.py"
for s in dispatch.sh check.sh new-worktree.sh doctor.sh rebuild-checkout.sh; do
  install -m 755 "$HERE/$s" "$D/bin/$s"
done
install -m 644 "$HERE"/codex-hooks/*.py "$D/bin/codex-hooks/"
install -m 644 "$HERE/AGENTS-workspace.md" "$D/AGENTS.md"
install -m 644 "$HERE"/manifests/*.json "$D/models/" 2>/dev/null || true
install -m 644 "$HERE/manifests/areal-tau2-data.json" "$D/_datasets/areal-tau2-data.json"
rm -f "$D/models/areal-tau2-data.json"  # dataset manifest belongs under _datasets only

# uv, codex and agent-wt come from the phai-relay workspace install; link them rather than keep a second copy.
for t in uv codex agent-wt; do ln -sfn "/root/phai-dev/home/.local/bin/$t" "$D/home/.local/bin/$t"; done

# The Codex hook config must point at the workspace copy of the guard.
mkdir -p "$D/home/.codex-cpa"
[ -s "$D/home/.codex-cpa/config.toml" ] || cp /root/phai-dev/home/.codex-cpa/config.toml "$D/home/.codex-cpa/config.toml"
chmod 600 "$D/home/.codex-cpa/config.toml"
cat > "$D/home/.codex-cpa/hooks.json" <<JSON
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {"type": "command", "command": "python3 $D/bin/codex-hooks/guard.py", "timeout": 10, "statusMessage": "AReaL Codex guard"}
        ]
      }
    ]
  }
}
JSON

echo "installed into $D"
python3 "$D/bin/codex-hooks/test_guard.py"
exec "$D/bin/doctor.sh"
