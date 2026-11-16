#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: install.sh   Installs this directory's tooling into the AReaL workspace, then runs the guard tests
# and doctor.sh. Idempotent: overwrites the workspace copies, never touches cpa/keys or the Codex
# config.toml once present. Run it from the checkout on the machine:
#   cd $AREAL_WORKSPACE_ROOT/Areal && scripts/astro/install.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ -r "$HERE/workspace.env" ] && . "$HERE/workspace.env"
: "${AREAL_WORKSPACE_ROOT:?Set AREAL_WORKSPACE_ROOT in workspace.env or environment}"
D="$AREAL_WORKSPACE_ROOT"
mkdir -p "$D/bin/codex-hooks" "$D/bin/manifests" "$D/_notes" "$D/tmp" "$D/home/.local/bin" "$D/home/.cache/uv" \
         "$D/home/.codex-cpa" "$D/_artifacts" "$D/_worktrees" "$D/models" "$D/_datasets"
install -m 644 "$HERE/workspace.env" "$D/bin/workspace.env"
install -m 644 "$HERE/lib.sh" "$D/bin/lib.sh"
install -m 644 "$HERE/fetch_model.py" "$D/bin/fetch_model.py"
install -m 644 "$HERE"/manifests/*.json "$D/bin/manifests/" 2>/dev/null || true
for s in dispatch.sh check.sh run-job.sh new-worktree.sh doctor.sh; do install -m 755 "$HERE/$s" "$D/bin/$s"; done
install -m 700 "$HERE/cpactl.sh" "$D/bin/cpactl"
ln -sfn "$D/bin/cpactl" "$D/cpa/cpactl" 2>/dev/null || true
install -m 644 "$HERE"/codex-hooks/*.py "$D/bin/codex-hooks/"
install -m 644 "$HERE/AGENTS-workspace.md" "$D/AGENTS.md"

# config.toml (CPA provider, env exclusion of *KEY*) is placed once from the AiLab backup; keep it if present.
[ -s "$D/home/.codex-cpa/config.toml" ] || { echo "missing $D/home/.codex-cpa/config.toml (see scripts/astro/README.md)" >&2; exit 1; }
chmod 600 "$D/home/.codex-cpa/config.toml"
cat > "$D/home/.codex-cpa/hooks.json" <<JSON
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {"type": "command", "command": "python3 $D/bin/codex-hooks/guard.py", "timeout": 10, "statusMessage": "AReaL astro guard"}
        ]
      }
    ]
  }
}
JSON
echo "installed into $D"
python3 "$D/bin/codex-hooks/test_guard.py"
exec "$D/bin/doctor.sh"
