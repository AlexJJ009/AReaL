#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: scripts/install.sh
# Sets up the Python environment for AReaL training and evaluation.
# Run from the repository root. Machine-independent: works on any host
# that has uv, python3.12+ and git.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# 1. Prerequisites
for t in uv python3 git; do
  command -v "$t" >/dev/null || { echo "$t not found; install it first" >&2; exit 1; }
done
py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
  echo "python $py_ver OK"
else
  echo "python 3.12+ required (found $py_ver)" >&2; exit 1
fi

# 2. Create/update venv and install AReaL dependencies.
echo "running uv sync --extra cuda ..."
uv sync --extra cuda

# 3. Install tau2 from the pinned revision in examples/tau2/contracts.py.
TAU2_REV=$(python3 -c "
import ast, pathlib
src = pathlib.Path('examples/tau2/contracts.py').read_text()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == 'OFFICIAL_TAU2_REVISION':
                print(ast.literal_eval(node.value))
")
TAU2_REPO=$(python3 -c "
import ast, pathlib
src = pathlib.Path('examples/tau2/contracts.py').read_text()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == 'OFFICIAL_TAU2_REPOSITORY':
                print(ast.literal_eval(node.value))
")
echo "installing tau2 @ ${TAU2_REPO}@${TAU2_REV}"
uv pip install "tau2 @ git+${TAU2_REPO}@${TAU2_REV}"

# 4. Pre-commit hooks.
echo "installing pre-commit hooks ..."
# Activate the venv so pre-commit sees the project's Python.
source .venv/bin/activate
pre-commit install --install-hooks

# 5. Verify tau2 provenance.
python3 -c "
from examples.tau2.contracts import validate_installed_tau2_revision
info = validate_installed_tau2_revision()
print(f\"tau2 {info['version']} @ {info['revision'][:12]} OK\")
"

echo "install complete"
