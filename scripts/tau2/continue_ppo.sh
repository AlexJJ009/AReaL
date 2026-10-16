#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Use a separately prepared immutable source snapshot and cloned recovery state.
set -euo pipefail
: "${TAU2_CONTINUATION_CONFIG:?Set the frozen continuation YAML}"
: "${TAU2_PINNED_SOURCE:?Set the pinned source snapshot with the recovery fix}"
: "${TAU2_RUN_ROOT:?Set the new trial output directory}"
: "${TAU2_TRIAL_NAME:?Set the new trial name}"
TOTAL_EPOCHS=${TOTAL_EPOCHS:-20}
if [[ ! "$TOTAL_EPOCHS" =~ ^[1-9][0-9]{0,3}$ ]] || (( TOTAL_EPOCHS <= 8 )); then
  echo 'TOTAL_EPOCHS must be an integer greater than 8.' >&2
  exit 2
fi
# Preserve full LR at the saved scheduler position 48, even for other horizons.
if (( TOTAL_EPOCHS * 6 * 15 / 1000 > 48 )); then
  echo 'Requested horizon would put recovered critic step 48 back in warmup.' >&2
  exit 2
fi
if (( $# > 1 )) || [[ $# == 1 && "$1" != --check-config ]]; then
  echo 'Usage: TOTAL_EPOCHS=20 launch.sh [--check-config]' >&2
  exit 2
fi
RECOVER_ROOT="$TAU2_RUN_ROOT/checkpoints/root/tau2-ppo/$TAU2_TRIAL_NAME"
"${AREAL_PYTHON:?Set AREAL_PYTHON}" - "$RECOVER_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
step = json.loads((root / "recover_info/step_info.json").read_text())
if step != dict(epoch=7, epoch_step=5, global_step=47, steps_per_epoch=6):
    raise SystemExit("Expected the prepared v3 step-47 recover state")
for role in ("default", "critic"):
    for name in (".metadata", "fsdp_engine_state.pt"):
        if not (root / role / "recover_checkpoint" / name).is_file():
            raise SystemExit(f"Missing recover file: {role}/{name}")
PY
export LITELLM_LOCAL_MODEL_COST_MAP=True UV_OFFLINE=1
export PYTHONPATH="$TAU2_PINNED_SOURCE"
cd "$TAU2_PINNED_SOURCE"
exec bash scripts/tau2/run.sh policy mixed ppo \
  --config "$TAU2_CONTINUATION_CONFIG" \
  "total_train_epochs=$TOTAL_EPOCHS" "$@"
