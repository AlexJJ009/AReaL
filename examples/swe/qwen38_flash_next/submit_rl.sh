#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: bash submit_rl.sh swe|rlvr [training config overrides...]
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export QWEN_REPO=${QWEN_REPO:-$(cd "$recipe_dir/../../.." && pwd)}
profile=${1:?Usage: submit_rl.sh swe|rlvr [overrides...]}
shift
case "$profile" in swe|rlvr) ;; *) echo 'Expected swe or rlvr' >&2; exit 2 ;; esac
if [[ -n ${QWEN_LAUNCH_ENV:-} ]]; then
  set -a; source "$QWEN_LAUNCH_ENV"; set +a
fi
for name in QWEN_OUTPUT_ROOT QWEN_MODEL QWEN_ACTOR_IMAGE QWEN_ROLLOUT_IMAGE \
  QWEN_RESERVATION QWEN_NODELIST QWEN_PARTITION QWEN_CONTROLLER_NODE \
  QWEN_MOUNTS QWEN_CONTROLLER_MOUNTS MCORE_BRIDGE_ROOT MEGATRON_ROOT; do
  : "${!name:?Set $name in the launch environment}"
  export "$name"
done
if [[ $profile == swe ]]; then
  for name in QWEN_PRIVATE_ENV QWEN_REPLAY64_ACCEPTANCE QWEN_CC_PROTOCOL_ACCEPTANCE; do
    : "${!name:?Set $name for SWE recovery}"
    test -f "${!name}"
    export "$name"
  done
  export QWEN_SWE_START_MODE=${QWEN_SWE_START_MODE:-recover}
  case "$QWEN_SWE_START_MODE" in
    fresh) ;;
    recover)
      : "${QWEN_RECOVER_SOURCE:?Set QWEN_RECOVER_SOURCE for SWE recovery}"
      test -f "$QWEN_RECOVER_SOURCE"
      export QWEN_RECOVER_SOURCE
      ;;
    *) echo 'QWEN_SWE_START_MODE must be fresh or recover' >&2; exit 2 ;;
  esac
else
  : "${QWEN_GSM8K_DATA:?Set the GSM8K dataset path}"
  export QWEN_GSM8K_DATA
fi
export QWEN_AWEX_FROZEN_CONTRACT=${QWEN_AWEX_FROZEN_CONTRACT:-$recipe_dir/fixtures/qwen4-exp-frozen-contract-v1.json}
export QWEN_ACTOR_PYTHONPATH="$recipe_dir/runtime${QWEN_TRAIN_EXTRA_PYTHONPATH:+:$QWEN_TRAIN_EXTRA_PYTHONPATH}:$MCORE_BRIDGE_ROOT/src:$QWEN_REPO"
export QWEN_ROLLOUT_PYTHONPATH="$recipe_dir/runtime${QWEN_INFER_EXTRA_PYTHONPATH:+:$QWEN_INFER_EXTRA_PYTHONPATH}:$MEGATRON_ROOT:$QWEN_REPO"
export QWEN_CONTROLLER_PYTHONPATH=$QWEN_ACTOR_PYTHONPATH
export SBATCH_PARTITION=$QWEN_PARTITION SBATCH_RESERVATION=$QWEN_RESERVATION
export AREAL_APPTAINER_STAGGER_SECONDS=2
mkdir -p "$QWEN_OUTPUT_ROOT"
exec sbatch --partition="$QWEN_PARTITION" --reservation="$QWEN_RESERVATION" \
  --nodelist="$QWEN_CONTROLLER_NODE" --chdir="$QWEN_REPO" \
  --job-name="qwen38-$profile-controller" --output="$QWEN_OUTPUT_ROOT/controller-%j.log" \
  --export=ALL "$recipe_dir/rl_controller.sbatch" "$profile" "$@"
