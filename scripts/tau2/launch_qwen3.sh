#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prepared comparison recipes. Only check-config is CPU-only.
# Usage: launch_qwen3.sh sao NAME {formal|check-config} [Hydra overrides...]
#        launch_qwen3.sh grpo NAME {formal|check-config} {default|matched} [overrides...]
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 3 ]]; then
  echo 'Usage: launch_qwen3.sh {sao|grpo} NAME {formal|check-config} [GRPO budget: default|matched] [overrides...]' >&2
  exit 2
fi
ALGORITHM=$1 NAME=$2 MODE=$3; shift 3
case "$ALGORITHM" in sao|grpo) ;; *) echo 'Algorithm must be sao or grpo' >&2; exit 2 ;; esac
CHECK=()
case "$MODE" in
  formal) ;;
  check-config) CHECK=(--check-config) ;;
  *) echo 'Mode must be formal or check-config' >&2; exit 2 ;;
esac
BUDGET_OVR=()
if [[ "$ALGORITHM" == grpo ]]; then
  case "${1:-}" in
    default) BUDGET_OVR=(gconfig.n_samples=8 train_dataset.batch_size=8 train_batch_episodes=64 total_train_epochs=2) ;;
    matched) BUDGET_OVR=(gconfig.n_samples=8 train_dataset.batch_size=16 train_batch_episodes=128 total_train_epochs=1) ;;
    *) echo 'GRPO requires an explicit budget: default or matched' >&2; exit 2 ;;
  esac
  shift
fi
: "${TAU2_PREP_ARTIFACT_ROOT:?Set TAU2_PREP_ARTIFACT_ROOT}"
: "${TAU2_AREAL_PACK:?Set TAU2_AREAL_PACK to the AReaL v2 pack}"
export TAU2_RUN_ROOT="$TAU2_PREP_ARTIFACT_ROOT/$NAME" TAU2_TRIAL_NAME="$NAME"
export TAU2_ACTOR_PATH="${TAU2_ACTOR_PATH:-Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554}"
export TAU2_CRITIC_PATH="${TAU2_CRITIC_PATH:-$TAU2_ACTOR_PATH}"
export AREAL_PYTHON="${AREAL_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=True UV_OFFLINE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$REPO_ROOT"
exec bash scripts/tau2/run.sh policy mixed "$ALGORITHM-qwen3" \
  "train_task_packs=[$TAU2_AREAL_PACK]" \
  experiment_mode=formal "${BUDGET_OVR[@]}" "${CHECK[@]}" "$@"
