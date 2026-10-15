#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prepared recipe; running smoke/formal starts training. check-config is CPU-only.
# Usage: launch_ppo.sh NAME {smoke|formal} [--check-config] [Hydra overrides...]
#        launch_ppo.sh NAME check-config {smoke|formal} [Hydra overrides...]
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo 'Usage: launch_ppo.sh NAME {smoke|formal|check-config} [mode] [overrides...]' >&2
  exit 2
fi
NAME=$1 MODE=$2; shift 2
CHECK=()
if [[ "$MODE" == check-config ]]; then
  MODE=${1:-formal}
  if [[ $# -gt 0 ]]; then shift; fi
  CHECK=(--check-config)
fi
case "$MODE" in
  smoke) MODE_OVR=(experiment_mode=qualification total_train_epochs=1 total_train_steps=2 evaluator.eval_before_train=false) ;;
  formal) MODE_OVR=(experiment_mode=formal) ;;
  *) echo 'Mode must be smoke or formal' >&2; exit 2 ;;
esac
: "${TAU2_PPO_ARTIFACT_ROOT:?Set TAU2_PPO_ARTIFACT_ROOT}"
: "${TAU2_AREAL_PACK:?Set TAU2_AREAL_PACK to the AReaL v2 task pack}"
export TAU2_RUN_ROOT="$TAU2_PPO_ARTIFACT_ROOT/$NAME" TAU2_TRIAL_NAME="$NAME"
export TAU2_ACTOR_PATH="${TAU2_ACTOR_PATH:-Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
PACKS="[$TAU2_AREAL_PACK]"
DOMAIN_OVR=()
if [[ -n "${TAU2_TELECOM_PACK:-}" ]]; then
  PACKS="[$TAU2_AREAL_PACK,$TAU2_TELECOM_PACK]"
  DOMAIN_OVR=('domains=[airline,retail,telecom]')
fi
cd "$REPO_ROOT"
exec bash scripts/tau2/run.sh policy mixed ppo \
  "train_task_packs=$PACKS" \
  "samples_per_task_per_epoch=${TAU2_SAMPLES_PER_TASK:-1}" \
  "evaluator.freq_steps=${TAU2_EVAL_EVERY:-10}" \
  "saver.freq_steps=${TAU2_EVAL_EVERY:-10}" \
  "recover.freq_steps=${TAU2_EVAL_EVERY:-10}" \
  "num_critic_only_steps=${TAU2_CRITIC_WARMUP_STEPS:-0}" \
  "${DOMAIN_OVR[@]}" "${MODE_OVR[@]}" "${CHECK[@]}" "$@"
