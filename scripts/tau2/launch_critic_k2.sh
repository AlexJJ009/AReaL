#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# K2: full-parameter Qwen3 critic for PPO. Fresh online train episodes at T=1;
# fixed-v3 A/R test 60 monitors only; use the final checkpoint, never test-best.
# Usage: launch_critic_k2.sh NAME [--check-config] [Hydra overrides...]
# Defaults retain ADR-0012's shared head/backbone lr 5e-6, batch 16, two epochs
# (747 tasks -> 94 updates). E-results are pending; no evidence yet for a larger
# lr or batch. K1 (the run-480 single-variable launcher) is independent.
# Head prior: run 480 Qwen3 T=1 TRAIN episodes, 927 / 1494 = 0.6204819277108434.
# Same 747 tasks, twice each; fixed-v2/v3 train task definitions are identical. No test rows.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: launch_critic_k2.sh NAME [--check-config] [overrides...]' >&2
  exit 2
fi
NAME=$1; shift
: "${TAU2_PPO_ARTIFACT_ROOT:?Set TAU2_PPO_ARTIFACT_ROOT}"
: "${TAU2_AREAL_PACK:?Set TAU2_AREAL_PACK to the AReaL v2 task pack}"
export TAU2_RUN_ROOT="$TAU2_PPO_ARTIFACT_ROOT/$NAME" TAU2_TRIAL_NAME="$NAME"
export TAU2_ACTOR_PATH="${TAU2_ACTOR_PATH:-Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LITELLM_LOCAL_MODEL_COST_MAP=True
cd "$REPO_ROOT"
exec bash scripts/tau2/run.sh data critic-head \
  experiment_name=tau2-critic-k2 \
  'domains=[airline,retail]' experiment_mode=formal \
  econfig.benchmark_variant=fixed \
  "critic_task_pack=$TAU2_AREAL_PACK" \
  rollout.agent.tool_call_parser=qwen25 \
  critic.freeze_critic_attention=false \
  "critic_head_bias_init=${TAU2_CRITIC_HEAD_BIAS_INIT:-0.6204819277108434}" \
  critic_head_lr=null \
  "critic.optimizer.lr=${TAU2_CRITIC_LR:-5e-6}" \
  "train_batch_episodes=${TAU2_CRITIC_BATCH:-16}" \
  "train_dataset.batch_size=${TAU2_CRITIC_BATCH:-16}" \
  "rollout.max_concurrent_rollouts=${TAU2_CRITIC_BATCH:-16}" \
  "rollout.queue_size=$((2 * ${TAU2_CRITIC_BATCH:-16}))" \
  "total_train_epochs=${TAU2_CRITIC_EPOCHS:-2}" \
  "evaluator.freq_steps=${TAU2_EVAL_EVERY:-10}" \
  "saver.freq_steps=${TAU2_EVAL_EVERY:-10}" \
  "recover.freq_steps=${TAU2_EVAL_EVERY:-10}" "$@"
