#!/usr/bin/env bash
# Run this script as a pueue gpu8 command; it never submits or changes jobs.
# Required: AREAL_PYTHON and ART. All Python imports resolve to this checkout.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${AREAL_PYTHON:?Set AREAL_PYTHON to the Python executable with AReaL dependencies}"
: "${ART:?Set ART to the offline experiment artifact directory}"
export PYTHONPATH="$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$REPO_ROOT"
MODE="${1:?Usage: launch_critic_offline.sh features|fit|sim [arguments...]}"
shift
mkdir -p "$ART"
case "$MODE" in
  features)
    exec "$AREAL_PYTHON" -m scripts.tau2.critic_features "$@"
    ;;
  fit)
    : "${NPROC_PER_NODE:?Set NPROC_PER_NODE to the allocated critic GPU count}"
    exec "$AREAL_PYTHON" -m torch.distributed.run --standalone \
      --nproc_per_node="$NPROC_PER_NODE" -m scripts.tau2.offline_critic_fit "$@"
    ;;
  sim)
    export CUDA_VISIBLE_DEVICES=''
    exec "$AREAL_PYTHON" -m scripts.tau2.critic_head_sim "$@"
    ;;
  *)
    echo "Unknown mode: $MODE (expected features, fit, sim)" >&2
    exit 2
    ;;
esac
