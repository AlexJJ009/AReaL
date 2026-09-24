#!/usr/bin/env bash
set -euo pipefail
# Reuse PPO's runtime setup and logging; keep SAO's algorithm/config independent.
# Optional hybrid critic freezing: append critic.freeze_critic_attention=true.
# Default false preserves full-parameter training. Use a new run for policy changes.
export SAO_TRAIN_MODULE=examples.math.gsm8k_sao
export SAO_TRAIN_CONFIG=examples/math/gsm8k_sao.yaml
# Inherit PPO's pretrained SAO_CRITIC_PATH; no LR or critic-only warmup.
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_ppo.sh" "$@"
