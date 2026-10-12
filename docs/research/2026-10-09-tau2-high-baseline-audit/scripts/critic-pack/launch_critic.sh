#!/usr/bin/env bash
# critic₀ on AReaL pack v2 (648) + official fixed train (99), 2 epochs, 94 updates.
# Same recipe as critic-formal-v1 (ADR 0005) except: fixed variant, Airline+Retail
# only, pack tasks first in every epoch. Validation: official fixed test, 60 tasks.
# DeepSeek-Flash for user simulator and NL judge. Uses all 8 GPUs (4 train + 4 rollout).
# Usage: launch_critic.sh NAME [extra hydra overrides...]
set -euo pipefail
NAME=$1; shift
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src-1076023b"
OUT="$ROOT/$NAME"
test ! -e "$OUT/tau2-critic-fit-evidence/summary.json" || { echo "refusing to overwrite $OUT"; exit 2; }
mkdir -p "$OUT"
cd "$SRC"
export SAO_ARTIFACT_ROOT=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math
export HF_HUB_CACHE=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub
export TAU2_DATA_DIR=/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v2/data
export TAU2_DEEPSEEK_ENV_FILE=/data_storage/yl_test/lgx/home/.config/recuris-seta-ledger-rft/deepseek.env
export TAU2_RUN_ROOT="$OUT" TAU2_TRIAL_NAME="$NAME"
export TAU2_USAGE_LOG="$OUT/user-usage.jsonl"
export AREAL_RUNTIME_AUDIT_DIR="$OUT/runtime-audit"
PACK=$(readlink -f /data_storage/yl_test/lgx/data-1/code/_cache/tau2-areal-pack-latest)
echo "$PACK" > "$OUT/pack.txt"
git rev-parse HEAD > "$OUT/git-head.txt"
git status --short > "$OUT/git-status.txt"
echo "$@" > "$OUT/args.txt"
set +e
scripts/tau2/run.sh data critic \
  experiment_name=tau2-critic-pack \
  econfig.benchmark_variant=fixed \
  "critic_task_pack=$PACK" \
  "$@" > "$OUT/run.log" 2>&1
echo "EXIT_CODE=$?" > "$OUT/exit-code.txt"
