#!/usr/bin/env bash
# critic₀ fit with the biased, mean-initialised value head (docs/research/tau2/critic/value-head.md, docs/research/tau2/critic/data-mix.md).
# Copy of tau2-training-20261012-critic-data/launch_critic.sh, changed only in:
# entry `data critic-head`, pinned source, head settings, and the DATA choice:
#   v2   AReaL pack v2 (648) + official fixed train (99), pack first, fixed
#        order, 2 epochs = 94 steps. Same data as run 479; compare head only.
#   mix  critic-mix pack (122 informative + 99 easy, 16.14.5) + official train
#        (99) = 320 rows, reshuffled together every epoch, 4 epochs = 80 steps.
# Bias init = expected T=1 success of the training rows (479 rollouts). The head
# trains at the backbone lr (5e-6): with Adam a 1e-4 head lr moved the zero-init
# weight enough to shift every prediction by ~0.15 in one step (smoke 498).
# Uses all 8 GPUs (4 train + 4 rollout); queue it in the pueue gpu8 group.
# Usage: launch_critic_head.sh NAME {v2|mix} [extra hydra overrides...]
set -euo pipefail
NAME=$1 DATA=$2; shift 2
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src-a88d480e"
OUT="$ROOT/$NAME"
CACHE=/data_storage/yl_test/lgx/data-1/code/_cache
PACKS=/data_storage/yl_test/lgx/data-1/dataset/tau2
test ! -e "$OUT/tau2-critic-fit-evidence/summary.json" || { echo "refusing to overwrite $OUT"; exit 2; }
case $DATA in
  v2)  PACK=$PACKS/tau2-areal-pack-20261011-v2
       DATA_OVR=(critic_head_bias_init=0.91 total_train_epochs=2 train_dataset.shuffle=false) ;;
  mix) PACK=$PACKS/tau2-areal-pack-20261013-critic-mix-v1
       DATA_OVR=(critic_head_bias_init=0.78 total_train_epochs=4 train_dataset.shuffle=true) ;;
  *) echo "bad data $DATA"; exit 2 ;;
esac
export TAU2_ACTOR_PATH=Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
mkdir -p "$OUT"
cd "$SRC"
# The shared .venv has areal.pth -> the main worktree; put the pinned tree first
# so scripts/, examples/ and areal/ all come from $SRC.
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export SAO_ARTIFACT_ROOT=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math
export HF_HUB_CACHE=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub
# fixed-v3 (11.2.1): the pinned code's `fixed` variant checks the v3 task hashes.
export TAU2_DATA_DIR=$CACHE/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data
export TAU2_DEEPSEEK_ENV_FILE=/data_storage/yl_test/lgx/home/.config/recuris-seta-ledger-rft/deepseek.env
export TAU2_RUN_ROOT="$OUT" TAU2_TRIAL_NAME="$NAME"
export TAU2_USAGE_LOG="$OUT/user-usage.jsonl"
export AREAL_RUNTIME_AUDIT_DIR="$OUT/runtime-audit"
echo "$PACK" > "$OUT/pack.txt"
git rev-parse HEAD > "$OUT/git-head.txt"
git status --short > "$OUT/git-status.txt"
echo "$DATA $*" > "$OUT/args.txt"
OVR=(experiment_name=tau2-critic-head econfig.benchmark_variant=fixed "critic_task_pack=$PACK"
     "${DATA_OVR[@]}" "$@")
if [[ " $* " == *" --check-config "* ]]; then
  exec scripts/tau2/run.sh data critic-head "${OVR[@]}"
fi
set +e
scripts/tau2/run.sh data critic-head "${OVR[@]}" > "$OUT/run.log" 2>&1
code=$?
echo "EXIT_CODE=$code" > "$OUT/exit-code.txt"
exit $code
