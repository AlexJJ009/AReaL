#!/usr/bin/env bash
# SAO on the critic-mix pack (README 16.14.6). Qwen3.5-4B actor, critic = final
# checkpoint of critic run 500 (critic-head-qwen35-mix), read from its summary.json.
# Train: mix pack 221 + official fixed train 99 = 320 tasks (Airline 180, Retail 140),
# 4 episodes per task per epoch, 2 epochs, batch 64 -> 40 steps, 2560 episodes.
# critic_updates_before_actor=2 and critic lr 5e-6 come from config_sao.yaml.
# Eval: fixed-v3 test 60 x 4 at policy T=0, before training and every 10 steps.
# Modes: smoke (2 steps, eval and save at step 2, no eval before training) | formal.
# Uses all 8 GPUs; queue it in the pueue gpu8 group.
# Usage: launch_sao_pack.sh NAME {smoke|formal} [extra hydra overrides...]
set -euo pipefail
NAME=$1 MODE=$2; shift 2
ROOT="$(cd "$(dirname "$0")" && pwd)"
SHA=fccceaa1
SRC="$ROOT/src-$SHA"
OUT="$ROOT/$NAME"
CACHE=/data_storage/yl_test/lgx/data-1/code/_cache
PACK=/data_storage/yl_test/lgx/data-1/dataset/tau2/tau2-areal-pack-20261013-critic-mix-v1
CRITIC_RUN=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261013-critic-head/critic-head-qwen35-mix-r2
case $MODE in
  smoke)  MODE_OVR=(total_train_steps=2 evaluator.eval_before_train=false evaluator.freq_steps=2 saver.freq_steps=2) ;;
  formal) MODE_OVR=(evaluator.freq_steps=10) ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac
test ! -e "$OUT/exit-code.txt" || { echo "refusing to overwrite $OUT"; exit 2; }
[[ "$(git -C "$SRC" rev-parse --short=8 HEAD)" == "$SHA" ]] || { echo "src HEAD is not $SHA"; exit 2; }
CRITIC=$(python3 -c "
import json, sys
s = json.load(open('$CRITIC_RUN/tau2-critic-fit-evidence/summary.json'))
if s.get('status') != 'completed' and s.get('completed_steps') != s.get('full_planned_steps'):
    sys.exit('critic run 500 is not complete: %s' % s.get('status'))
print(s['final_critic_checkpoint'])")
mkdir -p "$OUT"
cd "$SRC"
# The shared .venv has areal.pth -> the main worktree; put the pinned tree first.
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export SAO_ARTIFACT_ROOT=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math
export HF_HUB_CACHE=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub
export TAU2_DATA_DIR=$CACHE/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v3/data
export TAU2_DEEPSEEK_ENV_FILE=/data_storage/yl_test/lgx/home/.config/recuris-seta-ledger-rft/deepseek.env
export TAU2_ACTOR_PATH=Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export TAU2_CRITIC_PATH="$CRITIC"
export TAU2_RUN_ROOT="$OUT" TAU2_TRIAL_NAME="$NAME"
export TAU2_USAGE_LOG="$OUT/user-usage.jsonl"
export AREAL_RUNTIME_AUDIT_DIR="$OUT/runtime-audit"
echo "$CRITIC" > "$OUT/critic.txt"
echo "$PACK" > "$OUT/pack.txt"
git rev-parse HEAD > "$OUT/git-head.txt"
git status --short > "$OUT/git-status.txt"
echo "$MODE $*" > "$OUT/args.txt"
.venv/bin/python scripts/sao/check_critic_head.py --checkpoint "$CRITIC" --output "$OUT/critic-head-preflight.json"
OVR=(econfig.benchmark_variant=fixed "train_task_pack=$PACK" effective_episodes=null
     domain_effective_episodes.airline=720 domain_effective_episodes.retail=560 total_train_epochs=2
     "${MODE_OVR[@]}" "$@")
if [[ " $* " == *" --check-config "* ]]; then
  exec scripts/tau2/run.sh policy-pack mixed sao "${OVR[@]}"
fi
set +e
scripts/tau2/run.sh policy-pack mixed sao "${OVR[@]}" > "$OUT/run.log" 2>&1
code=$?
echo "EXIT_CODE=$code" > "$OUT/exit-code.txt"
exit $code
