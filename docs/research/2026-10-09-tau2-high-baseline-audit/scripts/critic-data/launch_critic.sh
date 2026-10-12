#!/usr/bin/env bash
# critic₀ fit on AReaL pack v2 (648) + official fixed train (99), Airline+Retail.
# Recipe of critic-formal-v1 (ADR 0005, config_critic_production.yaml) except:
# fixed variant, no Telecom, pack tasks first in every epoch. Validation: fixed
# official test (formal) or dev (tune). DeepSeek-Flash for user and NL judge.
# Uses all 8 GPUs (4 train + 4 rollout); queue it in the pueue gpu8 group.
# Usage: launch_critic.sh NAME {qwen35|instruct} [extra hydra overrides...]
set -euo pipefail
NAME=$1 ACTOR=$2; shift 2
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src-04be68ab"
OUT="$ROOT/$NAME"
test ! -e "$OUT/tau2-critic-fit-evidence/summary.json" || { echo "refusing to overwrite $OUT"; exit 2; }
case $ACTOR in
  qwen35) export TAU2_ACTOR_PATH=Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a; ACTOR_OVR=() ;;
  # Same parser as the Instruct step 0 evals (README 15.2).
  instruct) export TAU2_ACTOR_PATH=Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554
            ACTOR_OVR=(rollout.agent.tool_call_parser=qwen25) ;;
  *) echo "bad actor $ACTOR"; exit 2 ;;
esac
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
echo "$ACTOR $*" > "$OUT/args.txt"
OVR=(experiment_name=tau2-critic-data econfig.benchmark_variant=fixed "critic_task_pack=$PACK" "${ACTOR_OVR[@]}" "$@")
if [[ " $* " == *" --check-config "* ]]; then
  exec scripts/tau2/run.sh data critic "${OVR[@]}"
fi
set +e
scripts/tau2/run.sh data critic "${OVR[@]}" > "$OUT/run.log" 2>&1
code=$?
echo "EXIT_CODE=$code" > "$OUT/exit-code.txt"
exit $code
