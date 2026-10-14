#!/usr/bin/env bash
# Repeated eval for the critic EV ceiling (docs/research/tau2/critic/ev-ceiling.md): fixed official test (60 tasks) x 4,
# with the same sampling as the critic's validation rollouts: policy T=1, DeepSeek-Flash user T=0,
# DeepSeek-Flash NL judge T=0, thinking off. Protocol copied from user-sim-20261012/api-user/launch.sh
# (run 475, deepseek-test-u0p1), pinned to the critic snapshot src-04be68ab. Uses 8 GPUs (pueue gpu8).
# Usage: launch_ceiling.sh NAME {qwen35|instruct}     CHECK_ONLY=1 resolves the config only.
set -euo pipefail
NAME=$1 ACTOR=$2
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$(dirname "$ROOT")/src-04be68ab"
OUT="$ROOT/$NAME"
case $ACTOR in
  qwen35) MODEL=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
          PARSER=qwen3_coder ;;
  # Same parser as the Instruct critic run (launch_critic.sh) and the 15.2 Instruct evals.
  instruct) MODEL=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
            PARSER=qwen25 ;;
  *) echo "bad actor $ACTOR"; exit 2 ;;
esac
test ! -e "$OUT/summary.json" || { echo "refusing to overwrite $OUT"; exit 2; }
mkdir -p "$OUT"
cd "$SRC"
test "$(git rev-parse HEAD)" = 04be68ab892710c2132779637a3e099b92ded9af
git rev-parse HEAD > "$OUT/git-head.txt"; git status --short > "$OUT/git-status.txt"; echo "$@" > "$OUT/args.txt"
export SAO_ARTIFACT_ROOT=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math
export HF_HUB_CACHE=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub
export TAU2_DATA_DIR=/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v2/data
source scripts/sao/runtime_env.sh
export PATH="$SRC/.venv/bin:$PATH" PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 AREAL_ALLOW_DEFAULT_ADMIN_KEY=1 WANDB_MODE=offline
export AREAL_RUNTIME_AUDIT_DIR="$OUT/runtime-audit" TAU2_RUN_ROOT="$OUT" TAU2_TRIAL_NAME="$NAME"
export TAU2_USAGE_LOG="$OUT/user-usage.jsonl"
KEY_FILE=/data_storage/yl_test/lgx/home/.config/recuris-seta-ledger-rft/deepseek.env
test "$(stat -c %a "$KEY_FILE")" = 600
set -a; source "$KEY_FILE"; set +a
: "${DEEPSEEK_API_KEY:?DeepSeek key missing}"
CMD=(.venv/bin/python scripts/tau2/evaluate.py
  --config examples/tau2/config_grpo.yaml --model-path "$MODEL" --output-root "$OUT"
  --task-set test --repeats 4 --temperature 1 ${CHECK_ONLY:+--check-config}
  econfig.benchmark_variant=fixed 'domains=[airline,retail]' econfig.domain=mixed econfig.enable_thinking=false
  evaluation_rollout.agent.tool_call_parser=$PARSER
  evaluation_rollout.backend=sglang:d8p1t1 evaluation_rollout.max_concurrent_rollouts=256
  econfig.user_llm=deepseek-flash econfig.user_llm_base_url=https://api.deepseek.com
  econfig.user_llm_api_key_env=DEEPSEEK_API_KEY econfig.user_llm_num_retries=3 econfig.user_llm_args.temperature=0
  econfig.nl_judge_llm=deepseek-flash econfig.nl_judge_llm_base_url=https://api.deepseek.com
  econfig.nl_judge_llm_api_key_env=DEEPSEEK_API_KEY econfig.nl_judge_llm_args.temperature=0.0)
if [ -n "${CHECK_ONLY:-}" ]; then exec "${CMD[@]}"; fi
set +e
"${CMD[@]}" > "$OUT/run.log" 2>&1
rc=$?
echo "EXIT_CODE=$rc" > "$OUT/exit-code.txt"
exit $rc
