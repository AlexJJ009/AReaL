#!/usr/bin/env bash
# One offline τ² eval: DeepSeek Flash user/judge, policy temperature 0, base set x4.
# Usage: launch_eval.sh NAME MODEL_DIR GPUS VARIANT DOMAINS THINKING PARSER
#   DOMAINS: comma list (airline,retail[,telecom]); THINKING: 0|1; PARSER: qwen25|qwen3_coder
# Runs from a worktree pinned at a6abe781 so edits in the main checkout cannot reach workers.
set -euo pipefail
NAME=$1 MODEL=$2 GPUS=$3 VARIANT=$4 DOMAINS=$5 THINKING=$6 PARSER=$7
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src-a6abe781"
OUT="$ROOT/$NAME"
test ! -e "$OUT/summary.json" || { echo "refusing to overwrite $OUT"; exit 2; }
mkdir -p "$OUT"
cd "$SRC"
export SAO_ARTIFACT_ROOT=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math
export HF_HUB_CACHE=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub
case $VARIANT in
  fixed) export TAU2_DATA_DIR=/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v2/data ;;
  official) export TAU2_DATA_DIR=/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619/data ;;
  *) echo "bad variant $VARIANT"; exit 2 ;;
esac
source scripts/sao/runtime_env.sh
export PATH="$SRC/.venv/bin:$PATH"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$GPUS
NGPU=$(awk -F, "{print NF}" <<<"$GPUS")
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1 WANDB_MODE=offline
export AREAL_RUNTIME_AUDIT_DIR="$OUT/runtime-audit"
export TAU2_RUN_ROOT="$OUT" TAU2_TRIAL_NAME="$NAME"
export TAU2_USAGE_LOG="$OUT/user-usage.jsonl"
KEY_FILE=/data_storage/yl_test/lgx/home/.config/recuris-seta-ledger-rft/deepseek.env
test "$(stat -c %a "$KEY_FILE")" = 600
set -a; source "$KEY_FILE"; set +a
: "${DEEPSEEK_API_KEY:?DeepSeek key missing}"
NDOM=$(awk -F, "{print NF}" <<<"$DOMAINS")
if [ "$NDOM" = 1 ]; then EDOMAIN=$DOMAINS; else EDOMAIN=mixed; fi
if [ "$THINKING" = 1 ]; then TPL=hf; EXP=individual; THINK=true; else TPL=concat; EXP=concat; THINK=false; fi
AGENT_OVR=()
for r in rollout evaluation_rollout; do
  AGENT_OVR+=("$r.agent.chat_template_type=$TPL" "$r.agent.export_style=$EXP" "$r.agent.tool_call_parser=$PARSER")
done
git rev-parse HEAD > "$OUT/git-head.txt"
git status --short > "$OUT/git-status.txt"
echo "$@" > "$OUT/args.txt"
set +e
.venv/bin/python scripts/tau2/evaluate.py \
  --config examples/tau2/config_grpo.yaml \
  --model-path "$MODEL" \
  --output-root "$OUT" --task-set base --repeats ${REPEATS:-4} --temperature 0 ${SMOKE_TASKS:+--tasks "$SMOKE_TASKS"} \
  econfig.benchmark_variant=$VARIANT \
  "domains=[$DOMAINS]" econfig.domain=$EDOMAIN \
  econfig.enable_thinking=$THINK \
  "${AGENT_OVR[@]}" \
  evaluation_rollout.backend=sglang:d${NGPU}p1t1 \
  evaluation_rollout.max_concurrent_rollouts=$((NGPU*32)) \
  > "$OUT/run.log" 2>&1
echo "EXIT_CODE=$?" > "$OUT/exit-code.txt"
