#!/usr/bin/env bash
# Step-0 eval of Qwen3.5-4B with an API user simulator and the fixed DeepSeek NL judge (commit 9e7082dc).
# Usage: launch.sh NAME USER TASK_SET POLICY_T USER_T
#   USER: deepseek | deepseek-nothink | doubao;  TASK_SET: test | base;  fixed variant, 4 repeats, 8 GPUs.
set -euo pipefail
NAME=$1 USER_SIM=$2 TASK_SET=$3 POLICY_T=$4 USER_T=$5
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/user-sim-20261012/pinned-worktrees-b2/AReaL/validation-tau2-selfhost-user-b2-20261012
SHA=9e7082dcc4532048523a464f50fb7f65ba8b0c8f
RUNTIME_VENV=/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/.venv
MODEL=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
OUT="$ROOT/$NAME"
mkdir "$OUT" || { echo "refusing existing run $OUT"; exit 2; }
exec > "$OUT/launch.log" 2>&1
trap 'rc=$?; printf "EXIT_CODE=%s\n" "$rc" > "$OUT/exit-code.txt"' EXIT
cd "$SRC"
test "$(git rev-parse HEAD)" = "$SHA"
test -z "$(git status --porcelain)"
echo "$@" > "$OUT/args.txt"; git rev-parse HEAD > "$OUT/git-head.txt"
export SAO_ARTIFACT_ROOT=/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math
export HF_HUB_CACHE=/data_storage/yl_test/lgx/data-1/.cache/huggingface/hub
export TAU2_DATA_DIR=/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619-fixed-v2/data
source scripts/sao/runtime_env.sh
export CPATH="$RUNTIME_VENV/lib/python3.12/site-packages/nvidia/curand/include"
export PATH="$RUNTIME_VENV/bin:$PATH" PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 AREAL_ALLOW_DEFAULT_ADMIN_KEY=1 WANDB_MODE=offline
export AREAL_RUNTIME_AUDIT_DIR="$OUT/runtime-audit" TAU2_RUN_ROOT="$OUT" TAU2_TRIAL_NAME="$NAME"
export TAU2_USAGE_LOG="$OUT/user-usage.jsonl"
load_key() { test "$(stat -c %a "$1")" = 600; set -a; source "$1"; set +a; }
load_key /data_storage/yl_test/lgx/home/.config/recuris-seta-ledger-rft/deepseek.env
: "${DEEPSEEK_API_KEY:?}"
case $USER_SIM in
  deepseek) USER_OVR=(econfig.user_llm=deepseek-flash econfig.user_llm_base_url=https://api.deepseek.com
              econfig.user_llm_api_key_env=DEEPSEEK_API_KEY econfig.user_llm_num_retries=3) ;;
  deepseek-nothink) USER_OVR=(econfig.user_llm=deepseek-flash econfig.user_llm_base_url=https://api.deepseek.com
              econfig.user_llm_api_key_env=DEEPSEEK_API_KEY econfig.user_llm_num_retries=3
              +econfig.user_llm_args.extra_body.thinking.type=disabled) ;;
  doubao) load_key /data_storage/yl_test/lgx/home/.config/tau2-doubao/ark.env; : "${ARK_API_KEY:?}"
          USER_OVR=(econfig.user_llm=doubao-seed-2-1-turbo-260628 econfig.user_llm_base_url=https://ark.cn-beijing.volces.com/api/v3
              econfig.user_llm_api_key_env=ARK_API_KEY econfig.user_llm_num_retries=2
              +econfig.user_llm_args.reasoning_effort=medium '+econfig.user_llm_args.allowed_openai_params=[reasoning_effort]') ;;
  *) echo "bad user $USER_SIM"; exit 2 ;;
esac
set +e
"$RUNTIME_VENV/bin/python" scripts/tau2/evaluate.py \
  --config examples/tau2/config_grpo.yaml --model-path "$MODEL" --output-root "$OUT" \
  --task-set "$TASK_SET" --repeats 4 ${CHECK_ONLY:+--check-config} --temperature "$POLICY_T" \
  econfig.benchmark_variant=fixed 'domains=[airline,retail]' econfig.domain=mixed econfig.enable_thinking=false \
  evaluation_rollout.backend=sglang:d8p1t1 evaluation_rollout.max_concurrent_rollouts=256 \
  "${USER_OVR[@]}" econfig.user_llm_args.temperature="$USER_T" \
  econfig.nl_judge_llm=deepseek-flash econfig.nl_judge_llm_base_url=https://api.deepseek.com \
  econfig.nl_judge_llm_api_key_env=DEEPSEEK_API_KEY econfig.nl_judge_llm_args.temperature=0.0 \
  > "$OUT/run.log" 2>&1
rc=$?
set -e
test "$rc" = 0; [ -z "${CHECK_ONLY:-}" ] || exit 0
# Pass only if every planned episode completed and every judge call went to DeepSeek.
"$RUNTIME_VENV/bin/python" - "$OUT" <<'PY'
import json, sys, collections
out = sys.argv[1]
s = json.load(open(f"{out}/summary.json"))
assert s["status"] == "complete" and s["completed_episodes"] == s["planned_episodes"], s
assert s.get("infra_excluded_episodes", 0) == 0, s
hosts = collections.Counter(c["base_url_host"] for l in open(f"{out}/user-usage.jsonl")
                            for c in json.loads(l).get("nl_judge_calls", []))
assert set(hosts) <= {"api.deepseek.com"}, hosts
json.dump({"passed": True, "episodes": s["completed_episodes"], "judge_hosts": hosts},
          open(f"{out}/outcome-check.json", "w"), indent=2)
PY
