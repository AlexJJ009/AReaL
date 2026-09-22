#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
profile=${1:?Expected swe, swe-eval or rlvr}; shift
recipe_dir="$QWEN_REPO/examples/swe/qwen38_flash_next"
export NO_PROXY='*' no_proxy='*'
export PYTHONPATH=$QWEN_CONTROLLER_PYTHONPATH
cd "$QWEN_REPO"
case "$profile" in
  swe|swe-eval)
    set -a; source "$QWEN_PRIVATE_ENV"; set +a
    : "${ARENA_OPENAPI_BASE:?Set Arena endpoint in private environment}"
    : "${ARENA_LLM_API_KEY:?Set Arena LLM gateway credentials in private environment}"
    export ARENA_OPENAPI_BASE ARENA_LLM_API_KEY
    export QWEN_ARENA_TRIAL=${QWEN_ARENA_TRIAL:-claude_awex256_$SLURM_JOB_ID}
    export QWEN_ARENA_OUTPUT="$QWEN_OUTPUT_ROOT/$QWEN_ARENA_TRIAL"
    export SWE_RL_ADMIN_API_KEY
    SWE_RL_ADMIN_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    mkdir -p "$QWEN_ARENA_OUTPUT"
    config=${QWEN_CONFIG:-$recipe_dir/swe_rl_256k.yaml}
    if [[ $profile == swe-eval ]]; then
      config=${QWEN_CONFIG:-$recipe_dir/swe_mm_eval.yaml}
      : "${QWEN_ARENA_TASK_IDS_FILE:?Set the exact reference task manifest}"
      test -f "$QWEN_ARENA_TASK_IDS_FILE"
    fi
    exec python3 -m examples.swe.qwen38_flash_next.train_rl "$profile" \
      --config "$config" "$@"
    ;;
  rlvr)
    export QWEN_GSM8K_TRIAL=${QWEN_GSM8K_TRIAL:-gsm8k256_$SLURM_JOB_ID}
    export QWEN_GSM8K_OUTPUT="$QWEN_OUTPUT_ROOT/$QWEN_GSM8K_TRIAL"
    mkdir -p "$QWEN_GSM8K_OUTPUT"
    export QWEN_GSM8K_PROXY_KEY
    QWEN_GSM8K_PROXY_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    exec python3 -m examples.swe.qwen38_flash_next.train_rl rlvr \
      --config "$recipe_dir/rlvr_gsm8k_256k.yaml" "$@"
    ;;
  *) echo 'Expected swe, swe-eval or rlvr' >&2; exit 2 ;;
esac
