#!/usr/bin/env bash
set -euo pipefail
# Shared runtime primitive. Direct invocation selects PPO; run_sao.sh selects SAO.
SAO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAO_TRAIN_MODULE="${SAO_TRAIN_MODULE:-examples.math.sao_ppo}"
SAO_TRAIN_CONFIG="${SAO_TRAIN_CONFIG:-examples/math/sao_ppo.yaml}"
: "${SAO_ARTIFACT_ROOT:?Set the data-volume artifact directory}"
export SAO_CRITIC_PATH="${SAO_CRITIC_PATH:-${SAO_ARTIFACT_ROOT}/models/critic-dapo-step50-hf}"
: "${SAO_MODEL_PATH:?Set the local Base snapshot directory}"
: "${SAO_RUN_ROOT:?Set a unique experiment directory}"
if [[ "${SAO_TRAIN_MODULE}" == "examples.math.sao_ppo" ]]; then
  : "${SAO_DATA_PATH:?Set the audited DatasetDict directory}"
  : "${SAO_TRIAL_NAME:?Set the experiment trial name}"
fi
cd "${SAO_REPO_ROOT}"
source scripts/sao/runtime_env.sh
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH="${SAO_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Resolve and validate without starting training workers or creating a run.
if [[ " $* " == *" --check-config "* ]]; then
  exec python3 -m "${SAO_TRAIN_MODULE}" --config "${SAO_TRAIN_CONFIG}" "$@"
fi
mkdir -p "${SAO_RUN_ROOT}"
python3 -m "${SAO_TRAIN_MODULE}" --config "${SAO_TRAIN_CONFIG}" "$@" \
  2>&1 | tee -a "${SAO_RUN_ROOT}/controller.log"
