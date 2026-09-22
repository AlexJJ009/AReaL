#!/usr/bin/env bash
set -euo pipefail
SAO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAO_ARTIFACT_ROOT:?Set the data-volume artifact directory}"
: "${SAO_MODEL_PATH:?Set the local Base snapshot directory}"
# Step 50 is exported as a standalone HF critic under the task artifact root.
# Override only when intentionally selecting a different pretrained critic.
export SAO_CRITIC_PATH="${SAO_CRITIC_PATH:-${SAO_ARTIFACT_ROOT}/models/critic-dapo-step50-hf}"
: "${SAO_DATA_PATH:?Set the audited DatasetDict directory}"
: "${SAO_RUN_ROOT:?Set a unique experiment directory}"
: "${SAO_TRIAL_NAME:?Set the experiment trial name}"
cd "${SAO_REPO_ROOT}"
source scripts/sao/runtime_env.sh
export WANDB_MODE=online
source .venv/bin/activate
# Resolve and validate without creating a run or starting any training workers.
if [[ " $* " == *" --check-config "* ]]; then
  exec python examples/math/sao_ppo.py --config examples/math/sao_ppo.yaml "$@"
fi
mkdir -p "${SAO_RUN_ROOT}"
python examples/math/sao_ppo.py --config examples/math/sao_ppo.yaml "$@" \
  2>&1 | tee -a "${SAO_RUN_ROOT}/controller.log"
