#!/usr/bin/env bash
# Run only inside the approved gpu8 queue slot, after the existing SAO job.
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "Usage: $0 PYTHON WORKTREE POOL BASE_CHECKPOINT TRAINED_CRITIC OUTPUT" >&2
  exit 2
fi
python_bin=$1
worktree=$2
pool=$3
base=$4
trained=$5
out=$6
export PYTHONPATH="$worktree"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
mkdir -p "$out"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m scripts.tau2.critic_preexp_extract --mode estimate --checkpoint "$base" --pool "$pool" --output "$out/extraction-estimate.json"
if [[ ! -e "$out/planted-rule.json" ]]; then
  CUDA_VISIBLE_DEVICES='' "$python_bin" -m scripts.tau2.critic_preexp_extract --mode rule --pool "$pool" --tokenizer "$base" --output "$out/planted-rule.json"
fi
for model in base r510; do
  checkpoint=$base
  [[ $model != r510 ]] || checkpoint=$trained
  pids=()
  for gpu in {0..7}; do
    CUDA_VISIBLE_DEVICES=$gpu "$python_bin" -m scripts.tau2.critic_preexp_extract \
      --mode extract --pool "$pool" --checkpoint "$checkpoint" --tokenizer "$base" \
      --rule-file "$out/planted-rule.json" --model-name "$model" \
      --shard-index "$gpu" --shard-count 8 --output "$out/parts-$model/part-$gpu" \
      >"$out/extract-$model-$gpu.log" 2>&1 &
    pids+=("$!")
  done
  failure=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failure=1
  done
  [[ $failure -eq 0 ]] || exit 1
  CUDA_VISIBLE_DEVICES='' "$python_bin" -m scripts.tau2.critic_preexp_extract \
    --mode merge --pool "$pool" --model-name "$model" --shard-count 8 \
    --parts "$out/parts-$model" --tokenizer "$base" --output "$out/caches"
done
