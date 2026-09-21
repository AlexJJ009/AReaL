# Qwen Flash Next 256K SWE recipe

`swe_rl_256k.yaml` uses 64 GPUs with TP8 / PP4 / CP2 / EP16. The actor's
`--distribution=block:block` keeps each TP8 group on one eight-GPU node. Keep the
existing model, image, mount, Arena and frozen-weight-contract environment settings used
by `submit_rl.sh`; private credentials remain in the external environment file.

The long-context settings are:

- Chunked LM-head loss: `enable_chunked_logits: true`, chunk size 1024.
- PLE causal chunks: `QWEN_PLE_CHUNK_TOKENS=8192`; overlapping causal history is
  retained within each sequence, and shared weight gradients accumulate in FP32.
- QSA query chunks: `QWEN_QSA_QUERY_CHUNK_SIZE=1024`.
- Actor and rollout allocators: expandable segments disabled. The tested actor PyTorch
  2.9.1 and rollout PyTorch 2.13.0 cannot exchange expandable CUDA IPC handles. CPU Adam
  offload and full layer recomputation remain enabled.
- `QWEN_GDN_CP_COMPAT=1` installs the recipe-scoped Megatron-Core 0.17 GDN CP
  compatibility path, including the bridge's packed-sequence divisor correction. Use the
  clean pinned bridge checkout, not an experiment-patched bridge. Unexpected runtime
  source layouts fail explicitly. No installed source is rewritten.

The controller allows eight hours for `ppo_update` with one attempt. A timed-out
optimizer update must not be retried on the same live worker: its original request may
still finish. Resource readiness allows 24 hours. These settings use existing scheduler
arguments and are scoped to this recipe.

The default acceptance run is ten steps, batch16 groups with eight samples each,
seed1234 and a 262144-token context limit. `QWEN_ARENA_TASK_IDS_FILE` selects an ordered
task subset for comparison; use the same model, tasks and sampling settings as the
reference. The generation budget remains 65536 tokens with natural EOS.

Validation on 2026-09-21 completed one synthetic optimizer update with 256 sequences of
exactly 262144 tokens on 64 GPUs, with expandable segments enabled. All ranks reported
successful updates and changed parameters; peak allocated/reserved memory was
98.64/110.54 GiB. The first real SWE optimizer update also succeeded, but the following
AWEX weight synchronization failed to deserialize expandable IPC handles. A same-GPU
cross-image probe passes with expandable segments disabled and fails when enabled on the
actor, independently of chunking or CP.

The allocator settings above restore the working IPC path. The ten-step quality
comparison and full-length memory gate with this corrected allocator setting still
require validation; the earlier synthetic pass cannot establish either result.
