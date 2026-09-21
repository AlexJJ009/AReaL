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
- Actor allocator: expandable segments enabled; rollout retains disabled expandable
  segments. CPU Adam offload and full layer recomputation remain enabled.
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
exactly 262144 tokens on 64 GPUs. All ranks reported successful updates and changed
parameters. Peak allocated/reserved memory was 98.64/110.54 GiB. This validates the
full-length memory configuration, not ten-step SWE quality or steady-state speed. The
real SWE comparison is still pending.
