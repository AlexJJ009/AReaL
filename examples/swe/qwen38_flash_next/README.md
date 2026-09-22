# Qwen Flash Next 256K SWE recipe

`swe_rl_256k.yaml` uses 64 GPUs with TP8 / PP4 / CP2 / EP16. The actor's
`--distribution=block:block` keeps each TP8 group on one eight-GPU node. Keep the
existing model, image, mount, Arena and frozen-weight-contract environment settings used
by `submit_rl.sh`; private credentials remain in the external environment file.

The long-context settings are:

- Chunked LM-head loss: `enable_chunked_logits: true`, chunk size 1024.
- PLE causal chunks: `QWEN_PLE_CHUNK_TOKENS=8192`; overlapping causal history is
  retained within each sequence, and shared weight gradients accumulate in FP32.
- QSA index-score workspace: the pinned upstream bridge limits it to 1 GiB.
- Actor allocator: `expandable_segments:True`; rollout: `False`. AWEX allocates only its
  IPC staging buffers with expandable segments temporarily disabled, restoring all
  runtime allocator settings before serialization, including on packing errors. Training
  allocations retain expandable storage. CPU Adam offload and full recomputation remain
  enabled.
- `QWEN_GDN_CP_COMPAT=1` installs the recipe-scoped Megatron-Core 0.17 GDN CP
  compatibility path, including the bridge's packed-sequence divisor correction. Use the
  clean pinned bridge checkout, not an experiment-patched bridge. Unexpected runtime
  source layouts fail explicitly. No installed source is rewritten.

The controller allows eight hours for `ppo_update` with one attempt. A timed-out
optimizer update must not be retried on the same live worker: its original request may
still finish. Resource readiness allows 24 hours. These settings use existing scheduler
arguments and are scoped to this recipe.

Arena integration uses the standard `econfig.arena_streams` and
`econfig.arena_streams_file` fields. Supply your own stream, harness and reward
references using `QWEN_ARENA_STREAMS_FILE`; this recipe does not bundle a private
benchmark selection. Configure the harness request timeout to cover colocated training
pauses, and verify that the harness honors the requested timeout.

The actor rejects nonfinite optimizer gradient norms before clipping or updating
weights. On failure it reports rank-local element finiteness and a chunked FP64 norm to
distinguish nonfinite gradients from FP32 norm overflow. This guard prevents an invalid
update; it does not repair the underlying gradient instability.

The default acceptance run is ten steps, batch16 groups with eight samples each,
seed1234 and a 262144-token context limit. `QWEN_ARENA_TASK_IDS_FILE` selects an ordered
task subset using explicit `env:key@version` references for both RL and evaluation. Pin
the benchmark versions: the stream's latest versions can include diagnostic tasks under
the same environment keys. Use the same model, tasks and sampling settings as the
reference. The generation budget remains 65536 tokens with natural EOS.

Actor and rollout images must use compatible CUDA IPC serialization. The IPC staging
helper disables expandable segments only while allocating exported buffers; validate
cross-process weight equality with your actual image pair before training.

## Vision RL

For the Qwen4Exp ModelScope bridge, set `actor.megatron.language_model_only: false`,
`sglang.enable_multimodal: true`, and `sglang.skip_tokenizer_init: true`. Use the OpenAI
chat-completions proxy with base64 image inputs and processor-produced modality IDs. The
Qwen4Exp `WRAPPER_THD` path supports the chunked LM head: it preserves the visual
forward and mRoPE preparation while bypassing only the language output projection.

Vision actors require frozen-contract schema 2 with `language_model_only: false`. Both
sides load the same checkpoint. The actor keeps the HF visual tower frozen on PP0's
first virtual stage; AWEX excludes it on both sides and preserves the receiver's visual
state across offload/resume. Before the first exchange, the binder compares actor visual
values and shapes with the checkpoint. Every exchange validates the live original
parameters, frozen state, and ownership. Schema 1 remains text-only.

A reduced random Qwen4Exp with actual image processing and visual forward passed
chunk/full loss and gradient comparisons at CP1 and CP2 on 2026-09-21. This is a
numerical qualification, not evidence of full-model RL quality or benchmark parity.

### Diagnostic batch snapshots

For a small diagnostic run, set `QWEN_BATCH_SNAPSHOT_DIR` to a new, empty directory on
shared storage. The controller saves the output of each `prepare_batch` call, and the
input and output of each `compute_advantages` call as CPU `.pt` files. Snapshots retain
all batch fields, including image tensors, masks, log probabilities, and version fields.
Filenames count calls, not optimizer steps. No credentials from the experiment
configuration are copied into snapshot metadata.

This opt-in path adds CPU memory, remote tensor reads, and storage overhead. Publication
is atomic and refuses to overwrite existing files; a save failure stops the diagnostic
run. Load snapshots with `load_batch_snapshot(path)` from `batch_snapshot.py`; it uses
`torch.load(..., weights_only=True)` and restores rollout group metadata. These files
support input replay investigations, but do not contain model/optimizer state or RNG
state and do not by themselves reproduce an optimizer step. Repeated training on an old
snapshot is not an on-policy RL experiment.

For one diagnostic update without new rollout, set `QWEN_BATCH_REPLAY_PATH` to a
`prepare_batch-NNNN.output.pt` snapshot from any captured call, `total_train_steps=1`,
`recover.mode=disabled`, and `evaluator.eval_before_train=false`. Use a fresh
trial/output directory and the same initial checkpoint and sample count as the captured
run. The recipe validates model path and sample count, replaces preparation once, and
fails if another batch is requested. It still initializes the engines and exercises the
regular training and weight-update path. Selecting a later batch does not restore the
preceding optimizer or AWEX cycles. This is not exact RNG/optimizer recovery or a reward
evaluation; it does not verify that checkpoint bytes at the same path are unchanged.
Never resume this diagnostic trial as a normal RL run. Optional new snapshots must use a
different directory from the input snapshot.

To diagnose state carried across weight transfers, use `QWEN_BATCH_REPLAY_PATHS` instead
of the single-path variable: a JSON array of snapshot paths in call order, starting at
call zero from one source trial. Set `total_train_steps` to the number of paths. Each
batch is consumed once, with the normal optimizer update and AWEX transfer between
batches; exhaustion cannot fall back to live rollout. This reproduces the sequence of
inputs, not exact RNG, concurrent generation, or optimizer recovery. It is a diagnostic,
not an RL learning or evaluation run.

## Pinned bridge and multimodal recipes

`runtime.env` pins upstream ModelScope mcore-bridge to
`bc58ea9cf9b1dd2314637703973904f359e67c75`. `submit_rl.sh` rejects a different or dirty
bridge checkout before submitting. Supply its path through `MCORE_BRIDGE_ROOT`. This
main revision already includes QSA/PLE int64 offsets, checkpoint helpers and QSA indexer
freezing; do not apply the duplicate local QSA patch. The earlier SWE recipe used an
AReaL-local checkpoint validator, explaining why its older bridge worked without the
newly imported checkpoint module.

Set `QWEN_CONFIG` to `swe_mm_rl.yaml` for the small ten-step vision RL recipe (batch2,
samples4, CP2, chunk loss1024, PLE chunk8192, staleness2). Paths, images, reservation,
streams and the schema2 frozen contract remain external environment settings. Use fresh
trial names and disable diagnostic replay for real RL.

Use `submit_rl.sh swe-eval` with `swe_mm_eval.yaml` for one full validation pass. Set
`QWEN_ARENA_TASK_IDS_FILE` to explicit reference Env versions, `QWEN_EVAL_TASK_COUNT` to
their count, and `QWEN_ARENA_STREAMS_FILE` to the exact reference Harness and Reward
configuration. Both sample counts are one; training steps are zero, recovery is
disabled, and validation is unshuffled without dropping tasks. No optimizer update or
asynchronous training prefetch occurs. The evaluation recipe uses 32768 response tokens,
temperature 1 and medium reasoning effort. Match these settings to the benchmark you
intend to compare.

Compare the complete task set, including failed tasks as zero where required by the
reference aggregation. A single sampled pass need not produce identical trajectories or
exactly the same score. Do not interpret a zero-gradient replay as proof of effective RL
learning, checkpoint correctness, or AWEX value equality.

## AWEX dependency for this branch

This branch pins AWEX to commit `dc665c44d61c193fd83997bdda43a7dc8e86cd62` from
[inclusionAI/Awex#121](https://github.com/inclusionAI/Awex/pull/121) for bounded
transport and CUDA IPC staging allocation. Both dependency declarations and lockfiles
use this exact revision. Actor and rollout images must use the same pinned dependency;
the released `awex==0.8.1` does not contain these APIs. The AReaL colocate reader
selects the bounded transport for all supported models, preserving the original
experimental behavior; AWEX's default transport is unchanged.
