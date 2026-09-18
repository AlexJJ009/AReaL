# Qwen3.8 Flash Next recipes

These recipes use 8 nodes / 64 GPUs for RL: actor TP8/PP8/EP8, rollout 16 replicas at
TP4/EP4, batch 16 × 8 samples, context 262144 tokens and generation limit 65536. SFT
uses 3 nodes / 24 GPUs, TP8/PP3/EP8. All paths and cluster placement are external.

| Workload | Entry                                    | Configuration                     |
| -------- | ---------------------------------------- | --------------------------------- |
| SWE RL   | `bash submit_rl.sh swe`                  | `swe_rl_256k.yaml`                |
| GSM8K RL | `bash submit_rl.sh rlvr`                 | `rlvr_gsm8k_256k.yaml`            |
| SWE SFT  | `sbatch sbatch_sft_qwen38_flash_next.sh` | `sft_qwen38_flash_next.yaml`      |
| Math SFT | Same, set `SFT_CONFIG`                   | `sft_qwen38_flash_next_math.yaml` |

## Dependencies

Use the model integration from
[mcore-bridge](https://github.com/dingzhiqiang/mcore-bridge/tree/fix/qwen38-qsa-query-chunk)
(commit `03f5634`), Megatron-Core `0.17.0`, and unmodified AWEX `0.8.1`. Megatron-Core
must also be importable by inference workers because AWEX imports its writer eagerly.
Model implementation changes belong in mcore-bridge, not copied into these recipes.

The validated Qwen runtime is SGLang `0.5.19.dev125+g119b5ffe4` with stable QSA top-k
and the compress-gather bounds fix prepared in the inference image. Patch scripts are
not shipped or applied by these recipes. An unmodified image at that version is
insufficient; obtain the prepared runtime from the model provider.

Upstream [#39446](https://github.com/sgl-project/sglang/pull/39446), commit
`d72e59508b7554045cb51827f9b8d0f08c7a3abc`, is a candidate for replacing the bounds
patch. It is not yet an accepted AWEX runtime or a validated patch-free replacement. Do
not bypass the runtime version check to use it.

## RL launch

Set `QWEN_LAUNCH_ENV` to a private shell file, or export:

- Paths: `QWEN_REPO`, `QWEN_OUTPUT_ROOT`, `QWEN_MODEL`, `QWEN_ACTOR_IMAGE`,
  `QWEN_ROLLOUT_IMAGE`, `MCORE_BRIDGE_ROOT`, `MEGATRON_ROOT`.
- Placement: `QWEN_PARTITION`, `QWEN_RESERVATION`, `QWEN_NODELIST` (eight nodes),
  `QWEN_CONTROLLER_NODE`.
- Mounts: `QWEN_MOUNTS` and `QWEN_CONTROLLER_MOUNTS`, including shared files and the
  site's Slurm/munge configuration, sockets, commands and libraries.
- `QWEN_AWEX_FROZEN_CONTRACT`: a validated frozen-weight manifest for the exact model.
  Obtain it with the checkpoint from its provider: it binds checkpoint hashes, frozen
  PLE values and preserved visual parameters for the supported TP sizes. There is
  currently no public generator or bundled artifact; these recipes are not
  self-contained without that input. Runtime validation must not be bypassed.
- Optional prepared dependency paths: `QWEN_TRAIN_EXTRA_PYTHONPATH` and
  `QWEN_INFER_EXTRA_PYTHONPATH`. No packages are installed at launch.
- SWE: `QWEN_PRIVATE_ENV` with Arena credentials, `ARENA_OPENAPI_BASE` and
  `ARENA_LLM_API_KEY`; `QWEN_ARENA_STREAMS_FILE` in the standard
  [Arena streams format](../README.md). Streams define harness, protocol and reward.
- GSM8K: `QWEN_GSM8K_DATA`; install the standard MathAgent dependencies in the runtime.

SWE trains on the configured streams. For a controlled single-stream comparison,
`QWEN_ARENA_TASK_IDS_FILE` may point to an external JSON list of data IDs. The entry
rejects duplicate or missing IDs and preserves the list order. Keep evaluation tasks out
of the selected training data. No benchmark task IDs are shipped here. Gateway traffic
defaults to `ARENA_OPENAPI_BASE/api`. When the deployment has a separate LLM gateway,
set `ARENA_LLM_BASE_URL` in the private environment sourced by the controller and
rollout workers. Use an absolute HTTP(S) URL; supply credentials separately through
`ARENA_LLM_API_KEY`.

```bash
bash examples/swe/qwen38_flash_next/submit_rl.sh swe total_train_steps=10
```

Use a new trial/output directory for a fresh run. Recovery uses normal AReaL `recover.*`
configuration; there are no experiment-specific recovery/acceptance gates. Evaluation is
disabled in these short RL recipes. Logs and metrics are under `QWEN_OUTPUT_ROOT`.

## Functional helpers

`train_rl.py` uses standard Arena/MathAgent workflows and sets SGLang PLE embedding CPU
offload and FlashInfer linear attention. `actor_worker.py` selects FlashAttention and
full CPU Adam offload to fit the 256K memory budget. These narrowly scoped runtime
overrides cover options not exposed by the current AReaL configuration schema.

`proxy.py` and `template_defaults.py` supply thinking defaults (`enable_thinking=true`,
`reasoning_effort=medium`), honoring explicit request switches. SWE preserves unique
cache salts per generation request; GSM8K retains native prefix-cache reuse. Neither
helper captures tokens, gradients, wire payloads or diagnostic snapshots.

## SFT launch

Export `AREAL_DIR`, `MCORE_BRIDGE_ROOT`, `AREAL_IMAGE`, `TRAIN_RUNTIME_DEPS`,
`MODEL_PATH`, `FILERoot`, `QWEN_MOUNTS`, and `SFT_DATASET` (or `MATH_DATASET`). Set
`SFT_CONFIG` to the repository-relative math YAML for math SFT. Pass site partition,
reservation, nodes and log destination as `sbatch` options. Math SFT expects formatted
SFT records, not the raw GSM8K RL dataset.

## Historical validation

The earlier source `3426e938c` completed ten fresh 256K SWE steps without OOM. Mean
training reward was `0.9359375`, matching the internal run; first-step absolute logp
difference was `0.038055`, ten-step mean `0.0544257`. This used a selected training
subset and SGLang `0.5.19.dev125+g119b5ffe4` with two local patches. These results do
**not** validate the cleaned recipes or patch-free upstream source; a new run is
required.
