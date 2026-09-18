# Qwen3.8 Flash Next runtime dependencies

The 256K SWE experiment uses the Qwen model integration in AReaL and
[mcore-bridge's query-chunk branch](https://github.com/dingzhiqiang/mcore-bridge/tree/fix/qwen38-qsa-query-chunk),
commit `03f5634`. That bridge includes trainable PLE export, bounded QSA query
workspace, and shorter PLE backward temporary lifetimes. The tested Megatron-Core
version is `0.17.0`.

## SGLang startup patch

The tested SGLang package is `0.5.19.dev125+g119b5ffe4`. The experiment applies
`patch_sglang_qsa_topk.py` to its disposable inference environment before spawning
SGLang workers:

```bash
export QWEN_QSA_STABLE_TOPK=1
python3 examples/swe/qwen38_flash_next/patch_sglang_qsa_topk.py
```

Run this in the rollout worker's startup commands, after configuring its Python
environment and before starting inference. The script finds the installed package;
`--target PATH` supports an explicit disposable source copy. It checks the original
source SHA256 and function signature and refuses unknown or already-patched source. Use
a fresh disposable runtime for each launch; do not patch a shared installation.

Stable selection resolves equal scores by lower relative key index while preserving
causal bounds. Without the environment flag, the wrapper returns native selection. This
patch preserves the previous experiment's behavior; it is not a demonstrated fix for the
CUDA illegal access observed after AWEX resume.

The project metadata and lockfiles pin this experiment-specific bridge revision. The
launch wrapper also accepts its checkout through `MCORE_BRIDGE_ROOT`.

## AWEX inference imports

Use unmodified AWEX `0.8.1` and make Megatron-Core available in the inference Python
environment. AWEX eagerly imports its writer at package initialization; that writer
imports `TransformerConfig` from Megatron even when only the reader is needed. Importing
a reader submodule does not bypass package initialization.

The historical experiment used lazy AWEX exports to work around missing Megatron in the
inference image. An isolated check in the same SGLang image succeeded with unmodified
AWEX and Megatron-LM revision `f007db77b` available on `PYTHONPATH`: `awex`,
`awex.reader.nccl_reader`, and `areal.engine.awex.colocate_reader` all imported
successfully. The alternative needs no AWEX source patch. This verifies imports only;
GPU weight transfer and generation must still be validated with that environment before
a full run.

## Training recipes

All entrypoints below are committed with their runtime helpers in `runtime/`.
`dependencies.json` records the experimental bridge and installed package versions. Use
unmodified AWEX in the inference environment; do not reuse a historically patched AWEX
overlay without restoring its upstream source.

| Workload   | Configuration                     | Submit entry                             |
| ---------- | --------------------------------- | ---------------------------------------- |
| SWE RL     | `swe_rl_256k.yaml`                | `bash submit_rl.sh swe`                  |
| GSM8K RLVR | `rlvr_gsm8k_256k.yaml`            | `bash submit_rl.sh rlvr`                 |
| SWE SFT    | `sft_qwen38_flash_next.yaml`      | `sbatch sbatch_sft_qwen38_flash_next.sh` |
| Math SFT   | `sft_qwen38_flash_next_math.yaml` | Same SFT submit script, set `SFT_CONFIG` |

Run the commands from this directory. Both RL recipes preserve 8 nodes / 64 GPUs, actor
TP8/PP8/EP8/CP1, rollout 16 replicas at TP4/EP4, 16 groups x 8 samples, 262144 context
tokens, 65536 output tokens, temperature 1, and inference static memory fraction 0.70
(SWE) / 0.65 (RLVR). SWE uses the pinned sixteen-task acceptance subset, not a full
Verified evaluation. Its rollout queue, cache isolation, diagnostics, harness and
recovery checks are retained. RLVR preserves its own queue and cache settings. The
historical RLVR entry disables evaluation even though a validation dataset is present in
its YAML.

### Open-source main compatibility

The recipes use main's `actor.min_usable_group_size=8` and retain rollout-time mean-only
reward normalization with `gconfig.reward_normalization_use_std=false`. This option
defaults to true for existing workflows; mean-only normalization requires the v1 rollout
backend. `TOTAL_TRAIN_STEPS` controls the training limit (default 10).

The runtime helpers explicitly supply thinking template defaults through
`extra_body.chat_template_kwargs` (`enable_thinking=true`, `reasoning_effort=medium`,
`thinking_option=null`). A request-level thinking switch overrides the default switches.
SWE installs these defaults in its diagnostic proxy; RLVR supplies them in the MathAgent
request. These replace inner-source-only AgentConfig fields.

### RL launch environment

Export these variables, or set `QWEN_LAUNCH_ENV` to an untracked shell file defining
them (the submit wrapper exports variables sourced from that file):

- Paths: `QWEN_OUTPUT_ROOT`, `QWEN_MODEL`, `QWEN_ACTOR_IMAGE`, `QWEN_ROLLOUT_IMAGE`,
  `MCORE_BRIDGE_ROOT`, `MEGATRON_ROOT`.
- Placement: `QWEN_PARTITION`, `QWEN_RESERVATION`, `QWEN_NODELIST` (eight worker nodes),
  `QWEN_CONTROLLER_NODE`.
- Mounts: `QWEN_MOUNTS` for workers; `QWEN_CONTROLLER_MOUNTS` for the controller.
  Include the shared repository, data, outputs, dependency paths, and container access
  to the site's Slurm/munge configuration, sockets, commands and libraries.
- Optional overlays: `QWEN_TRAIN_EXTRA_PYTHONPATH`, `QWEN_INFER_EXTRA_PYTHONPATH`. The
  wrapper prepends committed runtime helpers and adds the bridge/repository. The
  inference path also includes `MEGATRON_ROOT` for unmodified AWEX imports.
- RLVR: `QWEN_GSM8K_DATA`. `math_verify` and OpenAI client dependencies must be
  installed in the inference/proxy runtime; `dependencies.json` also records the three
  historical math package versions. Include their overlay in
  `QWEN_INFER_EXTRA_PYTHONPATH` when needed. No installation happens at launch.
- SWE: `QWEN_PRIVATE_ENV`, `QWEN_REPLAY64_ACCEPTANCE`, `QWEN_CC_PROTOCOL_ACCEPTANCE`.
  The private environment supplies credentials, `ARENA_OPENAPI_BASE`, and
  `QWEN_ARENA_LLM_BASE`. Never commit that file.

The bundled frozen-weight contract and task/probe fixtures are specific to the
historical model. Set `QWEN_AWEX_FROZEN_CONTRACT` to override the contract for a
separately validated model. They are input fixtures, not fresh acceptance results. The
recovery source JSON must contain `fileroot`, `experiment_name`, `trial_name`,
`expected_saved_global_step`, and `expected_restored_weight_version` referencing an
existing checkpoint. Both acceptance files must come from real validation; the
entrypoint checks their status and the Claude harness version. It refuses a missing
recovery checkpoint or a mismatching restored step/version.

Set `QWEN_SWE_START_MODE=fresh` for a complete run from the initial HF model:

```bash
QWEN_SWE_START_MODE=fresh bash submit_rl.sh swe total_train_steps=10
```

Use a new output directory/trial. Fresh mode rejects any automatically discovered
checkpoint or nonzero initial weight version. The default `recover` mode requires
`QWEN_RECOVER_SOURCE`; its step limit is the final total step, not the number of
additional steps. Restoring completed step 5 with `total_train_steps=10` runs only five
new steps and validates recovery, not a fresh ten-step training run.

Training overrides are forwarded intact, for example:

```bash
bash submit_rl.sh rlvr total_train_steps=11 recover.trial_name=previous-trial \
  recover.fileroot="$PREVIOUS_RUN_ROOT"
```

### SFT launch environment

Export `AREAL_DIR`, `MCORE_BRIDGE_ROOT`, `AREAL_IMAGE`, `TRAIN_RUNTIME_DEPS`,
`MODEL_PATH`, `FILERoot`, `QWEN_MOUNTS`, and `SFT_DATASET` (or `MATH_DATASET` for math
SFT). Math SFT expects SFT-formatted records; the raw GSM8K dataset belongs to the RLVR
entry. Set `SFT_CONFIG` to the math YAML's repository-relative path to select it. Pass
site-specific partition, reservation, nodelist and output path as `sbatch` options. The
default allocation is 3 nodes, 8 GPUs each, TP8/PP3/EP8/CP1. The worker preserves the
original SPMD `torchrun` launch and requires a prepared runtime; it does not install or
upgrade CUDA dependencies. CP greater than one is not enabled by this recipe.

### Validation scope

The original controller and trainer used different bridge checkouts. These portable RL
scripts use the explicitly selected bridge for both. The `QWEN_QSA_STABLE_TOPK` flag is
implemented on inference only; the trainer still uses native `scores.topk`. SWE applies
the QSA startup patch; the historical RLVR recipe did not, and that behavior is
preserved. Changing these numerical choices requires separate validation. The packaged
recipes require runtime validation for the chosen start mode; a completed recovery run
does not validate a fresh ten-step run or establish a hardware cause for the historical
CUDA illegal access.

Use CLI `total_train_steps=...` to change RL duration. The historical SWE
`rollout_only_steps` field is unused by this PPO entrypoint.

### Full training pool

Set `QWEN_SWE_START_MODE=fresh`, `QWEN_SWE_TASK_SCOPE=training_pool` and
`TOTAL_TRAIN_STEPS=11` for eleven updates from the initial model over the entire pinned
training pool. The heldout partition remains excluded. The default
`QWEN_SWE_TASK_SCOPE=acceptance` retains the historical sixteen-task subset for
compatibility. `training-inventory.json` records the selected scope and task IDs.
Compare training rewards with evaluation on matched tasks; the historical fast subset
and asynchronous completion can substantially bias aggregate rewards.

### Fresh 256K SWE validation (2026-09-18)

Both branches completed ten fresh training steps without OOM, using 16 groups x 8
samples, 262144 context tokens and 65536 output tokens. The ten-step aggregate training
reward matched exactly: 1198/1280 (0.9359375) for both branches.

| Metric                                 |  Internal | Open-source |
| -------------------------------------- | --------: | ----------: |
| First-step absolute logp difference    |  0.038508 |    0.038055 |
| Ten-step mean absolute logp difference | 0.0487336 |   0.0544257 |

These are `ppo_actor/update/logp_abs_diff/avg` values, of order 1e-2. The actual
training trajectories differ; matching aggregate reward does not mean identical tokens
or numerical equivalence. Initial fixed probes matched tokens and logprobs exactly.
Post-update probe discrepancies remain unresolved.

Validated source revisions: open-source `3426e938c`, internal `3ccb512ff`. The runtime
used SGLang `0.5.19.dev125+g119b5ffe4`, retaining the stable top-k patch and applying
upstream compress-gather fix #38346, commit `1cdc5bca5e97b7134136a535c90c6037bd2001fa`.
This was not a patch-free stable-tag validation. Reward is from the acceptance training
subset, not an independent SWE-bench evaluation.

### Reproduce the validated SGLang bounds fix

`patch_sglang_qsa_compress_gather.py` packages the exact one-line upstream #38346 fix
used by the ten-step run. It clamps padded compress-gather indices to the available
source keys. This is separate from `patch_sglang_qsa_topk.py`, which controls tied-score
selection. Both remain experiment-specific helpers here; SGLang itself is an external
dependency.

Run both helpers only in the disposable inference environment, before workers start. The
bounds-fix helper verifies the original and patched SHA256 hashes, rejects unknown or
already-patched source, and writes provenance to a new audit file. It does not modify
the launcher's behavior automatically.

```bash
python3 examples/swe/qwen38_flash_next/patch_sglang_qsa_compress_gather.py \
  --audit compress-gather-fix.json
```

For an isolated source copy, pass `--target /path/to/qsa_indexer.py`. Do not run against
a shared installation. A newer stable SGLang tag containing this fix still requires
separate compatibility and training validation; this result does not establish that
either helper can be removed.
