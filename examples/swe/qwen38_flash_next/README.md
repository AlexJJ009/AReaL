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

| Workload         | Configuration                     | Submit entry                             |
| ---------------- | --------------------------------- | ---------------------------------------- |
| SWE RL, recovery | `swe_rl_256k.yaml`                | `bash submit_rl.sh swe`                  |
| GSM8K RLVR       | `rlvr_gsm8k_256k.yaml`            | `bash submit_rl.sh rlvr`                 |
| SWE SFT          | `sft_qwen38_flash_next.yaml`      | `sbatch sbatch_sft_qwen38_flash_next.sh` |
| Math SFT         | `sft_qwen38_flash_next_math.yaml` | Same SFT submit script, set `SFT_CONFIG` |

Run the commands from this directory. Both RL recipes preserve 8 nodes / 64 GPUs, actor
TP8/PP8/EP8/CP1, rollout 16 replicas at TP4/EP4, 16 groups x 8 samples, 262144 context
tokens, 65536 output tokens, temperature 1, and inference static memory fraction 0.65.
SWE uses the pinned sixteen-task acceptance subset, not a full Verified evaluation. Its
rollout queue, cache isolation, diagnostics, harness and recovery checks are retained.
RLVR preserves its own queue and cache settings. The historical RLVR entry disables
evaluation even though a validation dataset is present in its YAML.

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
- SWE: `QWEN_PRIVATE_ENV`, `QWEN_RECOVER_SOURCE`, `QWEN_REPLAY64_ACCEPTANCE`,
  `QWEN_CC_PROTOCOL_ACCEPTANCE`. The private environment supplies credentials,
  `ARENA_OPENAPI_BASE`, and `QWEN_ARENA_LLM_BASE`. Never commit that file.

The bundled frozen-weight contract and task/probe fixtures are specific to the
historical model. Set `QWEN_AWEX_FROZEN_CONTRACT` to override the contract for a
separately validated model. They are input fixtures, not fresh acceptance results. The
recovery source JSON must contain `fileroot`, `experiment_name`, `trial_name`,
`expected_saved_global_step`, and `expected_restored_weight_version` referencing an
existing checkpoint. Both acceptance files must come from real validation; the
entrypoint checks their status and the Claude harness version. It refuses a missing
recovery checkpoint or a mismatching restored step/version.

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
recipes have not yet completed a new multi-node training run and do not establish a
hardware cause for the historical CUDA illegal access.

Use CLI `total_train_steps=...` to change RL duration. The historical SWE
`rollout_only_steps` field is unused by this PPO entrypoint.
