# Offline tau2 critic experiments

These tools reuse saved training rollouts and fixed validation rollouts. They do not
start an actor, simulator, SGLang server, or API client. Feature extraction and full
FSDP replay need GPUs; probes and head simulation run on CPU.

Set `PYTHONPATH` to this checkout even when using another checkout's Python environment.
Keep all outputs outside the source checkout and original run directories.

## Registered pre-experiment (E1–E9)

Use `critic_preexp_extract` and `critic_preexp` for the frozen 2026-10-14 protocol. The
older entry points below remain general diagnostics and do not implement its complete
selection and experiment requirements. The protocol document is authoritative; these
tools never edit it.

Every registered command requires the audited `pool-manifest.json` and its `.sha256`
sidecar. The loader verifies source hashes, identities, labels and masks, rejects
train/validation task overlap, and restores r510's numeric rollout order. Cache merge
checks the manifest-selected trajectories and every sampled position. Training uses up
to 128 uniformly sampled action prefixes per trajectory with seed 20261014; validation
retains all action prefixes. Each sampled training token carries its original action
count divided by retained count as loss weight.

E1 stores the final value-head input and decoder hidden states 8, 16 and 24 for the
32-layer base model. Hidden-state index 8 is the output of decoder block 7, matching
`output_hidden_states[8]`; similarly for 16 and 24. These correspond to 25%, 50% and 75%
depth. Final features include the final norm. The trained backbone must come from
`critic/epoch2epochstep14globalstep54`, not the sibling `default/` actor checkpoint. The
cache retains source ID, task, domain, run, action-token progress and turn index. When
original turn IDs are unavailable, turn indices mean contiguous action-mask spans.

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# POOL, BASE, TRAINED, ART and AREAL_PYTHON are absolute local paths.
"$AREAL_PYTHON" -m scripts.tau2.critic_preexp_extract \
  --mode estimate --pool "$POOL" --output "$ART/extraction-estimate.json"
```

The eight-GPU wrapper is `scripts/tau2/launch_critic_preexp.sh`, taking positional
arguments `PYTHON WORKTREE POOL BASE_CHECKPOINT TRAINED_CRITIC OUTPUT`. It launches one
process per GPU with disjoint trajectory shards, first for the base and then for the
trained backbone, and merges all five caches. It never submits a pueue job. Run it only
inside the allocated `gpu8` slot after the existing SAO job. Its estimate assumes
aggregate throughput, not per-GPU throughput; CPU text decoding and initialization time
are additional. Retaining process parts and merged caches doubles feature disk use.

After extraction, the complete CPU command is:

```bash
CUDA_VISIBLE_DEVICES='' "$AREAL_PYTHON" -m scripts.tau2.critic_preexp \
  --pool-manifest "$POOL" --features "$ART/e1/caches/base-final/manifest.json" \
  --layer-cache "base-layer-8=$ART/e1/caches/base-layer-8/manifest.json" \
  --layer-cache "base-layer-16=$ART/e1/caches/base-layer-16/manifest.json" \
  --layer-cache "base-layer-24=$ART/e1/caches/base-layer-24/manifest.json" \
  --layer-cache "r510-final=$ART/e1/caches/r510-final/manifest.json" \
  --r510-reference-json "$R510/tau2-critic-fit-evidence/validation-step-000080.json" \
  --out "$ART/results" --threads 4
```

This writes `E2.json` through `E9.json`, paired 20,000-draw task-bootstrap intervals,
seed summaries, the head initialization artifacts, `hypothesis-decisions.json` and
`results-table-draft.md`. `--experiments E2 E9` selects named experiments; the full run
is the default. All regularization and learning-rate choices use task-grouped five-fold
training CV. E4 chooses rates independently for each batch/budget. H2 compares the
baseline at step 80 against the same r510 validation trajectories, while primary reports
use all 120 validation trajectories. H2 failure marks simulations as reference evidence.

E8 selects a literal visible-prefix text rule on training prefixes only, records all
candidate rates, and refuses a selected rate outside 20–60%. Validation uses the same
rule on its own prefixes. E6 task means count trajectories once, regardless of length;
its 2/6-trajectory strata refer to training task groups, since validation tasks are
separate. Paired bootstrap always moves the two validation trajectories of a task
together. The shared ridge is the declared primary for ambiguous shared/domain
hypothesis comparisons; both heads are reported, and validation never selects one.

E9 exports `ridge-head.pt` for `offline_critic_fit --head-init` and a head-only
`ridge-head-sao/` bundle for the SAO preflight. The bundle contains no backbone and must
not be used as a full checkpoint. The runner executes both consumer CLIs and checks
exact readback. The standalone CPU loader check is:

```bash
CUDA_VISIBLE_DEVICES='' "$AREAL_PYTHON" -m scripts.tau2.offline_critic_fit \
  --check-head-init --head-init "$ART/results/ridge-head.pt" --checkpoint "$BASE"
```

For a CPU dry run, first generate a fixture with
`python -m scripts.tau2.critic_preexp_synthetic --out "$ART/synthetic-fixture"`, then
use its pool, five cache manifests and `synthetic-r510-step80.json` in the command above
with `--synthetic`. It keeps 2774/120 trajectories and 747/60 task groups but uses six
action tokens and eight feature dimensions. It executes all registered arms and
bootstrap draws; synthetic values cannot establish H1–H8.

## Legacy single-cache feature extraction

```bash
export PYTHONPATH="$PWD"
# R510: original critic-head-qwen35-mix-r2 directory
# MODEL: local pinned Qwen3.5-4B snapshot or compatible critic checkpoint
# ART: new experiment artifact directory
python -m scripts.tau2.critic_features \
  --train "r510=$R510/logs/root/*/*/rollout/0/*.jsonl" \
  --validation "r510=$R510/tau2-critic-fit-evidence/fixed-validation-rollouts.pt" \
  --train-k 128 --estimate-only --output "$ART/estimate.json"
python -m scripts.tau2.critic_features \
  --train "r510=$R510/logs/root/*/*/rollout/0/*.jsonl" \
  --validation "r510=$R510/tau2-critic-fit-evidence/fixed-validation-rollouts.pt" \
  --checkpoint "$MODEL" --train-k 128 --output "$ART/features"
```

Repeat `--train` and `--validation` with distinct run names to include other runs.
Validation tensors require their adjacent fixed-validation manifest. The reader checks
identities and rejects training/validation task overlap. Retry rollout IDs are mapped
back to original logical IDs before sorting.

Features are the final-normalized `model.language_model` outputs supplied to the
adapter's `score` layer. An action token at position $t$ is supervised by the feature at
$t-1$, matching the production loss mask. The cache preserves the position, trajectory,
run, task, domain, logical batch and original token count. Validation always retains
every action prefix. `--train-k 0` also retains every training prefix; otherwise
deterministic uniform sampling stores at most K and weights each retained prefix by the
original count divided by its sample count. That weighting preserves trajectory token
mass, but sampled gradients remain an approximation. Sampling saves storage, not
backbone forward computation.

## Fit and simulate heads on CPU

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.tau2.critic_head_sim \
  --features "$ART/features/manifest.json" --out "$ART/head-simulation"
```

Ridge and logistic regularization are chosen on a task-grouped holdout from training,
then refitted using training only. Test validation never selects a hyperparameter or
checkpoint. Reports reuse the production metric row and EV aggregation; AUC uses
trajectory-mean predictions. Bootstrap draws resample task clusters, preserving repeated
rollouts of a task and using paired draws for model differences.

Adam replay uses saved batch membership, the production unhalved token-mean MSE, AdamW
with zero weight decay, and gradient clipping. Sweeps include learned/frozen bias and
warmup. The frozen backbone, reduced feature precision and optional prefix sampling
differ from online full-model training. A matching head norm is only a sanity check, not
proof of equivalent learning. The JSON records this limit.

## Full offline replay and a planted control

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.tau2.offline_critic_fit \
  --config "$R510/tau2-critic-fit-evidence/resolved-config.json" \
  --train-rollouts "$R510/logs/root/*/*/rollout/0/*.jsonl" \
  --validation-rollouts "$R510/tau2-critic-fit-evidence/fixed-validation-rollouts.pt" \
  --output-dir "$ART/replay" --head-lr 1e-4 --backbone-lr 5e-6 \
  --warmup-steps 10 --epochs 1 --check-config
```

For GPU execution remove `--check-config` and launch with `torchrun`, matching its
process count to `critic.backend` (the original r510 configuration has four training
ranks). The entry point uses `Tau2HeadFSDPPPOCritic`, its optimizer and PPO update, the
original `_value_target_row`, and the official validation evaluator. One epoch means one
pass over **saved trajectories**: r510's 1,280 saved rows already contain four original
task-pack epochs and make 80 optimizer steps at batch size 16.

`--label-override module:function` changes training targets on copies of rows. Use
`--validation-label-override module:function` explicitly when measuring a synthetic
target; otherwise validation keeps official rewards. The provided `first_token_parity`
hook is a simple binary target visible from the first token, intended for wiring checks;
it can be constant on real chat templates. A useful real planted control should vary
across examples and be visible at the prefixes being evaluated. An eventual tool error
may not be predictable before that tool response appears. Label hooks run trusted local
Python code.

The ridge export can initialize replay via `--head-init`; the base checkpoint and head
shapes must match. Logistic probe weights require a sigmoid and are not a compatible
initialization for this linear value head.

## Queue wrapper and verification

`scripts/tau2/launch_critic_offline.sh features|fit|sim` sets checkout-local imports,
offline Hugging Face mode, and invokes the corresponding entry point. Supply
`AREAL_PYTHON`, `ART`, and, for `fit`, `NPROC_PER_NODE`. It never submits jobs. After
resources are available, it can be used as the command of a pueue `gpu8` job. Estimate
first and direct stdout/stderr into a new file under `ART`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_tau2_critic_features.py tests/test_tau2_critic_head_sim.py \
  tests/test_tau2_offline_critic.py
```

The tiny CPU smoke exercises the actual model, optimizer, scheduler and loss. It cannot
validate FSDP sharding, GPU kernels, distributed numerics or runtime.
