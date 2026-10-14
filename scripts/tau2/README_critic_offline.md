# Offline tau2 critic experiments

These tools reuse saved training rollouts and fixed validation rollouts. They do not
start an actor, simulator, SGLang server, or API client. Feature extraction and full
FSDP replay need GPUs; probes and head simulation run on CPU.

Set `PYTHONPATH` to this checkout even when using another checkout's Python environment.
Keep all outputs outside the source checkout and original run directories.

## Extract features

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
