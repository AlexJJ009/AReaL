# Miles hyperparameters with AReaL asynchronous GRPO

The existing `sao_grpo.py --config examples/math/sao_grpo.yaml` entrypoint now uses the
user-selected Miles hyperparameters. This is a configuration migration, not a claim of
reproduced convergence or a numerically identical Miles loss.

Source:
[radixark/miles@366f614](https://github.com/radixark/miles/blob/366f614b8ba3904984d431291dd396be48ae1aed/examples/infra_features/fully_async/run_qwen3_5_4b_fully_async_eval.py).

| Setting                                                     | Explicit value                                |
| ----------------------------------------------------------- | --------------------------------------------- |
| Training prompt batch / samples per prompt                  | 32 / 8 (256 trajectories per update)          |
| Response / total sequence limit                             | 8192 / 9216                                   |
| Temperature / top-p                                         | 1 / 1                                         |
| Adam LR / betas / weight decay / epsilon                    | 1e-6 / (0.9, 0.98) / 0.1 / 1e-8               |
| Scheduler / warmup / gradient clipping                      | constant / 0 / 1                              |
| Ratio clip                                                  | \[0.8, 1.28\]                                 |
| Reward scaling / bias / KL                                  | 1 / 0 / 0                                     |
| Group reward normalization / extra advantage whitening      | group mean and std / disabled                 |
| Discount / GAE lambda / optimizer minibatches               | 1 / 1 / 1                                     |
| Compute / master weights and optimizer / gradient reduction | BF16 / FP32 / FP32                            |
| Async staleness / logprob recompute / decoupled loss        | 2 / enabled / enabled                         |
| Async correction                                            | token ratio mask, upper 5, no lower threshold |

Zero warmup replaces the old five-update warmup: the first optimizer update uses 1e-6.
The inherited official GSM8K recipe remains unchanged. Switching to its 6e-6 LR is a
future explicit recipe change; this script never changes LR based on observed reward.

## Deliberate local differences

- Keep the existing Qwen3.5-4B-Base snapshot, nonthinking workflow and audited math
  scorer. Miles uses Qwen3.5-4B with its DAPO reward implementation.
- Keep one finite training epoch, rather than silently adopting Miles's 3000 rollouts.
  On 17,157 prompts, dropping the incomplete prompt batch gives 536 optimizer updates,
  17,152 consumed prompts, 137,216 trajectories and 5 dropped prompts. Each consumed
  prompt has all 8 samples.
- Use 4 FSDP training GPUs + 3 SGLang rollout GPUs + 1 dedicated evaluation GPU. The
  example adapter reuses native controllers. Training publishes XCCL weights to its
  three rollout servers; evaluation loads saved HF checkpoints separately.
- Keep the existing 700-problem validation, 2 answers per problem, 8192 response limit,
  evaluation every 20 updates and at epoch end. This differs from Miles's AIME
  evaluation, 8 samples, 16384 response limit and interval 5. Evaluation runs in a
  serial background worker on its own GPU. Training can progress meanwhile; run
  completion waits for all scheduled evaluations.
- Keep native AReaL token-mean loss reduction. The pinned Miles script defaults to
  averaging each answer's token loss then averaging answers. This AReaL worktree does
  not expose that reduction option. Therefore the learning rate is a candidate copied
  from Miles, not a guarantee of identical gradients.
- Retain AReaL's proximal/behavior correction, as requested. Do not multiply a second
  TIS factor into it.

## TIS mapping

Miles's built-in TIS uses `clamp(exp(train_old_logp - rollout_logp), 0, 2)`. AReaL
already supports the corresponding token-weight clamp via
`rejection_sampling: {level: token, action: clamp, metric: ratio, lower: 0, upper: 2}`.
That would replace the current mask correction. It is not enabled in this recipe, and
the contract intentionally rejects that unapproved switch. Masking and clamping differ:
for a ratio of 6, the current mask gives weight 0, while TIS clamp gives weight 2. Loss
reduction remains a separate difference.

## Validation and launch

Set the existing `SAO_MODEL_PATH`, `SAO_DATA_PATH`, `SAO_RUN_ROOT` and `SAO_TRIAL_NAME`
environment variables, then use the existing configured runtime:

```bash
python examples/math/sao_grpo.py --config examples/math/sao_grpo.yaml --check-config
```

This resolves the typed AReaL configuration and validates the recipe without
initializing training. Removing `--check-config` starts training and is not part of this
configuration-only task. The historical launch manifest and historical run evidence
remain unchanged; they do not authorize this new configuration. `launch_grpo_run.py` now
derives group size, LR and evaluation versions from the new run's saved
configuration/evidence rather than N4/134-step constants.

## Dedicated evaluation GPU (4 / 3 / 1)

`rollout.backend=sglang:d3p1t1` and the example-local
`evaluation_rollout.backend=sglang:d1p1t1` reserve distinct inference workers. The
model-wide AReaL config API and global trainer are unchanged. The SAO entrypoint loads
`SaoGRPOConfig`, which includes this evaluation configuration.

With eight GPUs visible in the usual order, native local scheduling assigns actor
devices 0–3, training rollout devices 4–6, and evaluation device 7. The runtime requires
eight visible devices to prevent the scheduler's round-robin allocator from silently
wrapping onto training GPUs. The run records the actual allocation; a YAML parse alone
does not prove physical GPU placement.

Evaluation loads only the initial model or completed, synchronous HF saves. It never
follows ongoing training weight updates, and it does not delete the checkpoint after
loading. Each evaluation publishes its own completion/failure record under
`evidence/async-eval/`; a training step record is not evidence that that version's
evaluation has finished. The existing 700-question snapshot validator checks all 4
samples per question and policy-version consistency. Evaluation errors propagate to
training checks and final completion.

CPU acceptance covers typed config loading, native allocator simulation, background
dispatch, stable version binding, error propagation, and cleanup. Actual one-GPU
checkpoint loading, CUDA kernels, memory use, 4/3/1 concurrency, and throughput
improvement require a separately executed GPU run. The shared HF save still pauses
training briefly; background evaluation removes the wait for evaluating all questions,
not the checkpoint-save cost.
