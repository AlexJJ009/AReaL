# PPO v3: official GSM8K optimizer and asynchronous objective

User authorization (2026-09-21): align the PPO configuration with GSM8K, use five warmup
updates, include the official reward transform, pass acceptance, and launch the
replacement experiment. This amendment supersedes the v2 optimizer/loss/reward choices;
it does not alter the independent critic-pretraining plan.

- Load the same Qwen3.5-4B-Base snapshot afresh, never resume the deleted v2 run.
- Keep the audited DAPO 17,157 prompts, one epoch, 128 prompts × 4 trajectories, 8,192
  response tokens, five evaluation sets × 4 samples and four-plus-four GPUs.
- Inherit the official actor optimizer (LR 1.7e-5, AdamW decay 0.017), actor clip 0.4,
  reward `(R - 0.5) * 10`, decoupled objective, proximal logprob recomputation, and
  token rejection for proximal/behavior ratio greater than 5.
- Critic uses the same optimizer configuration; value clip remains 0.5.
- Explicit exception: fixed `warmup_steps=5` overrides the inherited ratio 0.001 for
  both models. Both models train during warmup; this is not critic pretraining.
- Explicit exception: this finite-budget math workflow disables future bootstrap at 8K.
  Preserve the actual terminated/truncated flags and emit a separate false bootstrap
  mask. Other workflows retain their existing bootstrap defaults.
- Raw scorer results and accuracy remain 0/1. With gamma=lambda=1 and no KL, every valid
  response-token return must equal the transformed outcome, -5 or +5.

Acceptance before launch: CPU tests for composed configuration, warmup resolution,
reward transform, finite-horizon GAE and legacy compatibility; complete pre-commit;
independent review; native eight-GPU small-batch test with the same model, 8K limit,
N=4, loss and optimizer settings, checking actual return tensors, successful joint
updates, and published-policy parity. Small-batch preflight is not a full-epoch quality
result. Formal first five updates retain the existing per-step supervision and GPU
overlap checks. Save/evaluate at updates 20/40/.../135 and at epoch end.

Freeze candidate SHA, source/config/command hashes, dataset identity, authorization,
preflight evidence and remaining task-specific differences before formal launch. Keep
evidence and logs outside the source checkout. No extra epoch or seed sweep is
authorized. Launch success does not establish validation improvement.
