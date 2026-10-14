# SAO hybrid critic attention freezing

The optional `critic.freeze_critic_attention` flag defaults to `false`. It applies only
to an FSDP critic without LoRA. The MVP requires both `self_attn` and `linear_attn`
modules, as in the Transformers Qwen3.5 hybrid backbone; it is not a generic
architecture-independent freeze policy.

With the usual SAO environment variables configured, inspect the enabled recipe:

```bash
bash scripts/sao/run_sao.sh --check-config critic.freeze_critic_attention=true
```

For a separately authorized training run, use the same override without
`--check-config`. Set a new run directory and trial name. Omitting the override, or
setting it to `false`, preserves full-parameter critic training.

## Scope

Freeze every parameter under both attention modules, including internal norms,
convolution, and gates. Keep MLP, embedding, decoder-level norms, and scalar head
trainable. Do not detach attention outputs or wrap attention in `no_grad`: upstream
trainable layers still need gradients through these operations.

The rule is applied before FSDP2 wrapping. Optimizers contain only trainable parameters.
Checkpoint engine sidecars record the rule and matched parameter names/counts. Optimizer
resume rejects a changed rule before loading weights; legacy sidecars without a rule are
accepted only for the disabled setting. Weight-only initialization is allowed across
rules and starts a new experiment.

The SAO paper freezes attention and optimizes MoE projections. Qwen3.5-4B is dense
hybrid: this configuration is a documented adaptation, not an exact reproduction of the
paper's parameter topology.

## Experiment design

For the first ablation, reuse the original initial actor and pretrained critic. Do not
retrain the critic first: that would change initialization as well as freezing. Preserve
rewards, horizon, optimizer, batch, seeds, update order, and evaluation settings.
Disclose step-zero evaluation and compare equal budgets.

If the actor or task changes, first evaluate critic calibration on trajectories from the
new policy using held-out data (MSE, explained variance, bias, and
truncated/nontruncated subsets). The existing critic's historical scores do not
establish compatibility with a new policy. Retraining with frozen attention is a
separate ablation, not a prerequisite for freezing during online SAO.

## Qualification boundary

CPU tests cover update invariance, gradient propagation through frozen modules,
configuration rejection, and recovery-policy checks. Before formal training, qualify the
actual multi-GPU hybrid model: frozen shards unchanged after two critic updates,
MLP/head updated, finite gradients, actor unaffected, unchanged critic-critic-actor
ordering, and fresh-process DCP recovery with the same rule. Measure memory rather than
assuming savings proportional to frozen parameters. Keep the one-real-sequence
microbatch limit. Ensure the final step is saved.
