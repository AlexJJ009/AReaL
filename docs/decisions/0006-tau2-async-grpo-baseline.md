# Keep a critic-free asynchronous GRPO recipe for τ²-Bench

Date: 2026-10-09

Status: Accepted

Applies to `examples/tau2/config_grpo.yaml`, `examples/tau2/train.py`,
`scripts/tau2/train_mixed.sh`, `scripts/tau2/train_{airline,retail,telecom}.sh` and
`scripts/tau2/run.sh policy <domain> grpo`.

## Context and Problem Statement

SAO results on τ²-Bench need a reference trained from the same initial policy under the
same environment, template, budgets and evaluation. That reference must not inherit
SAO's critic, so a weak critic₀ cannot affect it. What is the reference recipe, and how
does it relate to the critic and SAO jobs?

## Considered Options

The sources record no alternative to GRPO as the τ² reference. The open choices were
within the recipe, such as dynamic success filtering and the update recipe inherited
from the math GRPO line.

## Decision Outcome

Chosen option: asynchronous GRPO with no critic, reference model, teacher or KL. It
retrains from the initial policy and is independent of critic₀. The two jobs share the
8-GPU queue only for resources and have no success dependency.

Recipe, enforced by `examples/tau2/train.py`:

- 8 prompts × 8 complete episodes per update (64 episodes). A tail batch keeps whole
  groups; groups never merge across tasks or epochs. Training shuffle is
  seed-controlled.
- Group reward normalization (mean and std over the 8 samples), no extra advantage
  normalization. Dynamic success filtering is off, so all-success groups are kept.
- Decoupled asynchronous loss with recomputed logprobs and token-level ratio rejection
  above 5
  (`rejection_sampling: {level: token, action: mask, metric: ratio, upper: 5.0}`). PPO
  clip 0.2 with upper clip 0.28 is restored from the math GRPO recipe.
- One PPO minibatch, `token_mean`. Adam LR 1e-6, constant, no warmup, weight decay 0.1,
  beta2 0.98. `max_head_offpolicyness: 2`.
- Layout: 4 training GPUs, 3 rollout GPUs (`max_concurrent_rollouts: 128`, queue 256,
  SGLang `max_running_requests: 64`, `mem_fraction_static: 0.75`) and 1 dedicated
  evaluation GPU. Training workers set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to limit fragmentation from
  variable-length 32K episodes; this requires TMS offload to stay disabled.
- Save, recovery and evaluation every 5 updates and at the final update, with an initial
  evaluation that consumes the step-0 trigger. Recovery restores the main rollout and
  lazily started proxy versions together and refuses to resume without the saved
  unconsumed raw inputs.

GRPO and SAO share the in-training validation configuration
([ADR 0003](0003-tau2-data-and-evaluation.md)). SAO inherits `config_grpo.yaml` for
everything except its own algorithm settings
([ADR 0004](0004-sao-implementation-semantics.md)).

### Consequences

- Good, because the comparison isolates the algorithm: the same initial policy, harness,
  template, budgets and evaluator, with no critic dependency.
- Good, because exact recipe checks at startup stop a silently changed loss,
  normalization or rejection rule.
- Bad, because groups with identical rewards give no advantage. On the saturated pool
  most groups gave no signal, which is why Telecom was removed.
- Bad, because `max_head_offpolicyness: 2` bounds admission staleness, not measured
  per-token lag, and its throughput effect on long agentic episodes is unmeasured.
- Neutral: a fair "SAO beats GRPO" claim still needs equal episode budgets, per-domain
  exposure, seeds and a frozen evaluation matrix agreed in advance.
