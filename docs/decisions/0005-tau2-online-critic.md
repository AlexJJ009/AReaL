# Train the τ² initial critic online against a fixed policy

Date: 2026-10-09

Status: Accepted

Applies to `scripts/tau2/train_critic.py`, `scripts/tau2/train_critic.sh`,
`scripts/tau2/run.sh data critic`, `examples/tau2/config_critic_production.yaml`, and
the SAO critic input `TAU2_CRITIC_PATH` in `examples/tau2/config_sao.yaml`.

## Context and Problem Statement

SAO starts from a pretrained scalar critic. The τ² policy is a different model from the
earlier math work (see [ADR 0002](0002-tau2-policy-generation-protocol.md)), so a new
critic, critic₀, has to be fitted to the initial τ² policy before SAO consumes it. How
is critic₀ trained, what does "two epochs" mean, and what does finishing training prove?

## Considered Options

- Collect episodes once from the fixed policy, then fit the critic offline over the
  stored episodes for several epochs (the first implementation).
- Online: each epoch walks the same task order batch by batch, samples fresh episodes
  from the fixed policy for each batch, and updates the critic once per batch.

## Decision Outcome

Chosen option: "online, fresh episodes per batch in every epoch", which is the user's
explicit definition of two epochs: the second epoch uses the same order but samples
again. The collect-once offline replay path is retired, and stored episode dumps are not
inputs to the current loop.

Recipe:

- The actor is the fixed initial policy (`Qwen/Qwen3.5-4B`, non-thinking). It is never
  updated (actor LR 0) and has no optimizer state saved. The critic is the same backbone
  with a new scalar head.
- Target: the episode's terminal official 0/1 return for every valid pre-action state,
  with gamma 1 and bootstrap 0 at budget ends. The full context is the input; user/tool
  observations and padding are excluded from the loss.
- Loss and optimizer: plain MSE, `token_mean`, one minibatch, no value clipping; Adam LR
  5e-6, constant, no warmup, gradient clipping 1. All `self_attn` and `linear_attn`
  parameters are frozen (`freeze_critic_attention: true`); MLP, embeddings, norms and
  head train.
- Schedule: one episode per training task per epoch, batch 16 in fixed task order
  (`shuffle: false`), one critic update per complete batch, two epochs, tail batch kept
  with true sample weights. This is independent of SAO's two-critic-one-actor loop. With
  the Airline+Retail pool ([ADR 0003](0003-tau2-data-and-evaluation.md)), formal mode is
  104 tasks → 208 training episodes and 14 updates; tune mode is 83 train / 21 dev.
- Validation: trajectories from the fixed policy are collected once and reused at step
  0, every 5 updates and the final update. Formal mode uses official test and tune mode
  uses dev. Validation runs no optimizer step, and test never selects a checkpoint or
  tunes anything.
- Metrics: MSE, explained variance and gradient norm, overall and per domain. Training
  and validation gradients are reported separately. EV is undefined when the targets are
  constant.
- Recovery saves critic weights, optimizer, scheduler, RNG, update count, task cursor,
  unconsumed raw inputs and validation-set identity. After resume, unconsumed tasks are
  sampled again; old generated episodes are never reused and committed updates are never
  replayed.
- Layout: 4 training GPUs with actor and critic colocated and offloaded by phase, plus 4
  rollout GPUs. Validation reuses the training critic. No GPU is dedicated to
  evaluation.

"Training completed", "numerically healthy" and "useful predictions" are reported
separately. No hard MSE or EV threshold was agreed, and the pipeline does not add epochs
on its own when quality is low. Before SAO consumes a critic, it must be explicitly
bound to a named final checkpoint of a completed non-thinking critic₀ run trained on the
same domain mix, with its head checked. A short tune probe or a critic trained under the
thinking template is not acceptable.

### Consequences

- Good, because each epoch's targets come from fresh interaction, so the second epoch
  cannot memorise the first epoch's trajectories.
- Good, because the critic sees the same task mix, template and termination protocol SAO
  will train under.
- Bad, because every epoch pays full environment and user-simulator cost, and the
  periodic validation backward for per-domain gradient norms can cost more than the
  training updates.
- Bad, because the small task pool gives few updates. Critics produced so far reached
  negative explained variance; finishing the schedule does not qualify a critic.
- Neutral: changing the domains, backbone, template or reward protocol requires
  retraining critic₀.
