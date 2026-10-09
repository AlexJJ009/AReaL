# Carry explicit episode termination and bootstrap flags into GAE

Date: 2026-10-09

Status: Accepted

Applies to `areal/trainer/ppo/actor.py` (advantage/return construction),
`areal/trainer/ppo/trajectory.py`, `areal/trainer/ppo/gae.py`,
`areal/trainer/ppo/validation.py`, and the producers `areal/workflow/sao_math.py`,
`examples/math/sao_ppo.py` and `examples/tau2/agent.py`.

## Context and Problem Statement

Upstream AReaL infers whether a sequence was cut off from the padded batch width: a row
that fills the batch is treated as not having reached EOS, and its last value is
bootstrapped. The first formal async PPO run on DAPO math was discarded after 33 updates
because of this. The GAE return of the same finished trajectory changed with the length
of other sequences in its batch. The project also needs a way to say that a token or
step budget is the true end of the task, so the return there is the observed reward with
no value bootstrap, while still recording that generation stopped on length. How should
episode boundaries and bootstrap be represented?

## Considered Options

- Keep upstream inference from padding width or total token length.
- Treat the response budget (8K tokens for math) as a time-limit truncation and
  bootstrap the true final state value `V(s_T)`.
- Producers emit explicit per-row `terminated`/`truncated` flags from the real stop
  reason, plus a separate `bootstrap_mask` that decides whether a truncated row
  bootstraps.

## Decision Outcome

Chosen option: "explicit flags plus a separate bootstrap mask". The first async PPO
protocol used time-limit bootstrap at 8K. The user then made budget ends terminal for
value targets in the math PPO, critic pretraining and SAO recipes, and the τ² recipe
uses the same convention for context and step budgets. Padding-width inference is
rejected because it makes returns depend on batch composition.

Contract (enforced in `trajectory.py` and `actor.py` whenever a batch carries
`terminated` or `truncated`):

- One episode per row, right-padded. Exactly one of `terminated` and `truncated` is
  true. Every action has its preceding state inside the row. Pre-packed multi-episode
  rows are rejected; optional `episode_ids` detect accidental packing.
- `bootstrap_mask` (bool, one per row) may be true only for truncated rows. It defaults
  to `~terminated` when absent. A truncated row bootstraps from the value at its last
  real position only when its mask is true; otherwise the bootstrap value is 0.
- The outcome reward is placed at the last action token, not inferred from sequence
  length. Per-token tool rewards must be mapped to action positions.
- Math (`AuditedMathWorkflow` in `areal/workflow/sao_math.py`) keeps the real
  `truncated` flag for 8K cutoffs and emits `bootstrap_mask=false`. τ² marks
  `context_limit` episodes `truncated` with reward 0 and also exports
  `bootstrap_mask=false`.
- Return probes (`validation.py`) check, on real batches, that the number of
  bootstrapped rows never exceeds the number of cutoffs.

With gamma = lambda = 1, no KL and the mask false, every valid action-token return
equals the episode reward. Tests and native preflights use this as an independent
oracle.

### Consequences

- Good, because a trajectory's return no longer depends on padding or batch composition.
  The same row gives the same targets alone, padded, or mixed with other lengths.
- Good, because "stopped on length" (data) and "continue the value past the cutoff"
  (target protocol) are separate fields that can be checked independently.
- Bad, because every producer must supply the flags. A batch without them silently falls
  back to the upstream padding heuristic, which is acceptable only for recipes that
  never hit a cutoff.
- Bad, because a critic trained under one bootstrap convention is not valid under the
  other. Switching a budget back to time-limit semantics requires new value targets and
  a new critic.
