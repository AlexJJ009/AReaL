# Implement SAO as Direct DIS, dual-lambda GAE and critic-first updates

Date: 2026-10-09

Status: Accepted

Applies to `areal/trainer/ppo/` (`dis.py`, `lambda_fn.py`, `gae.py`, `trajectory.py`,
`update.py`, `value_checkpoint.py`, `actor.py`), `areal/trainer/rl_trainer.py`,
`areal/models/transformers/scalar_value.py`, the critic options in
`areal/api/cli_args.py`, `examples/math/gsm8k_sao.*` and
`examples/tau2/config_sao.yaml`.

## Context and Problem Statement

SAO ([arXiv 2607.07508](https://arxiv.org/html/2607.07508v1)) trains with one rollout
per prompt, a pretrained scalar critic and a direct importance-sampling loss. The paper
leaves several details open: stop-gradient placement, reduction, update order, gamma and
normalization. AReaL's default PPO loss is not SAO. Which implementation source do we
trust, and which semantics does the AReaL port fix for those open points?

## Considered Options

Implementation source:

- Apply the community fooSynaptic SAO patch.
- Port from
  [yyht/miles-values@aa2071f](https://github.com/yyht/miles-values/tree/aa2071f834ff9666a9bee124f7ed62e260b7488c).
- Port five mechanisms from miles-values as a design reference, correct each against the
  paper, and implement Direct DIS from the paper.

Direct DIS gradient:

- Literal `-f(r) A log π` with the ratio inside autograd.
- Surrogate `-r A`.
- Score-function form with stop-gradient on the importance weight.

## Decision Outcome

Implementation source: chosen option "miles-values as design reference, corrected
against the paper", user-chosen. The community patch is not used and no download/apply
path for it exists. miles-values is not a complete SAO: its actor still uses PPO clipped
surrogate with TIS/IcePop corrections, it has no Direct DIS, its adaptive lambda
defaults to `alpha=1` with a short-sequence `lambda=1` special case, and its colocated
path hands the actor values saved before the critic update. The port does not import
Miles/Megatron/Ray. It reuses AReaL's PPOTrainer, FSDP engine, SGLang weight sync and
checkpointing.

Direct DIS gradient: chosen option "score-function form with stop-gradient", confirmed
by the user after a numerical check. The surrogate form has the same first-order
gradient but a different loss value. The literal form adds a `-rA log π ∇log π` term
that can flip the gradient sign. The paper does not specify this; it is a project
convention, not a claim about the authors' code. Minimised loss:

```text
r_t  = exp(log π_θ(a_t|s_t) - log π_behavior,t(a_t|s_t))
m_t  = 1[(1 - ε_low) < r_t < (1 + ε_high)]     ε_low = 0.3, ε_high = 5.0  →  (0.7, 6.0), strict
L_π  = - Σ_t M_t · sg(m_t r_t A_t) · log π_θ(a_t|s_t) / Σ_t M_t
```

- Behavior logprobs are the per-token values returned at generation time. A trajectory
  may span several policy versions. No proximal/old-policy forward is used for DIS
  (`recompute_logprob: false`, `use_decoupled_loss: false`).
- The denominator `Σ M_t` counts all valid action tokens before DIS rejection.
- Rejected tokens get zero policy gradient regardless of the sign of `A`. A batch where
  all tokens are rejected skips the optimizer step, including Adam moments and decay,
  and is reported as no actor learning. Non-finite ratios fail loudly; they are never
  clamped or replaced by 1.
- DIS is a separate loss path. It is not combined with PPO clipping, decoupled PPO,
  SAPO/CISPO, KL or teacher terms.

Advantage and return semantics:

- Dual lambda: the actor uses `λ_π = 1 - 1/(α L)` with `α = 1.5`
  (`sao_length_adaptive_gae`), and the critic target uses `critic_gae_lambda: 1.0`.
  Critic targets are computed separately from actor advantages, never back-derived from
  normalised advantages. `critic_gae_lambda: None` keeps upstream behaviour for other
  recipes.
- `L` counts generated action tokens including EOS, excluding prompt, observations and
  padding. `L = 1` uses the same formula, and empty trajectories are rejected.
- Skip-observation GAE: the recursion runs over the action-token subsequence of one
  episode. Tool/user observation tokens consume no discount step and enter no loss, but
  later values still condition on them. Rewards stay aligned to action positions, and
  nothing propagates across episodes or packed samples.
- gamma 1, raw rewards, no reward or advantage normalization, plain token-mean critic
  MSE without value clipping, constant LR without warmup. Actor LR is 1e-6 and critic LR
  5e-6, the paper's math values. These were user-confirmed when the paper did not settle
  them.

Update order (user-confirmed, K = 2 from the paper; `critic_updates_before_actor: 2`):

1. Freeze behavior logprobs and batch identity.
1. Build lambda-1 critic targets from the pre-update critic and detach them; the target
   does not drift between the two fits.
1. Run two full critic optimizer updates on the same batch. Microbatches are not counted
   as updates.
1. Recompute values with the updated critic and compute actor advantages with `λ_π`.
1. Run one actor update, then publish policy weights.

Attempted, successful and skipped updates are counted from engine reports, not inferred
from call counts. The critic LR scheduler budget scales with K so schedules do not
exhaust halfway. `examples/tau2/train.py` refuses an SAO config that deviates from these
values.

Critic loading: the critic is a scalar-head model loaded from its own path (actor and
critic paths may differ). A sealed value artifact (`value_manifest.json`, checked
against `value_contract`) must contain both backbone and head; a missing head is never
silently re-initialised, and identity, tokenizer or target-protocol mismatches are
rejected. A Base backbone with a fresh head is allowed only by explicit opt-in (math:
`--allow-base-critic`) and is labelled an unpretrained cold start. Pretraining optimizer
state is not carried into online SAO. The critic keeps training online.

Attention freezing: the paper freezes attention and trains MoE projections. The first
math port deliberately left this out (user decision). It was later added as opt-in
`critic.freeze_critic_attention`, which freezes every parameter under both `self_attn`
and `linear_attn` of the Qwen3.5 hybrid critic before FSDP wrapping. It is enabled in
the τ² SAO and critic recipes at the user's request and disabled in `gsm8k_sao.yaml`.
This is a dense-model adaptation, not the paper's parameter topology; see
[docs/research/sao-critic-attention-freeze.md](../research/sao-critic-attention-freeze.md).

### Consequences

- Good, because each open paper detail has one written, tested choice, and the τ²
  entrypoint rejects silent drift from it.
- Good, because Direct DIS needs no extra old-policy forward, and asynchronous staleness
  is corrected token by token from the real behavior logprobs.
- Bad, because the stop-gradient convention, the token-sum reduction and the update
  order are project interpretations. Claims of reproducing the paper must say so, and
  the scope excludes the paper's MoE, 128K tool-integrated reasoning setting and scores.
- Bad, because recovery must keep unconsumed raw inputs and regenerate them. In-flight
  trajectories cannot be resumed with stale behavior logprobs.
- Neutral: skip-observation semantics are verified on synthetic multi-segment
  trajectories. Their benefit on real τ² episodes is a separate empirical question.
