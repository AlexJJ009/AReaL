# Critic-only warmup for PPO

`PPOConfig.num_critic_only_steps` skips actor optimizer and learning-rate scheduler
steps during the first N training rounds. The critic trains normally on each
round's rollouts. This follows the ordinary scalar-PPO warmup design in
[miles-values at aa2071f8](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/train_async.py#L46),
not its SAO, adaptive-lambda, DIS, or classification-value branches. This is an
AReaL-native adaptation, not a direct import of the Megatron/Ray implementation.

```yaml
num_critic_only_steps: 10
actor:
  gae_lambda: 0.95
  critic_gae_lambda: 1.0
  adv_norm: null
```

With N=10, zero-based rounds 0 through 9 train only the critic. Round 10 performs
the first actor update and publishes policy version 1. Warmup rounds count toward
the total training rounds and consume normal batches; they are not additional
epochs. For 17,157 prompts, batch size 16 and an un-dropped tail, one epoch has
1,073 rounds: 1,073 critic updates and 1,063 actor updates when each role uses one
optimizer minibatch. Eight responses per prompt means 128 trajectories per full
round and 1,280 trajectories used in ten warmup rounds.

## State and targets

The actor's parameters, optimizer moments and scheduler remain untouched during
warmup. A five-update actor LR warmup starts at the first actual actor update;
it is independent of ten critic-only rounds. Critic scheduler steps continue.
Scheduler horizons still use the experiment's total-round FinetuneSpec; the
example uses constant LR after its explicit five-update warmup.

`actor.critic_gae_lambda` optionally computes a separate return target for the
critic using the same rewards, masks, discount and bootstrap policy. Actor
advantages still use `actor.gae_lambda`. The default `None` preserves the old
shared-target behavior. With gamma=1, critic lambda=1, terminal bootstrap=0 and
only an outcome reward, valid critic targets equal the trajectory's Monte Carlo
outcome return. This does not automatically disable truncation bootstrapping:
the rollout workflow must supply the intended `bootstrap_mask` contract.

Advantage normalization is independently optional. `actor.adv_norm: null`
disables it and never changes critic targets. Separate target lambda requires a
critic; setting it on a critic-free algorithm is invalid.

During warmup, the real actor/rollout policy version does not advance and no
updated actor weights are published. The staleness manager credits consumed
prompt batches separately so fixed-policy rollout collection can continue
without false version labels. Accepted-rollout statistics are not reset.

Recovery checkpoints save both actor and critic, including optimizer state, and
`trainer_state.json` binds the warmup length and actual policy version. Resume
loads both roles, validates the warmup length/version relation and rebuilds
rollout capacity from the real policy version. Legacy checkpoints without this
file imply zero critic-only rounds. Changing warmup length on resume is rejected.
Actor evaluation is skipped during critic-only rounds because the actor is
unchanged. Checkpointing still runs, including the unchanged actor, so a warmup
interruption is recoverable.

## Scope and use

Warmup currently requires fixed prompt batches (`dynamic_bs: false`).
The supported layout is separate actor and rollout GPU groups, with a critic
optionally colocated with the actor. Actor-rollout colocation/AWEX is rejected
when warmup is requested: those paths require a separate no-update weight/KV
restore protocol. Defaults (`num_critic_only_steps: 0`,
`critic_gae_lambda: null`) retain normal PPO/GRPO update behavior.

`examples/math/gsm8k_ppo_critic_warmup.yaml` illustrates native-trainer settings.
It inherits GSM8K's model/data/backend settings and AReaL async correction. It is
not the approved 4B DAPO experiment or an exact reproduction of miles runtime
semantics. It can be consumed by the normal `examples/math/gsm8k_rl.py` entry;
the old `examples/math/sao_ppo.py` experiment intentionally validates the previous
GSM8K-aligned run and cannot be used with this recipe without updating its
experiment-specific contract/audit.

No new GPU experiment is launched by these changes. CPU tests cover the real
trainer loop with small optimizer-backed engines, frozen actor state, the phase
boundary, separate return targets, rollout capacity and recovery metadata/load.
GPU weight publication, FSDP checkpoint resume and model quality still require a
bounded hardware qualification before a formal experiment. Ten critic-only
rounds are a recipe parameter, not proof of value-model quality.
