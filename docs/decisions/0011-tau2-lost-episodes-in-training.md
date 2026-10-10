# Train on surviving group members and replay lost τ² tasks once per epoch

Date: 2026-10-10

Status: Accepted

Partly supersedes the training column of
[ADR 0009](0009-official-tau2-runner-semantics.md) for "any other exception" and "user
stop before any policy call".

Applies to `areal/infra/remote_inf_engine.py` (`GroupedRolloutWorkflow`),
`areal/infra/workflow_executor.py` (`BatchTaskDispatcher.active_submit_and_wait`,
`_RecoveryInputGenerator`), `areal/infra/controller/rollout_controller.py`,
`areal/experimental/openai/proxy/workflow.py`, `examples/tau2/agent.py`,
`examples/tau2/evaluation.py`, `examples/tau2/utils.py` and
`scripts/tau2/train_critic.py`.

## Context and Problem Statement

Training uses finite epochs, where every task is used exactly once per epoch, and
`fail_on_rejection=True`. Before this change, one episode that still failed after the
official retries rejected its whole GRPO group, and that aborted training. An episode
the user ended before any policy call did the same. On a multi-hour run this makes GRPO
fail outright. How should a lost member be handled without biasing the gradient and
without silently changing which tasks an epoch covers?

## Considered Options

- Keep the group if at least 2 members survive, otherwise skip the task.
- Drop the whole group and fill the batch with another task, as in DAPO dynamic
  sampling, OpenRLHF and ROLL.
- Score a lost member 0 and train on it, as in the upstream AReaL τ² example and Agent
  Lightning.
- Classify the failure, retry the member, train on survivors above a floor, and
  otherwise replay the task once at the end of the epoch under a skip budget. This is
  close to NeMo-RL, SkyRL and upstream AReaL `min_usable_group_size`.

The 2026-10-10 survey in
[audit §12.4](../research/2026-10-09-tau2-high-baseline-audit/README.md) weighs these:

- With two survivors and a binary reward, the advantage is either 0 or ±1.
- Losing six of eight members signals a broken task, and a floor of 2 hides that.
- Replacing the task skews the epoch towards easy tasks.
- Scoring infrastructure failures 0 punishes the policy for faults that are not its own.

## Decision Outcome

Chosen option: "classify, retry, train on survivors above a floor, replay once",
user-chosen on 2026-10-10.

- **Model and budget failures** are kept with reward 0. These are a context overflow and
  the training wall clock (ADR 0009 amendment).
- **Every other exception** replays that member from scratch, up to `infra_retries` (3).
  If the member still fails, the episode is lost: `reject_exhausted_episode` returns
  `None` and does not raise. A lost member is excluded from the group baseline and from
  training. An exception the agent does not mark retryable still raises.
- **A user stop before any policy call** exports no interaction. It is a lost member in
  the same way, because its reward does not depend on the policy.
- **Group floor.** `GroupedRolloutWorkflow` keeps a group with at least
  `min_usable_group_size` survivors, which defaults to $\\lceil n/2 \\rceil = 4$ for
  GRPO's 8. Reward normalisation then runs over the survivors through
  `traj_group_sizes`. A group with fewer survivors is rejected.
- **Replay at epoch end.** A rejected input is replayed once after the epoch's other
  inputs, so each task is still attempted in its own epoch. If the replay is rejected
  too, the task is skipped. More than `max_skipped_tasks_per_epoch` (2) skips abort
  training.
- **Critic training.** The critic uses one episode per task and strict per-step batches
  for recover, so it replays a rejected episode once within the same step, under the
  same budget.
- **Recover.** Inputs still waiting for their replay are saved in the recovery buffer
  and replay after a recover.
- **Metrics.** `tau2_batch/real_episodes` counts the surviving trajectories.
  `tau2_batch/epoch_deferred_tasks` and `tau2_batch/epoch_skipped_tasks` report replays
  and skips.

`fail_on_rejection` without a skip budget stays fatal, so other callers keep the old
behaviour.

### Consequences

- Good, because a provider outage on one episode no longer ends a GRPO run, and an epoch
  still visits every task.
- Good, because a task that keeps failing becomes visible as a skip. It is neither
  absorbed nor replaced.
- Bad, because failures that grow with conversation length still bias towards shorter
  conversations, both among survivors and among replays. Per-task failure rates and
  their correlation with turn count are the evidence for or against that. They are not
  yet reported per task.
- Bad, because a replayed task's gradient arrives later in the epoch than its dataloader
  position.
- Neutral: the floor of 4 and the budget of 2 are judgement calls. No surveyed framework
  uses these exact values.
