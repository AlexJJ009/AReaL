# Fix τ²-Bench data, domains and evaluation protocol

Date: 2026-10-09

Status: Accepted

Applies to `examples/tau2/train.py`, `examples/tau2/utils.py`,
`examples/tau2/config_*.yaml`, `scripts/tau2/evaluate.py`,
`scripts/tau2/upload_metrics.py` and `scripts/tau2/train_critic.py`.

## Context and Problem Statement

τ² training and evaluation need a fixed task pool, a held-out set that cannot leak into
model selection, and an evaluation protocol whose noise is small enough to compare
checkpoints. The official data has a few known reference-action errors, and one domain
turned out to be saturated by the initial policy. Which tasks do we train and score on,
how do we treat data errors, and how do we evaluate?

## Considered Options

Data errata:

- Patch the affected tasks (a local fix of Retail train task 105 was proposed).
- Adopt community errata (upstream PR 571 or a replacement dataset).
- Keep the pinned official data unchanged and record known issues.

Domains:

- Airline, Retail and Telecom (the original mix).
- Airline and Retail only.

Evaluation sampling:

- Temperature 1 with 8 samples per task (the original protocol).
- Temperature 0 with 4 samples per task.

## Decision Outcome

Data: chosen option "keep the pinned official data unchanged", user-chosen. The user
judged that patching one or two tasks was not worth it, and that tasks with wrong
references may simply be untrainable. No task is removed, no reward is changed and the
test denominator is not altered. Known issues are tracked so later analysis can report
affected tasks separately:

| Domain | Task    | Split        | Issue                                                                                                        |
| ------ | ------- | ------------ | ------------------------------------------------------------------------------------------------------------ |
| Retail | 2, 3, 4 | train        | Read-only gold action references missing product 6086499569                                                  |
| Retail | 105     | train        | Gold exchange fails the gift-card balance check ($17 \< $21.10) while the NL assertion requires the exchange |
| Retail | 64      | test         | Gold actions exchange a pending order before a valid modification                                            |
| Retail | 38      | not recorded | Gold cancellation reason fixed to "no longer needed"; the simulated user often says "ordered by mistake"     |

A known reference error does not mean the whole task is impossible. Any diagnostic that
excludes these tasks must be labelled and reported alongside full official metrics.

Splits: the official train split trains; the official test split only records results.
Test never drives fitting, tuning, early stopping or checkpoint selection. Task identity
is `(domain, task_id)`, and train/test are disjoint. Tuning mode
(`experiment_mode=tune`) uses a seed-42, domain-stratified train/dev split carved from
official train. Gold actions and the user's private goals never reach the policy. Task
rows are not duplicated to inflate the pool.

Domains: chosen option "Airline and Retail only" (`domains: [airline, retail]`), decided
on 2026-10-09 after an audit of the 2026-09-27 runs. The initial policy already solved
Telecom on every episode (320/320 before training, 160 of 166 exported GRPO groups
all-correct), so Telecom gave no GRPO group contrast while taking about 42% of the
rollout budget. Formal mode therefore trains on 104 official train tasks (30 Airline, 74
Retail) and evaluates 60 official test tasks (20 Airline, 40 Retail). An explicit
`domains=[...]` override still accepts Telecom for GRPO; the SAO recipe has no Telecom
budget.

Evaluation: chosen option "temperature 0 with 4 samples per task" for in-training
validation (`eval_gconfig`) and `scripts/tau2/evaluate.py` (`--temperature 0`,
`--repeats 4`). Training rollouts stay at temperature 1. The same audit found two step-0
evaluations of the same initial model differing by 4.6 percentage points at temperature
1, while every checkpoint's change against its own step 0 stayed inside that noise band.
Evaluation concurrency is 8 task groups × 4 repeats, keeping 32 episodes in flight. All
other evaluation settings (template, budgets, simulator, tools, data, scorer) match
training validation.

`evaluate.py --task-set test` (default) scores the held-out test split.
`--task-set base` scores the official leaderboard set (train plus test: 50 Airline, 114
Retail) after checking the task IDs against the official `base` list, and reports
`by_domain_split` so the held-out part stays visible.

Reporting: the primary metric is the mean success over all trials, overall and per
domain. pass^k or pass@k may be added only under their own names. Initial and final
models are compared paired by task, with uncertainty aggregated by task: 60 tasks × 4
repeats are not 240 independent tasks. Episodes that end as `infrastructure_error` after
the official retries are excluded from the denominator as the official runner does, and
their count and error types are always reported
([ADR 0009](0009-official-tau2-runner-semantics.md)). Official `avg_reward` and `pass^k`
are reported alongside the mean success. This protocol is not a full reproduction of the
official leaderboard.

### Consequences

- Good, because scores remain comparable with the official benchmark, and data issues
  are visible instead of hidden behind a patched denominator.
- Good, because removing a saturated domain spends the rollout budget on tasks with
  learning signal, and greedy evaluation lowers the noise between checkpoints.
- Bad, because results from the earlier protocol (three domains, temperature 1, 8
  repeats) cannot be compared with new results; they are kept under their original
  protocol label.
- Bad, because temperature-0 evaluation measures the policy's most likely behaviour, not
  the stochastic behaviour it is trained on.
- Bad, because the trained policy is no longer measured on Telecom. Reintroducing
  Telecom changes budgets, critic training data and baselines together.
