# Issue: why critic-only training on tau2 barely reaches positive explained variance

- Status: both case studies done (2026-10-14); synthesis in
  [ev-case-study.md](../critic/ev-case-study.md). Next: offline experiments (probe,
  planted signal, replay). its own report.
- Background: [critic/README.md](../critic/README.md) (all critic docs). Read it first;
  this file only adds the run index and the open questions.

## Run index

Run directories are under
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/`. Each has
`tau2-critic-fit-evidence/` (validation reports `validation-step-*.json`, per-step
`train-metrics/`, `fixed-validation-rollouts.pt`, `resolved-config.json`) and
`checkpoints/`.

| Name used in chat | pueue | Directory                                                      | Data                                              | Head              | Steps    | Status              |
| ----------------- | ----- | -------------------------------------------------------------- | ------------------------------------------------- | ----------------- | -------- | ------------------- |
| previous critic   | -     | `tau2-training-20260927-nonthinking/critic-formal-v1`          | official train 178, 3 domains                     | random, no bias   | 24       | done                |
| 479 (Qwen3.5)     | 479   | `tau2-training-20261012-critic-data/critic-qwen35-formal-v1`   | pack v2 648 then official 99                      | random, no bias   | 94       | done                |
| 480 (Instruct)    | 480   | `tau2-training-20261012-critic-data/critic-instruct-formal-v1` | same as 479                                       | random, bias 0    | 94       | done                |
| v2                | 499   | `tau2-training-20261013-critic-head/critic-head-qwen35-v2`     | same as 479, fixed order                          | zero w, bias 0.91 | 94       | done                |
| mix (first try)   | 500   | `tau2-training-20261013-critic-head/critic-head-qwen35-mix`    | mix pack 221 + official 99, reshuffled each epoch | zero w, bias 0.78 | 22 of 80 | failed (SGLang OOM) |
| mix r2            | 510   | `tau2-training-20261013-critic-head/critic-head-qwen35-mix-r2` | same as 500                                       | same as 500       | 80       | running / done      |

"v2" and "mix" are the two data choices of [data-mix.md](../critic/data-mix.md). "r2"
means the second attempt of the mix run. All runs use Qwen3.5-4B (non-thinking) as a
frozen actor at policy temperature 1 unless marked Instruct, DeepSeek-Flash as user
simulator and NL judge, critic lr 5e-6, batch 16 episodes, 0/1 terminal reward, gamma 1,
GAE lambda 1, and validation on fixed-v3 (fixed-v2 for 479/480) test: 60 tasks (Airline
20, Retail 40), one T=1 rollout each, collected once per run and reused. Each run
collects its own validation rollouts, so the validation sets differ across runs.

## What is known (see [critic/README.md](../critic/README.md))

- Problem A, level drift: fixed by zero-init w + mean-init bias
  ([value-head.md](../critic/value-head.md)); level is now stable in 499/510.
- Problem B, ranking within a domain: Airline predictions correlate with task difficulty
  (0.70 in 479), Retail does not (0.10).
- EV ceiling at the first token from task identity alone
  ([ev-ceiling.md](../critic/ev-ceiling.md)): Airline 0.61, Retail 0.20 \[0.03, 0.35\].
  More than half of Retail's reward variance is within-task (same task, different
  outcome), i.e. user-side randomness.
- 510 so far (step 50): Airline EV up to 0.137, Retail about 0. 499 final: Airline
  0.025, Retail -0.001.

## Questions for the case study

1. Is "Airline learns a bit, Retail learns nothing" explained by Retail's low ceiling
   and few failures (about 5 of 40 validation trajectories), or by something the critic
   could learn but does not?
1. What limits Airline: number of updates (80-94 steps of 16 episodes), lr, frozen
   attention, the token-level MSE objective, label noise, train/test shift (pack tasks
   are much easier than test), or validation noise (20 trajectories)? How large is the
   sampling noise of EV itself on 20/40 trajectories?
1. Token-level view: where along the trajectory does the critic's prediction change, and
   does it react to events that decide the outcome (wrong tool call, user giving wrong
   info, policy violation)?
1. How to reproduce the poor result cheaply: e.g. offline fitting on stored rollouts (no
   API), a linear probe on frozen hidden states, a larger lr / more epochs on the same
   data, a synthetic task with known value function. A cheap reproduction should let us
   tell "optimisation problem" from "no learnable signal".
1. What would make EV reliably positive, ranked by expected effect and cost.

## Constraints

- Do not launch GPU jobs, pueue tasks, or DeepSeek calls. GPU experiments go into the
  report as proposals with the exact command and expected cost; we queue them.
- Do not modify existing run directories. Write new analysis outputs only into your own
  case-study directory.
- Quote evidence with file paths; mark guesses as guesses.
