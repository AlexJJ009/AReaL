# Train and evaluate τ² on an official and a fixed benchmark variant

Date: 2026-10-10

Status: Accepted

Applies to `examples/tau2/benchmark_variant.py`, `examples/tau2/benchmark_fixes.json`,
`examples/tau2/agent.py`, `examples/tau2/train.py`, `scripts/tau2/evaluate.py`,
`scripts/tau2/train_critic.py`, `scripts/tau2/build_fixed_data.py` and
`scripts/tau2/check_gold_replay.py`.

## Context and Problem Statement

The step-0 failure audits on 2026-10-10
([audit §10–§11](../research/2026-10-09-tau2-high-baseline-audit/README.md)) found
problems in tau2-bench `b7ea907` that also affect the training reward, because training
uses the same evaluator:

- **A scoring hole.** On Airline refusal tasks with no gold write, transferring to a
  human leaves the database unchanged and scores 1. This accounts for about 30 pp of the
  Airline score.
- **Task and environment bugs with open upstream fixes:**
  - retail:105 cannot be completed with the gift card it requires;
  - retail:64 contains a gold step that cannot be executed;
  - `modify_pending_order_items` prices every item from the last variant in the loop.
- **Retail task defects that no upstream PR fixes:**
  - instructions leave open a payment method or cancel reason that gold fixes;
  - three tasks score 1 for doing nothing;
  - a gold transfer is copied from another task;
  - NL assertions restate writes that only the database check can see.
- **Leakage.** Five Retail train tasks have the same user and the same gold writes as a
  test task.

Upstream has merged none of the fixes. The official numbers must stay comparable with
the leaderboard. How do we train and evaluate on a corrected benchmark without losing
that comparison?

## Considered Options

- Keep only the official benchmark.
- Replace the official benchmark with a fixed one.
- Keep both as named variants and select one per run.

## Decision Outcome

Chosen option: "keep both as named variants", user-chosen on 2026-10-10.
`econfig.benchmark_variant` selects the variant. Every config states it explicitly, and
the default is `official`. Training uses `fixed`. The variant is recorded in row
provenance and in the evaluation manifest.

| Variant    | Data                                                                                                           | Environment                     | Scoring                               | Training pool                                             |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------- | --------------------------------------------------------- |
| `official` | tau2-bench `b7ea907`                                                                                           | unchanged                       | official evaluator                    | official train split                                      |
| `fixed`    | PR #411 reward bases (Airline 44 tasks, Retail 105); PR #571 Retail 64 and 105; our patches on 20 Retail tasks | PR #322 per-item variant prices | official, then the two transfer rules | official train split without Retail 37, 41, 67, 72 and 96 |

**Transfer rules** (fixed only), applied after the official evaluator:

- An agent transfer to a human scores 0 when no gold action transfers. In Retail, a
  transfer on a task with gold writes keeps its score: there it follows the finished
  task, at the user's request for something the agent cannot do. In Airline it dodges a
  refusal.
- An episode without an agent transfer scores 0 when a gold action transfers: Airline
  13, Retail 10, 12, 25 and 50.

**Data.** `benchmark_fixes.json` records the upstream PR head commits, the reason for
each of our patches, and the Retail train tasks left out of training.
`build_fixed_data.py` writes the two fixed `tasks.json` files; every other file is a
symlink to the official data. tau2 reads its data directory once, at import. Each
variant therefore pins the SHA256 of its `tasks.json` files, and evaluation, training
and critic startup refuse a `TAU2_DATA_DIR` that does not match. The excluded train
tasks stay in the benchmark, so `--task-set base` still scores all 164 tasks.

PR #580 is not applied as code. It refuses a database match when a gold write raises.
`check_gold_replay.py fixed` reports no raising gold write in Airline or Retail, so #580
would change no reward. The script stays as the check that this still holds.

### Consequences

- Good, because the fixed variant stops rewarding transfers that dodge a refusal,
  requires the transfers gold requires, and removes reward noise from underspecified
  Retail tasks. This holds in training as well as in evaluation.
- Good, because official numbers stay available and comparable with the leaderboard.
  Each policy can be trained on one variant and evaluated on both.
- Good, because the Retail test tasks that duplicate a train task are no longer trained
  on.
- Bad, because fixed numbers are not comparable with the leaderboard or with published
  results. They are always reported next to the official ones, never alone.
- Bad, because #411 moves 26 Airline tasks onto NL assertions. The NL judge is the
  user-simulator model (ADR 0009) and has reported upstream defects (#554, #436). Judge
  noise therefore enters the Airline training reward.
- Bad, because the transfer rule differs by domain. The Retail exemption rests on
  reading the 2026-10-10 step-0 transfers, not on a check the evaluator can make.
- Bad, because the #322 patch is process-global. One process cannot run both variants.
- Neutral: airline:7 and the order-sensitive Airline `payment_history` hash are not
  fixed anywhere and stay as they are.
