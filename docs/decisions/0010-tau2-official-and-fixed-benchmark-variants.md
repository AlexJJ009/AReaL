# Train and evaluate τ² on an official and a fixed benchmark variant

Date: 2026-10-10

Status: Accepted

Applies to `examples/tau2/benchmark_variant.py`, `examples/tau2/benchmark_fixes.json`,
`examples/tau2/agent.py`, `examples/tau2/train.py`, `scripts/tau2/evaluate.py`,
`scripts/tau2/train_critic.py`, `scripts/tau2/build_fixed_data.py` and
`scripts/tau2/check_gold_replay.py`.

## Context and Problem Statement

The step-0 failure audit on 2026-10-10
([audit §10](../research/2026-10-09-tau2-high-baseline-audit/README.md)) found two
groups of problems in tau2-bench `b7ea907`. Both groups also affect the training reward,
because training uses the same evaluator.

The first is a scoring hole. Airline tasks are scored on DB + COMMUNICATE. On refusal
tasks with no gold write, transferring to a human leaves the DB unchanged and scores 1.
This accounts for about 30 pp of the airline score.

The second is a set of task and environment bugs:

- retail:105 cannot be completed with the gift card it requires;
- retail:64 contains a gold step that cannot be executed;
- `modify_pending_order_items` prices every item from the last variant in the loop.

Upstream has merged none of the fixes; the corresponding PRs are open. The official
numbers must stay comparable with the leaderboard. How do we train and evaluate on a
corrected benchmark without losing that comparison?

## Considered Options

- Keep only the official benchmark.
- Replace the official benchmark with a fixed one.
- Keep both as named variants and select one per run.

## Decision Outcome

Chosen option: "keep both as named variants", user-chosen on 2026-10-10.
`econfig.benchmark_variant` selects the variant. Every config states it explicitly, and
the default is `official`. The variant is recorded in row provenance and in the
evaluation manifest.

| Variant    | Data                                                                           | Environment                     | Scoring                                                                          |
| ---------- | ------------------------------------------------------------------------------ | ------------------------------- | -------------------------------------------------------------------------------- |
| `official` | tau2-bench `b7ea907`                                                           | unchanged                       | official evaluator                                                               |
| `fixed`    | PR #411 reward bases (airline 44 tasks, retail 105); PR #571 retail 64 and 105 | PR #322 per-item variant prices | official evaluator, then 0 when the agent transfers and no gold action transfers |

The fixes are taken from the PR head commits recorded in `benchmark_fixes.json`.
`build_fixed_data.py` writes the two fixed `tasks.json` files; every other file is a
symlink to the official data. tau2 reads its data directory once, at import. Each
variant therefore pins the SHA256 of its `tasks.json` files, and evaluation, training
and critic startup refuse a `TAU2_DATA_DIR` that does not match.

PR #580 is not applied as code. It refuses a DB match when a gold write raises. After
#571, `check_gold_replay.py fixed` reports no raising gold write in Airline or Retail,
so #580 would change no reward. The script stays as the check that this still holds.

The fixes are not split-specific. In the official train split they change:

- the reward basis of 25/30 airline and 67/74 retail tasks;
- what is scored for 17 airline train tasks, which now score NL assertions;
- retail:105, replaced by #571;
- the gold DB hash of retail 20, 21 and 37, through #322.

The transfer rule applies wherever a gold action is not a transfer. Only airline 13 and
retail 10, 12, 26 and 50 transfer in gold.

### Consequences

- Good, because the fixed variant stops rewarding a transfer that dodges a refusal, in
  training as well as in evaluation.
- Good, because official numbers stay available and leaderboard-comparable. Each policy
  can be trained on one variant and evaluated on both.
- Bad, because fixed numbers are not comparable with the leaderboard or with published
  results. They are always reported next to the official ones and never alone.
- Bad, because #411 moves 26 airline tasks onto NL assertions. The NL judge is the
  user-simulator model (ADR 0009) and has reported upstream defects (#554, #436). Judge
  noise therefore enters the airline training reward.
- Bad, because the #322 patch is process-global. One process cannot run both variants.
- Neutral: airline:7, retail:38, retail:60, retail:110 and the order-sensitive airline
  `payment_history` hash are not fixed anywhere upstream and stay as they are.
