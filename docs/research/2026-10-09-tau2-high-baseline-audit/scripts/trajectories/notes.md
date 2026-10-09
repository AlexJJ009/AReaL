# tau2 step-0 trajectory audit (independent read, 2026-10-09)

No training/eval/LLM calls. Scripts here: `view.py` (dialogue + task spec), `muts.py`
(mutating calls per episode), `tail.py`, `tel.py`, `idx.py`/`index.tsv` (all step-0
episodes), `endscan.py`.

Sources (abbrev): G =
grpo-formal-v2/.../tau2-grpo-nonthinking-formal-v2-dedicated-eval/eval-rollout/0/, S =
sao-mixed-v1/.../tau2-sao-nonthinking-mixed-v1-dedicated-eval/eval-rollout/0/ under
/data_storage/yl_test/lgx/data-1/code/\_artifacts/AReaL/codex-sao-math/tau2-training-20260927-nonthinking/.
Ref = file:line (1-based JSONL line). Segment numbers \[k\] = view.py output index.
Tasks:
/data_storage/yl_test/lgx/data-1/code/\_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619/data/tau2/domains/

## Setup facts

- User sim = `deepseek-flash` @ api.deepseek.com, temperature 0.0, max_completion_tokens
  512 (config.yaml econfig). Official default is gpt-4.1 @ T=0. NL judge also
  deepseek-flash (agent.py:\_bind_nl_assertion_evaluator).
- User system prompt recorded in grpo-formal-v2/tau2-user-failures/255e...-0.json is
  byte-for-byte the official `simulation_guidelines.md` + `<scenario>`. Installed tau2
  1.0.1 @ b7ea907: 269/269 files match RECORD hashes (unmodified). RetryingUserSimulator
  only retries on empty/malformed output.
- Test split = 20/50 airline, 40/114 retail tasks (split_tasks.json). Airline: NL
  assertions present on all 20 but reward_basis = DB+COMMUNICATE (never scored, same as
  official). Retail: NL active on 11/40.
- Airline test: 6 no-mutation tasks avg 0.93, 14 mutating 0.74. Retail mutating 36 tasks
  0.78.

## Successes (16)

| ref          | task | class               | note                                                                                                                           |
| ------------ | ---- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| G 0.jsonl:1  | A2   | lenient             | Checks HAT006/HAT040 only, never finds the delayed flight; volunteers $50 compensation rule \[17\]. No mutation required -> 1. |
| G 1.jsonl:2  | A6   | clean (refusal)     | Refuses post-booking insurance repeatedly; user persistent per script.                                                         |
| G 5.jsonl:1  | A18  | clean               | 5 downgrades, each to original payment, total $23553 via calculate.                                                            |
| G 11.jsonl:6 | A29  | luck                | Only success of 16. User volunteers "I don't need insurance" \[26\] (not in instructions); gold needs insurance=no.            |
| S 17.jsonl:4 | A44  | lenient             | Quotes $1136 estimate, actual $1387 (NL wants 1380-1390, unscored); upgrades done, then transfer.                              |
| G 8.jsonl:1  | A24  | clean               | Exhaustive direct-flight search, picks cheapest JFK-SEA $106, gift-card order right.                                           |
| G 13.jsonl:2 | A31  | clean (refusal)     | Basic economy cannot change; no booking.                                                                                       |
| G 7.jsonl:1  | A22  | clean               | 3 updates, gift card per user preference.                                                                                      |
| G 20.jsonl:6 | R5   | clean               | Asks confirmation twice; user follows conditional script; returns water bottle.                                                |
| G 21.jsonl:1 | R9   | clean               | Mid-confirmation change of mind handled.                                                                                       |
| G 30.jsonl:2 | R38  | luck                | User picks "no longer needed" itself; 14/16 other samples user says "ordered by mistake" -> 0.                                 |
| G 43.jsonl:1 | R64  | luck/ambiguous gold | Two 4K waterproof variants under price (black 466.75 / silver 481.50); gold = black.                                           |
| G 22.jsonl:1 | R12  | clean-ish           | Tries PayPal refund first (tool rejects), then transfers when user refuses card.                                               |
| G 26.jsonl:2 | R27  | lenient             | Exchange then attempts return \[21\]; blocked only by tool error; gold = exchange only.                                        |
| G 24.jsonl:2 | R18  | clean               | Gray leather fixed-armrest exchange.                                                                                           |
| G 57.jsonl:1 | R102 | clean               | Address+modify+exchange all correct (muts.py).                                                                                 |

Clean 10/16 (2 refusal-type), grader-lenient 3/16, outcome decided by unscripted user
choice / ambiguous gold 3/16. 0/16 where the user did the agent's job.

## Failures (14)

| ref                        | task | cause                                                                                                     |
| -------------------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| G 20.jsonl:1, G 20.jsonl:3 | R5   | user sim deviates from conditional script (says "yes" to 2nd confirmation instead of switching to return) |
| G 21.jsonl:2, S 21.jsonl:1 | R9   | episode ends on agent's confirmation question -> user emitted STOP before "yes"                           |
| G 33.jsonl:4               | R42  | same premature STOP after first of two address changes                                                    |
| G 57.jsonl:7               | R102 | same premature STOP before exchange confirm                                                               |
| G 30.jsonl:1               | R38  | user-chosen reason "ordered by mistake" vs gold "no longer needed"                                        |
| G 43.jsonl:2               | R64  | agent proposes silver 4K variant, user accepts; gold-ambiguity                                            |
| G 5.jsonl:2                | A18  | agent misreads policy, refunds all 5 to one credit card despite "original payment for each"               |
| S 7.jsonl:1                | A22  | agent picks credit card without asking; user prefers gift card                                            |
| G 11.jsonl:1               | A29  | agent books insurance=yes ($312) unprompted                                                               |
| S 22.jsonl:2               | R12  | agent processes return to Visa without consent                                                            |
| G 24.jsonl:1               | R18  | agent exchanges to the same (unavailable) item id                                                         |
| S 26.jsonl:5               | R27  | agent returns before exchange; order-of-operations error                                                  |

Agent errors 6, user-sim errors 6, gold/task ambiguity 2. Population check (endscan.py):
51/131 retail failures and 9/65 airline failures end on an agent confirm-request with no
further turn (premature user STOP). This lowers success.

## Telecom (3 successes)

- G 82.jsonl:1 mobile_data Hard: user reports statuses honestly, one action per turn,
  refuses "Good" speed until VPN disconnect. Mild lead ("Should I turn that on?").
- G 71.jsonl:1 MMS Hard: stepwise roaming -> network mode -> SMS permission; user
  volunteers "abroad in France" (known_info).
- G 99.jsonl:1 service Easy: bill pay -> resume -> reboot -> reseat SIM; user reports
  "No SIM detected" truthfully. No volunteered fixes; device actions executed as
  instructed.
