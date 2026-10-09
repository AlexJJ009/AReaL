# Schedule τ² episodes with the official runner's retry, exclusion and metric rules

Date: 2026-10-09

Status: Accepted

Supersedes the failure-classification part of [ADR 0001](0001-official-tau2-harness.md).

Applies to `examples/tau2/official_runner.py`, `examples/tau2/agent.py`,
`examples/tau2/user_simulator.py`, `examples/tau2/evaluation.py`,
`examples/tau2/train.py`, `scripts/tau2/evaluate.py`, `scripts/tau2/upload_metrics.py`,
`BatchTaskDispatcher.wait_results` in `areal/infra/workflow_executor.py` and the retry
loop in `areal/experimental/openai/proxy/workflow.py`.

## Context and Problem Statement

ADR 0001 kept the official orchestrator, user simulator and evaluator for each episode,
but replaced everything above one episode with our own rules: only whitelisted
exceptions were retried, once; any unrecovered episode aborted the whole evaluation ("no
silent shrinking of the denominator"); context exhaustion and wall-clock overruns had
local meanings. That intent was wrong. The official `tau2 run` already defines how a run
survives infrastructure faults, and nothing in AReaL prevents following it. On
2026-10-09 a Doubao 2.1 Turbo evaluation aborted because the simulated user ended one
episode on its first turn: the episode was a valid `USER_STOP` with no policy tokens,
which our layer treated as an export error. Which scheduling semantics do τ² episodes
follow, and how do they fit AReaL's requirement that every result carries a token
trajectory?

## Considered Options

- Keep the local abort-on-failure rules from ADR 0001 and special-case policy-free
  episodes.
- Follow the official runner at revision `b7ea907` (`tau2/runner/progress.py`,
  `tau2/runner/batch.py`, `tau2/metrics/agent_metrics.py`) in one shared module that
  both offline and in-training evaluation use, with the smallest AReaL changes needed to
  attribute rejected results.

## Decision Outcome

Chosen option: "follow the official runner in one shared module", user-chosen on
2026-10-09. The scheduling layer becomes a template that other harnesses (for example a
future PHAI Relay integration) can copy: official semantics in one place, AReaL used
only as the controller that runs episodes.

Official semantics adopted:

- **Retry.** Any `Exception` replays the episode from scratch, at most
  `DEFAULT_RETRY_ATTEMPTS = 3` more times, `DEFAULT_RETRY_MIN_WAIT = 1.0` s apart (both
  read from `tau2.config`). Each replay owns a fresh environment and a fresh proxy
  session. Cancellation is not retried. In-place repairs (re-asking the user simulator
  after an empty reply or malformed tool JSON, re-scoring after an NL-judge JSON error)
  are removed; they fail the attempt and leave an incident record.
- **Exclusion.** An episode whose retries are exhausted becomes `infrastructure_error`.
  The run continues, the episode is excluded before any metric is computed, and the
  count and error types are reported. The run is `complete` and exits 0 once every
  planned episode has an outcome.
- **Scoring.** `AGENT_STOP` and `USER_STOP` are scored by the official evaluator. A user
  stop before any policy call is a scored episode, reported as a policy-free episode.
  `TIMEOUT` and `MAX_STEPS` score 0 and stay in the denominator.
- **Metrics.** `avg_reward` and `pass^k = C(c,k)/C(n,k)` averaged over tasks, with `k`
  clipped to the fewest evaluated trials of any task, overall and per domain, using the
  official `pass_hat_k`. The existing `pass1_mean` and per-domain/split views stay.

AReaL integration:

- `BatchTaskDispatcher.wait_results(..., with_task_ids=True)` returns
  `(task_id, result)`. A rejected result is otherwise a bare `None` and cannot be bound
  to its (task, trial).
- Every episode attempt writes one JSON record (official messages, `stop_reason`,
  `policy_turns`, reward info or error) to `trajectory_dir`, passed explicitly in
  workflow kwargs. An episode with no exported trajectory is resolved from its last
  attempt record: completed with `policy_turns == 0` is scored; anything else, including
  a missing record, is an infrastructure error.
- `episode_workflow_kwargs` sets `evaluation=True`, which selects the evaluation outcome
  rules below. Training keeps `evaluation=False`.

Evaluation and training differ only where official τ² has no training mode:

| Event                                                                             | Evaluation (`evaluation=True`)                      | Training (`evaluation=False`)                                                         |
| --------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Policy context exhausted (32K)                                                    | Exception → retries → excluded (official behaviour) | `context_limit`, reward 0, kept ([ADR 0002](0002-tau2-policy-generation-protocol.md)) |
| Wall clock (`eval_episode_timeout_seconds` 1800 / `episode_timeout_seconds` 1200) | Official `TIMEOUT`, scored 0 and counted            | Exception → retries                                                                   |
| Episode finished but over wall clock                                              | Kept and scored                                     | Exception → retries                                                                   |
| Any other exception                                                               | Retries ×3 → excluded                               | Retries ×3 → rejected; an incomplete GRPO group aborts                                |
| User stop before any policy call                                                  | Scored, counted as policy-free                      | Rejected (no tokens to train on)                                                      |

The 32K context stays for cost; the evaluation wall clock is a generous 1800 s against a
measured p99 of 538 s and a maximum of 651 s (656 DeepSeek temperature-0 episodes). The
user simulator gets only the official `temperature 0`; the local 512-token completion
cap is removed. Not aligned: no seed is passed (user decision); the NL judge still uses
the user-simulator model because no OpenAI key is available; the policy keeps its 4096
reply cap and `num_retries 0` behind the local proxy, with retries at the episode layer.

### Consequences

- Good, because one provider hiccup or one odd simulated user no longer discards a
  multi-hour evaluation, and our numbers follow the same exclusion rule as the official
  leaderboard.
- Good, because offline evaluation and in-training evaluation share one module and
  cannot drift apart.
- Good, because every attempt, including failed and policy-free ones, leaves a readable
  record of what the official orchestrator saw.
- Bad, because excluded episodes do shrink the denominator. The exclusion count is a
  first-class metric; a run with many exclusions is not comparable and must say so.
- Bad, because training still aborts when a GRPO group loses a member (excluded or
  policy-free). Changing group composition is an algorithm decision left open; it has
  not occurred with the DeepSeek simulator.
- Bad, because per-attempt records cost disk (measured in the PRD's smoke test) and are
  written for training rollouts too.
- Neutral: the evaluation and training columns above are a deliberate split. Changing
  either column is a protocol change and needs new baselines.
