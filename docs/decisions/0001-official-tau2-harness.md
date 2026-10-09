# Run τ²-Bench through the pinned official harness and AReaL proxy

Date: 2026-10-09

Status: Accepted

Applies to `examples/tau2/` (`agent.py`, `contracts.py`, `train.py`, `utils.py`),
`scripts/tau2/`, and the AReaL OpenAI-compatible proxy under
`areal/experimental/openai/`.

## Context and Problem Statement

τ²-Bench training needs a multi-turn environment with tools, mutable databases, a
simulated user and a task scorer. Training reward, in-training validation and offline
evaluation must all measure the same thing, and results must be comparable to the public
benchmark. Which environment and scorer do we train against, how does the policy reach
the local inference servers, and which episode outcomes count as model failures rather
than infrastructure failures?

## Considered Options

- Reimplement or adapt a τ² environment, scorer or synthetic data inside AReaL.
- Install the community `dhh/async-and-custom-completion` fork, which adds custom
  completion hooks and an async orchestrator.
- Install the official `sierra-research/tau2-bench` package at a fixed revision and use
  its synchronous orchestrator, tools, databases, user-simulator wiring and evaluator
  unchanged, with the policy reached through AReaL's local `/chat/completions` proxy.

## Decision Outcome

Chosen option: "official package at a fixed revision behind the AReaL proxy", because it
is the only option whose rewards are the benchmark's own scores, and it keeps
AReaL-specific code limited to request routing, trajectory export and failure handling.
The fork's hooks and async API are not part of the official package.

Contract:

- Revision `sierra-research/tau2-bench@b7ea9074c1cba482b30687fecdb5c8425fd6f619`
  (`OFFICIAL_TAU2_REVISION` in `examples/tau2/contracts.py`). Data, policies, tools and
  evaluator code come from that revision. Runs record digests of tasks, policies,
  databases and scoring code.
- Each episode runs in the official synchronous orchestrator on a worker thread so the
  AReaL async workflow loop stays non-blocking. Every episode, including every retry,
  resets the database, user state, tool environment and proxy session.
- The policy request path is τ² agent → LiteLLM/OpenAI HTTP → AReaL local
  `/chat/completions` proxy → chat template rendering → SGLang. The user simulator and
  the official NL-assertion judge use a separate provider route (currently DeepSeek,
  `deepseek-flash`, temperature 0). `agent.py` rebinds only the judge model; all other
  evaluator code is official. Provider credentials come only from the environment.
- The harness keeps the full official history: no summarization, sliding window, tool
  output truncation or extra context compression. The episode step budget is the
  official orchestrator's `max_steps=200` (orchestrator steps, not assistant replies).
- Reward is the official evaluator's terminal 0/1 score. There is no partial credit,
  length reward or format penalty (`invalid_format_penalty: 0.0`). τ² exports set
  `bootstrap_mask=false`.
- Each episode exports exactly one trajectory, the branch the conversation actually
  adopted (`export_style: concat`, `drop_retry_orphans: true`,
  `tau2_qwen_concat_prefix_matcher`). Abandoned retry branches never enter reward, loss
  or group membership. If the adopted branch cannot be identified, the episode fails as
  a bounded infrastructure error rather than being chosen by timestamp.

Failure classification:

| Event                                                                     | Treatment                                                                                                         |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Normal end                                                                | Official 0/1, full trajectory kept                                                                                |
| Policy context exhausted (`context_limit`) or `max_steps` reached         | Valid task failure, reward 0, kept for training, no retry                                                         |
| Policy emits bad tool arguments, loops, or empty output                   | Model failure inside the official protocol                                                                        |
| Episode wall clock reaches `episode_timeout_seconds` (1200)               | Infrastructure failure raised before scoring; bounded episode retry; never converted to reward 0                  |
| User-simulator empty message, malformed tool JSON, provider/network error | Infrastructure failure; request-level retries (`user_llm_num_retries: 3`, ≤120 s per request), then episode retry |
| Evaluator, database, or token/logprob mismatch                            | Infrastructure or implementation failure; retried only when the exception class is known to be retryable          |

Episode-level retries are `infra_retries: 1` (first attempt plus one). Request-level and
episode-level limits are separate layers and do not add up to one budget. A failed
attempt is never spliced with a later successful attempt. An unrecovered failed group
aborts fixed-dataset training rather than silently shrinking coverage.

### Consequences

- Good, because training reward, validation and offline evaluation share one scorer and
  any score difference cannot come from local scoring changes.
- Good, because context exhaustion is a legitimate outcome the policy can learn to
  avoid, while provider faults cannot leak in as false zeros.
- Bad, because the official evaluator's blind spots are inherited. Some tasks whose
  correct behaviour is to refuse or only read data score 1 for an agent that says
  "Done." without tool calls, and an empty NL-judge result list passes (`all([])` in
  `tau2/evaluator/evaluator_nl_assertions.py`).
- Bad, because the 1200 s limit is checked cooperatively at orchestrator turn
  boundaries. An in-flight request or tool can return later, and official scoring runs
  after the orchestrator returns, outside that timer. Python cannot kill the worker
  thread, so cancellation waits for the episode to finish before the proxy session
  closes.
- Bad, because results depend on a commercial user simulator whose alias does not pin a
  backend version. Its configured `max_completion_tokens: 512` is a request value, not a
  verified provider limit. Changing the simulator model is a protocol change and needs a
  new baseline.
