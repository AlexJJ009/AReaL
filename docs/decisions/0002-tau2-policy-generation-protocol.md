# Use non-thinking Qwen3.5-4B with a 32K episode context for τ²-Bench

Date: 2026-10-09

Status: Accepted

Applies to `examples/tau2/config_grpo.yaml`, `config_sao.yaml`,
`config_critic_production.yaml`, `examples/tau2/agent.py`, `examples/tau2/train.py`,
`scripts/tau2/evaluate.py` and the proxy chat-template path in
`areal/experimental/openai/`.

## Context and Problem Statement

The τ² policy, its critic and every evaluation must use one model, one chat template and
one length budget. Otherwise rewards, values and scores describe different agents. Which
model, reasoning mode and token budgets does the τ² recipe fix, and how is it enforced
end to end rather than only in YAML?

## Considered Options

- Keep the earlier math-line choice of Qwen3.5-4B-Base.
- Use Qwen3.5-4B (instruct) with thinking enabled.
- Use Qwen3.5-4B (instruct) with thinking disabled everywhere.

## Decision Outcome

Chosen option: "Qwen3.5-4B with thinking disabled everywhere", user-chosen. The user's
reason: current RL frameworks handle thinking tags poorly. The Base model was replaced
for the tool-use setting, so the critic had to be retrained on the new model (see
[ADR 0005](0005-tau2-online-critic.md)).

Fixed protocol:

| Item                       | Value                                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy and critic backbone | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, tokenizer from the same snapshot                                                                  |
| Reasoning mode             | `enable_thinking=false` for training rollouts, critic data, in-training validation and offline evaluation; `add_thinking_tool=false` (rejected by `train.py`) |
| Total context              | 32768 tokens for the final rendered prompt including tool schemas; one slot reserved for SGLang's strict bound, applied once                                  |
| Per reply                  | at most 4096 tokens and never more than the remaining context                                                                                                 |
| Training sampling          | temperature 1, top_p 1, top_k disabled by sentinel, frequency_penalty 0, non-greedy; global seed 42, request seed `null`                                      |

Per-request generation budget is `M = min(4096, 32768 - 1 - L)`, where `L` is the token
count of the actually rendered template. When `M <= 0` the proxy rejects the request
with a typed error. In training the episode ends as `context_limit` with reward 0 and no
retry ([ADR 0001](0001-official-tau2-harness.md)). Evaluation follows the official
runner instead: the error fails the attempt, is retried, and an episode that still
overflows is excluded ([ADR 0009](0009-official-tau2-runner-semantics.md)). Hitting 4096
tokens in one reply does not end the episode. Only complete, valid tool calls are
executed.

Enforcement is at the final consumer, not the config: the template flag
(`chat_template_kwargs` / `extra_body.chat_template_kwargs`) must pass through the local
HTTP proxy into the rendered template. An empty closed `<think></think>` prefix is valid
non-thinking output; an open thinking prefix or a reasoning delimiter in generated text
is a protocol violation. Generated thinking tokens must never be deleted afterwards
while reusing their old logprobs.

The reversal matters. Early τ² critic and GRPO runs were configured non-thinking, but
the proxy dropped `chat_template_kwargs`, so the template silently fell back to
thinking. Those weights and metrics are historical thinking results. They are not
non-thinking results, and re-evaluating them does not substitute for retraining.

The user-simulator and judge provider keep their own fixed settings and are not switched
to non-thinking with Qwen.

### Consequences

- Good, because policy, critic and evaluator agree on one template, and token/logprob
  alignment does not depend on parsing reasoning blocks.
- Good, because context exhaustion has one definition shared by the proxy, the workflow
  and the reward; only its outcome differs between training and evaluation (ADR 0009).
- Bad, because any thinking-mode comparison needs a separate protocol and new baselines;
  results across modes must not be mixed.
- Bad, because 32K is a length limit, not a capacity proof. Memory headroom for long
  episodes must still be measured per resource layout.
- Neutral: changing the backbone or tokenizer invalidates the trained critic and all
  baselines.
