# Tau2 non-thinking proxy fix and offline N=8 evaluation review

- Code candidate: `b793c8a2ba2cf2cee87dd225716a2bfe56e97cdd`.
- Independent reviewer: `/root/retail_regression_audit`, GPT-5.5, medium.
- Verdict: pass; latest review found no remaining findings.
- Scope: proxy chat extension forwarding, standalone official-test evaluator, one-shot
  W&B upload, focused tests, and the local launch protocol.
- Review rounds: proxy ingress review passed; offline review found a missing candidate
  SHA file; that launch-binding file was added. The reviewer then rechecked the
  committed code at the exact candidate above with no findings.
- Validation: 131 relevant tests passed; full pre-commit passed; CPU resolved-config
  check planned 800 episodes per model. GPU evaluation was not yet launched at review
  time.

The proxy previously discarded both nested and SDK-flattened template arguments. Actual
GRPO and critic rollout artifacts opened thinking prompts despite the non-thinking
config. The fix preserves supported template and context extensions, with
malformed/conflicting extensions rejected. Existing weights and metrics remain
historical thinking-affected results, not qualified non-thinking training results.

The offline evaluator preserves validation sampling parameters and runs eight trials
over all 100 held-out tasks for each model. Dispatcher task IDs preserve identity
despite concurrent completion order. Infra failure is not scored zero, existing results
cannot be overwritten, and W&B uploads only complete results using the mean across
trials rather than the best trial.

The local runner must bind its candidate SHA to the final branch HEAD before launch. The
initial and final actor use separate inference GPUs, with no optimizer or training pool.
Runtime non-thinking prompt readback is still required; CPU review does not establish
completed evaluation or downstream model quality.
