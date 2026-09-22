# Dense GSM8K SAO

`gsm8k_sao.py` uses the production `PPOTrainer` with FSDP and SGLang. The example
selects Direct DIS, action-length actor lambda with alpha 1.5, critic lambda 1, two
complete critic updates followed by refreshed values and one actor update. It uses one
answer per prompt and a logical batch of 128. Online learning rates are actor `1e-6` and
critic `5e-6`.

The approved engineering choices are gamma 1, raw rewards and advantages, plain critic
MSE, constant learning rates without warmup, and a DIS mean over original action tokens.
The DIS weight and advantage are detached in the score-function loss. These are explicit
interpretations where the paper does not specify code. Length stops in this
finite-budget recipe are terminal. Frozen attention, MoE, and paper score reproduction
are outside this implementation.

## Configuration

Set `SAO_MODEL_PATH` to the selected dense policy checkpoint and `SAO_RUN_ROOT` to an
external artifact directory. Use the existing AReaL runtime. The YAML's four training
GPUs plus four rollout GPUs are an example allocation; actor and critic share the
training allocation. Validate the resolved resource layout and sequence lengths before a
training run.

The user-authorized interim mode initializes a scalar head on the same Base backbone as
the actor:

```bash
python examples/math/gsm8k_sao.py --config examples/math/gsm8k_sao.yaml \
  --allow-base-critic
```

This mode is an untrained critic cold start. It is useful for integration checks and is
not a pretrained critic or a learning-quality acceptance result. No extra online
critic-only warmup is inserted into SAO.

For the independently trained critic, set `SAO_VALUE_PATH` and provide
`critic.value_contract` in a run-specific YAML. The contract contains `identity`
(backbone/tokenizer IDs), `protocol` (discount, horizon, thinking, reward, termination,
scorer/split/template digests and freeze policy), and `require_pretrained: true`. The
artifact must contain `value_manifest.json`, the complete backbone and scalar head,
tokenizer files and the hashed report from the separate pretraining workflow. Run
without `--allow-base-critic`. Missing weights, identity/protocol mismatches and
unqualified artifacts are rejected. Reuse the critic pretraining team's metrics and
evaluation report; this entry does not implement another value-quality evaluator.

## Verification scope

Mechanism tests live in
`tests/test_sao_{dual_lambda,adaptive_lambda,skip_observation, critic_updates,value_checkpoint,direct_dis}.py`.
Bounded FSDP probes live under `tests/torchrun/run_sao_*.py`.

Qwen3.5 uses one sequence per microbatch because its recurrent layers do not reset at
packed sequence boundaries. Requested `n_mbs=2` is a hint: actual counts depend on local
trajectories. Multiple microbatches still produce one optimizer update per
`train_batch`; SAO's two critic updates come from two explicit train batches.

Native preflight evidence must name the model/tokenizer, exact input fixture, actual
microbatch counts, optimizer steps and resource cleanup. Tiny-model tests and clipped
input fixtures do not qualify longer four-card training. Formal learning experiments
require their own approved budget and acceptance criteria.
