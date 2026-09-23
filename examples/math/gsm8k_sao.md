# Dense GSM8K SAO

`gsm8k_sao.py` uses the production `PPOTrainer` through the dedicated async evaluation
adapter, with FSDP and SGLang. The example uses FlashInfer for SGLang inference and
FA2/FLA for FSDP training. It selects Direct DIS, action-length actor lambda with alpha
1.5, critic lambda 1, two complete critic updates followed by refreshed values and one
actor update. It uses one answer per prompt and a logical batch of 128. Online learning
rates are actor `1e-6` and critic `5e-6`.

The approved engineering choices are gamma 1, raw rewards and advantages, plain critic
MSE, constant learning rates without warmup, and a DIS mean over original action tokens.
The DIS weight and advantage are detached in the score-function loss. These are explicit
interpretations where the paper does not specify code. Length stops in this
finite-budget recipe are terminal. Frozen attention, MoE, and paper score reproduction
are outside this implementation.

## Configuration

Set `SAO_MODEL_PATH` to the selected dense policy checkpoint and `SAO_RUN_ROOT` to an
external artifact directory. Use the existing AReaL runtime. The YAML reserves four GPUs
shared by actor and critic, three for training rollouts, and one for validation. Every
20 training steps it saves an HF checkpoint and queues validation against that
checkpoint on the dedicated GPU. Validation samples two answers per prompt; training
still samples one. Epoch/time triggers are disabled. Validation uses the test split of
the configured training dataset. Raw validation trajectories use AReaL's standard dump
mechanism, not the PPO experiment's custom 700-question sample ledger.

Set `SAO_ARTIFACT_ROOT` for runtime caches and use `scripts/sao/run_sao.sh` in an
activated AReaL environment. This thin wrapper calls `run_ppo.sh`, sharing runtime setup
and logging while selecting SAO's entry point and configuration. Calling `run_ppo.sh`
directly still selects PPO. Append `--check-config` to validate and print the resolved
configuration without starting workers. This resource layout still requires a GPU
preflight before a formal training run.

The default launcher selects the same pretrained critic as PPO via `SAO_CRITIC_PATH`,
defaulting to `${SAO_ARTIFACT_ROOT}/models/critic-dapo-step50-hf`. The existing HF
export is validated against `export-manifest.json`, including weight hashes, scalar
head, backbone dimensions and token-ID compatibility. It loads through the native
Qwen3.5 critic adapter; no synthetic `value_manifest.json` is generated and shared model
files are never rewritten. The selected export's validation metrics are provenance, not
a new qualification of SAO learning effectiveness.

Response length is 8192 and total context is 9216, matching the PPO/critic-pretraining
recipe. Validation inherits these limits. The 1024-token dataset filter applies to
prompts, not generated responses.

A sealed artifact may alternatively supply `critic.value_contract`. Base-critic cold
start remains available only by explicitly overriding `critic.path` to the actor
checkpoint and passing `--allow-base-critic`; it is not the default.

Actor LR stays at `1e-6` and critic LR at `5e-6`, with no LR warmup or critic-only
training stage, per the user's latest decision. The pretrained critic continues to
update twice per batch. This differs from the paper's reported 10-step value warmup.

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
