# SAO paper gap audit — 2026-09-23

Candidate: codex/sao-integration, based on e6da977d with uncommitted recipe/loader
changes. Paper: https://arxiv.org/html/2607.07508v1, sections 3 and 4.1.

| Item                   | Implementation / remaining limit                                                                                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single rollout         | 128 prompts × 1 response; one logical actor update per batch.                                                                                                                                                                   |
| Direct DIS             | Current/actual rollout token probability ratio, strict (0.7,6.0) mask, detached score-function weight; no extra PPO rejection sampling.                                                                                         |
| Value / actor updates  | Two full critic updates, refreshed values and advantages, one actor update. Order tests count optimizer steps.                                                                                                                  |
| Dual/adaptive lambda   | Critic lambda 1; actor lambda 1−1/(1.5 L), with L counting action tokens.                                                                                                                                                       |
| Skip-observation       | Primitive and component tests exist; current RLVR workflow is single-turn, not an agentic multi-turn acceptance run.                                                                                                            |
| Pretrained critic      | Same native HF export as PPO: critic-dapo-step50-hf, source step 50. Export hashes/head/token IDs checked; no forged sealed protocol.                                                                                           |
| Horizon                | Fixed SAO response 1024→8192, total context 2048→9216; both previously-run PPO and queued PPO configs already use 8192/9216. Prompt filter remains 1024.                                                                        |
| Warmup                 | Paper says 10-step value warmup. Latest user decision explicitly cancels warmup: actor 1e-6 and critic 5e-6 are constant from the start, with no critic-only stage.                                                             |
| Frozen attention / MoE | Not implemented for paper-exact reproduction. Dense Qwen3.5 trains the critic backbone; FlashAttention kernel selection is not freezing attention parameters.                                                                   |
| Experimental setting   | Paper Qwen3-30B-A3B-Thinking, TIR/Python and OpenHands, 128K context; current dense Qwen3.5 math recipe uses 8K responses and no tools.                                                                                         |
| Evaluation             | Dedicated GPU, every 20 steps, n=2 per user request; differs from paper benchmarks and 16/4 evaluation repetitions.                                                                                                             |
| Runtime evidence       | Prior short native tests do not qualify the newly changed pretrained-critic, 8K, 4/3/1 recipe. No new GPU training launched. Real mixed-version token behavior probabilities and long-running learning stability need evidence. |

The pretrained export records validation MSE 0.0893462 and explained variance 0.544691.
These are existing critic validation results, not SAO outcomes. Online critic training
continues after initialization; no pretraining optimizer state is implicitly resumed.

Core source anchors: `areal/trainer/ppo/dis.py`, `lambda_fn.py`, `gae.py`,
`trajectory.py`, `update.py`, `value_checkpoint.py`; entry and resource configuration:
`examples/math/gsm8k_sao.py`, `gsm8k_sao.yaml`, `scripts/sao/run_sao.sh`.

Unspecified paper details (e.g. exact minibatch split, gamma, normalization and token
reduction) are project choices; do not claim an official implementation specified them.

## Stability controls explicitly reported by the paper

Source: SAO v1 section 3.1 equations (1)–(3), sections 3.2 and 4.1.

| Control                 | Paper disclosure                                                                                                | Current math recipe                              |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| IS denominator          | Per-token log probability recorded by rollout; no separate old policy forward                                   | Same                                             |
| Mask rule               | Keep ratio itself within strict `(1-epsilon_low, 1+epsilon_high)`; otherwise zero, regardless of advantage sign | Same                                             |
| Math bounds             | epsilon low 0.3, high 5.0 → strict ratio `(0.7,6.0)`                                                            | Same                                             |
| Coding bounds           | epsilon low 0.8, high 3.0 → strict ratio `(0.2,4.0)`                                                            | Not used by math recipe                          |
| Additional TIS/clamp    | No extra capped-weight TIS term shown in DIS objective                                                          | None                                             |
| Additional PPO clipping | DIS substitutes its direct double-sided mask for PPO's sign-dependent surrogate clipping                        | None                                             |
| Value stability         | K=2, critic LR 5e-6, lambda=1, frozen-attention MoE critic                                                      | K/LR/lambda implemented; no frozen-attention MoE |
| Policy stability        | LR 1e-6; length-adaptive lambda with alpha=1.5                                                                  | Same                                             |
| Warmup                  | Value warmup 10 steps reported                                                                                  | Disabled by latest explicit user decision        |

The paper does not use MIS/TIS as separate named configuration switches. "MIS" in our
discussion means masked IS descriptively, not an official SAO flag. Its DIS formula is
masked importance weighting, not clamping a ratio to a threshold.

No concrete gradient-norm clipping threshold, Adam betas/epsilon/weight decay, KL
coefficient, entropy coefficient, reward/advantage normalization configuration, maximum
allowed version lag or queue capacity is specified in the inspected v1 paper. Do not
infer these are disabled: they are undisclosed. Our gradient clipping 1.0, Adam
0.9/0.999, weight decay 0, KL 0, raw reward/advantage and max-head lag 2 are project
engineering choices, not verified paper hyperparameters.

Equation (1) writes the calibrated ratio times advantage times log policy without an
explicit stop-gradient operator. Our detached coefficient implements a score-function
estimator; the precise official autograd graph cannot be established from that notation
alone. Retain this as an implementation interpretation.

Verification for this change: actual PPO export hash validation and SAO resolved-config
check succeeded on CPU, and 29 recipe/export/value/order tests passed. This does not
constitute 8K multi-GPU training or learning-quality acceptance.
