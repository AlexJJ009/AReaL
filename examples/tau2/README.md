# Customer Service Agent Training with Tau2 Benchmark

The scripts reuse AReaL's PPO trainer, official tau2 environment/evaluator, existing
dedicated evaluation controller, and ordinary Pueue. No separate launcher registry or
runtime manifest is required.

## Training entrypoints

| Entry                                                                  | Purpose                                                                                |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `scripts/tau2/train_mixed.sh`                                          | Async GRPO on all three official train domains                                         |
| `scripts/tau2/train_airline.sh`, `train_retail.sh`, `train_telecom.sh` | The same GRPO recipe restricted to one domain                                          |
| `scripts/tau2/collect_critic.sh`                                       | One frozen-policy episode per official train task, with task-disjoint train/dev labels |
| `scripts/tau2/train_critic.sh`                                         | Offline mixed critic fitting from collected JSONL                                      |
| `scripts/tau2/run.sh qualification mixed`                              | Historical three-episode SAO integration recipe                                        |

GRPO uses 4 training GPUs, 3 rollout GPUs, and 1 dedicated checkpoint-evaluation GPU. A
full training batch is 8 prompts x 8 trajectories = 64 episodes. The last epoch batch
may be smaller but always retains complete groups of eight. The default is two passes
over all 178 official train tasks (46 updates). It uses non-thinking, 32768 total
context, at most4096 tokens per response, policy temperature1, staleness2, recomputed
logprobs, decoupled loss, group reward normalization and the existing token ratio mask
above5. It does not load a critic. Save/recovery/evaluation cadence is20 steps plus
final, as in the previous math GRPO recipe. With `eval_before_train=true`, the initial
evaluation uses the starting model and consumes the initial evaluator trigger before
training; step1 does not request an unsaved checkpoint.

`experiment_mode=tune` selects142train/36dev, stratified by task and domain with seed42.
`experiment_mode=formal` trains on178officialtrain and evaluates 100officialtest tasks
with fixed settings; test results never select a checkpoint. The dedicated evaluator
reports per-domain coverage and rewards, and rejects missing/infra-failed episodes
instead of reporting a partial mean.

Critic collection defaults to one episode for each of178tasks:24/59/59train
and6/15/15dev. It mixes domains and retains valid model failures. Offline fit uses two
epochs, global batch16 including the tail (18 updates), validation at 0/5/10/15/18 and
checkpoints every5steps and final. Qwen3.5-4B initializes a fresh scalar head; attention
remains frozen. LR5e-6, token-mean plain MSE and optimizer settings match the latest SAO
recipe. Dev reports contain overall and per-domain MSE, explained variance, and
distributed gradient norm from a backward-only diagnostic (no optimizer step). The live
training gradient norm is reported separately. EV is undefined for constant targets.

## Runtime and configuration

```bash
export SAO_ARTIFACT_ROOT=/path/to/existing/areal-artifacts
export TAU2_RUN_ROOT=/path/to/a-unique-run
export TAU2_TRIAL_NAME=tau2-grpo-run
export HF_HUB_CACHE=/path/to/huggingface/hub
export TAU2_DATA_DIR=/path/to/pinned-tau2/data
export TAU2_DEEPSEEK_ENV_FILE=/path/to/deepseek.env
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

scripts/tau2/train_mixed.sh --check-config
scripts/tau2/train_mixed.sh
```

`--check-config` uses the official task loader but starts no GPU worker, resolves no
model snapshot and loads no provider credential. Ordinary overrides are forwarded to the
existing config parser. Run directories are stable so `recover.mode=auto` can resume the
same run. Qualification mode explicitly creates a new temporary attempt directory.

Set `TAU2_EPISODES` to the checked JSONL for critic fitting. The production critic
always starts from the pinned actor backbone rather than an old math critic. Use
`scripts/tau2/critic_data.py` to check/combine dumps, and `--require-full-coverage` on
fitting when enforcing the whole official pool. Collection and fit are separate Pueue
jobs; offline fit never calls DeepSeek or replays tools. Existing AReaL initialization
still reserves rollout workers for the replay workflow; this is a reuse tradeoff, not a
critic throughput optimum.

The runner sources `scripts/sao/runtime_env.sh` and the existing `.venv`. Do not run
`uv run` in this qualified environment: lock synchronization would replace the installed
tau2-compatible LiteLLM/Uvicorn overrides. It uses the trusted-host
`AREAL_ALLOW_DEFAULT_ADMIN_KEY=1`; the internal proxy key remains separate from
`DEEPSEEK_API_KEY`, loaded only from a mode0600 env file.

Transient provider failures get at most one fresh-session episode retry by default
(`infra_retries`). Request-level provider retries remain separate. Scored task failures
are not retried. Deterministic code/config failures do not become reward0. An
unrecovered failed group aborts fixed-dataset training rather than silently shrinking
coverage. Each retry resets both the environment and the proxy session, so abandoned
interactions cannot enter successful exports.

## Evidence boundary

The historical Pueue159 run qualified3episodes/oneSAOstep at actual sequence
lengths10K–20K. It did not qualify full32K batches, batch64GRPO throughput or critic
learning quality. Current delivery evidence and queued job identities are recorded in
`docs/plans/tau2-sao/training-delivery.md`. `eval_matrix.py` plans/verifies matrix
coverage; it does not run a scientific comparison.

## Overview

This example demonstrates how to train customer service agents using the
[$\\tau^2$-Bench](https://github.com/sierra-research/tau2-bench) with AReaL's PPO/GRPO
training pipeline. The $\\tau^2$-Bench provides realistic customer service simulation
environments across multiple domains (retail, airline, telecom) where agents must help
with user's request by both using agent tools and guiding users using their tools.

## Code Architecture

- `train.py`: Training script that creates tau2 datasets and runs PPO training with the
  `Tau2AgentWorkflow`.
- `agent.py`: Implements `Tau2AgentWorkflow` which runs tau2 simulations. The
  implementation is completely independent from AReaL (except for logging, which you can
  replace with other logging tools). AReaL's proxy server will automatically connects to
  the workflow and runs it with self-hosted inference servers for RL training.
- `utils.py`: Common utilities including `Tau2EnvConfig`, `Tau2PPOConfig`, and
  `Tau2RunInfo` dataclasses. Also patches tau2's cost calculation to silently handle
  self-hosted models.

## Running the Example

### Prerequisites

Please make sure AReaL is setup and working following the
[installation guide](https://areal-project.github.io/AReaL/en/tutorial/installation.html).

1. Install the pinned official tau2-bench revision. The index below keeps PyPI package
   traffic on the Tsinghua mirror; GitHub traffic should use the operator-approved proxy
   route:

```bash
pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  'tau2 @ git+https://github.com/sierra-research/tau2-bench.git@b7ea9074c1cba482b30687fecdb5c8425fd6f619'
```

The workflow uses the official synchronous orchestrator and runs one complete episode on
a worker thread so the AReaL async workflow loop stays non-blocking. Do not install the
old `dhh/async-and-custom-completion` fork: its custom completion hooks and async
orchestrator API are not part of the pinned official package. The policy proxy session
key and user-simulator provider key are required and routed independently.

1. Setup the `TAU2_DATA_DIR` environment variable:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

For multi-node experiment with slurm, this can be set in the config file under
`actor.scheduling_spec[0].env_vars.TAU2_DATA_DIR`.

### Configuration Files

Four example configurations are provided:

| Config                         | Model           | Cluster           | Allocation                                             | Use Case                                |
| ------------------------------ | --------------- | ----------------- | ------------------------------------------------------ | --------------------------------------- |
| `config_1.7b_airline.yaml`     | Qwen3-1.7B      | 1 node, 8 GPUs    | `sglang:d6+archon:d2`                                  | Small-scale local training              |
| `config_8b_airline.yaml`       | Qwen3-8B        | 3 nodes, 24 GPUs  | `sglang:d16+archon:d8`                                 | Multi-node Slurm training               |
| `config_30b_moe_airline.yaml`  | Qwen3-30B-A3B   | 8 nodes, 64 GPUs  | `sglang:d8t4+megatron:(attn:d4p4t2\|ffn:d2p4e4)`       | Multi-node Slurm training for MOE model |
| `config_235b_moe_airline.yaml` | Qwen3-235B-A22B | 10 nodes, 80 GPUs | `sglang:d4t8+megatron:(attn:d1p12t4c1\|ffn:d1p12t1e4)` | Multi-node Slurm training for MOE model |

### Prepare User Simulator Server

You need to setup a user simulator server if using self-hosted LLMs. For example, using
[Qwen with SGLang](https://qwen.readthedocs.io/en/latest/deployment/sglang.html):

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser qwen25 \
    --chat-template ./qwen3_nonthinking.jinja \
    --dp-size 2 \
    --tp-size 4
```

Update the `econfig.user_llm_base_url` in your config to point to this server.

### Training Commands

NOTE: Following commands should be executed from root directory of this repository.

#### Single Node (1.7B Model)

The commands in this legacy section document the upstream generic example. These YAMLs
do not supply the pinned actor and explicit episode-budget fields now required by
`train.py`, so they cannot be run unchanged with this adapter. Use the qualification
wrappers above for the verified host runtime path.

On a single 8x GPU node with our official image
(ghcr.io/areal-project/areal-runtime:latest), run:

```bash
python3 examples/tau2/train.py \
    --config examples/tau2/config_1.7b_airline.yaml \
    experiment_name=$experiment_name \
    trial_name=$trial_name \
    econfig.user_llm_base_url=http://localhost:8000/v1/ # your user LLM address
```

#### Multi-Node Slurm

On a SLURM cluster with at least 3 8x GPU nodes, directly run from a intermediate server
with AReaL and SLURM cli installed:

```bash
python3 examples/tau2/train.py \
    --config examples/tau2/config_8b_airline.yaml \
    experiment_name=$experiment_name \
    trial_name=$trial_name \
    cluster.fileroot=/path/to/shared/storage \
    cluster.name_resolve.nfs_record_root=/path/to/shared/storage/name_resolve \
    econfig.user_llm_base_url=http://localhost:8000/v1/ # your user LLM address
```

### Tau2 Related Configuration Options

| Option                           | Default   | Description                                                                     |
| -------------------------------- | --------- | ------------------------------------------------------------------------------- |
| `econfig.domain`                 | `airline` | Tau2 domain: `airline`, `retail`, `telecom`, or the shared launcher's `mixed`   |
| `econfig.max_steps`              | `100`     | Maximum number of steps per trajectory                                          |
| `econfig.add_thinking_tool`      | `false`   | Whether to use thinking as a tool for the agent                                 |
| `econfig.solo_mode`              | `false`   | If true, agent handles both agent and user roles (no user simulator needed)     |
| `econfig.user_llm_base_url`      | `null`    | Base URL of the user simulator LLM server                                       |
| `econfig.user_llm`               | `null`    | Model name for user simulator (e.g., `openai/self-hosted-Qwen2.5-72B`)          |
| `econfig.user_llm_args`          | `null`    | Arguments for user LLM (e.g., `{temperature: 0.0, max_completion_tokens: 512}`) |
| `econfig.turn_discount`          | `1.0`     | Discount factor for turn-based learning                                         |
| `econfig.invalid_format_penalty` | `0.1`     | Penalty for invalid format in completions                                       |

## Results

The following figure shows the training reward curves for the two configurations,
trained using the Archon engine:

<p align="left">
  <img src="reward.png" width="400">
</p>

- **Green line**: Qwen3-1.7B model (`config_1.7b_airline.yaml`)
- **Purple line**: Qwen3-8B model (`config_8b_airline.yaml`)

We also provide example configs for MoE models: `config_30b_moe_airline.yaml`
(Qwen3-30B-A3B) and `config_235b_moe_airline.yaml` (Qwen3-235B-A22B). The following
figure shows the training reward curve of Qwen3-30B-A3B model trained using the Megatron
engine:

<p align="left">
  <img src="reward_moe.png" width="400">
</p>

For reward curves of experiments on a larger scale, please refer to the
[AReaL Tau2 paper](https://arxiv.org/abs/2601.22607).

## Notes

1. **Trajectory logging**: Trajectories are dumped as `json` and `txt` files in the
   `generated/` directory under `cluster.fileroot`. You can analyze these for debugging
   and evaluation.

1. **Tree training**: The configs enable `enable_tree_training=true` by default, which
   optimizes training by sharing prefix computations across rollouts with the same
   prompt. This option can largely accelerate training but will possibly increase GPU
   memory usage if `actor.mb_spec.max_tokens_per_mb` is large. And this setting may
   cause instability during the training of the MoE model.

## Customization

We have released the training data and a trained model from this pipeline. You can use
the
[open-source Tau2 dataset](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)
to reproduce results from the [AReaL Tau2 paper](https://arxiv.org/abs/2601.22607), or
directly download the resulting model
[AReaL-SEA-235B-A22B](https://huggingface.co/inclusionAI/AReaL-SEA-235B-A22B) trained
with this data and pipeline.

The current non-speculative SGLang server requires prompt plus requested completion
strictly below its32768-token context window. The policy request leaves one slot unused;
exhausted requests return HTTP400 `context_length_exceeded`, never a rate-limit retry.
The episode receives reward0 with `truncated=true`, `bootstrap_mask=false`, and its
existing generated prefix is retained.

Critic collection and fitting use two training ranks (plus six rollout ranks), so
batch16 and the official2/14-row tails dispatch without dropping or duplicating tasks.
Their rollout input iterators stop at epoch boundaries.

Collection now drives the existing rollout controller directly: no actor backward,
optimizer step, or weight update runs merely to export an episode. The existing trainer
initialization is reused, but learning starts only in offline critic fitting. Both GRPO
and critic fitting use zero warmup steps and zero warmup proportion.

GRPO consumes finite epochs. For the two-prompt tail on four training ranks, complete
prompt groups are uniformly replicated for physical dispatch only; no extra episodes are
sampled. This repeats both the numerator and denominator of the token-mean loss, so the
gradient is unchanged. `tau2_batch/real_prompts` and `real_episodes` measure coverage;
worker sequence/token counters describe physical computation. The normal batch remains
eight prompts times eight samples, and the real tail is sixteen episodes.
