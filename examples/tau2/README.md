# Customer Service Agent Training with Tau2 Benchmark

The scripts reuse AReaL's PPO trainer, official tau2 environment/evaluator, the native
`areal/trainer/async_eval.py` dedicated evaluation controller, native metrics
collection, and ordinary Pueue. Tau2 adds only domain-level evaluation summarization on
top of the native async evaluator. No separate launcher registry or runtime manifest is
required.

## Training entrypoints

| Entry                                                                  | Purpose                                                     |
| ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| `scripts/tau2/train_mixed.sh`                                          | Async GRPO on the airline and retail official train domains |
| `scripts/tau2/train_airline.sh`, `train_retail.sh`, `train_telecom.sh` | The same GRPO recipe restricted to one domain               |
| `scripts/tau2/train_critic.sh`                                         | Online mixed critic training with fresh episodes each epoch |
| `scripts/tau2/run.sh qualification mixed`                              | Historical three-episode SAO integration recipe             |
| `scripts/tau2/run.sh policy {mixed,airline,retail} sao`                | Formal SAO recipe using the native async evaluator          |

GRPO uses 4 training GPUs, 3 rollout GPUs, and 1 dedicated checkpoint-evaluation GPU. A
full training batch is 8 prompts x 8 trajectories = 64 episodes. The last epoch batch
may be smaller but always retains complete groups of eight. The default is two passes
over the 104 official airline and retail train tasks (26 updates). Telecom is excluded
from the default mix: the initial policy already solves every Telecom episode, so it
adds no GRPO group contrast. Explicit `domains=[...]` overrides still accept it. It uses
non-thinking, 32768 total context, at most4096 tokens per response, policy temperature1,
staleness2, recomputed logprobs, decoupled loss, group reward normalization and the
existing token ratio mask above5. It does not load a critic. Saving, recovery
checkpoints and dedicated evaluation run every 5 optimizer steps plus the final step
(`saver`, `recover` and `evaluator` `freq_steps: 5`). The math GRPO recipe used 20
steps; the τ² train set is small enough (26 updates over two epochs) that 20 would leave
a single intermediate checkpoint. With `eval_before_train=true`, the initial evaluation
uses the starting model and consumes the initial evaluator trigger before training;
step1 does not request an unsaved checkpoint.

SAO uses the same 4/3/1 GPU split, one trajectory per prompt, batch64, staleness2,
TMS-enabled actor/critic colocation, a restored scalar critic from `TAU2_CRITIC_PATH`,
and frozen critic attention. The mixed formal schedule is explicit in the config: 240
airline and 592 retail episodes per epoch. Two epochs give 1664 total episodes and 26
updates. The critic must be trained on the same airline/retail task mix. Single-domain
SAO entrypoints prune only the unselected domain budget; the SAO recipe has no Telecom
budget; `train.py` derives `effective_episodes` from the resolved schedule after
overrides, task limits, epochs and partial `total_train_steps`. SAO saves,
recovery-checkpoints, and dedicated evaluation run every5 optimizer steps plus the final
forced checkpoint/evaluation. The SAO policy recipe defaults W&B online; the inherited
GRPO and critic recipes remain offline by default unless `TAU2_WANDB_MODE` overrides
them.

`experiment_mode=tune` selects83train/21dev, stratified by task and domain with seed42.
`experiment_mode=formal` trains on 104 official train and evaluates 60 official test
tasks (20 airline, 40 retail), four repeats each, with fixed settings; test results
never select a checkpoint. The dedicated evaluator reports per-domain coverage and
rewards, and rejects missing/infra-failed episodes instead of reporting a partial mean.

Policy training uses temperature 1; new checkpoint evaluations use temperature 0. The
standalone `scripts/tau2/evaluate.py` entrypoint defaults to `--temperature 0` even when
loading a historical resolved training config. Use `--temperature 1` explicitly to
reproduce the old sampling protocol. Report the temperature alongside the scores and
keep results from the two protocols separate.

`evaluate.py --task-set test` (default) scores the official test split only, the
held-out set for trained checkpoints. `--task-set base` scores the official leaderboard
set, train plus test (50 airline and 114 retail tasks when Telecom is excluded). Its
summary reports `by_domain_split` so the held-out test part stays visible. Base scores
for trained checkpoints include tasks seen during training.

Positional Hydra overrides after the flags are applied to `--config`, for example
`econfig.user_llm=<model>` to evaluate with a different user simulator. The output root
records the user model, endpoint and arguments in `manifest.json`, and per-episode
user-simulator prompt/completion tokens in `user-usage.jsonl`; `summary.json` reports
their totals.

Critic formal training uses the 104 official airline/retail train tasks and fixed
official test60 trajectories. Tuning uses83train/21dev (24/59 train and6/15dev). Each of
two epochs samples a fresh episode for each training task, in the same task order and
batch membership. Each batch waits for its complete ordered results, then updates the
critic once. Batch16 retains the tail: formal has208real training episodes, 14updates
and an8-task tail; tune has166episodes,12updates and a3-task tail. The actor remains
fixed. Physical tail replication preserves equal weighting and is not counted as
additional sampling.

Validation trajectories are collected once and reused at step0, every5steps and final;
checkpoints are saved every5steps and final. Formal test metrics never select a
checkpoint or tune parameters. Qwen3.5-4B initializes a fresh scalar head with frozen
attention. LR5e-6, constant/no-warmup schedule and token-mean plain MSE retain the
agreed critic recipe. Overall and per-domain metrics include MSE, explained variance and
gradient norm; validation diagnostics do not update parameters. EV is undefined for
constant targets. Completing training is distinct from qualifying critic quality.

The previous collect-once/offline-train replay recipe is retired. Old episode dumps and
checkpoints are historical evidence, not inputs to the current training loop. Resource
candidate:4training GPUs with actor/critic colocation and phase offload, plus4rollout
GPUs. Validation reuses the training critic. This candidate still needs native32K, tail,
save and recovery verification before capacity claims.

## Runtime and configuration

Formal critic and GRPO recipes record W&B locally by default, using `areal-tau2-critic`
and `areal-tau2-grpo`. Formal SAO uses `areal-tau2-sao` and defaults online because it
is an explicit policy recipe. Authentication uses the existing W&B login; never put API
keys in these recipes. Override `TAU2_WANDB_ENTITY`, `TAU2_WANDB_PROJECT`, or
`TAU2_WANDB_MODE` when needed. The runtime wrapper overrides the shared SAO
environment's disabled mode with the selected τ² mode. Offline critic/GRPO runs can be
uploaded with `scripts/tau2/upload_metrics.py`; offline SAO W&B runs should be synced
with W&B's native `wandb sync` flow.

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
model snapshot and loads no provider credential. `AREAL_PYTHON` may point this CPU check
at a compatible interpreter when the worktree `.venv` is absent. It is not a full GPU
runtime override: `scripts/sao/runtime_env.sh`, scheduler worker commands, PATH, and the
repo-local environment still have to be qualified before launch. Ordinary overrides are
forwarded to the existing config parser. Run directories are stable so
`recover.mode=auto` can resume the same run. Qualification mode explicitly creates a new
temporary attempt directory.

Use `scripts/tau2/train_critic.sh --check-config` to inspect the online critic recipe,
and `scripts/tau2/train_critic.sh` to run it. The entrypoint loads the same DeepSeek
credentials as policy training. It does not accept sealed training episodes as a
substitute for new epoch sampling. Resume uses the same run directory and native
recovery state, preserving the fixed validation corpus and the task split.

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

## Design decisions

The reasons behind the harness, generation protocol, data/evaluation protocol, critic,
GRPO and SAO choices above are recorded in
[docs/decisions/](../../docs/decisions/README.md). `eval_matrix.py` plans/verifies
matrix coverage; it does not run a scientific comparison.

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

The τ² recipes used by the entrypoints above are `config_grpo.yaml`, `config_sao.yaml`,
`config_sao_qualification.yaml` and `config_critic_production.yaml` (Qwen3.5-4B, one
8-GPU node). The four upstream generic configurations below are kept for reference:

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

Update the `econfig.user_llm_base_url` in your config to point to this server. The τ²
recipes in this fork instead use a hosted DeepSeek user simulator whose key is loaded
from `TAU2_DEEPSEEK_ENV_FILE`.

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

| Option                           | Default            | Description                                                                     |
| -------------------------------- | ------------------ | ------------------------------------------------------------------------------- |
| `econfig.domain`                 | `airline`          | Tau2 domain: `airline`, `retail`, `telecom`, or the shared launcher's `mixed`   |
| `econfig.max_steps`              | `100`              | Official orchestrator step limit per episode; the τ² recipes set `200`          |
| `econfig.add_thinking_tool`      | `false`            | Whether to use thinking as a tool for the agent                                 |
| `econfig.enable_thinking`        | `false`            | Policy chat-template thinking; `true` is rejected                               |
| `econfig.context_window_tokens`  | `32768`            | Rendered prompt plus completion budget; other values are rejected               |
| `econfig.max_completion_tokens`  | `4096`             | Tokens per assistant response, at most 4096                                     |
| `econfig.solo_mode`              | `false`            | If true, agent handles both agent and user roles (no user simulator needed)     |
| `econfig.user_llm_base_url`      | `null`             | Base URL of the user simulator LLM server                                       |
| `econfig.user_llm`               | `null`             | Model name for user simulator (e.g., `openai/self-hosted-Qwen2.5-72B`)          |
| `econfig.user_llm_api_key_env`   | `DEEPSEEK_API_KEY` | Environment variable holding the user simulator key                             |
| `econfig.user_llm_args`          | `null`             | Arguments for user LLM (e.g., `{temperature: 0.0, max_completion_tokens: 512}`) |
| `econfig.user_llm_num_retries`   | `3`                | Maximum retries for one user-simulator request                                  |
| `econfig.turn_discount`          | `1.0`              | Discount factor for turn-based learning                                         |
| `econfig.invalid_format_penalty` | `0.1`              | Penalty for invalid format in completions                                       |

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

1. **Tree training**: `config_1.7b_airline.yaml` enables `enable_tree_training=true`
   (the τ² recipes leave it off), which optimizes training by sharing prefix
   computations across rollouts with the same prompt. This option can largely accelerate
   training but will possibly increase GPU memory usage if
   `actor.mb_spec.max_tokens_per_mb` is large. And this setting may cause instability
   during the training of the MoE model.

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

GRPO consumes finite epochs. For the two-prompt tail on four training ranks, complete
prompt groups are uniformly replicated for physical dispatch only; no extra episodes are
sampled. This repeats both the numerator and denominator of the token-mean loss, so the
gradient is unchanged. `tau2_batch/real_prompts` and `real_episodes` measure coverage;
worker sequence/token counters describe physical computation. The normal batch remains
eight prompts times eight samples, and the real tail is sixteen episodes.

### Empty simulator responses

Training and evaluation both retry an empty user-simulator response once, with the same
request and history. A tool-only response is valid. The retry occurs before any returned
user tool call is executed, so policy actions and tools are not replayed. Existing
provider transport retries remain unchanged. Two consecutive empty responses raise an
infrastructure error; they do not become reward zero.

Failed simulator requests and the recovery response are written under
`$TAU2_RUN_ROOT/tau2-user-failures/` (or `./tau2-user-failures/` when unset), with a
unique incident ID and attempt number. The JSON includes the full conversation, tool
schemas, generation arguments, raw response (including reasoning, response ID and finish
reason), and error body. Authentication arguments are excluded. Debug-write failures
emit a warning and do not prevent the one recovery attempt.
