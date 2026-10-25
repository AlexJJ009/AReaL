# Offline PPO checkpoint queue

`checkpoint_eval.py` evaluates saved actor HF checkpoints with one Qwen3.5-4B user
simulator on GPU 7 and up to seven policy evaluators on GPUs 0–6. Run this queue after
training releases the eight GPUs. The runner does not reserve GPUs against other jobs;
the caller's `gpu8` queue group must provide that exclusion.

Use the Python environment already installed for the training run. No extra dependencies
are required. Describe the host paths in a **nonsecret** JSON file:

```json
{
  "python": "/absolute/venv/bin/python",
  "hf_hub_cache": "/absolute/huggingface/hub",
  "sao_artifact_root": "/absolute/shared-runtime-artifacts",
  "tau2_data_dir": "/absolute/fixed-tau2-data",
  "ark_plan_env_file": "/absolute/ark-plan-20261020.env"
}
```

The key file must have mode `0600` and export `ARK_PLAN_API_KEY`. It is sourced only
inside evaluator subprocesses during execution. The runner uses the Volcano plan
endpoint and `deepseek-v4-1-flash-261017`. It does not load another provider's key file.
The user server needs no judge credential. Both server kinds source
`scripts/sao/runtime_env.sh` with the supplied SAO runtime root and venv cuRAND headers,
matching the training environment.

Preview the full plan without starting processes, loading model code, sourcing runtime
or key files, probing ports, or creating output directories:

```bash
CUDA_VISIBLE_DEVICES='' /absolute/venv/bin/python scripts/tau2/checkpoint_eval.py \
  /absolute/run-directory --runtime /absolute/runtime.json --dry-run
```

Alternatively pass a trial name and `--artifact-root /absolute/run-parent`. The default
config is `RUN/logs/root/tau2-ppo/TRIAL/config.yaml`; `--config` can select another
resolved YAML with the same trial. Planning reads it with `yaml.safe_load`, without
calling `load_expr_config`. The base actor comes from its `actor.path`: a pinned
`repo@revision` resolves strictly inside the supplied local HF cache. No model is
downloaded.

The default completed-step list is `0,10,20,...,120,126`. Step 0 is the base actor; step
N selects `globalstep<N-1>`. With nine steps per epoch, step 10 selects
`default/epoch1epochstep0globalstep9`. `--steps`, `--steps-per-epoch`, and
`--final-step` select another schedule explicitly.

To execute, remove `--dry-run`. `--workers 1..7` limits GPU slots. Each slot runs its
assigned checkpoints serially, so a failed policy is stopped before another attempt
starts on that GPU. Default ports are:

| Role                 | HTTP  | NCCL  | AReaL RPC, callback and proxy pool (inclusive) |
| -------------------- | ----- | ----- | ---------------------------------------------- |
| GPU 0 policy         | 21000 | 21001 | 21002–21099                                    |
| GPU 1 policy         | 21100 | 21101 | 21102–21199                                    |
| GPU 2 policy         | 21200 | 21201 | 21202–21299                                    |
| GPU 3 policy         | 21300 | 21301 | 21302–21399                                    |
| GPU 4 policy         | 21400 | 21401 | 21402–21499                                    |
| GPU 5 policy         | 21500 | 21501 | 21502–21599                                    |
| GPU 6 policy         | 21600 | 21601 | 21602–21699                                    |
| GPU 7 user simulator | 40080 | 40081 | —                                              |

`--port-base`, `--user-port`, and `--user-nccl-port` can move these allocations. RPC
workers inherit `AREAL_PORT_RANGE=min:max`; normal AReaL runs without this variable
retain the original port selection behavior. The runner checks all planned TCP/UDP ports
before startup, waiting up to 180 seconds for a previous run to release them.
`--port-wait-timeout SECONDS` changes this bound. Each policy attempt also waits for its
entire slot before starting, including retries. Waiting responds to cancellation; a
timeout names the occupied ports and protocols. Another application can still claim a
port after that check; startup failure is reported, with no fallback to a shared policy
port.

Every evaluation uses `--repeats 4 --temperature 0`, `qwen25` policy tool parsing, and a
900-second episode timeout. `--task-set ppo` (default) is the fixed 100-task set over
airline, retail and telecom; `--task-set test` is the fixed test split of the run
config's `domains`. For a run trained on airline and retail it is the same 60 tasks as
the ppo set without telecom (checked 2026-10-25 on `run3p-std-beta1-notel-c128`). Policy radix cache
is enabled. `--judge-concurrency` caps concurrent episodes per process (default 1),
hence at most seven sequential NL judge streams by default. Increasing it also increases
user-simulator and judge traffic.

The shared user simulator uses the pinned Qwen3.5-4B snapshot, served name
`qwen35-4b-user`, memory fraction 0.75, `qwen3` reasoning parsing and `qwen3_coder` tool
parsing. User requests use temperature 0, top-p 1, 4096 output tokens and
`enable_thinking=false`. A supervisor restarts a crashed user server after five seconds.
Policy episodes retain the evaluator's existing infra retry handling during an outage.

Outputs default to `RUN/offline-checkpoint-eval/`:

- `plan.json`, `summary.json`, and `summary.md` describe the entire queue.
- `user-server.log` records the single shared server.
- `step-NNN/attempt-NNN/` preserves each evaluator's manifest, summary, episodes,
  trajectories, process log and runner receipt. A retry uses a fresh directory.
- Log/fileroot/name-resolve overrides point into each attempt **before** config loading,
  because `load_expr_config` writes a config snapshot while loading. It cannot overwrite
  the original training config through these write paths.

A checkpoint is finished only after evaluator exit code 0, a complete 400-episode PPO
summary (scored plus infra-excluded), and a runner success receipt matching the config
bytes, model path, evaluation settings, runtime paths and runner/evaluator code.
Partial, failed, check-config, or mismatched outputs are never skipped as finished.
Successful checkpoints need not still exist on disk to appear in the summary. At most
two attempts per checkpoint/protocol are allowed across restarts. To intentionally
repeat an exhausted evaluation, choose a fresh `--output-root`; failed evidence is
preserved.

Missing or incomplete HF checkpoints are reported immediately by default; this makes the
overall exit code nonzero while other checkpoints continue. `--wait-for-checkpoints`
enables polling, with `--wait-timeout` seconds **per checkpoint** (default 86400).
Readiness checks JSON metadata, tokenizer presence and all weight-index shards without
loading weights. It does not certify weight integrity or provide an atomic training-save
completion marker.

The JSON summary retains the evaluator's full per-domain metrics, including
`pass_hat_k_tasks` denominators. Markdown shows mean@4, pass^1 through pass^4, episode
totals, infra-excluded and context-limit counts. Infra errors remain excluded from
scores. A missing pass^k denominator displays `—`, never a made-up zero. Partial
attempts retain their evaluator summaries but have no final score in the queue table.

The runner locks its output directory against duplicate invocations. User and evaluator
subprocesses each get an owned process session and PGID file. On completion, exception,
TERM, INT or HUP, it sends TERM to the entire session, then KILL after a grace period.
TERM from `pueue kill` follows this path. SIGKILL and host failure cannot run cleanup
handlers. Reusing an output directory does not kill a stale process based solely on an
old PGID file.

## DeepSeek official backend

`--user-backend deepseek-official` replaces the local user and the Volcano judge with
the DeepSeek official API: user and judge are both `deepseek-flash` at
`https://api.deepseek.com`, temperature 0, in the model's default (thinking) mode, as in
the PPO runs before 2026-10-17. No user server starts, so `--workers` accepts 1..8 and
GPU 7 runs a policy. The runtime JSON replaces `ark_plan_env_file` with
`deepseek_env_file`, a mode-`0600` file that exports `DEEPSEEK_API_KEY`. The
`deepseek-flash` alias can change model versions; record the returned model name.

`--checkpoint LABEL=/absolute/hf/path` (repeatable) adds checkpoints from other runs,
evaluated with this run's config and protocol and written to `ckpt-LABEL/`. Use it to
compare runs under one user and judge backend instead of mixing their old configs.

A provider outage or exhausted balance turns episodes into infra exclusions, and a
checkpoint still counts as finished when scored plus excluded episodes reach 400. Check
`infra_excluded` in `summary.md` before using a score.

CPU verification uses mocks for policy/user startup plus real sleeping CPU process trees
for signal cleanup; it never loads weights or calls a judge:

```bash
CUDA_VISIBLE_DEVICES='' LITELLM_LOCAL_MODEL_COST_MAP=True \
  /absolute/venv/bin/python -m pytest -q \
  tests/test_tau2_checkpoint_runner.py tests/test_tau2_eval_ports.py \
  tests/test_tau2_eval_policy_ports.py
```
