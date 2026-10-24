<!-- Go-to brief for AI coding agents working on AReaL. -->

# AGENTS.md -- AReaL Agent Operations Guide

## Research fork `codex/sao-math`

This branch is a research fork, not an upstream contribution. The rules in this section
apply to every agent (Codex on phai, Claude Code on the local clone) and take precedence
over the upstream AReaL guidance that follows.

Topic: interactive RL on τ²-bench with AReaL (PPO with a privileged, script-aware
critic; user-simulator effects; offline checkpoint evaluation). Current state and
conclusions: `docs/research/tau2/README.md`.

### Who decides what (user rule, 2026-10-24)

The bottleneck is the user's understanding, not execution speed: a run or a design the
user did not understand has cost hours to half a day. The user designs the method; the
agent explains, implements and runs what the user approved.

- **Method choices are the user's.** Reward, advantage, loss, critic inputs, new
  components, non-default hyperparameters, and any deviation from the paper or the
  standard PPO/GAE implementation (clip, minibatches, value clip, head lr). Default to
  the paper / standard configuration. Present options with trade-offs and a
  recommendation; do not implement or launch until the user explicitly agrees.
- **Run card before every GPU run, eval or paid API job**, then wait for the user's go:
  the question it answers; the one variable that differs from the reference run; the
  expected result and which result would be uninformative; time, GPU and API cost; the
  stop condition. If no outcome would change a decision, do not propose the run.
- **Explain before asking for agreement:** the goal in one plain sentence; every symbol
  defined; one numeric example; how it differs from standard practice and why. Mark what
  was read versus inferred. If the user does not understand, restart from the
  prerequisite with an example; do not add new terms.
- **Report results in layers:** observed / inferred / unverified, and state the metric
  definition every time (EV definition, mean@k vs pass^k, temperature, which episodes
  were excluded and why).
- **Do not over-ask either.** Infra fixes, cleanup, smoke tests, monitoring and
  documentation: act and record. At most two questions per message, each with a
  recommended answer.

### Branches, worktrees and runs

- All work happens on phai. The main worktree is
  `/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math` on
  `codex/sao-math`; each task gets a branch `codex/<topic>` in its own worktree
  `_worktrees/AReaL/<topic>`, which is merged back into `codex/sao-math`. After the
  merge, remove the worktree and delete the branch.
- Never push to GitHub and never open PRs. The branch's GitHub upstream is a public
  fork; internal research content must not go there. The upstream PR workflow, the
  `create-pr` / `review-pr` skills and "squash before PR" do not apply.
- Run pre-commit on the changed files only (`pre-commit run --files <changed>`), not
  `--all-files`, which would reformat unrelated upstream code.
- Run and test only on phai. Every `pueue add` passes `-w <the task's own artifact dir>`
  and launch scripts `cd` there before starting Python: pueue runs the task in that
  directory and Python puts it on `sys.path`, so a stray `/tmp/grp.py` once shadowed the
  stdlib (pueue 653).
- Run outputs go to
  `/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/<worktree>/`, one directory
  per run; never into the repository or `/tmp`.
- Status scripts read the tail of logs and only the newest trajectory steps; one that
  re-parsed every trajectory JSON took 42 s per call.

### LLM APIs for users and judges

- Models: DeepSeek only, as `deepseek-flash` on the official API (preferred: cache hits
  make it much cheaper) or `deepseek-v4-1-flash-261017` on the Volcano plan. Never
  `deepseek-v4-pro`, and never switch models or endpoints without the user's approval.
- Keys live under `/data_storage/yl_test/lgx/home/.config/tau2-doubao/` (mode 600; never
  print values): `deepseek-official.env` (`DEEPSEEK_API_KEY`, shared balance with
  phai-relay, so check `GET https://api.deepseek.com/user/balance` before and after
  large jobs) and the Volcano plan files `ark-plan-20261023.env` (current) and
  `ark-plan-20261020.env` (`ARK_PLAN_API_KEY`, endpoint
  `https://ark.cn-beijing.volces.com/api/plan/v3`, weekly quota shared with training).
  Never use `phai-relay-ark-colleague.env`; never use the Volcano pay-as-you-go
  endpoint.
- Quota errors (`AccountQuotaExceeded`, balance exhausted) stop the job; no unbounded
  retries.
- Providers can change the model behind a name. Record the returned model name
  (`raw_data.model`) and stop if it changes.

______________________________________________________________________

## Quick reference

**Tech stack**: Python 3.12+ | PyTorch | FSDP2 / Megatron / Archon | SGLang / vLLM

```bash
# Environment
uv sync --extra cuda            # CUDA + SGLang inference (default); for vLLM: cp pyproject.vllm.toml pyproject.toml && cp uv.vllm.lock uv.lock && uv sync --extra cuda
uv sync --extra sandbox         # Daytona cloud sandbox backend (optional)
source .venv/bin/activate        # activate venv BEFORE pre-commit or git commit if venv exists
pre-commit install --install-hooks  # hooks: Ruff, clang-format, mdformat, nbstripout, conventional-commits
pre-commit run --files <changed files>  # this fork: changed files only, never --all-files

# Tests
uv run pytest tests/test_<topic>.py

# CLI docs
uv run python docs/generate_cli_docs.py

# Docs build (canonical, release-aligned)
./docs/build_all.sh
# Do NOT use `jupyter-book build docs/en|docs/zh` directly for final preview/release,
# because it skips AReaL-specific static setup and output packaging.
```

**Hard rules** -- never violate:

- No wildcard imports (`from x import *`).
- No hardcoded secrets, paths, or endpoints.
- No skipping pre-commit hooks.
- No guessing cluster configs or rebuilding CUDA/driver stacks.
- Integration tests require multi-node hardware -- explain skips explicitly.

**Always do**:

- Read relevant files before modifying code.
- Run pre-commit on the changed files before committing.
- Follow existing code patterns in the same module.
- Add tests for new functionality.
- Ask for decisions and clarifications with short, structured options instead of broad
  open-ended questions. Use the platform's native question/clarification tool if
  available.

**Ask first** before:

- Modifying config structures in `areal/api/cli_args.py`.
- Adding new dependencies.
- Changing launcher or scheduler logic.
- Deleting or renaming public APIs.

When unsure, leave a `TODO(agent)` comment and note the constraint in your response.

______________________________________________________________________

## Repository map

```
areal/                     Core Python package
|-- api/                   Config dataclasses, contracts, IO structs
|-- dataset/               Stateful dataset loaders (GSM8K, Geometry3K, CLEVR, ...)
|-- engine/                Training backends (FSDP2, Megatron) + inference adapters
|-- experimental/          Prototype engines/workflows (Archon MoE engine)
|-- infra/                 Launchers (Local/Ray/Slurm), schedulers, utilities
|-- models/                Model adapters (Megatron-Core, Transformers, custom heads)
|-- reward/                Built-in reward functions + math parsers
|-- tests/                 Unit/integration test suites
|-- trainer/               High-level orchestrators (PPOTrainer, SFTTrainer)
|-- utils/                 Cross-cutting helpers (logging, data, checkpoints, RL ops)
+-- workflow/              RolloutWorkflow implementations (RLVR, multi-turn, vision)

docs/                      Jupyter Book docs (https://areal-project.github.io/AReaL/)
examples/                  Training scripts and launcher recipes
```

______________________________________________________________________

## Code style & patterns

- **Composition over inheritance** -- keep hierarchies \<= 2 levels; prefer delegation.

| Type             | Pattern         | Example                                   |
| ---------------- | --------------- | ----------------------------------------- |
| Config dataclass | `XxxConfig`     | `GRPOConfig`, `FSDPEngineConfig`          |
| Engine class     | `XxxEngine`     | `FSDPEngine`, `ArchonEngine`              |
| Workflow class   | `XxxWorkflow`   | `RLVRWorkflow`, `MultiTurnWorkflow`       |
| Reward function  | `xxx_reward_fn` | `gsm8k_reward_fn`, `geometry3k_reward_fn` |

**Logging**: `areal.utils.logging.getLogger(name)` with **PascalCase** names -- never
`print` or `logging.__name__`. Per-rank format: `[{Component} Rank {N}]`. Register new
loggers with color in `areal/utils/logging.py`.

**Performance**:

- No GPU-CPU sync in hot paths (`.item()`, `.tolist()`, `print(tensor)`).
- Batch ops over Python loops on tensor elements.
- Explicit `dtype`/`device`; `torch.Size` assertions for shape validation.

**Typing & imports**: explicit type hints; reuse `areal/api/cli_args.py` dataclasses; no
wildcard imports; heavy optional deps inside functions.

**Async**: rollout workflows must stay non-blocking (`await` + `aiofiles`); no sync I/O
in `arun_episode`.

______________________________________________________________________

## Domain experts & skills

Fire the appropriate **expert subagent** or **load a skill** based on what you're
working on. Experts are read-only consultants with deep domain knowledge; skills are
step-by-step implementation guides.

| Working on...                | Fire subagent      | Load skill          |
| ---------------------------- | ------------------ | ------------------- |
| FSDP engine code             | `fsdp-expert`      | --                  |
| Archon engine / new model    | `archon-expert`    | `add-archon-model`  |
| Megatron engine code         | `megatron-expert`  | --                  |
| RL algorithms / PPO / GRPO   | `algorithm-expert` | --                  |
| Launcher / scheduler / infra | `launcher-expert`  | `debug-distributed` |
| New reward function          | --                 | `add-reward`        |
| New dataset loader           | --                 | `add-dataset`       |
| New rollout workflow         | --                 | `add-workflow`      |
| Unit tests                   | --                 | `add-unit-tests`    |
| Distributed debugging        | --                 | `debug-distributed` |

**How to invoke experts and skills** (platform-specific):

| Platform | Fire expert subagent                                                               | Load skill                                         |
| -------- | ---------------------------------------------------------------------------------- | -------------------------------------------------- |
| OpenCode | `task(subagent_type="<name>", load_skills=[], run_in_background=true, prompt="…")` | `skill(name="<name>")` or `load_skills=["<name>"]` |
| Codex    | Invoke registered subagent by canonical name (see `.codex/config.toml`)            | Reference `.agents/skills/<name>/SKILL.md`         |

**Harness layout**:

| Component         | OpenCode                                | Codex                                                  |
| ----------------- | --------------------------------------- | ------------------------------------------------------ |
| Root instructions | `AGENTS.md`                             | `AGENTS.md`                                            |
| Agent configs     | `.opencode/agents/*.md` (frontmatter)   | `.codex/config.toml` + `.codex/agents/*.toml` + `*.md` |
| Skills            | `.opencode/skills/` + `.agents/skills/` | `.agents/skills/<name>/SKILL.md`                       |

Directly executable workflows (both platforms): `add-workflow`, `translate-doc-zh`. The
`create-pr` and `review-pr` skills are upstream PR tooling and are not used on this
fork.

______________________________________________________________________

## Core concepts

**Trainer** orchestrator (`areal/trainer/`, `PPOTrainer`, `SFTTrainer`): manages the
training loop, dataset loading, and workflow execution. Entry point:
`examples/math/gsm8k_rl.py`.

**Rollout workflows** (`areal/workflow/`, `RolloutWorkflow.arun_episode`): define how
episodes are generated. Use `add-workflow` skill for step-by-step guide.

**Engines**: *Inference engines* handle async generation via `engine.agenerate()` and
manage weight updates. *Training engines* consume rollout tensors, compute PPO/GRPO
updates, and broadcast weight versions (FSDP2, Megatron, or Archon).

**Weight versioning**: async workflows require version alignment via `WeightUpdateMeta`
(`areal/api/engine_api.py`). Critical for correctness across distributed training.

**Observability**: emit metrics via `stats_tracker.get()`, persist artifacts under
`dump_dir`, checkpoint via `areal/utils/saver.py` / `recover.py`.

**Launcher / scheduler**: training requires cluster setup (local / Ray / Slurm) via
configs in `areal/infra/launcher/`. See `launcher-expert` for deployment guidance.

______________________________________________________________________

## API & config rules

*Applies to: `areal/api/**`*

- **Field ordering**: required -> common optional -> rare optional -> internal (`_`
  prefix).
- **Validation**: `__post_init__` with `ValueError` and clear message.
- **Backward compat**: add fields with defaults; deprecate before removing; avoid type
  changes.
- **CLI**: use `Literal` for enum choices; all public configs need docstrings with
  constraints.

______________________________________________________________________

## Distributed code rules

*Applies to: `areal/engine/**`, `areal/experimental/**`*

- Never create global process groups at module level; always pass `process_group`
  explicitly.
- `dist.get_rank(group)` not `dist.get_rank()` when group matters.
- DeviceMesh dimensions must match `ArchonParallelDims`: `dp_shard`, `tp`, `cp`, `ep`,
  `etp`.
- All-reduce: all ranks must call. Broadcast: explicit `src`. Barrier: debugging only.

| Issue         | Cause                            | Fix                       |
| ------------- | -------------------------------- | ------------------------- |
| Hang          | Mismatched collective calls      | All ranks call same op    |
| Wrong results | Incorrect `ReduceOp`             | Check SUM vs MEAN         |
| OOM           | Unsharded tensor on wrong device | Verify DTensor placements |

Debug env vars: `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `NCCL_DEBUG=INFO`,
`CUDA_LAUNCH_BLOCKING=1`. See the `debug-distributed` skill for the full workflow.

______________________________________________________________________

## Testing rules

*Applies to: `**/tests/**`, `test_*.py`*

| Marker                                  | When                             |
| --------------------------------------- | -------------------------------- |
| `@pytest.mark.slow`                     | > 10s (excluded from default CI) |
| `@pytest.mark.slow` + `@pytest.mark.ci` | Slow but must run in CI          |
| `@pytest.mark.asyncio`                  | Async tests                      |

- Naming: `test_<what>_<condition>_<expected>()` with Arrange/Act/Assert.
- GPU: skip gracefully (`@pytest.mark.skipif(not CUDA_AVAILABLE, reason="...")`).
- Distributed mocking: `torch.distributed.fake_pg`; don't mock FSDP/DTensor internals.
- Assertions: `torch.testing.assert_close()` with explicit `rtol`/`atol`; prefer
  `tmp_path`, `monkeypatch`.

| Suite       | Command                       | GPU       |
| ----------- | ----------------------------- | --------- |
| Unit        | `pytest tests/test_*.py`      | No        |
| GRPO        | `pytest tests/grpo/`          | Yes       |
| FSDP        | `pytest tests/test_fsdp_*.py` | Yes       |
| Distributed | `pytest tests/torchrun/`      | Multi-GPU |

______________________________________________________________________

## Collaboration & review

- **Branches**: `codex/<topic>` in their own worktree, merged into `codex/sao-math` (see
  the research section). No PRs and no GitHub pushes.
- **Commits**: Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `style:`), ~72 char
  subject, imperative voice; load the `commit-conventions` skill before `git commit`.
- **Pre-merge**: pre-commit on the changed files; doc-only edits need at least mdformat.

______________________________________________________________________

## Reference material

- **Docs portal**: <https://areal-project.github.io/AReaL/>
- **Quickstart**: `docs/tutorial/quickstart.md`
- **Architecture**: `docs/tutorial/gsm8k_grpo.md`
- **Customization**: `docs/customization/*.md`
- **Algorithms**: `docs/algorithms/*.md`
- **Best practices**: `docs/best_practices/*.md`
- **CLI reference**: `docs/cli_reference.md`
- **Agent workflow**: `docs/customization/agent.md`
