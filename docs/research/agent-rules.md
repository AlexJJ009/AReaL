# Research fork rules (`codex/sao-math`)

Rules for every agent on this branch: Codex on phai and Claude Code on the local clone.
They override the upstream AReaL guidance in `AGENTS.md` / `CLAUDE.md` where the two
disagree.

The branch is a research fork, not an upstream contribution: interactive RL on τ²-bench
with AReaL (PPO with a privileged, script-aware critic; user-simulator effects; offline
checkpoint evaluation). Current state and conclusions: `docs/research/tau2/README.md`.

## Who decides what (user rule, 2026-10-24)

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

## Repository and runs

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

## LLM APIs for users and judges

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
