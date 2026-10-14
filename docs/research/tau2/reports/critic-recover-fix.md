---
task_id: 20261014T032515Z-tau2-critic-recovery-34b395ac
report_id: 20261014T034550Z-final-71c659
kind: final
workspace: /data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-critic-recover
generated_at: '2026-10-14T03:45:50Z'
window_start: '2026-10-14T03:45:50Z'
window_end: '2026-10-14T03:45:50Z'
---

# Tau2 critic-only recovery fix

## Result and intended use {#goal}

The two requested recovery fixes are implemented and CPU-validated: τ² critic-only
recovery restores its checkpoints and native trainer state without calling
actor-to-inference `update_weights`, and resumed epochs use the configured shuffle seed.
**217 tests passed; all ten deliberate mutations failed as intended.** Changes remain
uncommitted in `codex/critic-recover-fix`, based on `315617b9`.

The original `critic-head-qwen35-mix` run needs one additional launch preparation:
install the verified fixed-validation manifest candidate described below. Its tensor
cache is unchanged, but the strict manifest includes source-file hashes, so using the
patched source with the old manifest would be rejected. No original run artifacts were
modified, and no GPU job, pueue task or DeepSeek call was launched. The GPU resume smoke
remains for the user to queue.

## Changed files and restored state {#progress}

| File                                                                                                                                                 | Change and purpose                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [scripts/tau2/train_critic.py](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-critic-recover/scripts/tau2/train_critic.py)             | Before worker startup, require actor LR zero, an explicit positive step limit entirely inside the critic-only window, a critic, and no SAO actor-update rounds. Opt into skipping the recovery transfer. Set sampler seed on both fresh and recovered initialization. |
| [areal/trainer/rl_trainer.py](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-critic-recover/areal/trainer/rl_trainer.py)               | Add the private initializer keyword `skip_recovery_weight_update=False` and forward it to recovery. Ordinary PPO/GRPO/SAO callers retain the default.                                                                                                                 |
| [areal/utils/recover.py](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-critic-recover/areal/utils/recover.py)                         | Add `skip_inference_weight_update=False`. Reject skipping unless recovered policy version is zero and optimizer roles are exactly `['critic']`. Bypass only the inference transfer protocol, preserving checkpoint loads and version setters.                         |
| [tests/test_tau2_critic_recovery.py](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-critic-recover/tests/test_tau2_critic_recovery.py) | Add 14 CPU cases covering real PPO initialization/flag forwarding, real recovery, Adam/scheduler/model restoration, recording transfer calls, unsafe requests and actual shuffled continuation.                                                                       |

The actor checkpoint is still loaded, with the existing model-only policy; critic
weights, Adam state and scheduler restore normally. Recovery still restores the
dataloader, saver, evaluator, statistics logger, recovery cadence and saved trainer
metadata. Actor, critic and inference version setters still run. Existing
staleness-manager and rollout-input recovery remain in the same initialization path.
Keeping the actor load preserves checkpoint validation and the existing state contract;
it also preserves final-summary expectations that both recovery checkpoints exist.

## Why skipping and seed restoration are safe {#decisions}

The wrapper establishes that this run cannot perform an actor optimizer step. Recovery
independently checks the saved version and the existing optimizer-role policy, so a
nonzero recovered policy or trainable-actor checkpoint cannot silently take the skip
path. The safety premise is the existing τ² contract: inference serves the same pinned
HF snapshot from which the frozen actor started. These guards validate training
history/configuration; they are not a new cryptographic comparison of every actor tensor
against HF. Keep the original actor snapshot and configuration when resuming. The
fixed-validation manifest checks the actor path as part of its unchanged non-source
payload.

For this run, the inspected checkpoint records `epoch=0`, `epoch_step=19`,
`global_step=19`, `steps_per_epoch=20`, `num_critic_only_steps=80`, `policy_version=0`,
and optimizer roles `['critic']`. It resumes at logical step 20, the start of epoch 1.
The step-20 metrics record zero skipped tasks. This directly satisfies the recovery
guards. The initial `connect_engine` call is retained: a weight-update group may still
be initialized, but recovery does not call `update_weights`.

A pure wrapper workaround that supplied no inference engine would also suppress the
existing version setters. The two small default-off core parameters let the wrapper
select the intended behavior without losing that restoration. The ordinary policy path
still loads and transfers exactly as before.

For shuffling, installed torchdata's `StatefulDataLoader.load_state_dict` clears the
iterator and stores `next_iter_state`; it does not fast-forward immediately.
Fast-forward happens at the next `iter(dataloader)`. Therefore setting
`sampler.seed=config.seed` after `super()._init_impl` is correct, provided it precedes
`_next_strict_rows`, as it now does. The saved epoch is retained: at an epoch boundary,
the restored exhausted iterator first raises `StopIteration`, and the strict wrapper
advances to the next epoch once. The tests compare remaining batches and later epochs
for resumes after 2, 4, 6 and 8 batches, with shuffled 13-row data, batch size 4 and a
tail batch. Production and tests use `num_workers=0`.

## Verification and failure evidence {#evidence}

The final command used this worktree's import path, the requested dataset directory and
hidden CUDA devices:

```bash
export PYTHONPATH=/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-critic-recover
export CUDA_VISIBLE_DEVICES=
export TAU2_DATA_DIR=/data_storage/yl_test/lgx/data-1/code/_cache/tau2-bench-b7ea9074c1cba482b30687fecdb5c8425fd6f619/data
.venv/bin/python -m pytest -q \
  tests/test_tau2_critic_recovery.py \
  tests/test_tau2_critic_training.py tests/test_tau2_critic_head.py \
  tests/test_tau2_sao_pack_training.py tests/test_tau2_task_pack.py \
  tests/test_tau2_sao_contract.py tests/test_sao_critic_head_guard.py \
  tests/test_recover.py tests/test_ppo_warmup_recovery.py \
  tests/test_rollout_recovery_inputs.py tests/test_sao_snapshot_recovery.py
```

[Final suite log](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/final-cpu-suite.log):
**217 passed, zero failures or skips, 54.16 seconds**. The 43 warnings concern the
existing Transformer Engine/Apex fallback and warmup-setting precedence.
[Verification record](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/final-verification.json)
confirms `areal.__file__` resolves inside the requested worktree and all three
production source hashes match after mutation restoration.
[Ruff/format/diff checks](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/final-lint.log)
pass.
[Independent code review](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/final-code-review.md)
found no blocking issue.

All three implementation files were temporarily restored to `315617b9`.
[Baseline log](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/baseline-red.log)
shows **5 failed, 1 passed**: the critic test fails on the recorded transfer call
itself, and all four sampler cases fail on actual row order. The policy case passes. The
request's requirement that test 2 both preserve old behavior and fail on unchanged old
code is contradictory; its valid negative control is deliberately disabling policy
transfer, which makes that test fail.

[Mutation results](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/mutation-results.json)
record ten isolated expected failures. Individual logs are in the artifact root:

| Deliberate break                            | Red log                         |
| ------------------------------------------- | ------------------------------- |
| Remove version-zero check                   | `guard-version-red.log`         |
| Remove critic-only optimizer-role check     | `guard-optimizer-roles-red.log` |
| Remove frozen-LR check                      | `guard-frozen-lr-red.log`       |
| Allow steps beyond critic-only window       | `guard-critic-window-red.log`   |
| Allow SAO actor updates                     | `guard-sao-red.log`             |
| Allow missing critic                        | `guard-critic-present-red.log`  |
| Allow zero training steps                   | `guard-positive-steps-red.log`  |
| Allow missing explicit step limit           | `guard-explicit-limit-red.log`  |
| Disable policy recovery transfer            | `policy-transfer-red.log`       |
| Drop the initializer's skip-flag forwarding | `trainer-forwarding-red.log`    |

[The mutation runner](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/run_mutations.py)
restores source bytes in `finally`; every targeted mutation exited with pytest code 1,
and the final complete suite ran after restoration. Development logs also retain fixture
mistakes and an initial broad invocation missing `TAU2_DATA_DIR` (16 missing-data
failures). Those are superseded by the correctly configured final run, not counted as
passing evidence.

The
[historical failure excerpt](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/historical-xccl-excerpt.log)
confirms successful step-20 metadata recovery at original log line 1160, followed by
illegal-memory-access watchdog failures in `update_weight_group` at lines 1165 and 1182
and SGLang distributed-update stacks. This establishes the unnecessary transfer path
involved in the crash, not the root cause of the CUDA error. No xccl implementation was
changed.

## Resume audit, cache compatibility and limits {#scope}

The implementation is bounded to the two requested fixes. `areal/api/cli_args.py`,
`areal/infra/**`, engine weight-sync code and other worktrees were not edited.
GPU/distributed integration was intentionally not run because the user prohibited it and
task 510 owns the GPUs.

| Resume concern                             | Finding and disposition                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_tau2_skipped`                            | Still resets on initialization and is not checkpointed. A mid-epoch restart can reset the rejection budget. Fixing this requires checkpoint-bound history and a legacy-checkpoint policy; reported, not changed. This run resumes at an epoch boundary and its saved metrics show zero skips.                             |
| `_tau2_next_train_step` and trace task IDs | Already starts at `last_step_info.next().global_step`; retained and tested. With batch size 16, this run restarts task IDs at 320. Recomputed postcheckpoint steps reuse their logical IDs, so retain attempt-specific audit logs.                                                                                        |
| Fixed validation                           | Source hashes invalidate the old manifest under patched code. Prepared and verified a specific replacement manifest; see below. No recollection or API calls.                                                                                                                                                             |
| Best checkpoint                            | In tune mode, `best-validation.json` can point beyond the restored checkpoint; missing metadata can label a recovered candidate `critic-initial`. These remain report-only defects requiring reconciliation with saved checkpoints. The actual run is formal and does not select a best checkpoint.                       |
| Postcheckpoint metrics                     | Successful replay overwrites per-step metrics and validation JSON for recomputed steps. Until replay finishes, files newer than the recovery checkpoint are evidence from a previous attempt, not committed current state. Preserve them separately before launching.                                                     |
| Final summary                              | Existing guards remain, including final-step, critic-only state, policy-version-zero and actor/critic checkpoint existence checks. A crash after the final recovery save but before final metrics can leave no work to replay and a missing final-metrics file; summary correctly fails rather than inventing completion. |
| Resume baseline statistics                 | The baseline validation JSON uses the recovered completed step, while its initial statistics commit still logs step zero. Reporting issue retained.                                                                                                                                                                       |

The historical source is `src-a88d480e`. Its ten recorded source hashes all match that
pinned source. Three differ from the current worktree: `train_critic.py`,
`examples/tau2/train.py`, and `examples/tau2/utils.py`. The latter two are pre-existing
branch changes: SAO entrypoint changes and a default-None `train_task_pack` field.
Relevant validation/dataset helpers have unchanged AST; the moved task-pack helper
produces the same 221 ordered rows at saved seed 42. No blanket source-hash exception
was added.

[Cache compatibility evidence](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/cache-compatibility.json)
and
[final recheck log](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/cache-recheck.log)
verify the complete non-source manifest payload, all 60 cached row identities/shapes and
the tensor SHA. The real cache loader accepted the candidate with collection explicitly
forbidden. Deliberately wrong manifest digest, tensor hash and tensor shape were
rejected. Historical manifest and tensor bytes stayed unchanged.

[Candidate manifest](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/fixed-validation-manifest.resume-candidate.json)
SHA-256: `abd502a99f6ad76e0552d61d7a5d043c1e15e9c2778d35ae0f4eb4bfb963b5e4`. Historical
manifest SHA-256: `c3a2c164159d998e2e8c795dad86730419a822a4292fbeb56d83d6c5a1bdd949`.
Tensor SHA-256 stays `2efe1f1a2746b3880c714199f9e7a15c69694c6bc2cd3692ceff42894dd7e19e`.
The
[offline reproducer](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/prepare_cache_candidate.py)
pins reviewed historical and target hashes. Recheck after any source edits, especially
when incorporating the separate xccl work; this candidate is not permission to accept
future source changes.

Reporting-tool limit: a real independent intent Judge ran and its exact decision was
passed to `report_runtime.py register`. Registration rejected the user-specified
external artifact task directory with `git_root_missing`;
[the failure is preserved](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-critic-recover-fix/report-registration-result.json).
The requested external report location is retained. No automatic reporting obligation
registration is claimed.

## Operator handoff and proposed GPU smoke {#next_steps}

Before resuming the original run, preserve its current manifest, recovery
metadata/checkpoint pair, and files/traces from steps beyond 20. Verify the historical
manifest SHA above, then install the candidate as
`tau2-critic-fit-evidence/fixed-validation-manifest.json` using a same-directory
temporary file and atomic rename. Keep `fixed-validation-rollouts.pt` unchanged. This
installation has **not** been performed. The compatibility evidence and exact source
pins apply only to the reviewed worktree state.

A resume launch should keep `TAU2_RUN_ROOT`, experiment name, trial name, actor
snapshot, dataset/pack, seed and training-step plan unchanged. Use this patched source
as `SRC` and set `PYTHONPATH="$SRC"`. Retain the original runtime environment, including
the pinned fixed-v3 dataset and simulator configuration. The previous launch script
redirects to `run.log`, rewrites `args.txt` and writes the same exit-code file; use
distinct attempt names instead. For example, after reconstructing the original `OVR`
array from the saved launch/config:

```bash
# Proposed launch pattern only; not executed by this task.
cd "$SRC"
export PYTHONPATH="$SRC"
ATTEMPT="resume-$(date -u +%Y%m%dT%H%M%SZ)"
OVR+=(recover.mode=auto)
printf '%q ' scripts/tau2/run.sh data critic-head "${OVR[@]}" \
  > "$OUT/args.$ATTEMPT.txt"
printf '\n' >> "$OUT/args.$ATTEMPT.txt"
git rev-parse HEAD > "$OUT/git-head.$ATTEMPT.txt"
git diff --binary > "$OUT/source.$ATTEMPT.patch"
# Also archive new tests/untracked files if retaining a complete source snapshot.
set +e
scripts/tau2/run.sh data critic-head "${OVR[@]}" \
  > "$OUT/run.$ATTEMPT.log" 2>&1
code=$?
printf 'EXIT_CODE=%s\n' "$code" > "$OUT/exit-code.$ATTEMPT.txt"
exit "$code"
```

For `critic-head-qwen35-mix`, preserve the 320-row mix, `critic_head_bias_init=0.78`,
`total_train_epochs=4`, `train_dataset.shuffle=true` and the original run identity. The
previous resume attempt used `sglang.mem_fraction_static=0.5`; do not accidentally drop
that override. Do not start in a new run root, which would lose native recovery/cache
discovery. Attempt-specific logs should also cover simulator usage and runtime audit
evidence, where supported, because logical task IDs after step 20 are replayed.

Proposed GPU acceptance, to be queued by the user after GPUs are free:

1. Use new tune smoke roots with one fixed configuration targeting four critic-only
   steps and recovery saves every step. Run one uninterrupted reference and one
   interrupted run, stopping the latter after a complete step-2 checkpoint and its
   step-2 metrics are present. Resume with exactly the same four-step plan and cache; do
   not change the total/warmup step count between attempts.
1. Before its next update, compare restored critic tensors, Adam moments/step and
   scheduler state with the saved checkpoint. Check next task IDs/row order,
   saver/evaluator cadence and actor/critic/inference version zero. Assert no recovery
   weight-transfer request reaches SGLang; weight-group initialization alone is not a
   transfer. Confirm no extra validation collection and the same cache tensor SHA.
1. Complete step 4 and check final metrics, validation, checkpoints and summary. Repeat
   an epoch-boundary interruption using a run long enough to cross the configured
   `ceil(train_rows / batch_size)` boundary, without changing the plan on resume.
1. For a strict final-weight comparison, replay identical recorded training rollout
   tensors. Fresh online simulator episodes can differ even when task order matches;
   without replay, compare restoration and control-flow invariants rather than claiming
   bitwise-equivalent training.

CPU verification establishes the guarded recovery path and sampler ordering. The
proposed smoke must still establish actual GPU/inference behavior; this report does not
claim that the separate xccl defect is fixed or that the user has accepted the patch.
