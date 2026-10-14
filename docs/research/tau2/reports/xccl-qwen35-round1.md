---
task_id: 20261014T033356Z-qwen3-5-xccl-weight-synchronization-investigatio-d7b63af5
report_id: 20261014T033356Z-final-b721b7
kind: final
workspace: /data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35
generated_at: '2026-10-14T03:40:21Z'
window_start: '2026-10-14T03:40:21Z'
window_end: '2026-10-14T03:40:21Z'
---

# Qwen3.5 FSDP → SGLang XCCL: partial result, crash unresolved

## Goal and current boundary {#goal}

**The requested NCCL crash fix is not complete.** This delivery fixes a separate,
CPU-proven precision defect in Qwen3.5 weight synchronization and supplies a two-GPU
reproduction/verification command. The illegal-address root cause remains unproven. No
GPU verification has run, and the precision patch must not be treated as a demonstrated
crash fix or as readiness evidence for the SAO baseline.

The target is `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, FSDP actor →
SGLang over `update_weight_group`, including the recovery push. Work stayed in the
requested checkout plus the requested artifact directory. There were no commits, GPU
jobs, queued pueue tasks, DeepSeek calls, dependency changes, or edits to `cli_args.py`,
`recover.py`, or `train_critic.py`.

## Delivered change and checks {#progress}

[fsdp_engine.py](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/areal/engine/fsdp_engine.py:2119)
now preserves the source dtype of `.linear_attn.A_log` only when the model is in the
Qwen3.5 family and the receiver is SGLang. SGLang explicitly stores this parameter in
FP32 even under BF16 inference. Previously, an FP32 actor value passed through BF16
before being copied back to FP32, permanently losing precision. All other
model/backend/parameter casts remain unchanged; no disk fallback was introduced.

[CPU tests](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/tests/test_fsdp_xccl_dtype.py:1)
exercise the actual FSDP send loop, bucket boundaries, SGLang metadata serialization,
and both single-group and multi-group branches, replacing only transport/device
operations. They cover FP32/BF16 actor storage, SGLang/vLLM, and
Qwen3.5/Qwen3.5-text/Qwen3. They verify names, shapes, wire dtypes and exact retained
values. They do **not** simulate NCCL or demonstrate CUDA allocator correctness.

| Check                                                                                             | Observed result                                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Exact baseline sender from `git show 315617b9:areal/engine/fsdp_engine.py`, new CPU test retained | [4 failed, 20 passed](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/cpu-red-315617b9.log); the failures are FP32 Qwen3.5 `A_log` cases                       |
| Restored precision fix, same test                                                                 | [24 passed](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/cpu-green-initial.log)                                                                             |
| Relevant CPU suites, including both requested τ² suites                                           | [232 passed, 3 deselected](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/cpu-green-regression.log)                                                           |
| Additional available weight-update CPU suites                                                     | [56 passed, 18 skipped](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/cpu-weight-update-available.log); skips depend on optional DTE/AWEX availability       |
| Additional `test_nccl_group.py`                                                                   | [Collection error](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/cpu-weight-update-regression.log): optional `megatron` package missing through AWEX imports |
| Ruff lint/format; Git whitespace                                                                  | [Passed](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/lint.log); `git diff --check` returned 0                                                              |
| GPU script static and deliberate negative controls                                                | [Passed](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/gpu-verification/cpu-validation.md); GPU execution remains pending                                    |

The baseline run physically restored only the changed engine file from Git, ran pytest,
and restored the saved fixed file using an EXIT trap. No stash, branch switch or commit
was used. Both requested test modules, `tests/test_tau2_sao_contract.py` and
`tests/test_tau2_critic_training.py`, ran in the 232-test batch. The three deselected
tests are GPU optimizer-dtype tests; no GPU suite was launched.

## Why this is a limited correction {#decisions}

The precision correction follows a demonstrated receiving-parameter requirement, not a
guessed communication mismatch.
[SGLang's constructor](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/.venv/lib/python3.12/site-packages/sglang/srt/models/qwen3_5.py:193)
gives `A_log` shape `[32]` and explicit `torch.float32`, while `dt_bias` follows the
inference dtype. The sender derives `ParamSpec` from the actual transmitted tensor and
computes bucket sizes in bytes, so retaining FP32 changes both payload dtype and byte
accounting consistently.

A
[snapshot inspection](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/snapshot-dtypes.log)
found `A_log: F32 [32]`, `dt_bias: BF16 [32]`, `conv1d.weight: BF16 [8192,1,4]`, and
`linear_attn.norm.weight: F32 [128]`. The SGLang norm uses its default inference dtype,
unlike explicit FP32 `A_log`. **The untouched snapshot's first-layer `A_log` values
happen to be exactly BF16-representable**: its measured round-trip difference was zero.
Thus the snapshot alone is not a precision regression test, and this correction cannot
explain the frozen-weight recovery crash. The tests use non-BF16 values; the GPU script
adds a `0.0001` FP32 residual to every `A_log`, and its CPU probe verified sensitivity
for all 24 such tensors.

“Exact served weights” in the verifier means the canonical actor values loaded into
SGLang's native dtype and packed parameter layout. For BF16 destination parameters,
ordinary FP32-master → BF16 conversion remains expected. FP32 `A_log` must retain every
actor FP32 bit. Literal FP32 master equality for every BF16 destination is not
representable.

## Crash investigation evidence {#evidence}

The recovery log reports the update-group CUDA illegal access at
[line 1165](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261013-critic-head/critic-head-qwen35-mix/run.attempt2-resume.log:1165).
Its main-thread stacks are still at `ModelRunner.update_weights_from_distributed`, line
1593, at
[log line 1275](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261013-critic-head/critic-head-qwen35-mix/run.attempt2-resume.log:1275)
and
[1315](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261013-critic-head/critic-head-qwen35-mix/run.attempt2-resume.log:1315).
In the installed source, that is `torch.empty` within the receive loop. Earlier
asynchronous broadcasts may already be running, but `load_weights` has not yet been
called for this request. This narrows the observed failure to outstanding
transport/allocation work; it does not identify the originating CUDA instruction.

| Hypothesis                                            | Inspected evidence and consequence                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Receiver allocates from its own parameter dtype/shape | [model_runner.py:1589](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/.venv/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py:1589) allocates from the request's `dtypes` and `shapes`; [sender:1837](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/areal/engine/fsdp_engine.py:1837) derives them from the actual tensor. No static mismatch found. |
| Flattened bucket offsets disagree                     | [sglang_remote.py:248](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/areal/engine/sglang_remote.py:248) sends no `flattened_bucket` format. Both sides broadcast each tensor separately.                                                                                                                                                                                                            |
| Convolution `[C,1,K]` versus `[C,K]`                  | [Qwen3.5 receiver:147](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/.venv/lib/python3.12/site-packages/sglang/srt/models/qwen3_5.py:147) adds the singleton dimension. Actor snapshot and receiver expect `[8192,1,4]` at TP1.                                                                                                                                                                     |
| VLM prefix, fused projections, tied/MTP weights       | [Qwen3.5 loader:1361](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/.venv/lib/python3.12/site-packages/sglang/srt/models/qwen3_5.py:1361) maps `model.language_model.` to `model.`, maps split `qkv/z` and `b/a`, handles tied embeddings, and skips MTP. This is applied after receive completion. No evidence connects these mappings to the recorded pre-loader crash.                           |
| Sender drops buffers too early                        | [sender:1908](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/areal/engine/fsdp_engine.py:1908) retains the tensors in a pending bucket; [1915](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/areal/engine/fsdp_engine.py:1915) waits handles and the HTTP future before clearing them. No static lifetime defect found.                                                   |
| Recovery uses another transport                       | [recover.py:521](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/areal/utils/recover.py:521) and [rl_trainer.py:1056](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-xccl-qwen35/areal/trainer/rl_trainer.py:1056) both invoke `update_weights`; no recovery source change is needed for the dtype correction.                                                                          |

The installed versions are PyTorch `2.9.1+cu129`, NCCL `2.27.5`, SGLang `0.5.10.post1`,
Transformers `5.3.0`, and TorchMemorySaver `0.0.9`
([readback](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/snapshot-dtypes.log)).

**Remaining hypothesis, not a root-cause conclusion:** the actor uses globally enabled
TorchMemorySaver (TMS), which intercepts CUDA allocations. The exact PyPI 0.0.9 source
[entrypoint.cpp:13](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/torch_memory_saver-0.0.9/csrc/entrypoint.cpp:13)
initializes an allocation flag independently in each thread;
[utils.h:183](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/torch_memory_saver-0.0.9/csrc/utils.h:183)
creates virtual-memory allocations with no export-handle flag and access for the
allocating GPU only. This warrants checking NCCL's allocation/peer-access path, but the
logs do not show which allocation or NCCL operation failed. A calling-thread `disable()`
wrapper does not establish that NCCL background-thread allocations bypass TMS, so no
speculative wrapper or NCCL flag change was shipped.

## Unverified scope and risks {#scope}

The required crash-root fix, old-code two-GPU crash reproduction, and new-code GPU
success are **not yet achieved**. The CPU red/green pair satisfies the precision-logic
regression only, not the full crash requirement. A separate source audit and independent
code review found no demonstrated metadata, convolution, prefix, or pending-buffer
cause. This absence is not proof that XCCL cannot be made correct; disk fallback is
therefore unjustified.

The GPU harness has only CPU/static validation. It uses one FSDP actor rank and one TP1
server, as requested; the historical failures involved more ranks. It enables actor TMS,
FP32 master storage, production NCCL flags and repeated offload/onload cycles, but does
not run a training optimizer step or concurrent τ² rollouts. The old two-GPU case may
fail to reproduce the historical crash, or both versions may crash. Either outcome is a
suite FAIL requiring further investigation.

Readback hashes every live served parameter before constructing a reference. It then
poisons destination parameters with NaN and loads canonical actor tensors through
SGLang's loader, catching skipped destinations and partial fused loads, and restores the
original served values. This checks actual tensor contents, not just HTTP success. It
reuses the installed loader to construct the reference, so it cannot detect a loader
defect common to both paths; it does not claim HF/logit equivalence. The recovery check
uses actual DCP save, zeroes actor parameters, then calls `RecoverHandler.load` and
verifies the served values again.

## Deferred GPU command and acceptance {#next_steps}

The
[runner](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/gpu-verification/run_later.sh),
[implementation](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/gpu-verification/run_suite.py),
and
[usage details](/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/gpu-verification/README.md)
are ready for a later GPU attempt. Old source is archived from `315617b9`; current
source is snapshotted when the command starts. The shared checkout and venv are not
edited by the runner. Source hashes and environment provenance accompany fresh logs
under `gpu-verification/run-<timestamp>-<pid>/`. Full oracles, readbacks and DCP
checkpoints may occupy approximately 100 GB.

**Pueue-ready command; shown only, not executed.** This uses physical GPUs 0 and 1 after
task 510 succeeds; they must be reserved for this run at execution time:

```bash
pueue add --group gpu8 --after 510 -- 'bash /data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/gpu-verification/run_later.sh 0,1'
```

`--after 510` is a success dependency: if 510 fails, pueue will fail this dependent task
instead of starting it. Once the two GPUs are independently reserved, the direct command
is:

```bash
bash /data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/tau2-training-20261014-xccl-qwen35-fix/gpu-verification/run_later.sh 0,1
```

Suite exit 0 requires old code to fail **during an unfinished actual XCCL push** with
the exact `update_weight_group` watchdog illegal-access signature, and the current code
to pass full native-dtype served-weight comparisons after both direct and recovery
pushes. Missing reproduction, startup errors, timeouts, later oracle errors or value
mismatches return nonzero. Guard tests deliberately reject these false-positive cases,
including Python optimization mode. A precision-only patch that leaves the crash
unresolved must fail this suite.

The next evidence needed is that GPU result. If both cases crash, inspect the fresh
active-push window and NCCL/TMS allocation behavior; do not declare success from CPU
tests or change to disk updates without establishing the remaining cause.
