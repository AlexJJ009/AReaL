# Issue: xccl weight sync from the FSDP actor to SGLang crashes on Qwen3.5-4B

- Status: root cause found and fixed (2026-10-14, d8f72c51, 6bf1d96e; README 16.14.8).
  SAO smoke 526 is the first real-training check. are stashed).
- Owner: Codex round 2 (pueue 516, GPU), review by Claude.
- Branch with work in progress: `codex/xccl-qwen35-fix` (worktree
  `_worktrees/AReaL/codex-xccl-qwen35`).

## Symptom

Any actor -> SGLang weight push over `weight_update_mode: xccl` on
`Qwen/Qwen3.5-4B@851bf6e8` kills the SGLang schedulers. The NCCL watchdog of the
`update_weight_group` reports `CUDA error: an illegal memory access was encountered`,
the scheduler aborts (exit -6), and the HTTP call `/update_weights_from_distributed`
fails. The first names in the failing payload are
`model.language_model.layers.0.linear_attn.{dt_bias,A_log,conv1d.weight}`.

No successful xccl push on Qwen3.5-4B is on record. Critic-only training never pushes
during normal steps, which is why runs 479, 498, 499, 500 and 510 were not affected.

| When       | Run                                                   | Path that pushed           | Outcome                                           |
| ---------- | ----------------------------------------------------- | -------------------------- | ------------------------------------------------- |
| 2026-09-27 | SAO `tau2-training-20260927-nonthinking/sao-mixed-v1` | per-step actor update      | crash, `rollout.log` ~line 1958                   |
| 2026-10-14 | critic resume 506 (`critic-head-qwen35-mix`)          | `RecoverHandler.load` push | crash, `run.attempt2-resume.log` lines ~1145-1450 |

Paths are under
`/data_storage/yl_test/lgx/data-1/code/_artifacts/AReaL/codex-sao-math/`.

## Production layout where it crashed

- 8 GPUs on one node. Training side: GPUs 0-3, FSDP, actor and critic colocated, the
  actor offloaded and onloaded around steps, TorchMemorySaver enabled on the train side.
  Rollout side: GPUs 4-7, 4 SGLang servers, TP1 each, `mem_fraction_static` 0.65 (0.5 in
  506), context 32768.
- Versions: PyTorch 2.9.1+cu129, NCCL 2.27.5, SGLang 0.5.10.post1, Transformers 5.3.0,
  torch_memory_saver 0.0.9.
- Actor master weights fp32, compute bf16.
- Configs: `examples/tau2/config_critic_production.yaml`,
  `examples/tau2/config_sao.yaml`; the resolved config of the crashed run is
  `.../critic-head-qwen35-mix/logs/root/tau2-critic-head/critic-head-qwen35-mix/config.yaml`.

## What round 1 (Codex, CPU only) established

Report: `.../tau2-training-20261014-xccl-qwen35-fix/codex-report.md`.

- Ruled out by source inspection: sender/receiver shape or dtype mismatch (the receiver
  allocates from the request metadata, which the sender derives from the real tensor),
  flattened-bucket offsets (not used), conv1d `[C,1,K]` layout, the
  `model.language_model.` prefix and fused-projection mapping (applied after receive),
  and the sender freeing buffers early.
- The crash stack sits in `ModelRunner.update_weights_from_distributed` at the
  `torch.empty` of the receive loop, before `load_weights`; the failing CUDA instruction
  is not identified.
- Open hypothesis: TorchMemorySaver intercepts actor-side CUDA allocations
  (virtual-memory allocations without an export handle, access only for the allocating
  GPU), which may be incompatible with NCCL. Not proven.
- Separate finding: the sender casts every tensor to bf16, but SGLang keeps Qwen3.5
  `linear_attn.A_log` in fp32, so actor updates to `A_log` are rounded away on the
  rollout side. This does not explain the crash (the snapshot values are bf16-exact).
  The round-1 fix special-cases model + backend + parameter name; review asked for a
  generic rule.
- Prepared a 2-GPU reproduction and verification suite:
  `gpu-verification/run_later.sh ACTOR_GPU,SERVER_GPU`.

## Done criteria

1. The crash is reproduced on `315617b9` in a stated layout, without the tau2 API.
1. Root cause explained with evidence from controlled runs (one variable at a time).
1. Fixed code: direct push and recovery push both complete, and every SGLang-served
   parameter equals the actor's weights in the receiver's native dtype, in the same
   layout.
1. CPU tests that fail on `315617b9` and pass after the fix; GPU red/green logs kept in
   the artifact dir.
1. No behaviour change for other models.
