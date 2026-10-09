# Run Qwen3.5 FSDP2/SGLang training from a locked uv environment on A100

Date: 2026-10-09

Status: Accepted

Applies to `pyproject.toml` (`sao-math` extra), `uv.lock`,
`scripts/sao/bootstrap_env.sh`, `scripts/sao/runtime_env.sh`,
`scripts/sao/check_dependencies.py`, `scripts/sao/run_ppo.sh`, `scripts/sao/run_sao.sh`,
`scripts/tau2/run.sh`, the Qwen3.5 microbatch path in `areal/engine/fsdp_engine.py`, and
the `sglang.attention_backend` setting in the math and τ² recipes.

## Context and Problem Statement

The fork trains Qwen3.5-4B, a hybrid of 24 Gated DeltaNet linear-attention layers and 8
full-attention layers, with FSDP2 and SGLang on one node of 8× A100-SXM4-80GB (SM80).
The host has a CUDA 13 compat driver stack and other experiments' environments that must
not change. Upstream AReaL v2.1.0 ships a Docker image, and its plain `uv sync` does not
reproduce that image: FlashAttention 2 and causal-conv1d are installed manually in the
Dockerfile, and causal-conv1d is excluded by a pyproject override. How is the runtime
built, and which hardware- and model-specific constraints does it fix?

## Considered Options

- Pull the full upstream AReaL Docker image.
- Plain `uv sync` of upstream extras.
- A thin project extra `sao-math` on the root `pyproject.toml`/`uv.lock`, installed in
  two stages (Torch first, then compiled extensions) into a data-volume `.venv`.

## Decision Outcome

Chosen option: "thin `sao-math` extra with a two-stage bootstrap". It avoids the
Docker-disk cost of the full image. It does not change the machine CUDA toolkit or older
environments, and it stays reproducible from the lock file.

- Torch 2.9.1+cu129 and SGLang 0.5.10.post1 follow the v2.1.0 SGLang lock.
  FlashAttention 2 2.8.3, FLA and causal-conv1d are built or matched for cp312/Torch
  2.9/CUDA 12/SM80 with a project-local CUDA 12.9 toolchain
  (`TORCH_CUDA_ARCH_LIST=8.0`). Megatron, Hopper-only kernels (FA3, FlashMLA, DeepGEMM)
  and agentic sandbox dependencies are not built.
- `uv pip check` is expected to fail on exactly four deliberate overrides, and
  `check_dependencies.py` accepts only these: SGLang's `openai` 2.30.0, `soundfile`
  0.12.1, `torchao` 0.15.0, and Torch's `nvidia-cudnn-cu12` 9.16.0.29.
- Launch goes through the `scripts/sao/` and `scripts/tau2/` runners, which source
  `runtime_env.sh` and the existing `.venv`. Do not use `uv run` in this environment:
  lock synchronisation drops the extra and replaces the τ²-compatible LiteLLM/Uvicorn
  overrides.
- SGLang uses `attention_backend: flashinfer`. The upstream default `fa3` is Hopper-only
  and does not apply to SM80.
- The FSDP critic and actor use `flash_attention_2`, with FLA/causal-conv1d kernels for
  the linear-attention layers.
- Qwen3.5 forwards one real sequence per microbatch. The Transformers 5.3 GatedDeltaNet
  and convolution path does not reset recurrent state at packed `cu_seqlens`, so packing
  would leak state across episodes. `fsdp_engine.py` overrides `n_mbs` to the batch row
  count for Qwen3.5. `n_mbs` is a hint and never permission to mix sequences.
- The Qwen3.5 scalar critic reuses the pretrained text backbone with a new `score` head
  (`areal/models/transformers/token_critic.py`, strict variant `scalar_value.py`). Only
  the head may be missing, and only when initialising from a policy backbone; a lenient
  load must never hide backbone mis-loading.

### Consequences

- Good, because the environment is reproducible from `uv.lock` plus a recorded build
  step, and other projects on the host keep their CUDA and Python stacks.
- Good, because the one-sequence rule keeps hybrid-model logprobs and values correct for
  multi-turn episodes up to 32K.
- Bad, because a non-zero `uv pip check` is the normal state. A new conflict must be
  judged against the four accepted overrides, not ignored.
- Bad, because one sequence per microbatch gives up packing throughput, and the 32K
  memory headroom has to be measured per layout.
- Bad, because moving to Hopper GPUs, another Transformers version or a non-hybrid model
  means revisiting the attention backend, kernel builds and the packing rule.
