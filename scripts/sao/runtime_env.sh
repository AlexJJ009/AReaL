#!/usr/bin/env bash
# Source before probes or the native launcher. All writable caches live on data.
: "${SAO_ARTIFACT_ROOT:?Set the task artifact directory on the data volume}"
export CUDA_HOME="${SAO_ARTIFACT_ROOT}/env/cuda-12.9"
: "${SAO_CUDA_COMPAT_DIR:=/usr/local/cuda-13.0/compat}"
test -f "${SAO_CUDA_COMPAT_DIR}/libcuda.so.1" || return 1
export LD_LIBRARY_PATH="${SAO_CUDA_COMPAT_DIR}:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# This host's Python comes from Conda and otherwise loads its older libstdc++.
# FlashInfer JIT uses the system compiler; bind its matching C++ runtime locally.
SAO_CXX_RUNTIME="${SAO_CXX_RUNTIME:-$(c++ -print-file-name=libstdc++.so.6)}"
test -f "${SAO_CXX_RUNTIME}" || return 1
export LD_PRELOAD="${SAO_CXX_RUNTIME}${LD_PRELOAD:+:${LD_PRELOAD}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export UV_CACHE_DIR="${SAO_ARTIFACT_ROOT}/env/cache/uv"
export UV_LINK_MODE=copy
export TMPDIR="${SAO_ARTIFACT_ROOT}/env/tmp"
export TORCH_EXTENSIONS_DIR="${SAO_ARTIFACT_ROOT}/env/cache/torch-extensions"
export AREAL_CACHE_DIR="${SAO_ARTIFACT_ROOT}/env/cache/areal"
export TRITON_CACHE_DIR="${SAO_ARTIFACT_ROOT}/env/cache/triton"
export XDG_CACHE_HOME="${SAO_ARTIFACT_ROOT}/env/cache/xdg"
export FLASHINFER_WORKSPACE_BASE="${SAO_ARTIFACT_ROOT}/env/cache/flashinfer"
# CUDA compiler components are minimal; cuRAND headers come from the locked
# nvidia-curand-cu12 wheel and are needed by FlashInfer stochastic sampling.
SAO_RUNTIME_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CPATH="${SAO_RUNTIME_REPO}/.venv/lib/python3.12/site-packages/nvidia/curand/include${CPATH:+:${CPATH}}"
export HF_HOME="${SAO_ARTIFACT_ROOT}/env/hf"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# RPC/NCCL addresses are local to this host; avoid routing RPC through HTTP proxy.
export NO_PROXY='*' no_proxy='*'
export WANDB_MODE=disabled
