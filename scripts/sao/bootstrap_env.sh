#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAO_ARTIFACT_ROOT:?Set SAO_ARTIFACT_ROOT to the data-volume artifact directory}"
ENV_ROOT="${SAO_ARTIFACT_ROOT}/env"
LOG_ROOT="${ENV_ROOT}/logs"

mkdir -p \
  "${ENV_ROOT}/cache" \
  "${ENV_ROOT}/hf" \
  "${ENV_ROOT}/tmp" \
  "${LOG_ROOT}"

export TMPDIR="${ENV_ROOT}/tmp"
export TEMP="${ENV_ROOT}/tmp"
export TMP="${ENV_ROOT}/tmp"
export UV_CACHE_DIR="${ENV_ROOT}/cache/uv"
export UV_LINK_MODE=copy
export UV_PYTHON_DOWNLOADS=never
export UV_HTTP_TIMEOUT=30 UV_HTTP_RETRIES=2
export TORCH_EXTENSIONS_DIR="${ENV_ROOT}/cache/torch-extensions"
export TRITON_CACHE_DIR="${ENV_ROOT}/cache/triton"
export MAX_JOBS="${MAX_JOBS:-8}"
export TORCH_CUDA_ARCH_LIST=8.0
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export XDG_CACHE_HOME="${ENV_ROOT}/cache/xdg"

# Match the cu129 Torch ABI without changing the machine's CUDA13 toolkit.
fetch_cuda_component() {
  local component="$1" version="$2" digest="$3"
  local filename="${component}-linux-x86_64-${version}-archive.tar.xz"
  local archive="${ENV_ROOT}/downloads/${filename}"
  mkdir -p "${ENV_ROOT}/downloads" "${ENV_ROOT}/cuda-12.9"
  if ! test -f "${archive}"; then
    curl -fL --retry 3 --connect-timeout 15 \
      "https://developer.download.nvidia.com/compute/cuda/redist/${component}/linux-x86_64/${filename}" \
      -o "${archive}"
  fi
  echo "${digest}  ${archive}" | sha256sum --check
  tar -xJf "${archive}" --strip-components=1 -C "${ENV_ROOT}/cuda-12.9"
}
fetch_cuda_component cuda_nvcc 12.9.86 7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35
fetch_cuda_component cuda_cudart 12.9.79 1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99
fetch_cuda_component cuda_cccl 12.9.27 8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa
export CUDA_HOME="${ENV_ROOT}/cuda-12.9"
export PATH="${CUDA_HOME}/bin:${PATH}"
test -e "${CUDA_HOME}/lib64" || ln -s lib "${CUDA_HOME}/lib64"

cd "${ROOT}"

{
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=${ROOT}"
  echo "env_root=${ENV_ROOT}"
  uv --version
  python3 --version
} | tee "${LOG_ROOT}/bootstrap_env.log"

test -x .venv/bin/python || uv venv --python 3.12
uv lock --check
uv sync --locked --extra sao-math --group dev \
  --no-install-package causal-conv1d --no-install-package torch-memory-saver \
  2>&1 | tee "${LOG_ROOT}/uv-sync-base.log"
uv sync --locked --extra sao-math --group dev 2>&1 | tee "${LOG_ROOT}/uv-sync-sao-math.log"
uv pip check > "${LOG_ROOT}/uv-pip-check.log" 2>&1 || \
  echo "Native package metadata conflicts recorded in ${LOG_ROOT}/uv-pip-check.log"
.venv/bin/python scripts/sao/check_dependencies.py \
  --output "${LOG_ROOT}/dependency-overrides.json"
source scripts/sao/runtime_env.sh

.venv/bin/python - <<'PY' 2>&1 | tee "${LOG_ROOT}/cpu-import-readback.log"
import importlib
import sys

packages = [
    "torch",
    "sglang",
    "flash_attn",
    "fla",
    "causal_conv1d",
    "math_verify",
    "latex2sympy2_extended",
    "sympy",
]

print("python", sys.version.replace("\n", " "))
for name in packages:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name} {version}")
PY
