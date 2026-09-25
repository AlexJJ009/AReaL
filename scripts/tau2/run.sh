#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/tau2/run.sh qualification {collect|critic|mixed|airline|retail|telecom} [overrides...]
  scripts/tau2/run.sh data {collect|critic} [overrides...]
  scripts/tau2/run.sh policy {mixed|airline|retail|telecom} grpo [overrides...]

Required runtime variables:
  SAO_ARTIFACT_ROOT  shared AReaL runtime/cache root
  TAU2_RUN_ROOT      output root for this run
  HF_HUB_CACHE       Hugging Face cache containing the pinned actor snapshot
  TAU2_DATA_DIR      data directory of the pinned official tau2 package

Episode-producing run modes require TAU2_DEEPSEEK_ENV_FILE (mode 0600).
qualification critic requires TAU2_EPISODES and TAU2_CRITIC_INIT_PATH.

With --check-config, run.sh skips the runtime CUDA environment and DeepSeek key load.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

SELECTOR="$1"
shift

case "${SELECTOR}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
export TAU2_ACTOR_PATH="${TAU2_ACTOR_PATH:-Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"

has_check_config() {
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--check-config" ]]; then
      return 0
    fi
  done
  return 1
}

prepare_runtime_env() {
  : "${SAO_ARTIFACT_ROOT:?Set SAO_ARTIFACT_ROOT to the existing runtime/cache root}"
  : "${HF_HUB_CACHE:?Set HF_HUB_CACHE to the cache containing the pinned actor}"
  : "${TAU2_DATA_DIR:?Set TAU2_DATA_DIR to the pinned official tau2 data directory}"

  # Reuse the existing SAO host runtime setup. This supplies the compatible CUDA
  # and libstdc++ paths, local-RPC proxy bypass, and data-volume caches.
  # shellcheck source=../sao/runtime_env.sh
  source "${REPO_ROOT}/scripts/sao/runtime_env.sh"

  # TMS is preloaded by AReaL's local scheduler before the worker Python process
  # starts. Make its venv-provided CUDA runtime dependency visible to the dynamic
  # loader; importing torch later is too late for an LD_PRELOAD dependency.
  TMS_CUDA_RUNTIME_LIB="$(
    "${REPO_ROOT}/.venv/bin/python" -c \
      'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cuda_runtime/lib")'
  )"
  if [[ ! -f "${TMS_CUDA_RUNTIME_LIB}/libcudart.so.12" ]]; then
    printf 'TMS CUDA runtime library is missing: %s\n' "${TMS_CUDA_RUNTIME_LIB}" >&2
    exit 2
  fi
  export LD_LIBRARY_PATH="${TMS_CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

prepare_run_root() {
  local run_kind="$1"
  : "${TAU2_RUN_ROOT:?Set TAU2_RUN_ROOT to the current run output directory}"
  if [[ "${run_kind}" == "qualification" ]]; then
    # Qualification retries never share logs, name resolution, or checkpoints.
    # The caller supplies a stable prefix; the runner prints the resolved attempt root.
    mkdir -p "$(dirname -- "${TAU2_RUN_ROOT}")"
    TAU2_RUN_ROOT="$(mktemp -d "${TAU2_RUN_ROOT}.XXXXXX")"
  else
    mkdir -p "${TAU2_RUN_ROOT}"
  fi
  export TAU2_RUN_ROOT
  printf 'TAU2_RUN_ROOT=%s\n' "${TAU2_RUN_ROOT}"
}

load_deepseek_key() {
  : "${TAU2_DEEPSEEK_ENV_FILE:?Set TAU2_DEEPSEEK_ENV_FILE for tool-use episodes}"
  if [[ ! -f "${TAU2_DEEPSEEK_ENV_FILE}" ]]; then
    printf 'DeepSeek environment file does not exist: %s\n' "${TAU2_DEEPSEEK_ENV_FILE}" >&2
    exit 2
  fi
  if [[ "$(stat -c '%a' "${TAU2_DEEPSEEK_ENV_FILE}")" != "600" ]]; then
    printf 'DeepSeek environment file must have mode 0600: %s\n' "${TAU2_DEEPSEEK_ENV_FILE}" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${TAU2_DEEPSEEK_ENV_FILE}"
  set +a
  : "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is missing from TAU2_DEEPSEEK_ENV_FILE}"
}

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

case "${SELECTOR}" in
  qualification|data)
    MODE="$1"
    shift
    if ! has_check_config "$@"; then
      prepare_run_root "${SELECTOR}"
      prepare_runtime_env
    else
      : "${TAU2_RUN_ROOT:?Set TAU2_RUN_ROOT to the current run output directory}"
      export TAU2_RUN_ROOT
    fi
    ;;
  policy)
    MODE="$1"
    shift
    if [[ $# -lt 1 ]]; then
      printf 'Policy mode requires an explicit algorithm: grpo\n' >&2
      usage >&2
      exit 2
    fi
    ALGORITHM="$1"
    shift
    if [[ "${ALGORITHM}" != "grpo" ]]; then
      printf 'Unsupported tau2 policy algorithm in this entrypoint: %s\n' "${ALGORITHM}" >&2
      usage >&2
      exit 2
    fi
    case "${MODE}" in
      mixed|airline|retail|telecom) ;;
      *)
        printf 'Unsupported tau2 policy domain: %s\n' "${MODE}" >&2
        usage >&2
        exit 2
        ;;
    esac
    if ! has_check_config "$@"; then
      prepare_run_root policy
      prepare_runtime_env
    else
      : "${TAU2_RUN_ROOT:?Set TAU2_RUN_ROOT to the current run output directory}"
      export TAU2_RUN_ROOT
    fi
    ;;
  *)
    printf 'Unknown tau2 selector: %s\n' "${SELECTOR}" >&2
    usage >&2
    exit 2
    ;;
esac

case "${SELECTOR}:${MODE}" in
  data:collect)
    if ! has_check_config "$@"; then load_deepseek_key; fi
    exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/tau2/train.py" \
      --config "${REPO_ROOT}/examples/tau2/config_critic_collect.yaml" "$@"
    ;;
  data:critic)
    : "${TAU2_EPISODES:?Set TAU2_EPISODES to the checked episode JSONL}"
    export TAU2_CRITIC_INIT_PATH="${TAU2_ACTOR_PATH}"
    exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/tau2/train_critic.py" \
      --config "${REPO_ROOT}/examples/tau2/config_critic_production.yaml" \
      --episodes "${TAU2_EPISODES}" "$@"
    ;;
  qualification:collect)
    if ! has_check_config "$@"; then
      load_deepseek_key
    fi
    exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/tau2/train.py" \
      --config "${REPO_ROOT}/examples/tau2/config_critic_collect_qualification.yaml" \
      "$@"
    ;;
  qualification:critic)
    : "${TAU2_EPISODES:?Set TAU2_EPISODES to the checked six-episode JSONL file}"
    : "${TAU2_CRITIC_INIT_PATH:?Set TAU2_CRITIC_INIT_PATH to the critic initialization}"
    exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/tau2/train_critic.py" \
      --config "${REPO_ROOT}/examples/tau2/config_critic_qualification.yaml" \
      --episodes "${TAU2_EPISODES}" \
      "$@"
    ;;
  qualification:mixed)
    if ! has_check_config "$@"; then
      load_deepseek_key
    fi
    : "${TAU2_CRITIC_PATH:?Set TAU2_CRITIC_PATH to the restored critic checkpoint}"
    exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/tau2/train.py" \
      --config "${REPO_ROOT}/examples/tau2/config_sao_qualification.yaml" \
      "$@"
    ;;
  qualification:airline|qualification:retail|qualification:telecom)
    if ! has_check_config "$@"; then
      load_deepseek_key
    fi
    : "${TAU2_CRITIC_PATH:?Set TAU2_CRITIC_PATH to the restored critic checkpoint}"
    case "${MODE}" in
      airline) DOMAIN_PRUNE=("~domain_effective_episodes.retail" "~domain_effective_episodes.telecom") ;;
      retail) DOMAIN_PRUNE=("~domain_effective_episodes.airline" "~domain_effective_episodes.telecom") ;;
      telecom) DOMAIN_PRUNE=("~domain_effective_episodes.airline" "~domain_effective_episodes.retail") ;;
    esac
    exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/tau2/train.py" \
      --config "${REPO_ROOT}/examples/tau2/config_sao_qualification.yaml" \
      "domains=[${MODE}]" \
      "econfig.domain=${MODE}" \
      "train_batch_episodes=1" \
      "effective_episodes=1" \
      "domain_effective_episodes.${MODE}=1" \
      "${DOMAIN_PRUNE[@]}" \
      "train_dataset.batch_size=1" \
      "$@"
    ;;
  policy:mixed)
    if ! has_check_config "$@"; then
      load_deepseek_key
    fi
    exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/tau2/train.py" \
      --config "${REPO_ROOT}/examples/tau2/config_grpo.yaml" \
      "$@"
    ;;
  policy:airline|policy:retail|policy:telecom)
    if ! has_check_config "$@"; then
      load_deepseek_key
    fi
    exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/tau2/train.py" \
      --config "${REPO_ROOT}/examples/tau2/config_grpo.yaml" \
      "domains=[${MODE}]" \
      "econfig.domain=${MODE}" \
      "$@"
    ;;
  collect)
    printf 'Use explicit selector: scripts/tau2/run.sh qualification collect\n' >&2
    exit 2
    ;;
  *)
    printf 'Unknown tau2 mode: %s %s\n' "${SELECTOR}" "${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
