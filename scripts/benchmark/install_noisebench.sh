#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: scripts/benchmark/install_noisebench.sh [INSTALL_DIR]
#
# One-click install for AgentNoiseBench (tau2-bench fork with noise injection).
# Portable: works on any machine with uv, git, and python3.12+.
#
# What it does:
#   1. Clones the AgentNoiseBench repo (pinned revision)
#   2. Creates a venv in tau2-bench-main/ and installs dependencies via uv
#   3. Applies the JSON parsing patch to evaluator_nl_assertions.py
#   4. Generates model config templates (Qwen3.5-4B, Qwen3-4B)
#   5. Generates eval script templates
#   6. Verifies the installation
set -euo pipefail

# --- Configuration ---
NOISEBENCH_REPO="https://github.com/keven-cyber/agentnoisebench.git"
NOISEBENCH_REV="d24517cef3eb22e2b5dfc7f70c1a596d88c25a1f"
PYTHON_VERSION="3.12"

# Install directory: first argument, or default
INSTALL_DIR="${1:-${D:-/mnt/shared-storage-gpfs2/gpfs2-agent-deepresearch/lgx}/benchmark/agentnoisebench}"

# --- Prerequisites ---
echo "[install] Checking prerequisites..."
for t in uv git; do
  command -v "$t" >/dev/null || { echo "$t not found; install it first" >&2; exit 1; }
done

# --- Clone or update repo ---
if [ -d "$INSTALL_DIR/.git" ]; then
  echo "[install] Existing repo found at $INSTALL_DIR"
  cd "$INSTALL_DIR"
  CURRENT_REV=$(git rev-parse HEAD)
  if [ "$CURRENT_REV" = "$NOISEBENCH_REV" ]; then
    echo "[install] Already at pinned revision $NOISEBENCH_REV"
  else
    echo "[install] Fetching and checking out pinned revision..."
    git fetch origin
    git checkout "$NOISEBENCH_REV"
  fi
else
  echo "[install] Cloning AgentNoiseBench to $INSTALL_DIR..."
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone "$NOISEBENCH_REPO" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
  git checkout "$NOISEBENCH_REV"
fi

TAU2_DIR="$INSTALL_DIR/tau2-bench-main"

# --- Create venv and install dependencies ---
echo "[install] Creating venv in $TAU2_DIR/.venv ..."
uv venv --python "$PYTHON_VERSION" "$TAU2_DIR/.venv"

echo "[install] Installing tau2 dependencies..."
# The NoiseBench fork uses pdm-backend; uv pip handles it.
uv pip install --python "$TAU2_DIR/.venv/bin/python" -e "$TAU2_DIR"

# --- Apply JSON parsing patch ---
EVALUATOR="$TAU2_DIR/src/tau2/evaluator/evaluator_nl_assertions.py"
echo "[install] Applying JSON parsing patch to evaluator_nl_assertions.py..."

# Check if patch is already applied (idempotent)
if grep -q "raw_decode" "$EVALUATOR" 2>/dev/null; then
  echo "[install] Patch already applied, skipping."
else
  # The upstream file has:
  #   result_data = json.loads(assistant_message.content)
  # Replace with robust JSON extraction that handles thinking tokens.
  python3 -I - "$EVALUATOR" << 'PYEOF'
import sys, pathlib

evaluator_path = pathlib.Path(sys.argv[1])
content = evaluator_path.read_text()

OLD = "        result_data = json.loads(assistant_message.content)"
NEW = """\
        # Guard: generate() returns None on connection/request failure
        if assistant_message is None or assistant_message.content is None:
            return [
                NLAssertionCheck(
                    nl_assertion=assertion,
                    met=False,
                    justification="LLM evaluation failed: no response from model",
                )
                for assertion in nl_assertions
            ]

        # Robust JSON extraction: the model may output reasoning text
        # before/after the JSON object (thinking tokens, trailing explanation).
        # Use raw_decode to parse the first JSON object and ignore the rest.
        text = assistant_message.content.strip()
        try:
            result_data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            if start == -1:
                raise ValueError(
                    f"NL assertions evaluator: no JSON object in model response: {text[:200]}"
                )
            decoder = json.JSONDecoder()
            result_data, _ = decoder.raw_decode(text, start)"""

if OLD not in content:
    print("ERROR: Could not find patch target in evaluator_nl_assertions.py", file=sys.stderr)
    print("The file may have been modified. Check manually.", file=sys.stderr)
    sys.exit(1)

content = content.replace(OLD, NEW, 1)
evaluator_path.write_text(content)
print("Patch applied successfully.")
PYEOF
fi

# --- Generate model config templates ---
echo "[install] Generating model config templates..."

create_model_config() {
  local config_path="$1"
  local model_name="$2"

  if [ -f "$config_path" ]; then
    echo "[install] $config_path already exists, skipping."
    return
  fi

  cat > "$config_path" << YAMLEOF
# Models config for local SGLang serving $model_name
default:
  base_url: http://localhost:30001/v1/chat/completions
  temperature: 0.0
  max_input_tokens: 32768
  headers:
    Authorization: "Bearer dummy"
    Content-Type: "application/json"

models:
  - name: $model_name
    max_tokens: 8192
    max_input_tokens: 32768
    chat_template_kwargs:
      enable_thinking: false

  # NL assertions evaluator looks up gpt-4o-mini by name.
  # Override the model field so SGLang receives the served name.
  - name: gpt-4o-mini
    model: $model_name
    max_tokens: 8192
    max_input_tokens: 32768
    chat_template_kwargs:
      enable_thinking: false
YAMLEOF
  echo "[install] Created $config_path"
}

create_model_config "$INSTALL_DIR/models-qwen35-4b.yaml" "Qwen3.5-4B"
create_model_config "$INSTALL_DIR/models-qwen3-4b.yaml" "Qwen3-4B-Instruct-2507"

# --- Generate eval script template ---
echo "[install] Generating eval script template..."

TEMPLATE_DIR="$INSTALL_DIR/eval-templates"
mkdir -p "$TEMPLATE_DIR"

cat > "$TEMPLATE_DIR/run-noisebench.sh" << 'EVALEOF'
#!/bin/bash
# AgentNoiseBench evaluation template
# Usage: run-noisebench.sh --model-path PATH --served-name NAME --model-config YAML
#                          --output-dir DIR [--sglang-args "EXTRA_ARGS"]
#
# 12 noise conditions x 3 domains = 36 evaluations
# =====================================================
set -euo pipefail

# --- Parse arguments ---
MODEL_PATH=""
SERVED_NAME=""
MODEL_CONFIG=""
OUTBASE=""
SGLANG_EXTRA_ARGS=""
PORT=30001

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path)     MODEL_PATH="$2"; shift 2 ;;
    --served-name)    SERVED_NAME="$2"; shift 2 ;;
    --model-config)   MODEL_CONFIG="$2"; shift 2 ;;
    --output-dir)     OUTBASE="$2"; shift 2 ;;
    --sglang-args)    SGLANG_EXTRA_ARGS="$2"; shift 2 ;;
    --port)           PORT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

for var in MODEL_PATH SERVED_NAME MODEL_CONFIG OUTBASE; do
  [ -n "${!var}" ] || { echo "Missing required argument: --$(echo $var | tr '[:upper:]' '[:lower:]' | tr '_' '-')" >&2; exit 1; }
done

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TAU2_DIR="$SCRIPT_DIR/tau2-bench-main"
VENV="$TAU2_DIR/.venv"

# SGLang can come from AReaL's venv or the system PATH
SGLANG_PYTHON="${SGLANG_PYTHON:-$(command -v python3)}"

mkdir -p "$OUTBASE/logs"
LOG="$OUTBASE/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "AgentNoiseBench: $SERVED_NAME"
echo "Start: $(date)"
echo "=========================================="

# --- Environment ---
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Enable speculative decoding v2 when MTP is requested
if echo "$SGLANG_EXTRA_ARGS" | grep -q "speculative-algorithm"; then
    export SGLANG_ENABLE_SPEC_V2=1
    echo "[$(date)] SGLANG_ENABLE_SPEC_V2=1 (speculative decoding detected)"
fi

# NoiseBench config
export TAU2_DATA_DIR="$TAU2_DIR/data"
export TAU2_MODEL_CONFIG_PATH="$MODEL_CONFIG"
export NOISE_ENV_PATH=/dev/null

# Config_API credentials (noise injection + deviation audit)
export API_KEY=dummy
export BASE_URL=http://localhost:$PORT/v1
export MODEL=$SERVED_NAME
export EVAL=$SERVED_NAME

# litellm/openai fallback
export OPENAI_API_KEY=dummy
export OPENAI_API_BASE=http://localhost:$PORT/v1

# --- Start SGLang server ---
fuser -k $PORT/tcp 2>/dev/null || true
sleep 2

echo "[$(date)] Starting SGLang server: $SERVED_NAME (port $PORT)..."
# shellcheck disable=SC2086
$SGLANG_PYTHON -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --port $PORT \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --skip-server-warmup \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    $SGLANG_EXTRA_ARGS &
SGLANG_PID=$!
echo "[$(date)] SGLang PID: $SGLANG_PID"

# Wait for server readiness
echo "[$(date)] Waiting for SGLang health check (max 15 min)..."
for i in $(seq 1 180); do
    HTTP_CODE=$(curl -s -o /dev/null -w %{http_code} http://localhost:$PORT/health 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "[$(date)] SGLang server ready."
        break
    fi
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "[$(date)] FATAL: SGLang server died during startup"
        exit 1
    fi
    if [ "$i" -eq 180 ]; then
        echo "[$(date)] FATAL: SGLang server not ready after 15 min"
        kill $SGLANG_PID 2>/dev/null || true
        exit 1
    fi
    sleep 5
done

# Smoke test
echo "[$(date)] Smoke test..."
RESP=$(curl -s http://localhost:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer dummy" \
    -d "{\"model\": \"$SERVED_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}], \"max_tokens\": 32, \"temperature\": 0.0}")
echo "[$(date)] Smoke response: $(echo "$RESP" | head -c 300)"
if ! echo "$RESP" | python3 -I -c "import sys,json; d=json.load(sys.stdin); assert d.get(\"choices\")" 2>/dev/null; then
    echo "[$(date)] FATAL: Smoke test failed"
    kill $SGLANG_PID 2>/dev/null || true
    exit 1
fi

# --- Preflight: verify noise injection works ---
echo ""
echo "[$(date)] Preflight: airline x induce, 2 tasks, 1 trial..."
yes y | timeout 600 $VENV/bin/tau2 run \
    --domain airline \
    --num-tasks 2 --num-trials 1 \
    --max-concurrency 2 \
    --agent-llm "$SERVED_NAME" --user-llm "$SERVED_NAME" \
    --noise_category induce --noise_nums 100000 --max_single_tool_noise 10 \
    --user-noise original \
    --max-steps 50 --seed 300 \
    --save-to "$OUTBASE/preflight_induce" \
    --log-level ERROR \
    > "$OUTBASE/logs/preflight.log" 2>&1
PREFLIGHT_EC=$?
echo "[$(date)] Preflight exit code: $PREFLIGHT_EC"

python3 -I - "$OUTBASE" << 'PYEOF' || echo "[$(date)] WARNING: preflight verification issue"
import json, sys, pathlib
outbase = pathlib.Path(sys.argv[1])
p = outbase / "preflight_induce.json"
if not p.exists():
    print("PREFLIGHT: output file missing"); sys.exit(1)
d = json.loads(p.read_text())
sims = d.get("simulations", [])
if not sims:
    print("PREFLIGHT: no simulations completed"); sys.exit(1)
for s in sims:
    ri = s.get("reward_info") or {}
    ni = s.get("noise_injections") or []
    print(f"  task={s.get('task_id')}: noise_injections={len(ni)}, "
          f"traj_ok={ri.get('traj_ok')}, reward={ri.get('reward')}")
print("Preflight check passed.")
PYEOF

# --- Noise conditions ---
get_max_single_tool_noise() {
    case "$1" in
        others|fault) echo 1 ;;
        wrong) echo 2 ;;
        induce|redundant|incomplete) echo 10 ;;
        *) echo 1 ;;
    esac
}

TOTAL=0; PASS=0; FAIL=0

run_condition() {
    local domain="$1" noise_cat="$2" user_noise="$3" label="$4"
    local max_stn
    max_stn=$(get_max_single_tool_noise "$noise_cat")

    local num_tasks
    case "$domain" in
        airline)  num_tasks=50 ;;
        retail)   num_tasks=114 ;;
        telecom)  num_tasks=114 ;;
    esac

    local save_name="$OUTBASE/${SERVED_NAME}_${domain}_${label}"
    local log_file="$OUTBASE/logs/${domain}_${label}.log"

    echo ""
    echo "[$(date)] >>> $domain / $label (noise_cat=$noise_cat, user_noise=$user_noise, max_stn=$max_stn)"

    yes y | $VENV/bin/tau2 run \
        --domain "$domain" \
        --num-tasks "$num_tasks" \
        --num-trials 4 \
        --max-concurrency 32 \
        --agent-llm "$SERVED_NAME" --user-llm "$SERVED_NAME" \
        --noise_category "$noise_cat" --noise_nums 100000 --max_single_tool_noise "$max_stn" \
        --user-noise "$user_noise" \
        --max-steps 200 --seed 300 \
        --save-to "$save_name" \
        --log-level ERROR \
        > "$log_file" 2>&1
    local ec=$?
    TOTAL=$((TOTAL + 1))
    if [ $ec -eq 0 ]; then
        PASS=$((PASS + 1))
        echo "[$(date)] <<< $domain / $label : OK"
    else
        FAIL=$((FAIL + 1))
        echo "[$(date)] <<< $domain / $label : FAILED (exit $ec)"
    fi
}

DOMAINS="airline retail telecom"

# 1) Clean condition
for d in $DOMAINS; do
    run_condition "$d" others original clean
done

# 2) User noise conditions (tool=others)
for noise in ambiguity conflict redundancy topic_shift boundary cognition; do
    for d in $DOMAINS; do
        run_condition "$d" others "$noise" "user_${noise}"
    done
done

# 3) Tool noise conditions (user=original)
for noise in fault wrong induce redundant incomplete; do
    for d in $DOMAINS; do
        run_condition "$d" "$noise" original "tool_${noise}"
    done
done

echo ""
echo "=========================================="
echo "[$(date)] $SERVED_NAME evaluation complete."
echo "Results: $PASS/$TOTAL passed, $FAIL/$TOTAL failed"
echo "Output:  $OUTBASE/"
echo "=========================================="

# --- Stop SGLang ---
kill $SGLANG_PID 2>/dev/null || true
wait $SGLANG_PID 2>/dev/null || true
echo "[$(date)] SGLang stopped. Done."
EVALEOF

chmod +x "$TEMPLATE_DIR/run-noisebench.sh"
echo "[install] Created $TEMPLATE_DIR/run-noisebench.sh"

# --- Verification ---
echo ""
echo "[install] Verifying installation..."

# Check tau2 CLI is available
if "$TAU2_DIR/.venv/bin/tau2" --help > /dev/null 2>&1; then
  echo "[install] tau2 CLI: OK"
else
  echo "[install] ERROR: tau2 CLI not working" >&2
  exit 1
fi

# Check the patch was applied
if grep -q "raw_decode" "$EVALUATOR"; then
  echo "[install] JSON parsing patch: OK"
else
  echo "[install] ERROR: JSON parsing patch not applied" >&2
  exit 1
fi

# Check model configs exist
for cfg in "$INSTALL_DIR/models-qwen35-4b.yaml" "$INSTALL_DIR/models-qwen3-4b.yaml"; do
  if [ -f "$cfg" ]; then
    echo "[install] Model config $(basename "$cfg"): OK"
  else
    echo "[install] WARNING: $(basename "$cfg") not created"
  fi
done

echo ""
echo "[install] AgentNoiseBench installed at: $INSTALL_DIR"
echo "[install] tau2 venv: $TAU2_DIR/.venv"
echo "[install] Eval template: $TEMPLATE_DIR/run-noisebench.sh"
echo ""
echo "[install] Example usage (Qwen3.5-4B with MTP):"
echo "  $TEMPLATE_DIR/run-noisebench.sh \\"
echo "    --model-path /path/to/Qwen3.5-4B \\"
echo "    --served-name Qwen3.5-4B \\"
echo "    --model-config $INSTALL_DIR/models-qwen35-4b.yaml \\"
echo "    --output-dir /path/to/output \\"
echo "    --sglang-args '--dp 2 --tp 1 --speculative-algorithm NEXTN --mamba-scheduler-strategy extra_buffer'"
echo ""
echo "[install] Example usage (Qwen3-4B, no MTP):"
echo "  $TEMPLATE_DIR/run-noisebench.sh \\"
echo "    --model-path /path/to/Qwen3-4B-Instruct-2507 \\"
echo "    --served-name Qwen3-4B-Instruct-2507 \\"
echo "    --model-config $INSTALL_DIR/models-qwen3-4b.yaml \\"
echo "    --output-dir /path/to/output \\"
echo "    --sglang-args '--dp 4 --tp 1'"
echo ""
echo "[install] Done."
