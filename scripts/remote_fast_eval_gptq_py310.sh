#!/usr/bin/env bash
set -euo pipefail

ENV="${ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-quantcheck}"
SCRIPT_DIR="${SCRIPT_DIR:-/root/autodl-tmp/SOAR-Toolkit}"
FAST_SCRIPT="${FAST_SCRIPT:-${SCRIPT_DIR}/mini_test.sh}"
FAST_DATASET="${FAST_DATASET:-${SCRIPT_DIR}/eval_dataset/perf_public_fast_eval.jsonl}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/test_results/gptq_py310_fastcheck_$(date +%Y%m%d_%H%M%S)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"

export LD_LIBRARY_PATH="/root/miniconda3/envs/py310/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV}/bin:${PATH}"
export PYTHONUNBUFFERED=1

mkdir -p "${RESULT_DIR}"
cd "${SCRIPT_DIR}"

SERVER_LOG="${RESULT_DIR}/server.log"
FAST_LOG="${RESULT_DIR}/fast_eval.log"

ensure_eval_runtime() {
  local missing=()
  while read -r status name; do
    if [[ "${status}" == "MISSING" ]]; then
      missing+=("${name}")
    fi
  done < <("${ENV}/bin/python" - <<'PY'
mods = [
    ("IPython", "IPython"),
    ("pydantic", "pydantic"),
    ("pydantic_core", "pydantic-core"),
]
for import_name, package_name in mods:
    try:
        __import__(import_name)
        print("OK", package_name)
    except Exception:
        print("MISSING", package_name)
PY
)

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "[fastcheck] installing missing eval deps: ${missing[*]}" | tee -a "${FAST_LOG}"
    "${ENV}/bin/python" -m pip install -i "https://mirrors.aliyun.com/pypi/simple/" "${missing[@]}" >> "${FAST_LOG}" 2>&1
  fi
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[fastcheck] start $(date '+%F %T')" | tee "${FAST_LOG}"
echo "[fastcheck] env=${ENV}" | tee -a "${FAST_LOG}"
echo "[fastcheck] model=${MODEL_PATH}" | tee -a "${FAST_LOG}"
echo "[fastcheck] result_dir=${RESULT_DIR}" | tee -a "${FAST_LOG}"

ensure_eval_runtime

"${ENV}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --quantization gptq_marlin \
  --dtype float16 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --attention-backend flashinfer \
  --force-dense-minicpm \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --skip-server-warmup \
  --enable-request-time-stats-logging \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null; then
    echo "[fastcheck] server ready" | tee -a "${FAST_LOG}"
    break
  fi
  sleep 2
done

if ! curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null; then
  echo "[fastcheck] server failed to become ready" | tee -a "${FAST_LOG}"
  exit 1
fi

EVAL_USE_ALL_ROWS=1 \
EVAL_TASK_MAX_TOKENS='mcq:64,qa:128,niah:128,fwe:256,cwe:256' \
API_BASE="http://${HOST}:${PORT}" \
EVAL_DATA="${FAST_DATASET}" \
RESULT_DIR="${RESULT_DIR}" \
bash "${FAST_SCRIPT}" 2>&1 | tee -a "${FAST_LOG}"

echo "[fastcheck] done $(date '+%F %T')" | tee -a "${FAST_LOG}"
