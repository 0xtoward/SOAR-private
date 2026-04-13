#!/usr/bin/env bash
set -euo pipefail

ENV="${ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-quantcheck}"
SCRIPT_DIR="${SCRIPT_DIR:-/root/autodl-tmp/SOAR-Toolkit}"
MEDIUM_DATASET="${MEDIUM_DATASET:-${SCRIPT_DIR}/eval_dataset/perf_public_medium_eval_v1.jsonl}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/test_results/gptq_py310_mediumcheck_$(date +%Y%m%d_%H%M%S)}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
PREFERRED_SAMPLING_PARAMS="${PREFERRED_SAMPLING_PARAMS:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-}"

if [[ -z "${PREFERRED_SAMPLING_PARAMS}" && ( -n "${REPETITION_PENALTY}" || -n "${FREQUENCY_PENALTY}" || -n "${PRESENCE_PENALTY}" ) ]]; then
  PREFERRED_SAMPLING_PARAMS="$(
    python3 - <<PY
import json
payload = {}
if "${REPETITION_PENALTY}":
    payload["repetition_penalty"] = float("${REPETITION_PENALTY}")
if "${FREQUENCY_PENALTY}":
    payload["frequency_penalty"] = float("${FREQUENCY_PENALTY}")
if "${PRESENCE_PENALTY}":
    payload["presence_penalty"] = float("${PRESENCE_PENALTY}")
print(json.dumps(payload, separators=(",", ":")))
PY
  )"
fi

export LD_LIBRARY_PATH="/root/miniconda3/envs/py310/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV}/bin:${PATH}"
export PYTHONUNBUFFERED=1

mkdir -p "${RESULT_DIR}"
cd "${SCRIPT_DIR}"

SERVER_LOG="${RESULT_DIR}/server.log"
EVAL_LOG="${RESULT_DIR}/eval_model.log"

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
    ("annotated_types", "annotated-types"),
    ("typing_inspection", "typing-inspection"),
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
    echo "[mediumcheck] installing missing eval deps: ${missing[*]}" | tee -a "${EVAL_LOG}"
    "${ENV}/bin/python" -m pip install -i "https://mirrors.aliyun.com/pypi/simple/" "${missing[@]}" >> "${EVAL_LOG}" 2>&1
  fi
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[mediumcheck] start $(date '+%F %T')" | tee "${EVAL_LOG}"
echo "[mediumcheck] env=${ENV}" | tee -a "${EVAL_LOG}"
echo "[mediumcheck] model=${MODEL_PATH}" | tee -a "${EVAL_LOG}"
echo "[mediumcheck] result_dir=${RESULT_DIR}" | tee -a "${EVAL_LOG}"
echo "[mediumcheck] preferred_sampling_params=${PREFERRED_SAMPLING_PARAMS:-<unset>}" | tee -a "${EVAL_LOG}"

ensure_eval_runtime

pkill -f "sglang.launch_server --host ${HOST} --port ${PORT}" >/dev/null 2>&1 || true
pkill -f "eval_model.py --api_base http://${HOST}:${PORT}" >/dev/null 2>&1 || true

SERVER_ARGS=(
  --model-path "${MODEL_PATH}"
  --quantization gptq_marlin
  --dtype float16
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --attention-backend flashinfer
  --force-dense-minicpm
  --chunked-prefill-size 8192
  --disable-radix-cache
  --disable-cuda-graph
  --skip-server-warmup
  --enable-request-time-stats-logging
)

if [[ -n "${PREFERRED_SAMPLING_PARAMS}" ]]; then
  SERVER_ARGS+=(--preferred-sampling-params "${PREFERRED_SAMPLING_PARAMS}")
fi

"${ENV}/bin/python" -m sglang.launch_server \
  "${SERVER_ARGS[@]}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null; then
    echo "[mediumcheck] server ready" | tee -a "${EVAL_LOG}"
    break
  fi
  sleep 2
done

if ! curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null; then
  echo "[mediumcheck] server failed to become ready" | tee -a "${EVAL_LOG}"
  exit 1
fi

stdbuf -oL -eL "${ENV}/bin/python" eval_model.py \
  --model_path "${MODEL_PATH}" \
  --api_base "http://${HOST}:${PORT}" \
  --data_path "${MEDIUM_DATASET}" \
  --max_seq_len 262144 \
  --concurrency 8 \
  --verbose 2>&1 | tee -a "${EVAL_LOG}"

echo "[mediumcheck] done $(date '+%F %T')" | tee -a "${EVAL_LOG}"
