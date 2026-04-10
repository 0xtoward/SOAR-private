#!/usr/bin/env bash
set -euo pipefail

ENV="/root/autodl-tmp/sglang/sglang_t550_py312_env"
SERVE_ENV="/root/autodl-tmp/sglang/sglang_minicpm_sala_env"
SERVE_PY="${SERVE_ENV}/bin/python3"
SYNC_SCRIPT="/root/autodl-tmp/remote_env_sync_evalmatch.sh"
DRAFT_DIR="/root/autodl-tmp/codex-drafts/w4a16-marlin-draft"
SOAR_ROOT="/root/autodl-tmp/SOAR-Toolkit"
MODEL_IN="/root/autodl-tmp/models"
MODEL_OUT="/root/autodl-tmp/models-gptq-w4a16-t550-pg19-64"
CALIB="${DRAFT_DIR}/calibration_gptq_w4a16_pg19_v2.jsonl"
PORT="${PORT:-30003}"
FAST_DATA="${SOAR_ROOT}/eval_dataset/perf_public_fast_eval.jsonl"
RUN_ID="gptq_t550_fast_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SOAR_ROOT}/test_results/${RUN_ID}"
FAST_EVAL_TASK_CAPS="${FAST_EVAL_TASK_CAPS:-mcq:64,qa:128,niah:128,fwe:256,cwe:256}"

mkdir -p "${RUN_DIR}"
echo "RUN_ID=${RUN_ID}" | tee "${RUN_DIR}/launch.txt"
echo "ENV=${ENV}" | tee -a "${RUN_DIR}/launch.txt"
echo "SERVE_ENV=${SERVE_ENV}" | tee -a "${RUN_DIR}/launch.txt"
echo "MODEL_OUT=${MODEL_OUT}" | tee -a "${RUN_DIR}/launch.txt"
echo "FAST_EVAL_TASK_CAPS=${FAST_EVAL_TASK_CAPS}" | tee -a "${RUN_DIR}/launch.txt"

on_error() {
  echo "[quant-fast] failed" | tee -a "${RUN_DIR}/launch.txt"
}
trap on_error ERR

cleanup() {
  if [[ -f "${RUN_DIR}/server.pid" ]]; then
    kill "$(cat "${RUN_DIR}/server.pid")" >/dev/null 2>&1 || true
  fi
  pkill -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[1/4] sync eval-match py310 env via ${SYNC_SCRIPT}" | tee -a "${RUN_DIR}/launch.txt"
bash "${SYNC_SCRIPT}" 2>&1 | tee "${RUN_DIR}/env_sync.log"

echo "[2/4] verify transformers version" | tee -a "${RUN_DIR}/launch.txt"
"${ENV}/bin/python" - <<'PY' 2>&1 | tee -a "${RUN_DIR}/env_sync.log"
import importlib.metadata as im
print("transformers", im.version("transformers"))
print("gptqmodel", im.version("gptqmodel"))
print("accelerate", im.version("accelerate"))
print("sentencepiece", im.version("sentencepiece"))
print("huggingface_hub", im.version("huggingface_hub"))
PY

if ! "${ENV}/bin/python" - <<'PY'
import importlib.metadata as im
raise SystemExit(0 if im.version("transformers") == "5.5.0" else 1)
PY
then
  echo "[quant-fast] transformers is not 5.5.0" | tee -a "${RUN_DIR}/launch.txt"
  exit 1
fi

echo "[3/4] GPTQ quantization" | tee -a "${RUN_DIR}/launch.txt"
rm -rf "${MODEL_OUT}"
"${ENV}/bin/python" "${DRAFT_DIR}/quantize_gptq_w4a16.py" \
  --model-path "${MODEL_IN}" \
  --calibration-path "${CALIB}" \
  --output-path "${MODEL_OUT}" \
  --num-samples 64 \
  --dtype float16 \
  --bits 4 \
  --group-size 128 \
  --batch-size 1 \
  --max-shard-size 4GB \
  2>&1 | tee "${RUN_DIR}/quantize.log"

test -f "${MODEL_OUT}/quantize_config.json"

echo "[4/4] launch server + fast eval" | tee -a "${RUN_DIR}/launch.txt"
pkill -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1 || true
nohup "${SERVE_PY}" -m sglang.launch_server \
  --model-path "${MODEL_OUT}" \
  --quantization gptq_marlin \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --enable-request-time-stats-logging \
  > "${RUN_DIR}/server.log" 2>&1 &
echo $! > "${RUN_DIR}/server.pid"

READY=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "${READY}" -ne 1 ]]; then
  echo "[quant-fast] server readiness probe timed out" | tee -a "${RUN_DIR}/launch.txt"
  exit 1
fi

cd "${SOAR_ROOT}"
export PATH="${SERVE_ENV}/bin:${PATH}"
export VIRTUAL_ENV="${SERVE_ENV}"
API_BASE="http://127.0.0.1:${PORT}" TIMEOUT=0 MINI_PER_TASK=999 EVAL_USE_ALL_ROWS=1 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 \
  EVAL_TASK_MAX_TOKENS="${FAST_EVAL_TASK_CAPS}" \
  EVAL_DATA="${FAST_DATA}" \
  bash "${SOAR_ROOT}/mini_test.sh" 2>&1 | tee "${RUN_DIR}/fast_eval.log"

echo "[quant-fast] done" | tee -a "${RUN_DIR}/launch.txt"
