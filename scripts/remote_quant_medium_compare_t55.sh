#!/usr/bin/env bash
set -euo pipefail

ENV="/root/autodl-tmp/sglang/sglang_t550_py312_env"
SERVE_ENV="/root/autodl-tmp/sglang/sglang_minicpm_sala_env"
SERVE_PY="${SERVE_ENV}/bin/python3"
SOAR_ROOT="/root/autodl-tmp/SOAR-Toolkit"
DRAFT_DIR="/root/autodl-tmp/codex-drafts/w4a16-marlin-draft"
MASTER_LOG="/root/autodl-tmp/quant_fast_t55_master.log"
MEDIUM_DATA="${SOAR_ROOT}/eval_dataset/perf_public_medium_eval_v1.jsonl"
BASE_MODEL="/root/autodl-tmp/models"
MODEL_A="/root/autodl-tmp/models-gptq-w4a16-t550-pg19-64"
MODEL_B="/root/autodl-tmp/models-gptq-w4a16-t550-singlelong-49"
CALIB_B="${DRAFT_DIR}/calibration_gptq_w4a16_true160k_singlelong_v1.jsonl"
PORT_A="${PORT_A:-30004}"
PORT_B="${PORT_B:-30005}"
RUN_ID="gptq_t550_medium_compare_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SOAR_ROOT}/test_results/${RUN_ID}"

mkdir -p "${RUN_DIR}"
echo "RUN_ID=${RUN_ID}" | tee "${RUN_DIR}/launch.txt"
echo "MODEL_A=${MODEL_A}" | tee -a "${RUN_DIR}/launch.txt"
echo "MODEL_B=${MODEL_B}" | tee -a "${RUN_DIR}/launch.txt"
echo "CALIB_B=${CALIB_B}" | tee -a "${RUN_DIR}/launch.txt"

wait_for_master() {
  local waited=0
  while (( waited < 14400 )); do
    if [[ -f "${MASTER_LOG}" ]]; then
      if grep -q "\[quant-fast\] done" "${MASTER_LOG}"; then
        echo "[queue] detected current quant+fast completion" | tee -a "${RUN_DIR}/launch.txt"
        return 0
      fi
      if grep -q "\[quant-fast\] failed" "${MASTER_LOG}"; then
        echo "[queue] current quant+fast failed; skip compare queue" | tee -a "${RUN_DIR}/launch.txt"
        return 1
      fi
    fi
    sleep 15
    waited=$((waited + 15))
  done
  echo "[queue] timeout waiting for ${MASTER_LOG}" | tee -a "${RUN_DIR}/launch.txt"
  return 1
}

wait_for_env_ready() {
  local waited=0
  while (( waited < 3600 )); do
    if "${ENV}/bin/python" - <<'PY' >/tmp/quant_medium_env_ready.log 2>&1
import importlib.metadata as im
for p in ("transformers", "gptqmodel", "accelerate"):
    print(p, im.version(p))
PY
    then
      cat /tmp/quant_medium_env_ready.log | tee -a "${RUN_DIR}/launch.txt"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "[queue] eval-match env never became ready" | tee -a "${RUN_DIR}/launch.txt"
  return 1
}

run_medium() {
  local label="$1"
  local model_path="$2"
  local port="$3"
  local outdir="${RUN_DIR}/${label}"
  mkdir -p "${outdir}"

  pkill -f "sglang.launch_server --host 127.0.0.1 --port ${port}" >/dev/null 2>&1 || true
  nohup "${SERVE_PY}" -m sglang.launch_server \
    --model-path "${model_path}" \
    --quantization gptq_marlin \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "${port}" \
    --dtype float16 \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 8192 \
    --skip-server-warmup \
    --disable-radix-cache \
    --disable-cuda-graph \
    --dense-as-sparse \
    > "${outdir}/server.log" 2>&1 &
  echo $! > "${outdir}/server.pid"

  local ready=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "[${label}] server readiness probe timed out" | tee -a "${RUN_DIR}/launch.txt"
    return 1
  fi

  (
    cd "${SOAR_ROOT}"
    export PATH="${SERVE_ENV}/bin:${PATH}"
    export VIRTUAL_ENV="${SERVE_ENV}"
    stdbuf -oL -eL "${SERVE_PY}" eval_model.py \
      --model_path "${model_path}" \
      --model_name "${model_path}" \
      --api_base "http://127.0.0.1:${port}" \
      --data_path "eval_dataset/$(basename "${MEDIUM_DATA}")" \
      --max_seq_len 262144 \
      --concurrency 8 \
      --verbose 2>&1 | tee "${outdir}/medium_eval.log"
  )

  kill "$(cat "${outdir}/server.pid")" >/dev/null 2>&1 || true
  pkill -f "sglang.launch_server --host 127.0.0.1 --port ${port}" >/dev/null 2>&1 || true
}

echo "[1/3] wait for current quant+fast" | tee -a "${RUN_DIR}/launch.txt"
wait_for_master || exit 0

echo "[env] wait for eval-match env readiness" | tee -a "${RUN_DIR}/launch.txt"
wait_for_env_ready || exit 1

echo "[2/3] medium on current t55 candidate" | tee -a "${RUN_DIR}/launch.txt"
if [[ -f "${MODEL_A}/quantize_config.json" ]]; then
  run_medium "current_pg19_64" "${MODEL_A}" "${PORT_A}" || true
else
  echo "[queue] current model missing: ${MODEL_A}" | tee -a "${RUN_DIR}/launch.txt"
fi

echo "[3/3] quantize singlelong + medium" | tee -a "${RUN_DIR}/launch.txt"
rm -rf "${MODEL_B}"
"${ENV}/bin/python" "${DRAFT_DIR}/quantize_gptq_w4a16.py" \
  --model-path "${BASE_MODEL}" \
  --calibration-path "${CALIB_B}" \
  --output-path "${MODEL_B}" \
  --num-samples 64 \
  --dtype float16 \
  --bits 4 \
  --group-size 128 \
  --batch-size 1 \
  --max-shard-size 4GB \
  2>&1 | tee "${RUN_DIR}/quantize_singlelong.log"

test -f "${MODEL_B}/quantize_config.json"
run_medium "singlelong_49" "${MODEL_B}" "${PORT_B}"

echo "[queue] done" | tee -a "${RUN_DIR}/launch.txt"
