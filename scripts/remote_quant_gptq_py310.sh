#!/usr/bin/env bash
set -euo pipefail

ENV="${ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models}"
CALIB_PATH="${CALIB_PATH:-/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_pg19_v2.jsonl}"
OUT_PATH="${OUT_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-quantcheck}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
DTYPE="${DTYPE:-float16}"
SCRIPT_SRC="${SCRIPT_SRC:-/root/autodl-tmp/codex-drafts/quantize_gptq_w4a16.py}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_quantcheck.log}"

export LD_LIBRARY_PATH="/root/miniconda3/envs/py310/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "${LOG_PATH}")"

echo "[quantcheck] start $(date '+%F %T')" | tee "${LOG_PATH}"
echo "[quantcheck] env=${ENV}" | tee -a "${LOG_PATH}"
echo "[quantcheck] model=${MODEL_PATH}" | tee -a "${LOG_PATH}"
echo "[quantcheck] calib=${CALIB_PATH}" | tee -a "${LOG_PATH}"
echo "[quantcheck] out=${OUT_PATH}" | tee -a "${LOG_PATH}"
echo "[quantcheck] samples=${NUM_SAMPLES} dtype=${DTYPE}" | tee -a "${LOG_PATH}"

rm -rf "${OUT_PATH}"

"${ENV}/bin/python" "${SCRIPT_SRC}" \
  --model-path "${MODEL_PATH}" \
  --calibration-path "${CALIB_PATH}" \
  --output-path "${OUT_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --dtype "${DTYPE}" 2>&1 | tee -a "${LOG_PATH}"

echo "[quantcheck] done $(date '+%F %T')" | tee -a "${LOG_PATH}"
