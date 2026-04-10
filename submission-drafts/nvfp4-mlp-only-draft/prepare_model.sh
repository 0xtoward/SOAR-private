#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
  echo "Usage: bash prepare_model.sh --input <path> --output <path>" >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/quantize_modelopt_nvfp4_conservative.py" \
  --help >/dev/null

run_attempt() {
  local attempt_name="$1"
  local quant_dtype="$2"
  local calib_file="$3"
  local max_length="$4"
  local min_good_batches="$5"

  echo "[prepare_model] attempt=${attempt_name} dtype=${quant_dtype} max_length=${max_length} calib=$(basename "${calib_file}")"
  rm -rf "${OUTPUT}"

  if python3 "${SCRIPT_DIR}/quantize_modelopt_nvfp4_conservative.py" \
    --model-path "${INPUT}" \
    --export-dir "${OUTPUT}" \
    --cfg mlp_only \
    --dtype "${quant_dtype}" \
    --local-calib-files "${calib_file}" \
    --local-calib-samples 128 \
    --max-length "${max_length}" \
    --min-good-batches "${min_good_batches}" \
    --overwrite; then
    echo "[prepare_model] success on ${attempt_name}"
    return 0
  fi

  echo "[prepare_model] failed on ${attempt_name}"
  return 1
}

run_attempt \
  "curated16k_bf16" \
  "bfloat16" \
  "${SCRIPT_DIR}/calibration/calibration_curated_16k_v2.jsonl" \
  "16384" \
  "64" && exit 0

run_attempt \
  "curated8k_bf16" \
  "bfloat16" \
  "${SCRIPT_DIR}/calibration/calibration_curated_16k_v2.jsonl" \
  "8192" \
  "64" && exit 0

run_attempt \
  "legacy2048_fp16" \
  "float16" \
  "${SCRIPT_DIR}/calibration/calibration_data_merged.jsonl" \
  "2048" \
  "96" && exit 0

echo "[prepare_model] all attempts failed" >&2
exit 1
