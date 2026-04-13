#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/submission_env_common.sh"

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

echo "[prepare_model] start $(date '+%F %T')"
echo "[prepare_model] input=${INPUT}"
echo "[prepare_model] output=${OUTPUT}"
echo "[prepare_model] quant_env=${SOAR_QUANT_ENV}"

soar_ensure_uv
soar_create_overlay_env "${SOAR_QUANT_ENV}"

QUANT_PY="${SOAR_QUANT_ENV}/bin/python"
TMPDIR="${SOAR_RUNTIME_DIR}/tmp"
PATCHED_INPUT_DIR="${SOAR_RUNTIME_DIR}/patched_input_model"
HF_MODULES_CACHE_DIR="${SOAR_RUNTIME_DIR}/hf_modules_cache"
mkdir -p "${TMPDIR}"
mkdir -p "${HF_MODULES_CACHE_DIR}"
export TMPDIR
export HF_MODULES_CACHE="${HF_MODULES_CACHE_DIR}"

[[ -f "${SOAR_FLASH_ATTN_WHEEL}" ]] || soar_fail "[prepare_model] missing flash_attn wheel: ${SOAR_FLASH_ATTN_WHEEL}"
[[ -f "${SOAR_GPTQMODEL_WHEEL}" ]] || soar_fail "[prepare_model] missing gptqmodel wheel: ${SOAR_GPTQMODEL_WHEEL}"
[[ -d "${SOAR_INFLLM_DIR}" ]] || soar_fail "[prepare_model] missing bundled infllm_v2 source tree: ${SOAR_INFLLM_DIR}"

uv pip install --python "${QUANT_PY}" --upgrade pip "setuptools>=78.1.1" wheel packaging "cmake>=3.26"
soar_require_torch "${QUANT_PY}" "quant env"
uv pip install --python "${QUANT_PY}" --no-deps "${SOAR_FLASH_ATTN_WHEEL}" "${SOAR_GPTQMODEL_WHEEL}"
uv pip install --python "${QUANT_PY}" --no-deps "ninja"

uv pip install --python "${QUANT_PY}" --no-deps \
  "transformers==5.5.0" \
  "accelerate==1.13.0" \
  "optimum==2.1.0" \
  "sentencepiece==0.2.1" \
  "huggingface_hub==1.10.1" \
  "tokenizers==0.22.2" \
  "safetensors==0.7.0" \
  "datasets==4.8.4" \
  "pyarrow==23.0.1" \
  "numpy==2.2.6" \
  "threadpoolctl==3.6.0" \
  "device-smi==0.5.3" \
  "protobuf==7.34.1" \
  "pillow==12.2.0" \
  "pypcre==0.2.15" \
  "hf_transfer==0.1.9" \
  "hf_xet" \
  "tokenicer==0.0.10" \
  "logbar==0.2.3" \
  "maturin==1.13.1" \
  "torchao==0.17.0" \
  "kernels==0.12.3" \
  "defuser==0.0.16" \
  "regex" \
  "tqdm" \
  "pyyaml" \
  "requests" \
  "filelock" \
  "typing_extensions" \
  "fsspec" \
  "aiohttp" \
  "multidict" \
  "yarl" \
  "attrs" \
  "frozenlist" \
  "aiosignal" \
  "propcache" \
  "pandas" \
  "python-dateutil" \
  "pytz" \
  "tzdata" \
  "xxhash" \
  "multiprocess" \
  "dill" \
  "httpx" \
  "anyio" \
  "httpcore" \
  "h11" \
  "sniffio" \
  "certifi" \
  "charset-normalizer" \
  "idna" \
  "urllib3" \
  "six" \
  "einops"

soar_copy_vendor_fla "${QUANT_PY}"
uv pip install --python "${QUANT_PY}" --no-deps -e "${SCRIPT_DIR}/sglang/python"

soar_prepare_patched_model_dir "${INPUT}" "${PATCHED_INPUT_DIR}"
echo "[prepare_model] patched_input=${PATCHED_INPUT_DIR}"

"${QUANT_PY}" - <<'PY'
import torch
print("[prepare_model] torch import OK", torch.__version__)
PY

export PYTHONPATH="${SOAR_INFLLM_DIR}:${PYTHONPATH:-}"

if ! "${QUANT_PY}" - <<'PY'
import infllm_v2  # noqa: F401
print("[prepare_model] infllm_v2 import OK from vendored tree")
PY
then
  echo "[prepare_model] vendored infllm_v2 import failed; building from source"
  (
    cd "${SOAR_INFLLM_DIR}"
    "${QUANT_PY}" setup.py install
  )
fi

"${QUANT_PY}" "${SCRIPT_DIR}/quantize_gptq_w4a16.py" --help >/dev/null

run_attempt() {
  local attempt_name="$1"
  local calib_file="$2"
  local num_samples="$3"
  local quant_dtype="$4"

  echo "[prepare_model] attempt=${attempt_name} calib=$(basename "${calib_file}") samples=${num_samples} dtype=${quant_dtype}"
  rm -rf "${OUTPUT}"

  if PYTHONPATH="${SOAR_INFLLM_DIR}:${PYTHONPATH:-}" \
    "${QUANT_PY}" "${SCRIPT_DIR}/quantize_gptq_w4a16.py" \
      --model-path "${PATCHED_INPUT_DIR}" \
      --calibration-path "${calib_file}" \
      --output-path "${OUTPUT}" \
      --num-samples "${num_samples}" \
      --dtype "${quant_dtype}" \
      --bits 4 \
      --group-size 128 \
      --batch-size 1 \
      --max-shard-size 4GB; then
    if [[ -f "${OUTPUT}/quantize_config.json" ]]; then
      soar_sync_tokenizer_assets "${PATCHED_INPUT_DIR}" "${OUTPUT}"
      if [[ -x "${SOAR_EVAL_ENV}/bin/python" ]]; then
        "${SOAR_EVAL_ENV}/bin/python" - <<PY || soar_fail "[prepare_model] eval-env tokenizer validation failed after ${attempt_name}"
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("${OUTPUT}", trust_remote_code=True)
print("[prepare_model] eval-env tokenizer OK", tok.__class__.__name__)
PY
      fi
      echo "[prepare_model] success on ${attempt_name}"
      return 0
    fi
    echo "[prepare_model] ${attempt_name} finished but quantize_config.json is missing" >&2
  fi

  echo "[prepare_model] failed on ${attempt_name}" >&2
  return 1
}

run_attempt \
  "stopaligned64k_v1" \
  "${SCRIPT_DIR}/calibration_gptq_w4a16_stopaligned64k_v1.jsonl" \
  "64" \
  "float16" && exit 0

run_attempt \
  "true160k_fp16_32" \
  "${SCRIPT_DIR}/calibration_gptq_w4a16_true160k_v1.jsonl" \
  "32" \
  "float16" && exit 0

run_attempt \
  "pg19_32k_fp16_64" \
  "${SCRIPT_DIR}/calibration_gptq_w4a16_pg19_v2.jsonl" \
  "64" \
  "float16" && exit 0

echo "[prepare_model] all attempts failed" >&2
exit 1
