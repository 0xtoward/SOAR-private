#!/usr/bin/env bash

# This file is sourced by the SOAR platform. Keep it simple and avoid strict
# shell options that could leak into the platform entrypoint.

SOAR_BASE_PYTHON="${SOAR_BASE_PYTHON:-$(command -v python || true)}"
export SOAR_BASE_PYTHON

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/submission_env_common.sh"

echo "[prepare_env] start $(date '+%F %T')"
echo "[prepare_env] submission_root=${SOAR_SUBMISSION_ROOT}"
echo "[prepare_env] eval_env=${SOAR_EVAL_ENV}"
echo "[prepare_env] index=${SOAR_INDEX_URL}"

soar_ensure_uv
soar_create_overlay_env "${SOAR_EVAL_ENV}"

EVAL_PY="${SOAR_EVAL_ENV}/bin/python"

[[ -f "${SOAR_FLASH_ATTN_WHEEL}" ]] || soar_fail "[prepare_env] missing flash_attn wheel: ${SOAR_FLASH_ATTN_WHEEL}"

soar_uv pip install --python "${EVAL_PY}" --upgrade pip setuptools wheel packaging
soar_require_torch "${EVAL_PY}" "eval env"
soar_uv pip install --python "${EVAL_PY}" --no-deps "${SOAR_FLASH_ATTN_WHEEL}"

soar_uv pip install --python "${EVAL_PY}" --no-deps \
  "transformers==4.57.1" \
  "accelerate==1.13.0" \
  "sentencepiece==0.2.1" \
  "huggingface_hub==0.36.2" \
  "tokenizers==0.22.2" \
  "safetensors==0.7.0" \
  "triton==3.5.1" \
  "flashinfer_python==0.5.3" \
  "flashinfer_cubin==0.5.3" \
  "fastapi" \
  "uvicorn" \
  "uvloop" \
  "orjson" \
  "msgspec" \
  "psutil" \
  "pybase64" \
  "pydantic==2.12.0" \
  "pydantic-core==2.41.1" \
  "annotated-types" \
  "typing-inspection" \
  "setproctitle" \
  "interegular" \
  "partial-json-parser" \
  "python-multipart" \
  "prometheus-client" \
  "pyzmq" \
  "aiohttp" \
  "multidict" \
  "yarl" \
  "attrs" \
  "frozenlist" \
  "aiosignal" \
  "propcache" \
  "requests" \
  "packaging" \
  "filelock" \
  "typing_extensions" \
  "fsspec" \
  "numpy==2.2.6" \
  "scipy" \
  "einops" \
  "jinja2" \
  "markupsafe" \
  "httpcore" \
  "h11" \
  "sniffio" \
  "IPython" \
  "ninja"

soar_uv pip install --python "${EVAL_PY}" --no-deps -e "${SCRIPT_DIR}/sglang/python"
soar_copy_vendor_fla "${EVAL_PY}"

if ! "${EVAL_PY}" - <<'PY'
import importlib.util
import importlib.metadata as im

required = [
    "torch",
    "transformers",
    "sglang",
    "flash_attn",
    "flashinfer",
    "IPython",
    "pydantic",
    "pydantic_core",
    "fastapi",
    "uvicorn",
    "uvloop",
    "aiohttp",
    "orjson",
    "msgspec",
    "numpy",
    "psutil",
    "pybase64",
    "setproctitle",
    "einops",
    "fla",
]
for name in required:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing eval import: {name}")

print("[prepare_env] torch=", im.version("torch"))
print("[prepare_env] transformers=", im.version("transformers"))
print("[prepare_env] sglang=", im.version("sglang"))
print("[prepare_env] flash-attn=", im.version("flash-attn"))
PY
then
  soar_fail "[prepare_env] eval env validation failed"
fi

export SOAR_SUBMISSION_ROOT="${SOAR_SUBMISSION_ROOT}"
export SOAR_EVAL_ENV="${SOAR_EVAL_ENV}"
export SOAR_QUANT_ENV="${SOAR_QUANT_ENV}"
export VIRTUAL_ENV="${SOAR_EVAL_ENV}"
export PATH="${SOAR_EVAL_ENV}/bin:${PATH}"
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend flashinfer --force-dense-minicpm --chunked-prefill-size 8192 --skip-server-warmup --disable-cuda-graph --cuda-graph-max-bs 1 --quantization gptq_marlin --dtype bfloat16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done $(date '+%F %T')"
