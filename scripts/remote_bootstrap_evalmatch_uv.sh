#!/usr/bin/env bash
set -euo pipefail

# Minimal eval/serve bootstrap for the SOAR probe-like environment.
# Goal:
#   - keep torch 2.9.1+cu128 + transformers 4.57.1 from the base probe env
#   - add only flash_attn plus missing serve/runtime dependencies
#   - avoid any quantization-only stack such as transformers 5.5.0 / gptqmodel
#   - use uv and domestic mirrors only; no proxy, no network_turbo

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/sglang}"
BASE_ENV="${BASE_ENV:-/root/autodl-tmp/sglang/sglang_minicpm_sala_env}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/sglang_evalmatch_uv_py310_env}"
REQUIRED_PY="${REQUIRED_PY:-3.10}"
UV_INDEX_URL_DEFAULT="https://mirrors.aliyun.com/pypi/simple/"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-/root/autodl-tmp/SOAR-Toolkit/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl}"
SGLANG_SRC="${SGLANG_SRC:-${REPO_ROOT}/python}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy HF_ENDPOINT
export UV_INDEX_URL="${UV_INDEX_URL:-${UV_INDEX_URL_DEFAULT}}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-${UV_INDEX_URL_DEFAULT}}"
export PIP_EXTRA_INDEX_URL=""
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"

echo "============================================"
echo " MiniCPM-SALA Eval/Serve Bootstrap (uv, py310)"
echo "============================================"
echo "Base env:       ${BASE_ENV}"
echo "Eval env:       ${VENV_DIR}"
echo "PyPI mirror:    ${UV_INDEX_URL}"
echo "flash_attn:     ${FLASH_ATTN_WHEEL}"
echo "sglang source:  ${SGLANG_SRC}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[0/7] installing uv from domestic mirror..."
  python3 -m pip install -i "${UV_INDEX_URL}" uv
fi

if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
  echo "missing flash_attn wheel: ${FLASH_ATTN_WHEEL}" >&2
  exit 1
fi

if [[ ! -x "${BASE_ENV}/bin/python3" ]]; then
  echo "missing base python: ${BASE_ENV}/bin/python3" >&2
  exit 1
fi

echo "[1/7] choosing python ${REQUIRED_PY}..."
if command -v "python${REQUIRED_PY}" >/dev/null 2>&1; then
  UV_PYTHON="$(command -v python${REQUIRED_PY})"
else
  uv python install "${REQUIRED_PY}"
  UV_PYTHON="$(uv python find --python-preference only-managed "${REQUIRED_PY}")"
fi
echo "python=${UV_PYTHON} ($(${UV_PYTHON} --version))"

echo "[2/7] creating uv venv with base site-packages..."
rm -rf "${VENV_DIR}"
"${BASE_ENV}/bin/python3" -m venv --system-site-packages "${VENV_DIR}"

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
echo "venv_python=$(python --version)"

if command -v g++ >/dev/null 2>&1; then
  export CC=gcc
  export CXX=g++
fi
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

install_if_missing_module() {
  local module_name="$1"
  local package_name="$2"
  if python - "$module_name" <<'PY'
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
  then
    echo "[eval-env] ${module_name} already available"
    return 0
  fi

  echo "[eval-env] install ${package_name}"
  if [[ "${package_name}" == *.whl || "${package_name}" == /* ]]; then
    UV_INDEX_URL="${UV_INDEX_URL}" uv pip install --no-deps "${package_name}"
  else
    UV_INDEX_URL="${UV_INDEX_URL}" uv pip install "${package_name}"
  fi
}

echo "[3/7] pinning the probe-matching base..."
UV_INDEX_URL="${UV_INDEX_URL}" uv pip install --upgrade pip setuptools wheel packaging

echo "[4/7] installing flash_attn from bundled wheel..."
install_if_missing_module "flash_attn" "${FLASH_ATTN_WHEEL}"

echo "[5/7] installing editable sglang (no deps to avoid pulling quant stack)..."
UV_INDEX_URL="${UV_INDEX_URL}" uv pip install --no-deps -e "${SGLANG_SRC}"

echo "[6/7] filling only the serve/runtime modules that the probe did not already provide..."
for spec in \
  "IPython:IPython" \
  "fastapi:fastapi" \
  "uvicorn:uvicorn" \
  "uvloop:uvloop" \
  "aiohttp:aiohttp" \
  "orjson:orjson" \
  "msgspec:msgspec" \
  "numpy:numpy" \
  "psutil:psutil" \
  "pybase64:pybase64" \
  "pydantic:pydantic" \
  "pydantic_core:pydantic-core" \
  "requests:requests" \
  "setproctitle:setproctitle" \
  "packaging:packaging" \
  "einops:einops" \
  "scipy:scipy" \
  "sentencepiece:sentencepiece" \
  "huggingface_hub:huggingface_hub" \
  "tokenizers:tokenizers" \
  "safetensors:safetensors" \
  "tqdm:tqdm" \
  "interegular:interegular" \
  "partial_json_parser:partial-json-parser" \
  "multipart:python-multipart" \
  "prometheus_client:prometheus-client" \
  "zmq:pyzmq" \
  "PIL:pillow"
do
  module_name="${spec%%:*}"
  package_name="${spec#*:}"
  install_if_missing_module "${module_name}" "${package_name}"
done

echo "[7/7] final verification..."
python - <<'PY'
import importlib.metadata as im
import importlib.util

mods = [
    "torch",
    "transformers",
    "sglang",
    "flash_attn",
    "IPython",
    "fastapi",
    "uvicorn",
    "uvloop",
    "aiohttp",
    "orjson",
    "msgspec",
    "numpy",
    "psutil",
    "pybase64",
    "pydantic",
    "pydantic_core",
    "requests",
    "setproctitle",
    "packaging",
    "einops",
    "scipy",
    "sentencepiece",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "tqdm",
]

for name in mods:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing import: {name}")

from IPython.display import HTML, display  # noqa: F401
from pydantic import BaseModel  # noqa: F401

print("[eval-env] torch=", im.version("torch"))
print("[eval-env] transformers=", im.version("transformers"))
print("[eval-env] sglang=", im.version("sglang"))
print("[eval-env] flash-attn=", im.version("flash-attn"))
PY

echo ""
echo "============================================"
echo " Eval/serve uv installation complete!"
echo "============================================"
echo "source ${VENV_DIR}/bin/activate"
