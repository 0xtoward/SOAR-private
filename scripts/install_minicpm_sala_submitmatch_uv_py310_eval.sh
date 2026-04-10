#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/sglang}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/sglang_submitmatch_uv_py310_eval_env}"
SGLANG_SRC="${SGLANG_SRC:-${REPO_ROOT}/python}"
REQUIRED_PY="${REQUIRED_PY:-3.10}"

UV_INDEX_URL_DEFAULT="https://mirrors.aliyun.com/pypi/simple/"
TORCH_WHEEL="${TORCH_WHEEL:-/root/autodl-tmp/cache/torch-2.9.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-/root/autodl-tmp/SOAR-Toolkit/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy HF_ENDPOINT
export UV_INDEX_URL="${UV_INDEX_URL:-${UV_INDEX_URL_DEFAULT}}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-${UV_INDEX_URL_DEFAULT}}"
export PIP_EXTRA_INDEX_URL=""
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"
if [[ -d /root/miniconda3/envs/py310/lib ]]; then
  export LD_LIBRARY_PATH="/root/miniconda3/envs/py310/lib:${LD_LIBRARY_PATH:-}"
fi

echo "============================================"
echo " MiniCPM-SALA Eval/Serve Bootstrap (uv, py310)"
echo "============================================"
echo "Root Directory: ${REPO_ROOT}"
echo "Eval env:       ${VENV_DIR}"
echo "PyPI mirror:    ${UV_INDEX_URL}"
echo "Torch wheel:    ${TORCH_WHEEL}"
echo "flash_attn:     ${FLASH_ATTN_WHEEL}"
echo "sglang source:  ${SGLANG_SRC}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[0/6] installing uv from domestic mirror..."
  if [[ -x "${REPO_ROOT}/sglang_minicpm_sala_env/bin/pip" ]]; then
    "${REPO_ROOT}/sglang_minicpm_sala_env/bin/pip" install -i "${UV_INDEX_URL}" uv
    export PATH="${REPO_ROOT}/sglang_minicpm_sala_env/bin:${PATH}"
  else
    python3.10 -m ensurepip --upgrade
    python3.10 -m pip install -i "${UV_INDEX_URL}" uv
  fi
fi

for wheel_path in "${TORCH_WHEEL}" "${FLASH_ATTN_WHEEL}"; do
  if [[ ! -f "${wheel_path}" ]]; then
    echo "missing required wheel: ${wheel_path}" >&2
    exit 1
  fi
done

echo "[1/6] initializing submodules..."
cd "${REPO_ROOT}"
git submodule update --init --recursive

echo "[2/6] choosing python ${REQUIRED_PY}..."
if command -v "python${REQUIRED_PY}" >/dev/null 2>&1; then
  UV_PYTHON="$(command -v python${REQUIRED_PY})"
else
  uv python install "${REQUIRED_PY}"
  UV_PYTHON="$(uv python find --python-preference only-managed "${REQUIRED_PY}")"
fi
echo "python=${UV_PYTHON} ($(${UV_PYTHON} --version))"

echo "[3/6] creating uv venv..."
rm -rf "${VENV_DIR}"
uv venv --python "${UV_PYTHON}" "${VENV_DIR}"

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

echo "[4/6] installing wheel-pinned binary stack..."
uv pip install "cmake>=3.26"
uv pip install --upgrade pip "setuptools>=78.1.1" wheel
uv pip install --no-deps "${TORCH_WHEEL}"
uv pip install --no-deps "${FLASH_ATTN_WHEEL}"
uv pip install --no-deps \
  "nvidia-cusparselt-cu12" \
  "nvidia-nvshmem-cu12" \
  "nvidia-nvjitlink-cu12" \
  "nvidia-cublas-cu12" \
  "nvidia-curand-cu12" \
  "nvidia-cufft-cu12" \
  "nvidia-cusolver-cu12" \
  "nvidia-cusparse-cu12" \
  "nvidia-cufile-cu12" \
  "nvidia-cuda-nvrtc-cu12" \
  "nvidia-cuda-cupti-cu12" \
  "nvidia-nccl-cu12" \
  "nvidia-cudnn-cu12"

echo "[5/6] installing serve/runtime python stack..."
uv pip install --no-deps \
  "transformers==4.57.1" \
  "accelerate==1.13.0" \
  "sentencepiece==0.2.1" \
  "huggingface_hub==0.36.2" \
  "tokenizers==0.22.2" \
  "safetensors==0.7.0" \
  "triton==3.5.1" \
  "einops" \
  "numpy==2.2.6" \
  "sympy" \
  "networkx" \
  "threadpoolctl" \
  "protobuf" \
  "pillow" \
  "regex" \
  "tqdm" \
  "pyyaml" \
  "requests" \
  "packaging" \
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
  "certifi" \
  "charset-normalizer" \
  "idna" \
  "urllib3" \
  "six" \
  "fastapi" \
  "uvicorn" \
  "uvloop" \
  "orjson" \
  "msgspec" \
  "psutil" \
  "pybase64" \
  "pydantic" \
  "setproctitle" \
  "interegular" \
  "partial-json-parser" \
  "python-multipart" \
  "prometheus-client" \
  "pyzmq" \
  "anyio" \
  "httpcore" \
  "h11" \
  "sniffio" \
  "jinja2" \
  "markupsafe"

# These two families were missing in the live eval env and caused server startup
# failures even though editable sglang itself was installed successfully.
uv pip install \
  "IPython" \
  "pydantic==2.12.0" \
  "pydantic-core==2.41.1" \
  "annotated-types" \
  "typing-inspection"

uv pip install --no-deps -e "${SGLANG_SRC}"

echo "[6/6] final verification..."
python - <<'PY'
import importlib.metadata as im
import importlib.util

required = [
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
    "sentencepiece",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "tqdm",
]

for name in required:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing import: {name}")

from IPython.display import HTML, display  # noqa: F401
from pydantic import BaseModel  # noqa: F401

tf_version = im.version("transformers")
if tf_version != "4.57.1":
    raise SystemExit(f"transformers version mismatch: {tf_version} != 4.57.1")

print("[eval-env] torch=", im.version("torch"))
print("[eval-env] transformers=", tf_version)
print("[eval-env] sglang=", im.version("sglang"))
print("[eval-env] flash-attn=", im.version("flash-attn"))
PY

echo ""
echo "============================================"
echo " Eval/serve uv installation complete!"
echo "============================================"
echo "source ${VENV_DIR}/bin/activate"
