#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/sglang"
VENV_DIR="${REPO_ROOT}/sglang_submitmatch_uv_py310_env"
DEPS_DIR="${REPO_ROOT}/3rdparty"
FLASH_ATTN_WHEEL="/root/autodl-tmp/SOAR-Toolkit/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl"
UV_INDEX_URL_DEFAULT="https://mirrors.aliyun.com/pypi/simple/"
REQUIRED_PY="3.10"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy HF_ENDPOINT
export UV_INDEX_URL="${UV_INDEX_URL:-${UV_INDEX_URL_DEFAULT}}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-${UV_INDEX_URL_DEFAULT}}"
export PIP_EXTRA_INDEX_URL=""

echo "============================================"
echo " MiniCPM-SALA Submit-Match Installation (uv)"
echo "============================================"
echo "Root Directory: ${REPO_ROOT}"
echo "PyPI mirror:    ${UV_INDEX_URL}"
echo "flash_attn:     ${FLASH_ATTN_WHEEL}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[0/6] installing uv from domestic mirror..."
  python3 -m pip install -i "${UV_INDEX_URL}" uv
fi

if [[ ! -x "${FLASH_ATTN_WHEEL}" && ! -f "${FLASH_ATTN_WHEEL}" ]]; then
  echo "missing flash_attn wheel: ${FLASH_ATTN_WHEEL}" >&2
  exit 1
fi

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

echo "[4/6] installing core packages..."
uv pip install "cmake>=3.26"
uv pip install --upgrade pip setuptools wheel
uv pip install --no-deps "${FLASH_ATTN_WHEEL}"
uv pip install -e "${REPO_ROOT}/python[all]"

echo "[5/6] building repo-local CUDA deps..."
cd "${DEPS_DIR}/infllmv2_cuda_impl"
git submodule update --init --recursive
python setup.py install

cd "${DEPS_DIR}/sparse_kernel"
python setup.py install

echo "[6/6] installing extra libs from domestic mirror..."
uv pip install tilelang flash-linear-attention

echo ""
echo "============================================"
echo " Submit-match uv installation complete!"
echo "============================================"
echo "source ${VENV_DIR}/bin/activate"
