#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/sglang}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/sglang_submitmatch_uv_py310_gptq_env}"
DEPS_DIR="${REPO_ROOT}/3rdparty"
REQUIRED_PY="${REQUIRED_PY:-3.10}"

UV_INDEX_URL_DEFAULT="https://mirrors.aliyun.com/pypi/simple/"
TORCH_WHEEL="${TORCH_WHEEL:-/root/autodl-tmp/cache/torch-2.9.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-/root/autodl-tmp/SOAR-Toolkit/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl}"
GPTQMODEL_WHEEL="${GPTQMODEL_WHEEL:-/root/autodl-tmp/cache/gptqmodel-5.8.0+cu128torch2.9-cp310-cp310-linux_x86_64.whl}"
FLA_SOURCE_DIR="${FLA_SOURCE_DIR:-${REPO_ROOT}/sglang_minicpm_sala_env/lib/python3.12/site-packages/fla}"
SKIP_REPO_BUILD="${SKIP_REPO_BUILD:-0}"

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
echo " MiniCPM-SALA Submit-Match Installation (uv, py310 + GPTQ)"
echo "============================================"
echo "Root Directory: ${REPO_ROOT}"
echo "PyPI mirror:    ${UV_INDEX_URL}"
echo "UV cache:       ${UV_CACHE_DIR}"
echo "Torch wheel:    ${TORCH_WHEEL}"
echo "flash_attn:     ${FLASH_ATTN_WHEEL}"
echo "gptqmodel:      ${GPTQMODEL_WHEEL}"
echo "fla source:     ${FLA_SOURCE_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[0/7] installing uv from domestic mirror..."
  if [[ -x "${REPO_ROOT}/sglang_minicpm_sala_env/bin/pip" ]]; then
    "${REPO_ROOT}/sglang_minicpm_sala_env/bin/pip" install -i "${UV_INDEX_URL}" uv
    export PATH="${REPO_ROOT}/sglang_minicpm_sala_env/bin:${PATH}"
  else
    python3 -m ensurepip --upgrade
    python3 -m pip install -i "${UV_INDEX_URL}" uv
  fi
fi

for wheel_path in "${TORCH_WHEEL}" "${FLASH_ATTN_WHEEL}" "${GPTQMODEL_WHEEL}"; do
  if [[ ! -f "${wheel_path}" ]]; then
    echo "missing required wheel: ${wheel_path}" >&2
    exit 1
  fi
done

echo "[1/7] initializing submodules..."
cd "${REPO_ROOT}"
git submodule update --init --recursive

echo "[2/7] choosing python ${REQUIRED_PY}..."
if command -v "python${REQUIRED_PY}" >/dev/null 2>&1; then
  UV_PYTHON="$(command -v python${REQUIRED_PY})"
else
  uv python install "${REQUIRED_PY}"
  UV_PYTHON="$(uv python find --python-preference only-managed "${REQUIRED_PY}")"
fi
echo "python=${UV_PYTHON} ($(${UV_PYTHON} --version))"

echo "[3/7] creating uv venv..."
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

echo "[4/7] installing wheel-pinned binary stack..."
uv pip install "cmake>=3.26"
uv pip install --upgrade pip "setuptools>=78.1.1" wheel
uv pip install --no-deps "${TORCH_WHEEL}"
uv pip install --no-deps "${FLASH_ATTN_WHEEL}"
uv pip install --no-deps "${GPTQMODEL_WHEEL}"
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

echo "[5/7] installing pure-python stack from domestic mirror..."
uv pip install --no-deps \
  "transformers==5.5.0" \
  "accelerate==1.13.0" \
  "optimum==2.1.0" \
  "sentencepiece==0.2.1" \
  "huggingface_hub==1.10.1" \
  "tokenizers==0.22.2" \
  "safetensors==0.7.0" \
  "datasets==4.8.4" \
  "pyarrow==23.0.1" \
  "triton==3.5.1" \
  "einops"

uv pip install --no-deps \
  "numpy==2.2.6" \
  "sympy" \
  "networkx" \
  "threadpoolctl" \
  "device-smi" \
  "protobuf" \
  "pillow" \
  "pypcre" \
  "hf_transfer" \
  "psutil" \
  "httpx" \
  "jinja2" \
  "anyio" \
  "httpcore" \
  "h11" \
  "sniffio" \
  "tokenicer" \
  "logbar" \
  "maturin" \
  "torchao" \
  "kernels" \
  "defuser" \
  "regex" \
  "tqdm" \
  "pyyaml" \
  "requests" \
  "packaging" \
  "filelock" \
  "typing_extensions" \
  "fsspec" \
  "hf_xet" \
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
  "certifi" \
  "charset-normalizer" \
  "idna" \
  "urllib3" \
  "six"

echo "[5.5/7] reusing the existing local fla package..."
SITE_PACKAGES="$(python - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]
print(paths[0])
PY
)"
if [[ -d "${FLA_SOURCE_DIR}" ]]; then
  rm -rf "${SITE_PACKAGES}/fla"
  cp -a "${FLA_SOURCE_DIR}" "${SITE_PACKAGES}/fla"
else
  echo "missing fla source dir: ${FLA_SOURCE_DIR}" >&2
  exit 1
fi

echo "[6/7] installing sglang and repo-local CUDA deps..."
if [[ "${SKIP_REPO_BUILD}" == "1" ]]; then
  echo "[6/7] skipping sglang + local CUDA builds for smoke validation"
else
  uv pip install -e "${REPO_ROOT}/python[all]"

  cd "${DEPS_DIR}/infllmv2_cuda_impl"
  git submodule update --init --recursive
  python setup.py install

  cd "${DEPS_DIR}/sparse_kernel"
  python setup.py install

  echo "[7/7] installing extra libraries..."
  uv pip install tilelang flash-linear-attention
fi

python - <<'PY'
mods = [
    "torch",
    "transformers",
    "gptqmodel",
    "accelerate",
    "optimum",
    "sentencepiece",
    "huggingface_hub",
    "flash_attn",
    "einops",
]
if "SKIP_REPO_BUILD" not in __import__("os").environ or __import__("os").environ["SKIP_REPO_BUILD"] != "1":
    mods.append("sglang")
for m in mods:
    mod = __import__(m)
    print(m, getattr(mod, "__version__", "<no __version__>"))
PY

echo ""
echo "============================================"
echo " Submit-match uv installation complete!"
echo "============================================"
echo "source ${VENV_DIR}/bin/activate"
