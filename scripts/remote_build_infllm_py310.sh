#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/sglang}"
SRC_DIR="${SRC_DIR:-${REPO_ROOT}/3rdparty/infllmv2_cuda_impl}"
QUANT_ENV="${QUANT_ENV:-${REPO_ROOT}/sglang_submitmatch_uv_py310_gptq_env}"
EVAL_ENV="${EVAL_ENV:-${REPO_ROOT}/sglang_submitmatch_uv_py310_eval_env}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/infllm_v2_py310_build.log}"
DONE_PATH="${DONE_PATH:-/root/autodl-tmp/infllm_v2_py310_build.done}"
MAX_JOBS="${MAX_JOBS:-2}"
TMP_ROOT="${TMP_ROOT:-/root/autodl-tmp/tmp}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/torch-extensions}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[infllm-py310] start $(date '+%F %T')"
echo "[infllm-py310] repo=${REPO_ROOT}"
echo "[infllm-py310] src=${SRC_DIR}"
echo "[infllm-py310] quant_env=${QUANT_ENV}"
echo "[infllm-py310] eval_env=${EVAL_ENV}"

export LD_LIBRARY_PATH="/root/miniconda3/envs/py310/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS
mkdir -p "${TMP_ROOT}" "${TORCH_EXTENSIONS_DIR}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"
export TORCH_EXTENSIONS_DIR
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME="/usr/local/cuda"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

cd "${SRC_DIR}"

rm -f "${DONE_PATH}"
rm -f infllm_v2/C*.so
rm -f build/lib.linux-x86_64-cpython-310/infllm_v2/C*.so

echo "[infllm-py310] build_ext --inplace"
"${QUANT_ENV}/bin/python" setup.py build_ext --inplace

SO_PATH="$(find "${SRC_DIR}/infllm_v2" -maxdepth 1 -name 'C*.so' | head -n 1)"
if [[ -z "${SO_PATH}" ]]; then
  echo "[infllm-py310] missing built cp310 .so after build_ext" >&2
  exit 1
fi
echo "[infllm-py310] built_so=${SO_PATH}"

copy_into_env() {
  local env_dir="$1"
  local site_packages
  site_packages="$("${env_dir}/bin/python" - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]
print(paths[0])
PY
)"
  echo "[infllm-py310] install target env=${env_dir}"
  echo "[infllm-py310] site_packages=${site_packages}"
  rm -rf "${site_packages}/infllm_v2"
  cp -a "${SRC_DIR}/infllm_v2" "${site_packages}/infllm_v2"
  "${env_dir}/bin/python" - <<'PY'
import infllm_v2
from infllm_v2 import infllmv2_attn_stage1
print("import_ok", infllm_v2.__file__, infllmv2_attn_stage1.__name__)
PY
}

copy_into_env "${QUANT_ENV}"
copy_into_env "${EVAL_ENV}"

touch "${DONE_PATH}"
echo "[infllm-py310] done $(date '+%F %T')"
