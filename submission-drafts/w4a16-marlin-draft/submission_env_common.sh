#!/usr/bin/env bash

# Shared helpers for the dual-env SOAR submission.

soar_fail() {
  local code="${2:-1}"
  echo "$1" >&2
  return "${code}" 2>/dev/null || exit "${code}"
}

SOAR_SUBMISSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOAR_WHEEL_DIR="${SOAR_SUBMISSION_ROOT}/wheels"
SOAR_VENDOR_DIR="${SOAR_SUBMISSION_ROOT}/vendor"
SOAR_THIRD_PARTY_DIR="${SOAR_SUBMISSION_ROOT}/third_party"
SOAR_RUNTIME_DIR="${SOAR_SUBMISSION_ROOT}/runtime_envs"
SOAR_PATCHED_MODEL_CODE_DIR="${SOAR_SUBMISSION_ROOT}/patched_model_code"

SOAR_EVAL_ENV="${SOAR_RUNTIME_DIR}/eval_py310_env"
SOAR_QUANT_ENV="${SOAR_RUNTIME_DIR}/quant_py310_env"

SOAR_FLASH_ATTN_WHEEL="${SOAR_WHEEL_DIR}/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl"
SOAR_GPTQMODEL_WHEEL="${SOAR_WHEEL_DIR}/gptqmodel-5.8.0+cu128torch2.9-cp310-cp310-linux_x86_64.whl"
SOAR_VENDOR_FLA_DIR="${SOAR_VENDOR_DIR}/fla"
SOAR_INFLLM_DIR="${SOAR_THIRD_PARTY_DIR}/infllmv2_cuda_impl"
SOAR_PATCHED_CONFIG_PY="${SOAR_PATCHED_MODEL_CODE_DIR}/configuration_minicpm_sala.py"
SOAR_PATCHED_MODELING_PY="${SOAR_PATCHED_MODEL_CODE_DIR}/modeling_minicpm_sala.py"

SOAR_INDEX_URL="${SOAR_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy HF_ENDPOINT
export UV_INDEX_URL="${UV_INDEX_URL:-${SOAR_INDEX_URL}}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-${SOAR_INDEX_URL}}"
export PIP_EXTRA_INDEX_URL=""
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ -d /opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib ]]; then
  export LD_LIBRARY_PATH="/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi
if command -v gcc >/dev/null 2>&1; then
  export CC="${CC:-gcc}"
fi
if command -v g++ >/dev/null 2>&1; then
  export CXX="${CXX:-g++}"
fi

soar_ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  python -m pip install -i "${SOAR_INDEX_URL}" uv || soar_fail "[common] failed to install uv"
}

soar_base_python() {
  if [[ -n "${SOAR_BASE_PYTHON:-}" && -x "${SOAR_BASE_PYTHON}" ]]; then
    printf '%s\n' "${SOAR_BASE_PYTHON}"
    return 0
  fi
  if [[ -x /opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/python ]]; then
    printf '%s\n' /opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/python
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v python3.10 >/dev/null 2>&1; then
    command -v python3.10
    return 0
  fi
  soar_fail "[common] no suitable python interpreter found"
}

soar_base_site_packages() {
  local base_py="${1:-$(soar_base_python)}"
  "${base_py}" - <<'PY'
import os
import site

seen = set()
for path in site.getsitepackages():
    if path and os.path.isdir(path):
        real = os.path.realpath(path)
        if real not in seen:
            print(real)
            seen.add(real)

user_site = site.getusersitepackages()
if user_site and os.path.isdir(user_site):
    real = os.path.realpath(user_site)
    if real not in seen:
        print(real)
PY
}

soar_bridge_base_site_packages() {
  local py_bin="$1"
  local base_py="${2:-$(soar_base_python)}"
  local site_dir
  site_dir="$(soar_site_packages_dir "${py_bin}")"
  local pth_file="${site_dir}/soar_official_base_env.pth"
  : > "${pth_file}"

  local count=0
  while IFS= read -r path; do
    [[ -n "${path}" && -d "${path}" ]] || continue
    printf '%s\n' "${path}" >> "${pth_file}"
    count=$((count + 1))
  done < <(soar_base_site_packages "${base_py}")

  if [[ "${count}" -eq 0 ]]; then
    soar_fail "[common] failed to discover official base site-packages from ${base_py}"
  fi
}

soar_create_overlay_env() {
  local env_dir="$1"
  local base_py="${2:-$(soar_base_python)}"
  mkdir -p "${SOAR_RUNTIME_DIR}"
  if [[ ! -x "${env_dir}/bin/python" ]]; then
    "${base_py}" -m venv "${env_dir}" || soar_fail "[common] failed to create venv: ${env_dir}"
  fi
  soar_bridge_base_site_packages "${env_dir}/bin/python" "${base_py}"
}

soar_site_packages_dir() {
  local py_bin="$1"
  "${py_bin}" - <<'PY'
import site
paths = [p for p in site.getsitepackages() if p.endswith("site-packages")]
if not paths:
    raise SystemExit("no site-packages path found")
print(paths[0])
PY
}

soar_copy_vendor_fla() {
  local py_bin="$1"
  [[ -d "${SOAR_VENDOR_FLA_DIR}" ]] || soar_fail "[common] missing bundled vendor/fla directory"
  local site_dir
  site_dir="$(soar_site_packages_dir "${py_bin}")"
  rm -rf "${site_dir}/fla"
  cp -a "${SOAR_VENDOR_FLA_DIR}" "${site_dir}/fla" || soar_fail "[common] failed to copy bundled fla into ${site_dir}"
}

soar_prepare_patched_model_dir() {
  local input_dir="$1"
  local output_dir="$2"

  [[ -d "${input_dir}" ]] || soar_fail "[common] missing input model dir: ${input_dir}"
  [[ -f "${SOAR_PATCHED_CONFIG_PY}" ]] || soar_fail "[common] missing bundled patched config: ${SOAR_PATCHED_CONFIG_PY}"
  [[ -f "${SOAR_PATCHED_MODELING_PY}" ]] || soar_fail "[common] missing bundled patched modeling: ${SOAR_PATCHED_MODELING_PY}"

  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"

  local entry base
  for entry in "${input_dir}"/*; do
    base="$(basename "${entry}")"
    ln -s "${entry}" "${output_dir}/${base}"
  done

  cp -f "${SOAR_PATCHED_CONFIG_PY}" "${output_dir}/configuration_minicpm_sala.py"
  cp -f "${SOAR_PATCHED_MODELING_PY}" "${output_dir}/modeling_minicpm_sala.py"
}

soar_sync_tokenizer_assets() {
  local src_dir="$1"
  local dst_dir="$2"

  [[ -d "${src_dir}" ]] || soar_fail "[common] missing tokenizer source dir: ${src_dir}"
  [[ -d "${dst_dir}" ]] || soar_fail "[common] missing tokenizer destination dir: ${dst_dir}"

  local name src
  for name in \
    tokenizer_config.json \
    special_tokens_map.json \
    tokenizer.json \
    tokenizer.model \
    added_tokens.json \
    chat_template.jinja; do
    src="${src_dir}/${name}"
    if [[ -f "${src}" ]]; then
      cp -f "${src}" "${dst_dir}/${name}" || soar_fail "[common] failed to copy tokenizer asset ${name}"
    fi
  done
}

soar_python_has_torch() {
  local py_bin="$1"
  "${py_bin}" - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
}

soar_require_torch() {
  local py_bin="$1"
  local env_name="$2"
  if ! soar_python_has_torch "${py_bin}"; then
    "${py_bin}" - <<'PY' || true
import site
import sys

print("[common] torch import debug executable=", sys.executable)
print("[common] torch import debug sitepackages=", site.getsitepackages())
print("[common] torch import debug sys_path=", sys.path)
PY
    soar_fail "[common] ${env_name} cannot import torch from the official base environment"
  fi
}
