#!/usr/bin/env bash
set -euo pipefail

echo "[prepare_env] start $(date '+%F %T')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${SCRIPT_DIR}/wheels"

FLASH_ATTN_WHEEL="${WHEEL_DIR}/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl"
MODELOPT_WHEEL="${WHEEL_DIR}/nvidia_modelopt-0.42.0-py3-none-any.whl"
PULP_WHEEL="${WHEEL_DIR}/pulp-3.2.2-py3-none-any.whl"

has_module() {
  local module_name="$1"
  python - "$module_name" <<'PY'
import importlib.util
import sys

module_name = sys.argv[1]
raise SystemExit(0 if importlib.util.find_spec(module_name) else 1)
PY
}

can_import_modelopt() {
  python - <<'PY'
import sys

try:
    import modelopt.torch.export  # noqa: F401
    import modelopt.torch.quantization  # noqa: F401
except Exception as exc:
    print(f"[prepare_env] modelopt import check failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

raise SystemExit(0)
PY
}

install_if_missing() {
  local module_name="$1"
  local wheel_path="$2"
  local package_name="$3"

  if has_module "${module_name}"; then
    echo "[prepare_env] ${module_name} already available"
    return 0
  fi

  if [[ -n "${wheel_path}" && -f "${wheel_path}" ]]; then
    echo "[prepare_env] install ${package_name} from wheel"
    if uv pip install --no-deps "${wheel_path}" && has_module "${module_name}"; then
      return 0
    fi

    echo "[prepare_env] wheel install for ${package_name} did not produce importable ${module_name}; falling back to index"
  fi

  echo "[prepare_env] install ${package_name} from index"
  UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    uv pip install "${package_name}"
  has_module "${module_name}"
}

install_modelopt() {
  if can_import_modelopt; then
    echo "[prepare_env] modelopt runtime already available"
    return 0
  fi

  if [[ -f "${MODELOPT_WHEEL}" ]]; then
    echo "[prepare_env] install nvidia-modelopt from wheel with deps"
    UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
      uv pip install "${MODELOPT_WHEEL}"
  else
    echo "[prepare_env] install nvidia-modelopt from index"
    UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
      uv pip install "nvidia-modelopt==0.42.0"
  fi

  if can_import_modelopt; then
    echo "[prepare_env] modelopt runtime import OK"
    return 0
  fi

  echo "[prepare_env] repair common modelopt runtime deps from index"
  UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
    uv pip install rich packaging tqdm regex safetensors scipy ninja nvidia-ml-py pydantic pulp

  can_import_modelopt
}

install_if_missing "pulp" "${PULP_WHEEL}" "pulp"
install_modelopt
install_if_missing "flash_attn" "${FLASH_ATTN_WHEEL}" "flash-attn"

echo "[prepare_env] install editable sglang"
uv pip install --no-deps -e "${SCRIPT_DIR}/sglang/python"

export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --disable-cuda-graph --quantization modelopt --dtype float16"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done $(date '+%F %T')"
