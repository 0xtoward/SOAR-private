#!/usr/bin/env bash
set -euo pipefail

# Shared runtime for SOAR-Toolkit and local editable sglang on AutoDL.
# Usage:
#   source /root/autodl-tmp/use_soar_uv.sh
#   cd "$SOAR_HOME"    # optional

export AUTODL_ROOT="/root/autodl-tmp"
export SOAR_HOME="${AUTODL_ROOT}/SOAR-Toolkit"
export SGLANG_HOME="${AUTODL_ROOT}/sglang"
export SOAR_VENV="${SGLANG_HOME}/sglang_minicpm_sala_env"
export SOAR_PYTHON="${SOAR_VENV}/bin/python"

if [[ ! -x "${SOAR_PYTHON}" ]]; then
  echo "[ERROR] Missing shared runtime python: ${SOAR_PYTHON}" >&2
  return 1 2>/dev/null || exit 1
fi

export VIRTUAL_ENV="${SOAR_VENV}"
export PATH="${SOAR_VENV}/bin:${PATH}"

soar_python() {
  "${SOAR_PYTHON}" "$@"
}

soar_pip() {
  "${SOAR_PYTHON}" -m pip "$@"
}

echo "[soar-env] VIRTUAL_ENV=${SOAR_VENV}"
echo "[soar-env] python=${SOAR_PYTHON}"
