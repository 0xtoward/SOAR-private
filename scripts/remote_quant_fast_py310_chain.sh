#!/usr/bin/env bash
set -euo pipefail

BUILD_SCRIPT="${BUILD_SCRIPT:-/root/autodl-tmp/codex-drafts/remote_build_infllm_py310.sh}"
QUANT_SCRIPT="${QUANT_SCRIPT:-/root/autodl-tmp/codex-drafts/remote_quant_gptq_py310.sh}"
FAST_SCRIPT="${FAST_SCRIPT:-/root/autodl-tmp/codex-drafts/remote_fast_eval_gptq_py310.sh}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/quant_fast_py310_chain.log}"

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[chain-py310] start $(date '+%F %T')"
bash "${BUILD_SCRIPT}"
bash "${QUANT_SCRIPT}"
bash "${FAST_SCRIPT}"
echo "[chain-py310] done $(date '+%F %T')"
