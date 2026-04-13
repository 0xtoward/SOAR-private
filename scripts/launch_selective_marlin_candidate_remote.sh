#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <candidate-name>" >&2
  echo "candidates: odown-gs64 | skip-o-down64 | odown-kq-gs64 | odown-qkvall-gs64 | skip-o-down | noattn-skipdown151617-gs128" >&2
  exit 1
fi

CANDIDATE="$1"
REMOTE_HOST="${REMOTE_HOST:-rtx6000-2}"
TMUX_SESSION="${TMUX_SESSION:-codex-soar}"
REMOTE_DRAFT_ROOT="${REMOTE_DRAFT_ROOT:-/root/autodl-tmp/codex-drafts}"
REMOTE_SOAR_ROOT="${REMOTE_SOAR_ROOT:-/root/autodl-tmp/SOAR-Toolkit}"
REMOTE_MODEL_ROOT="${REMOTE_MODEL_ROOT:-/root/autodl-tmp/models}"
REMOTE_QUANT_ENV="${REMOTE_QUANT_ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env}"
CALIB_PATH="${CALIB_PATH:-/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_stopaligned64k_v1.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
DTYPE="${DTYPE:-float16}"
QUANT_EXTRA_ARGS="${QUANT_EXTRA_ARGS:-}"

case "${CANDIDATE}" in
  odown-gs64|skip-o-down64|odown-kq-gs64|odown-qkvall-gs64|skip-o-down|noattn-skipdown151617-gs128)
    ;;
  *)
    echo "unknown candidate: ${CANDIDATE}" >&2
    exit 1
    ;;
esac

WINDOW_NAME="sel-${CANDIDATE}"
RUN_ID="selective_${CANDIDATE}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${REMOTE_SOAR_ROOT}/test_results/${RUN_ID}_quant.log"
case "${CANDIDATE}" in
  noattn-skipdown151617-gs128)
    OUT_PATH="/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-v1"
    ;;
  *)
    OUT_PATH="/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1-${CANDIDATE}"
    ;;
esac
DYNAMIC_CONFIG_PATH="${REMOTE_DRAFT_ROOT}/configs/selective_marlin/${CANDIDATE}.json"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
tmux has-session -t "${TMUX_SESSION}" 2>/dev/null || tmux new-session -d -s "${TMUX_SESSION}" -n idle
tmux kill-window -t "${TMUX_SESSION}:${WINDOW_NAME}" >/dev/null 2>&1 || true
tmux new-window -t "${TMUX_SESSION}" -n "${WINDOW_NAME}" "source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && export ENV=${REMOTE_QUANT_ENV} && export MODEL_PATH=${REMOTE_MODEL_ROOT} && export CALIB_PATH=${CALIB_PATH} && export OUT_PATH=${OUT_PATH} && export NUM_SAMPLES=${NUM_SAMPLES} && export DTYPE=${DTYPE} && export SCRIPT_SRC=${REMOTE_DRAFT_ROOT}/quantize_gptq_w4a16.py && export DYNAMIC_CONFIG_PATH=${DYNAMIC_CONFIG_PATH} && export QUANT_EXTRA_ARGS='${QUANT_EXTRA_ARGS}' && export LOG_PATH=${LOG_PATH} && bash ${REMOTE_DRAFT_ROOT}/remote_quant_gptq_py310.sh"
printf '%s\n%s\n%s\n' "${RUN_ID}" "${LOG_PATH}" "${OUT_PATH}"
EOF
