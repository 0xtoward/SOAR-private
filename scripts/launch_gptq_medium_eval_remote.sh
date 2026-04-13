#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <label> <model-path> [dataset-path]" >&2
  exit 1
fi

LABEL="$1"
MODEL_PATH="$2"
DATASET_PATH="${3:-/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.jsonl}"

REMOTE_HOST="${REMOTE_HOST:-rtx6000-2}"
TMUX_SESSION="${TMUX_SESSION:-codex-soar}"
REMOTE_DRAFT_ROOT="${REMOTE_DRAFT_ROOT:-/root/autodl-tmp/codex-drafts}"
REMOTE_SOAR_ROOT="${REMOTE_SOAR_ROOT:-/root/autodl-tmp/SOAR-Toolkit}"
REMOTE_EVAL_ENV="${REMOTE_EVAL_ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env}"

WINDOW_NAME="eval-${LABEL}"
RUN_ID="${LABEL}_$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="${REMOTE_SOAR_ROOT}/test_results/${RUN_ID}"

ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
tmux has-session -t "${TMUX_SESSION}" 2>/dev/null || tmux new-session -d -s "${TMUX_SESSION}" -n idle
tmux kill-window -t "${TMUX_SESSION}:${WINDOW_NAME}" >/dev/null 2>&1 || true
tmux new-window -t "${TMUX_SESSION}" -n "${WINDOW_NAME}" "source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && export ENV=${REMOTE_EVAL_ENV} && export MODEL_PATH=${MODEL_PATH} && export MEDIUM_DATASET=${DATASET_PATH} && export RESULT_DIR=${RESULT_DIR} && bash ${REMOTE_DRAFT_ROOT}/remote_medium_eval_gptq_py310.sh"
printf '%s\n%s\n%s\n' "${RUN_ID}" "${RESULT_DIR}" "${DATASET_PATH}"
EOF
