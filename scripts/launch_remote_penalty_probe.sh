#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-106.120.183.117}"
SSH_PORT="${SSH_PORT:-50884}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-/Users/ql/.ssh/id_ed25519}"

TMUX_SOCKET="${TMUX_SOCKET:-/root/autodl-tmp/tmux/codex-soar.sock}"
TMUX_SESSION="${TMUX_SESSION:-codex-soar}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-stopaligned64k-v1}"
DATASET_PATH="${DATASET_PATH:-/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_overgen8_eval_v1.jsonl}"
REMOTE_RUNNER="${REMOTE_RUNNER:-/root/autodl-tmp/codex-drafts/remote_medium_eval_gptq_py310.sh}"
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-/root/autodl-tmp/SOAR-Toolkit/test_results}"

LABEL=""
WINDOW_NAME=""
PREFERRED_SAMPLING_PARAMS=""
REPETITION_PENALTY=""
FREQUENCY_PENALTY=""
PRESENCE_PENALTY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label)
      LABEL="$2"
      shift 2
      ;;
    --window-name)
      WINDOW_NAME="$2"
      shift 2
      ;;
    --sampling-params)
      PREFERRED_SAMPLING_PARAMS="$2"
      shift 2
      ;;
    --repetition-penalty)
      REPETITION_PENALTY="$2"
      shift 2
      ;;
    --frequency-penalty)
      FREQUENCY_PENALTY="$2"
      shift 2
      ;;
    --presence-penalty)
      PRESENCE_PENALTY="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${LABEL}" ]]; then
  echo "Usage: $0 --label <name> [--sampling-params '<json>' | --repetition-penalty <v> | --frequency-penalty <v> | --presence-penalty <v>] [--window-name <name>] [--model-path <path>] [--dataset-path <path>]" >&2
  exit 1
fi

if [[ -z "${WINDOW_NAME}" ]]; then
  WINDOW_NAME="${LABEL}"
fi

if [[ -z "${PREFERRED_SAMPLING_PARAMS}" && ( -n "${REPETITION_PENALTY}" || -n "${FREQUENCY_PENALTY}" || -n "${PRESENCE_PENALTY}" ) ]]; then
  PREFERRED_SAMPLING_PARAMS="$(
    python3 - <<PY
import json
payload = {}
if "${REPETITION_PENALTY}":
    payload["repetition_penalty"] = float("${REPETITION_PENALTY}")
if "${FREQUENCY_PENALTY}":
    payload["frequency_penalty"] = float("${FREQUENCY_PENALTY}")
if "${PRESENCE_PENALTY}":
    payload["presence_penalty"] = float("${PRESENCE_PENALTY}")
print(json.dumps(payload, separators=(",", ":")))
PY
  )"
fi

PARAMS_B64="$(printf '%s' "${PREFERRED_SAMPLING_PARAMS}" | base64)"

ssh -p "${SSH_PORT}" -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SSH_USER}@${SSH_HOST}" \
  "RUN_DIR=${REMOTE_RESULTS_DIR}/${LABEL}_\$(date +%Y%m%d_%H%M%S); \
   mkdir -p ${REMOTE_RESULTS_DIR}; \
   printf '%s\n' \"\$RUN_DIR\" > ${REMOTE_RESULTS_DIR}/latest_${LABEL}.txt; \
   tmux -S ${TMUX_SOCKET} new-window -d -t ${TMUX_SESSION} -n ${WINDOW_NAME} \
     \"mkdir -p \\\"\$RUN_DIR\\\" && \
      export MODEL_PATH='${MODEL_PATH}' && \
      export MEDIUM_DATASET='${DATASET_PATH}' && \
      export RESULT_DIR=\\\"\$RUN_DIR\\\" && \
      export PREFERRED_SAMPLING_PARAMS=\\\"\$(printf '%s' '${PARAMS_B64}' | base64 -d)\\\" && \
      bash ${REMOTE_RUNNER} 2>&1 | tee \\\"\$RUN_DIR/launcher.log\\\"\"; \
   tmux -S ${TMUX_SOCKET} list-windows -t ${TMUX_SESSION} | tail -n 3; \
   echo RUN_DIR=\$RUN_DIR"
