#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-rtx6000-2}"
REMOTE_DRAFT_ROOT="${REMOTE_DRAFT_ROOT:-/root/autodl-tmp/codex-drafts}"
REMOTE_SOAR_ROOT="${REMOTE_SOAR_ROOT:-/root/autodl-tmp/SOAR-Toolkit}"
REMOTE_MODEL_ROOT="${REMOTE_MODEL_ROOT:-/root/autodl-tmp/models}"

SAMPLES="${SAMPLES:-16}"
DTYPE="${DTYPE:-float16}"
LABEL="${LABEL:-gptq_w4a16_sanity16}"
WINDOW_NAME="${WINDOW_NAME:-gptq-q16}"
OUTPUT_PATH="${OUTPUT_PATH:-/root/autodl-tmp/models-gptq-w4a16-sanity16}"
CALIB_PATH="${CALIB_PATH:-/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_v1.jsonl}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_LOG="${REMOTE_SOAR_ROOT}/test_results/${LABEL}_${TIMESTAMP}.log"

ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
pkill -f "python3 eval_model.py --model_path ${REMOTE_MODEL_ROOT} --api_base http://127.0.0.1:30001 --model_name ${REMOTE_MODEL_ROOT} --max_seq_len 524288 --data_path eval_dataset/perf_public_fast_eval.jsonl --concurrency 8" >/dev/null 2>&1 || true
pkill -f "sglang.launch_server --model-path ${REMOTE_MODEL_ROOT} --trust-remote-code --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true
tmux kill-window -t codex-soar:${WINDOW_NAME} >/dev/null 2>&1 || true
tmux new-window -t codex-soar -n ${WINDOW_NAME} "source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1 && python ${REMOTE_DRAFT_ROOT}/quantize_gptq_w4a16.py --model-path ${REMOTE_MODEL_ROOT} --calibration-path ${CALIB_PATH} --output-path ${OUTPUT_PATH} --num-samples ${SAMPLES} --dtype ${DTYPE} > ${REMOTE_LOG} 2>&1"
printf '%s\n' "${REMOTE_LOG}"
EOF
