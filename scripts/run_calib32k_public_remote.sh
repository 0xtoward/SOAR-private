#!/usr/bin/env bash
set -euo pipefail

REMOTE_SCRIPT_DIR="/root/autodl-tmp/SOAR-Toolkit/scripts"
REMOTE_OUT="/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl"
REMOTE_LOG="/root/autodl-tmp/SOAR-Toolkit/test_results/build_calibration_curated_32k_public_v1.log"

mkdir -p "${REMOTE_SCRIPT_DIR}" /root/autodl-tmp/SOAR-Toolkit/calibration /root/autodl-tmp/SOAR-Toolkit/test_results

pkill -f "build_calibration_curated.py --model-path /root/autodl-tmp/models" >/dev/null 2>&1 || true
tmux kill-window -t codex-soar:calib32k-public >/dev/null 2>&1 || true

tmux new-window -d -t codex-soar -n calib32k-public "bash -lc '
source /etc/network_turbo >/dev/null 2>&1 || true
export HF_ENDPOINT=https://hf-mirror.com
source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1
python3 ${REMOTE_SCRIPT_DIR}/build_calibration_curated.py \
  --model-path /root/autodl-tmp/models \
  --public-path /root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl \
  --output-path ${REMOTE_OUT} \
  --public-quotas mcq:30,qa:30,niah:30,fwe:19,cwe:19 \
  --open-quotas cnn_dailymail:0,wikitext:0 \
  --max-sample-tokens 32768 \
  --head-tokens 4096 \
  --middle-tokens 4096 \
  --tail-tokens 24576 \
  > ${REMOTE_LOG} 2>&1
'"

tmux list-windows -t codex-soar | tail -n 4
