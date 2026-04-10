#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
DRAFT=$ROOT/codex-drafts/nvfp4-mlp-only-draft
OUT_DIR=$ROOT/models-nvfp4-mlp-weight-only-curated32k64-public-v1
LOG_DIR=$ROOT/SOAR-Toolkit/test_results
QUEUE_LOG=$LOG_DIR/quant_fast_queue.log
QLOG=$LOG_DIR/nvfp4_mlp_weight_only_32k64_public_v1_quant.log
FAST_BASE_LOG=$LOG_DIR/stock_base_fast_eval.log
FAST_QUANT_LOG=$LOG_DIR/nvfp4_32k64_fast_eval.log
MINI_LOG=$LOG_DIR/nvfp4_32k64_mini.log
BASE_SERVER_LOG=$LOG_DIR/stock_base_fast_server.log
SERVER_LOG=$LOG_DIR/nvfp4_32k64_server.log
FAST_EVAL_TASK_CAPS="mcq:64,qa:128,niah:128,fwe:256,cwe:256"

mkdir -p "$LOG_DIR"

has_technical_failure() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 0
  grep -Eiq "Request failed|ERROR|Traceback|Exception|server error|connection error|HTTP/[0-9.]+\" 5[0-9][0-9]" "$log_path"
}

has_empty_failure() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 0
  grep -Eq "empty=[1-9]" "$log_path"
}

has_eval_summary() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 1
  grep -q "avg_score=" "$log_path"
}

official_eval_active() {
  ps -eo args= | awk 'index($0, "python3 eval_model.py --model_path /root/autodl-tmp/models") && index($0, "run_quant_fast_queue.sh") == 0 {found=1} END {exit found ? 0 : 1}'
}

stock_server_active() {
  ps -eo args= | awk 'index($0, "sglang.launch_server --model-path /root/autodl-tmp/models") && index($0, "run_quant_fast_queue.sh") == 0 {found=1} END {exit found ? 0 : 1}'
}

echo "[queue] waiting for stock official eval / server to finish" | tee -a "$QUEUE_LOG"
while official_eval_active || stock_server_active; do
  echo "[queue] waiting for active official eval / server to finish" | tee -a "$QUEUE_LOG"
  sleep 30
done

echo "[queue] stock official finished; starting base fast eval" | tee -a "$QUEUE_LOG"
source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1
pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true
nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server \
  --model-path /root/autodl-tmp/models \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30001 \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse > "$BASE_SERVER_LOG" 2>&1 &
sleep 20
cd "$ROOT/SOAR-Toolkit"
API_BASE=http://127.0.0.1:30001 TIMEOUT=0 MINI_PER_TASK=999 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 EVAL_TASK_MAX_TOKENS="$FAST_EVAL_TASK_CAPS" \
  EVAL_DATA="$ROOT/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl" \
  bash mini_test.sh | tee "$FAST_BASE_LOG"
pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true

if has_technical_failure "$FAST_BASE_LOG" || has_technical_failure "$BASE_SERVER_LOG" || ! has_eval_summary "$FAST_BASE_LOG"; then
  echo "[queue] base fast unhealthy; aborting quant queue" | tee -a "$QUEUE_LOG"
  exit 1
fi

echo "[queue] starting v2 nvfp4 quantization" | tee -a "$QUEUE_LOG"
python3 "$DRAFT/quantize_modelopt_nvfp4_conservative.py" \
  --model-path /root/autodl-tmp/models \
  --export-dir "$OUT_DIR" \
  --cfg mlp_weight_only \
  --dtype bfloat16 \
  --local-calib-files "$ROOT/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl" \
  --local-calib-samples 64 \
  --max-length 32768 \
  --min-good-batches 48 \
  --overwrite | tee "$QLOG"

echo "[queue] quantization finished; starting fast eval" | tee -a "$QUEUE_LOG"
pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true
nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server \
  --model-path "$OUT_DIR" \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30001 \
  --dtype float16 \
  --quantization modelopt \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse > "$SERVER_LOG" 2>&1 &
sleep 20
cd "$ROOT/SOAR-Toolkit"
API_BASE=http://127.0.0.1:30001 TIMEOUT=0 MINI_PER_TASK=999 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 EVAL_TASK_MAX_TOKENS="$FAST_EVAL_TASK_CAPS" \
  EVAL_DATA="$ROOT/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl" \
  bash mini_test.sh | tee "$FAST_QUANT_LOG"

if has_technical_failure "$FAST_QUANT_LOG" || has_technical_failure "$SERVER_LOG" || ! has_eval_summary "$FAST_QUANT_LOG" || has_empty_failure "$FAST_QUANT_LOG"; then
  echo "[queue] fast unhealthy; skipping mini" | tee -a "$QUEUE_LOG"
elif has_eval_summary "$FAST_QUANT_LOG"; then
  echo "[queue] fast passed basic gate; starting mini" | tee -a "$QUEUE_LOG"
  API_BASE=http://127.0.0.1:30001 TIMEOUT=0 MINI_PER_TASK=4 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 \
    bash mini_test.sh | tee "$MINI_LOG"
else
  echo "[queue] fast failed basic gate; skipping mini" | tee -a "$QUEUE_LOG"
fi

echo "[queue] nvfp4 fast/mini queue done" | tee -a "$QUEUE_LOG"
