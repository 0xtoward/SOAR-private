#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-30001}"
MODEL_OUT="${MODEL_OUT:-/root/autodl-tmp/models-gptq-w4a16-sanity16}"
RUN_ID="gptq_marlin_fastmed_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/root/autodl-tmp/SOAR-Toolkit/test_results/${RUN_ID}"
FAST_DATA="/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl"
MEDIUM_DATA="/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.jsonl"
FAST_EVAL_TASK_CAPS="${FAST_EVAL_TASK_CAPS:-mcq:64,qa:128,niah:128,fwe:256,cwe:256}"

mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" | tee "$RUN_DIR/launch.txt"
echo "MODEL_OUT=$MODEL_OUT" | tee -a "$RUN_DIR/launch.txt"
echo "FAST_EVAL_TASK_CAPS=$FAST_EVAL_TASK_CAPS" | tee -a "$RUN_DIR/launch.txt"

source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1

cleanup() {
  if [[ -f "$RUN_DIR/server.pid" ]]; then
    kill "$(cat "$RUN_DIR/server.pid")" >/dev/null 2>&1 || true
  fi
  pkill -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

pkill -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1 || true

echo "[1/3] launch gptq_marlin server" | tee -a "$RUN_DIR/launch.txt"
nohup /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server \
  --model-path "$MODEL_OUT" \
  --quantization gptq_marlin \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --enable-request-time-stats-logging \
  > "$RUN_DIR/server.log" 2>&1 &
echo $! > "$RUN_DIR/server.pid"

READY=0
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "[gptq-fastmed] server readiness probe timed out" | tee -a "$RUN_DIR/launch.txt"
  exit 1
fi

echo "[2/3] fast eval" | tee -a "$RUN_DIR/launch.txt"
cd /root/autodl-tmp/SOAR-Toolkit
API_BASE="http://127.0.0.1:${PORT}" TIMEOUT=0 MINI_PER_TASK=999 EVAL_USE_ALL_ROWS=1 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 \
  EVAL_TASK_MAX_TOKENS="$FAST_EVAL_TASK_CAPS" \
  EVAL_DATA="$FAST_DATA" \
  bash /root/autodl-tmp/SOAR-Toolkit/mini_test.sh 2>&1 | tee "$RUN_DIR/fast_eval.log"

echo "[3/3] medium eval" | tee -a "$RUN_DIR/launch.txt"
stdbuf -oL -eL python3 eval_model.py \
  --model_path "$MODEL_OUT" \
  --model_name "$MODEL_OUT" \
  --api_base "http://127.0.0.1:${PORT}" \
  --data_path "eval_dataset/$(basename "$MEDIUM_DATA")" \
  --max_seq_len 262144 \
  --concurrency 8 \
  --verbose 2>&1 | tee "$RUN_DIR/medium_eval.log"

echo "[gptq-fastmed] done" | tee -a "$RUN_DIR/launch.txt"
