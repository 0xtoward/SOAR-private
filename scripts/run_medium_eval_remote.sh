#!/usr/bin/env bash
set -euo pipefail

RUN_ID="medium_eval_stock_base_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/root/autodl-tmp/SOAR-Toolkit/test_results/${RUN_ID}"

mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" | tee "$RUN_DIR/launch.txt"

pkill -f "sglang.launch_server --host 127.0.0.1 --port 30001" >/dev/null 2>&1 || true

source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1

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
  --dense-as-sparse \
  > "$RUN_DIR/server.log" 2>&1 &

echo $! > "$RUN_DIR/server.pid"

READY=0
for _ in $(seq 1 60); do
  if python3 - <<'PY' >/dev/null 2>&1
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:30001/v1/models", timeout=2) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
  then
    READY=1
    break
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "[medium-eval] server readiness probe timed out" | tee -a "$RUN_DIR/launch.txt"
  kill "$(cat "$RUN_DIR/server.pid")" >/dev/null 2>&1 || true
  exit 1
fi

echo "[medium-eval] server ready, starting eval" | tee -a "$RUN_DIR/launch.txt"

cd /root/autodl-tmp/SOAR-Toolkit
stdbuf -oL -eL python3 eval_model.py \
  --model_path /root/autodl-tmp/models \
  --api_base http://127.0.0.1:30001 \
  --data_path eval_dataset/perf_public_medium_eval_v1.jsonl \
  --max_seq_len 262144 \
  --concurrency 8 \
  --verbose 2>&1 | tee "$RUN_DIR/eval_model.log"
