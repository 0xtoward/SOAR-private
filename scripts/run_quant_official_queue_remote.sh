#!/bin/bash
set -euo pipefail

source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1
cd /root/autodl-tmp/SOAR-Toolkit

QUEUE_LOG="/root/autodl-tmp/SOAR-Toolkit/test_results/quant_official_queue.log"
PORT=30001

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

wait_until_idle() {
  while pgrep -f 'python3 eval_model.py' >/dev/null 2>&1 || \
        pgrep -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1; do
    log "waiting for active official eval / server to finish"
    sleep 30
  done
}

preflight_checkpoint() {
  local path="$1"
  local required_csv="$2"
  python3 - "$path" "$required_csv" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
required = [x for x in sys.argv[2].split(",") if x]
missing = [name for name in required if not (path / name).exists()]
weights = sorted(x.name for x in path.glob("*.safetensors"))
print(
    {
        "path": str(path),
        "missing": missing,
        "num_safetensors": len(weights),
        "sample_weights": weights[:3],
    }
)
raise SystemExit(0 if path.exists() and not missing and weights else 1)
PY
}

run_candidate() {
  local label="$1"
  local model_path="$2"
  local quantization="$3"
  local dtype="$4"
  local run_dir="/root/autodl-tmp/SOAR-Toolkit/test_results/${label}_$(date +%Y%m%d_%H%M%S)"

  mkdir -p "$run_dir"
  log "start candidate label=$label model=$model_path quant=$quantization run_dir=$run_dir"

  local server_cmd=(
    /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3
    -m sglang.launch_server
    --model-path "$model_path"
    --trust-remote-code
    --host 127.0.0.1
    --port "$PORT"
    --dtype "$dtype"
    --attention-backend minicpm_flashinfer
    --chunked-prefill-size 8192
    --skip-server-warmup
    --disable-radix-cache
    --disable-cuda-graph
    --dense-as-sparse
  )
  if [[ -n "$quantization" ]]; then
    server_cmd+=(--quantization "$quantization")
  fi

  "${server_cmd[@]}" > "$run_dir/server.log" 2>&1 &
  echo $! > "$run_dir/server.pid"

  until curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    sleep 2
  done
  log "server ready label=$label"

  local eval_cmd=(
    python3 eval_model.py
    --model_path "$model_path"
    --api_base "http://127.0.0.1:${PORT}"
    --max_seq_len 524288
    --data_path eval_dataset/perf_public_set.jsonl
    --concurrency 8
  )
  "${eval_cmd[@]}" | tee "$run_dir/eval_model.log"
  local status=${PIPESTATUS[0]}

  kill "$(cat "$run_dir/server.pid")" >/dev/null 2>&1 || true
  pkill -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1 || true
  log "done label=$label status=$status run_dir=$run_dir"
  return "$status"
}

log "queue start"
log "preflight gptq checkpoint"
preflight_checkpoint \
  /root/autodl-tmp/models-gptq-w4a16-v19 \
  config.json,quantize_config.json,tokenizer_config.json | tee -a "$QUEUE_LOG"

log "preflight nvfp4 checkpoint"
preflight_checkpoint \
  /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128 \
  config.json,tokenizer_config.json | tee -a "$QUEUE_LOG"

wait_until_idle
run_candidate official_gptq_marlin /root/autodl-tmp/models-gptq-w4a16-v19 gptq_marlin float16

wait_until_idle
run_candidate official_nvfp4_curated16k128 /root/autodl-tmp/models-nvfp4-mlp-only-curated16k-128 modelopt float16

log "queue finished"
