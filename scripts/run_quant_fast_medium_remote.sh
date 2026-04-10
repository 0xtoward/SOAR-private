#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-30001}"
CFG="${CFG:-mlp_weight_only}"
QUANT_DTYPE="${QUANT_DTYPE:-float16}"
SERVER_DTYPE="${SERVER_DTYPE:-float16}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
LOCAL_CALIB_SAMPLES="${LOCAL_CALIB_SAMPLES:-64}"
MIN_GOOD_BATCHES="${MIN_GOOD_BATCHES:-48}"
RUN_BASENAME="${RUN_BASENAME:-nvfp4_${CFG}_${MAX_LENGTH}_${LOCAL_CALIB_SAMPLES}_${QUANT_DTYPE}}"
MODEL_OUT="${MODEL_OUT:-/root/autodl-tmp/models-${RUN_BASENAME}}"
RUN_ID="quant_fastmed_${RUN_BASENAME}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/root/autodl-tmp/SOAR-Toolkit/test_results/${RUN_ID}"
FAST_DATA="/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl"
FAST_META="/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.meta.json"
MEDIUM_DATA="/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.jsonl"
QUANT_SCRIPT="/root/autodl-tmp/codex-drafts/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py"
CALIB_FILE="/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl"
DISABLE_DEEPGEMM="${DISABLE_DEEPGEMM:-0}"
FAST_EVAL_TASK_CAPS="${FAST_EVAL_TASK_CAPS:-mcq:64,qa:128,niah:128,fwe:256,cwe:256}"

mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR" | tee "$RUN_DIR/launch.txt"
echo "MODEL_OUT=$MODEL_OUT" | tee -a "$RUN_DIR/launch.txt"
echo "CFG=$CFG" | tee -a "$RUN_DIR/launch.txt"
echo "QUANT_DTYPE=$QUANT_DTYPE" | tee -a "$RUN_DIR/launch.txt"
echo "SERVER_DTYPE=$SERVER_DTYPE" | tee -a "$RUN_DIR/launch.txt"
echo "MAX_LENGTH=$MAX_LENGTH" | tee -a "$RUN_DIR/launch.txt"
echo "LOCAL_CALIB_SAMPLES=$LOCAL_CALIB_SAMPLES" | tee -a "$RUN_DIR/launch.txt"
echo "MIN_GOOD_BATCHES=$MIN_GOOD_BATCHES" | tee -a "$RUN_DIR/launch.txt"
echo "DISABLE_DEEPGEMM=$DISABLE_DEEPGEMM" | tee -a "$RUN_DIR/launch.txt"
echo "FAST_EVAL_TASK_CAPS=$FAST_EVAL_TASK_CAPS" | tee -a "$RUN_DIR/launch.txt"

source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1

cleanup() {
  if [ -f "$RUN_DIR/server.pid" ]; then
    kill "$(cat "$RUN_DIR/server.pid")" >/dev/null 2>&1 || true
  fi
  pkill -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

source_path = Path("/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_medium_eval_v1.jsonl")
target_path = Path("/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl")
meta_path = Path("/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.meta.json")
per_task = 4
caps = {
    "mcq": 64,
    "qa": 128,
    "niah": 128,
    "fwe": 256,
    "cwe": 256,
}

def bucket(prompt_tokens: int) -> str:
    if prompt_tokens < 4096:
        return "0-4K"
    if prompt_tokens < 16384:
        return "4-16K"
    if prompt_tokens < 32768:
        return "16K-32K"
    if prompt_tokens < 131072:
        return "32K-128K"
    return "128K-160K"

def evenly_pick(items, k):
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    picked = []
    used = set()
    n = len(items)
    for i in range(k):
        pos = round(i * (n - 1) / (k - 1)) if k > 1 else n // 2
        while pos in used and pos + 1 < n:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        if pos in used:
            continue
        used.add(pos)
        picked.append(items[pos])
    return picked

rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
by_task = {}
for row in rows:
    by_task.setdefault(row["task"], []).append(row)

selected = []
for task in sorted(by_task):
    task_rows = sorted(
        by_task[task],
        key=lambda row: (int(row.get("prompt_tokens", 0)), int(row.get("index", 0))),
    )
    selected.extend(evenly_pick(task_rows, per_task))

selected = sorted(
    selected,
    key=lambda row: (row.get("task", ""), int(row.get("prompt_tokens", 0)), int(row.get("index", 0))),
)

for row in selected:
    task = row.get("task")
    cap = int(caps[task])
    original_completion = int(row.get("completion_tokens", cap))
    row["completion_tokens_original"] = original_completion
    row["completion_tokens"] = min(original_completion, cap)
    row["fast_eval_cap"] = cap

target_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
    encoding="utf-8",
)

meta = {
    "source": str(source_path),
    "count": len(selected),
    "per_task": per_task,
    "completion_token_caps": caps,
    "by_task": dict(sorted(Counter(row["task"] for row in selected).items())),
    "by_bucket": dict(sorted(Counter(bucket(int(row.get("prompt_tokens", 0))) for row in selected).items())),
}
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False))
PY
echo "[fast-eval] dataset refreshed at $FAST_DATA" | tee -a "$RUN_DIR/launch.txt"

echo "[1/4] quantize ${CFG} maxlen=${MAX_LENGTH} samples=${LOCAL_CALIB_SAMPLES} dtype=${QUANT_DTYPE}" | tee -a "$RUN_DIR/launch.txt"
python3 "$QUANT_SCRIPT" \
  --model-path /root/autodl-tmp/models \
  --export-dir "$MODEL_OUT" \
  --cfg "$CFG" \
  --dtype "$QUANT_DTYPE" \
  --local-calib-files "$CALIB_FILE" \
  --local-calib-samples "$LOCAL_CALIB_SAMPLES" \
  --max-length "$MAX_LENGTH" \
  --min-good-batches "$MIN_GOOD_BATCHES" \
  --overwrite 2>&1 | tee "$RUN_DIR/quantize.log"

pkill -f "sglang.launch_server --host 127.0.0.1 --port ${PORT}" >/dev/null 2>&1 || true

echo "[2/4] launch quantized server" | tee -a "$RUN_DIR/launch.txt"
DG_CMD=(env)
if [ "$DISABLE_DEEPGEMM" = "1" ]; then
  DG_CMD=(
    env
    SGLANG_ENABLE_JIT_DEEPGEMM=0
    SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
    SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
  )
fi
nohup "${DG_CMD[@]}" /root/autodl-tmp/sglang/sglang_minicpm_sala_env/bin/python3 -m sglang.launch_server \
  --model-path "$MODEL_OUT" \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype "$SERVER_DTYPE" \
  --quantization modelopt \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  > "$RUN_DIR/server.log" 2>&1 &
echo $! > "$RUN_DIR/server.pid"

READY=0
for _ in $(seq 1 180); do
  if python3 - <<'PY' > "$RUN_DIR/model_probe.json.tmp" 2>/dev/null
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:30001/v1/models", timeout=2) as resp:
    payload = json.load(resp)
model_id = payload["data"][0]["id"]
if not model_id:
    raise SystemExit(1)
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
  then
    mv "$RUN_DIR/model_probe.json.tmp" "$RUN_DIR/model_probe.json"
    READY=1
    break
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "[quant-fastmed] server readiness probe timed out" | tee -a "$RUN_DIR/launch.txt"
  kill "$(cat "$RUN_DIR/server.pid")" >/dev/null 2>&1 || true
  exit 1
fi

echo "[3/4] fast eval" | tee -a "$RUN_DIR/launch.txt"
cd /root/autodl-tmp/SOAR-Toolkit
API_BASE="http://127.0.0.1:${PORT}" TIMEOUT=0 MINI_PER_TASK=999 EVAL_USE_ALL_ROWS=1 EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 \
  EVAL_TASK_MAX_TOKENS="$FAST_EVAL_TASK_CAPS" \
  EVAL_DATA="$FAST_DATA" \
  bash /root/autodl-tmp/SOAR-Toolkit/mini_test.sh 2>&1 | tee "$RUN_DIR/fast_eval.log"

echo "[4/4] medium eval" | tee -a "$RUN_DIR/launch.txt"
stdbuf -oL -eL python3 eval_model.py \
  --model_path "$MODEL_OUT" \
  --model_name "$MODEL_OUT" \
  --api_base "http://127.0.0.1:${PORT}" \
  --data_path "eval_dataset/$(basename "$MEDIUM_DATA")" \
  --max_seq_len 262144 \
  --concurrency 8 \
  --verbose 2>&1 | tee "$RUN_DIR/medium_eval.log"

cleanup
