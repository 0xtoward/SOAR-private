#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH=${DATA_PATH:-/root/autodl-tmp/SOAR-Toolkit/test_results/winning99_eagle3_topk2_nograph_smoke_nocap_after_full_20260512_152925/no_cap_smoke_data.jsonl}
ROW_ORDINAL=${ROW_ORDINAL:-5}
PORT=${PORT:-30238}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}

source /etc/network_turbo >/dev/null 2>&1 || true
source /root/autodl-tmp/use_soar_uv.sh
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PYTHONPATH=/root/autodl-tmp/codex-drafts/sglang-sala-tree-verify/python:${PYTHONPATH:-}
export SGLANG_ENABLE_SPEC_V2=False
export SGLANG_SALA_SKIP_TARGET_VERIFY_TEMPORAL_WRITE=1
unset SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM || true
unset SGLANG_ENABLE_DETERMINISTIC_INFERENCE || true
unset CUBLAS_WORKSPACE_CONFIG || true

PROMPT_TEXT="$(
  DATA_PATH="$DATA_PATH" ROW_ORDINAL="$ROW_ORDINAL" python3 - <<'PY'
import json
import os
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path(os.environ["DATA_PATH"]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(rows[int(os.environ["ROW_ORDINAL"])]["question"])
PY
)"

cd /root/autodl-tmp/codex-drafts/sglang-sala-tree-verify

BASE_ARGS=(
  --model-path /root/autodl-tmp/models
  --trust-remote-code
  --host 127.0.0.1
  --disable-radix-cache
  --force-dense-minicpm
  --chunked-prefill-size 8192
  --skip-server-warmup
  --disable-cuda-graph
  --dtype bfloat16
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --decode-log-interval 1
  --attention-backend triton
)

cleanup_server() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
      sleep 1
      kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

wait_ready() {
  local name="$1"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${PORT}/get_model_info" > "$RUN_DIR/${name}_server_info.json" 2> "$RUN_DIR/${name}_ready.err"; then
      return 0
    fi
    if [[ -f "$RUN_DIR/${name}_server.pid" ]]; then
      local pid
      pid="$(cat "$RUN_DIR/${name}_server.pid")"
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        tail -120 "$RUN_DIR/${name}_server.log" || true
        return 1
      fi
    fi
    sleep 1
  done
  tail -120 "$RUN_DIR/${name}_server.log" || true
  return 1
}

generate_prompt() {
  local name="$1"
  PROMPT_TEXT="$PROMPT_TEXT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" RUN_DIR="$RUN_DIR" CASE="$name" PORT_VALUE="$PORT" python3 - <<'PY'
import json
import os
import urllib.request
from pathlib import Path

payload = {
    "text": os.environ["PROMPT_TEXT"],
    "sampling_params": {
        "temperature": 0,
        "max_new_tokens": int(os.environ["MAX_NEW_TOKENS"]),
    },
}
req = urllib.request.Request(
    f"http://127.0.0.1:{os.environ['PORT_VALUE']}/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
out = urllib.request.urlopen(req, timeout=720).read().decode()
Path(os.environ["RUN_DIR"], f"{os.environ['CASE']}.json").write_text(out, encoding="utf-8")
PY
}

run_case() {
  local name="$1"
  printf "%q " python -m sglang.launch_server "${BASE_ARGS[@]}" --port "$PORT" > "$RUN_DIR/${name}_command.txt"
  echo >> "$RUN_DIR/${name}_command.txt"
  setsid python -m sglang.launch_server "${BASE_ARGS[@]}" --port "$PORT" > "$RUN_DIR/${name}_server.log" 2>&1 &
  echo "$!" > "$RUN_DIR/${name}_server.pid"
  if wait_ready "$name"; then
    generate_prompt "$name"
  fi
  cleanup_server "$RUN_DIR/${name}_server.pid"
}

{
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "tmux_session=baseline_repeat_idx6"
  echo "log_path=$RUN_DIR/baseline_repeat.log"
  date -Is
  run_case baseline_a
  run_case baseline_b
  python3 - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
objs = {name: json.loads((run / f"{name}.json").read_text()) for name in ["baseline_a", "baseline_b"]}
def ids(o):
    return o.get("output_ids") or o.get("meta_info", {}).get("output_ids") or []
a = ids(objs["baseline_a"])
b = ids(objs["baseline_b"])
mismatch = None
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        mismatch = {
            "index": i,
            "a": x,
            "b": y,
            "a_window": a[max(0, i - 12): i + 16],
            "b_window": b[max(0, i - 12): i + 16],
        }
        break
if mismatch is None and len(a) != len(b):
    mismatch = {"index": min(len(a), len(b)), "lens": [len(a), len(b)]}
summary = {"exact": mismatch is None, "mismatch": mismatch, "len_a": len(a), "len_b": len(b)}
(run / "baseline_repeat_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
PY
} 2>&1 | tee "$RUN_DIR/baseline_repeat.log"
