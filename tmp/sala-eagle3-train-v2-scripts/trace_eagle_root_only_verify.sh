#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

RUN_DIR=${RUN_DIR:-$SALA_EAGLE3_ROOT/eval/eagle_root_only_verify_$(timestamp)}
PORT=${PORT:-30218}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
SESSION_NAME=${SESSION_NAME:-sala_eagle3_root_only_verify}
HEAD_PATH=${HEAD_PATH:-$SALA_EAGLE3_ROOT/outputs/main2000_hot32k_lr5e5_ttt5/epoch_0_step_800}
PROMPT_TEXT=${PROMPT_TEXT:-"Human: Continue this pattern and output only the next segment: abc abc abc abc abc abc\n\nAssistant:"}
SPECULATIVE_ATTENTION_MODE=${SPECULATIVE_ATTENTION_MODE:-prefill}

mkdir -p "$RUN_DIR"
cd "$SGLANG_REPO"

export SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-False}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

BASE_COMMON=(
  --model-path "$MODEL_PATH"
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

SPEC_ARGS=(
  "${BASE_COMMON[@]}"
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path "$HEAD_PATH"
  --speculative-attention-mode "$SPECULATIVE_ATTENTION_MODE"
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
)

cleanup_server() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]]; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    fi
  fi
}

wait_ready() {
  local port="$1" name="$2"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/get_model_info" > "$RUN_DIR/${name}_server_info.json" 2> "$RUN_DIR/${name}_ready.err"; then
      return 0
    fi
    if [[ -f "$RUN_DIR/${name}_server.pid" ]]; then
      local pid
      pid=$(cat "$RUN_DIR/${name}_server.pid")
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "${name}: server exited before ready" >&2
        tail -180 "$RUN_DIR/${name}_server.log" >&2 || true
        return 1
      fi
    fi
    sleep 1
  done
  echo "${name}: server not ready" >&2
  tail -180 "$RUN_DIR/${name}_server.log" >&2 || true
  return 1
}

generate_repeat() {
  local port="$1" name="$2"
  PROMPT_TEXT="$PROMPT_TEXT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" RUN_DIR="$RUN_DIR" CASE="$name" PORT_VALUE="$port" python - <<'PY'
import json
import os
import urllib.request
from pathlib import Path

payload = {
    "text": os.environ["PROMPT_TEXT"].encode("utf-8").decode("unicode_escape"),
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
Path(os.environ["RUN_DIR"], f"{os.environ['CASE']}_repeat.json").write_text(out, encoding="utf-8")
PY
}

run_server_case() {
  local name="$1"
  local trace_path="$RUN_DIR/${name}_trace.jsonl"
  shift
  rm -f "$trace_path"
  echo "===== $name $(date -Iseconds) =====" | tee -a "$RUN_DIR/launch.log"
  printf '%q ' python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${name}_command.txt"
  echo >> "$RUN_DIR/${name}_command.txt"

  env \
    SGLANG_SALA_SPEC_TRACE=1 \
    SGLANG_SALA_SPEC_TRACE_PATH="$trace_path" \
    "${CASE_EXTRA_ENV[@]}" \
    setsid python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${name}_server.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_DIR/${name}_server.pid"

  if wait_ready "$PORT" "$name"; then
    generate_repeat "$PORT" "$name" || true
  fi
  cleanup_server "$RUN_DIR/${name}_server.pid"
  echo "$name done $(date -Iseconds)" | tee -a "$RUN_DIR/launch.log"
}

summarize() {
  python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])

def load_case(name):
    path = run_dir / f"{name}_repeat.json"
    return json.loads(path.read_text(encoding="utf-8"))

def ids(obj):
    return obj.get("output_ids") or obj.get("meta_info", {}).get("output_ids") or []

def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return [i, x, y]
    if len(a) != len(b):
        return [min(len(a), len(b)), None, None]
    return None

def trace_stats(name):
    path = run_dir / f"{name}_trace.jsonl"
    stats = {
        "trace_path": str(path),
        "exists": path.exists(),
        "root_only_events": 0,
        "verify_draft_token_nums": [],
        "mamba_update_draft_token_nums": [],
        "accept_length_per_req_cpu": [],
        "accepted_steps": [],
    }
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "eagle_root_only_verify_input":
            stats["root_only_events"] += 1
            stats["verify_draft_token_nums"].append(row.get("draft_token_num"))
        if row.get("event") == "eagle_mamba_verify_update":
            stats["mamba_update_draft_token_nums"].append(row.get("draft_token_num"))
            stats["accept_length_per_req_cpu"].append(row.get("accept_length_per_req_cpu"))
            stats["accepted_steps"].append(row.get("accepted_steps"))
    return stats

base = load_case("baseline_triton")
spec = load_case("eagle_root_only_force_reject")
base_ids = ids(base)
spec_ids = ids(spec)
summary = {
    "baseline_len": len(base_ids),
    "spec_len": len(spec_ids),
    "exact": base_ids == spec_ids,
    "mismatch": first_mismatch(base_ids, spec_ids),
    "baseline_window": None,
    "spec_window": None,
    "root_only_trace": trace_stats("eagle_root_only_force_reject"),
}
if summary["mismatch"]:
    i = summary["mismatch"][0]
    lo = max(0, i - 10)
    hi = i + 10
    summary["baseline_window"] = base_ids[lo:hi]
    summary["spec_window"] = spec_ids[lo:hi]
(run_dir / "root_only_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
}

{
  print_run_header "$SESSION_NAME" "$RUN_DIR/root_only_verify.log" "$0 RUN_DIR=$RUN_DIR HEAD_PATH=$HEAD_PATH PORT=$PORT"
  echo "head_path=$HEAD_PATH"
  echo "prompt_text=$PROMPT_TEXT"

  CASE_EXTRA_ENV=()
  run_server_case baseline_triton "${BASE_COMMON[@]}"

  CASE_EXTRA_ENV=(
    SGLANG_EAGLE_ROOT_ONLY_VERIFY=1
    SGLANG_FORCE_REJECT_EAGLE_DRAFT=1
  )
  run_server_case eagle_root_only_force_reject "${SPEC_ARGS[@]}"

  summarize
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_DIR/final_gpu.csv" || true
  echo "run_dir=$RUN_DIR"
  date -Is
} 2>&1 | tee "$RUN_DIR/root_only_verify.log"
