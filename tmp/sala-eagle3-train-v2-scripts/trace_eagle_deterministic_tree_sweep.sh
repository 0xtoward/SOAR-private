#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

RUN_DIR=${RUN_DIR:-$SALA_EAGLE3_ROOT/eval/eagle_det_tree_sweep_$(timestamp)}
PORT=${PORT:-30240}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
SESSION_NAME=${SESSION_NAME:-sala_eagle3_det_tree_sweep}
PROMPTS_JSONL=${PROMPTS_JSONL:-$RUN_DIR/prompts.jsonl}
REQUEST_BATCH_SIZE=${REQUEST_BATCH_SIZE:-1}
SPECULATIVE_ATTENTION_MODE=${SPECULATIVE_ATTENTION_MODE:-prefill}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}
BASELINE_CASE_NAME=${BASELINE_CASE_NAME:-baseline_${ATTENTION_BACKEND}}
DETERMINISTIC=${DETERMINISTIC:-1}
RUN_FORCE_REJECT_CASES=${RUN_FORCE_REJECT_CASES:-1}
ROOT_ONLY_VERIFY=${ROOT_ONLY_VERIFY:-0}
CONFIG_MATRIX=${CONFIG_MATRIX:-"s1_topk1_draft2:1:1:2 s3_topk1_draft4:3:1:4 s5_topk1_draft6:5:1:6 s5_topk1_draft8:5:1:8 s2_topk2_draft5:2:2:5 s2_topk4_draft9:2:4:9"}
DEFAULT_HOT32K_HEAD=${DEFAULT_HOT32K_HEAD:-$SALA_EAGLE3_ROOT/outputs/main2000_hot32k_lr5e5_ttt5/epoch_0_step_800}
HEAD=${HEAD:-}
if [[ -z "$HEAD" ]]; then
  if [[ -d "$DEFAULT_HOT32K_HEAD" ]]; then
    HEAD="$DEFAULT_HOT32K_HEAD"
  else
    HEAD="$OLD_EAGLE3_HEAD"
  fi
fi

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
  --attention-backend "$ATTENTION_BACKEND"
)

if [[ "$DETERMINISTIC" == 1 ]]; then
  export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
  BASE_COMMON+=(--enable-deterministic-inference)
fi

HEAD_ARGS=(
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path "$HEAD"
  --speculative-attention-mode "$SPECULATIVE_ATTENTION_MODE"
)

if [[ ! -s "$PROMPTS_JSONL" ]]; then
  python - "$PROMPTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
prompts = [
    {
        "id": "known_index59_repeat",
        "text": "Human: Continue this pattern and output only the next segment: abc abc abc abc abc abc\n\nAssistant:",
    },
    {
        "id": "natural_en",
        "text": "Human: Please explain speculative decoding in two concise sentences.\n\nAssistant:",
    },
    {
        "id": "natural_zh",
        "text": "Human: \u7528\u4e24\u53e5\u8bdd\u89e3\u91ca\u4ec0\u4e48\u662f\u6295\u673a\u89e3\u7801\uff0c\u4e0d\u8981\u5217\u8868\u3002\n\nAssistant:",
    },
    {
        "id": "code_structured",
        "text": "Human: Write a tiny Python function named add_one(x), then give one example call. Keep the answer short.\n\nAssistant:",
    },
    {
        "id": "copy_heavy",
        "text": "Human: Repeat exactly, then continue for one short clause: MiniCPM-SALA uses sparse attention and linear attention. MiniCPM-SALA uses sparse attention and linear attention.\n\nAssistant:",
    },
]
with path.open("w", encoding="utf-8") as fout:
    for row in prompts:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
fi

PROMPT_IDS_JSON=$(
  python - "$PROMPTS_JSONL" "$REQUEST_BATCH_SIZE" <<'PY'
import json
import sys

batch = int(sys.argv[2])
ids = []
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip():
        continue
    prompt_id = json.loads(line)["id"]
    for idx in range(batch):
        ids.append(prompt_id if batch == 1 else f"{prompt_id}__r{idx}")
print(json.dumps(ids))
PY
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
      wait "$pid" 2>/dev/null || true
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

generate_prompts() {
  local port="$1" name="$2"
  python - "$port" "$name" "$MAX_NEW_TOKENS" "$PROMPTS_JSONL" "$RUN_DIR" "$REQUEST_BATCH_SIZE" <<'PY'
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

port, case, max_new, prompt_path, run_dir, batch_size = sys.argv[1:7]
run_dir = Path(run_dir)
batch_size = int(batch_size)
tasks = []
for line in Path(prompt_path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    for idx in range(batch_size):
        prompt_id = row["id"] if batch_size == 1 else f"{row['id']}__r{idx}"
        tasks.append((prompt_id, row["text"]))

def one(prompt_id: str, text: str) -> None:
    payload = {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": int(max_new),
        },
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    out_path = run_dir / f"{case}_{prompt_id}.json"
    try:
        out = urllib.request.urlopen(req, timeout=720).read().decode()
    except Exception as exc:
        out = json.dumps({"error": str(exc)}, ensure_ascii=False)
    out_path.write_text(out, encoding="utf-8")

workers = max(1, min(batch_size, len(tasks)))
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(one, prompt_id, text) for prompt_id, text in tasks]
    for future in as_completed(futures):
        future.result()
PY
}

append_manifest() {
  CASE="$1" BASELINE_CASE="$2" CASE_KIND="$3" STEPS_VALUE="$4" TOPK_VALUE="$5" DRAFT_TOKENS_VALUE="$6" \
    FORCE_REJECT_VALUE="$7" TRACE_PATH_VALUE="$8" FORCE_REJECT_CASE_VALUE="$9" PROMPT_IDS_JSON="$PROMPT_IDS_JSON" \
    HEAD_PATH_VALUE="$HEAD" SPEC_MODE_VALUE="$SPECULATIVE_ATTENTION_MODE" DETERMINISTIC_VALUE="$DETERMINISTIC" \
    ATTENTION_BACKEND_VALUE="$ATTENTION_BACKEND" \
    REQUEST_BATCH_SIZE_VALUE="$REQUEST_BATCH_SIZE" ROOT_ONLY_VERIFY_VALUE="$ROOT_ONLY_VERIFY" \
    python - "$RUN_DIR/case_manifest.jsonl" <<'PY'
import json
import os
import sys

row = {
    "case": os.environ["CASE"],
    "baseline_case": os.environ["BASELINE_CASE"],
    "case_kind": os.environ["CASE_KIND"],
    "head_label": "eagle_head",
    "head_path": os.environ["HEAD_PATH_VALUE"],
    "head_vocab_mode": "hot" if "hot32k" in os.environ["HEAD_PATH_VALUE"] else "unknown",
    "backend": os.environ["ATTENTION_BACKEND_VALUE"],
    "spec_attention_mode": os.environ["SPEC_MODE_VALUE"],
    "deterministic": os.environ["DETERMINISTIC_VALUE"] == "1",
    "root_only_verify": os.environ.get("ROOT_ONLY_VERIFY_VALUE") == "1",
    "request_batch_size": int(os.environ["REQUEST_BATCH_SIZE_VALUE"]),
    "steps": None if os.environ["STEPS_VALUE"] == "" else int(os.environ["STEPS_VALUE"]),
    "topk": None if os.environ["TOPK_VALUE"] == "" else int(os.environ["TOPK_VALUE"]),
    "is_topk1": os.environ["TOPK_VALUE"] == "1",
    "draft_tokens": None if os.environ["DRAFT_TOKENS_VALUE"] == "" else int(os.environ["DRAFT_TOKENS_VALUE"]),
    "is_force_reject": os.environ["FORCE_REJECT_VALUE"] == "1",
    "force_reject_case": os.environ["FORCE_REJECT_CASE_VALUE"],
    "trace_path": os.environ["TRACE_PATH_VALUE"],
    "prompt_ids": json.loads(os.environ["PROMPT_IDS_JSON"]),
}
with open(sys.argv[1], "a", encoding="utf-8") as fout:
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

run_case() {
  local name="$1" baseline_case="$2" case_kind="$3" steps="$4" topk="$5" draft_tokens="$6" force_reject="$7"
  shift 7
  local trace_path="$RUN_DIR/${name}_trace.jsonl"
  local force_reject_case=""
  if [[ "$force_reject" != "1" && "$name" == accept_* && "$RUN_FORCE_REJECT_CASES" == 1 ]]; then
    force_reject_case="force_reject_${name#accept_}"
  fi
  rm -f "$trace_path"
  echo "===== $name $(date -Iseconds) =====" | tee -a "$RUN_DIR/launch.log"
  printf '%q ' python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${name}_command.txt"
  echo >> "$RUN_DIR/${name}_command.txt"
  append_manifest "$name" "$baseline_case" "$case_kind" "$steps" "$topk" "$draft_tokens" "$force_reject" "$trace_path" "$force_reject_case"

  env \
    SGLANG_SALA_SPEC_TRACE=1 \
    SGLANG_SALA_SPEC_TRACE_PATH="$trace_path" \
    SGLANG_FORCE_REJECT_EAGLE_DRAFT="$force_reject" \
    SGLANG_EAGLE_ROOT_ONLY_VERIFY="$ROOT_ONLY_VERIFY" \
    setsid python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${name}_server.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_DIR/${name}_server.pid"

  if wait_ready "$PORT" "$name"; then
    generate_prompts "$PORT" "$name" || true
  fi
  cleanup_server "$RUN_DIR/${name}_server.pid"
  echo "$name done $(date -Iseconds)" | tee -a "$RUN_DIR/launch.log"
}

{
  print_run_header "$SESSION_NAME" "$RUN_DIR/sweep.log" "$0 RUN_DIR=$RUN_DIR HEAD=$HEAD CONFIG_MATRIX='$CONFIG_MATRIX' DETERMINISTIC=$DETERMINISTIC REQUEST_BATCH_SIZE=$REQUEST_BATCH_SIZE"
  echo "head=$HEAD"
  echo "prompts=$PROMPTS_JSONL"
  echo "prompt_ids=$PROMPT_IDS_JSON"
  echo "attention_backend=$ATTENTION_BACKEND"
  echo "deterministic=$DETERMINISTIC"
  echo "speculative_attention_mode=$SPECULATIVE_ATTENTION_MODE"
  echo "request_batch_size=$REQUEST_BATCH_SIZE"
  echo "root_only_verify=$ROOT_ONLY_VERIFY"
  echo "config_matrix=$CONFIG_MATRIX"
  : > "$RUN_DIR/case_manifest.jsonl"

  run_case "$BASELINE_CASE_NAME" "$BASELINE_CASE_NAME" baseline "" "" "" 0 \
    "${BASE_COMMON[@]}"

  for spec in $CONFIG_MATRIX; do
    IFS=':' read -r label steps topk draft_tokens <<< "$spec"
    if [[ "$RUN_FORCE_REJECT_CASES" == 1 ]]; then
      run_case "force_reject_${label}" "$BASELINE_CASE_NAME" "force_reject_${label}" "$steps" "$topk" "$draft_tokens" 1 \
        "${BASE_COMMON[@]}" "${HEAD_ARGS[@]}" \
        --speculative-num-steps "$steps" \
        --speculative-eagle-topk "$topk" \
        --speculative-num-draft-tokens "$draft_tokens"
    fi
    run_case "accept_${label}" "$BASELINE_CASE_NAME" "accept_${label}" "$steps" "$topk" "$draft_tokens" 0 \
      "${BASE_COMMON[@]}" "${HEAD_ARGS[@]}" \
      --speculative-num-steps "$steps" \
      --speculative-eagle-topk "$topk" \
      --speculative-num-draft-tokens "$draft_tokens"
  done

  python "$SCRIPT_DIR/trace_minicpm_sala_spec_mismatch.py" \
    --run-dir "$RUN_DIR" \
    --manifest "$RUN_DIR/case_manifest.jsonl" \
    --output-md "$RUN_DIR/eagle_deterministic_tree_sweep.md" \
    --output-json "$RUN_DIR/eagle_deterministic_tree_sweep.json"
  cp "$RUN_DIR/eagle_deterministic_tree_sweep.md" "$SALA_EAGLE3_ROOT/reports/eagle_deterministic_tree_sweep.md"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_DIR/final_gpu.csv" || true
  echo "summary=$RUN_DIR/eagle_deterministic_tree_sweep.md"
  date -Is
} 2>&1 | tee "$RUN_DIR/sweep.log"
