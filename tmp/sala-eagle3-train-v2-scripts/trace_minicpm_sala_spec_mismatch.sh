#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

RUN_DIR=${RUN_DIR:-$SALA_EAGLE3_ROOT/eval/eagle_forensics_$(timestamp)}
PORT=${PORT:-30178}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
SESSION_NAME=${SESSION_NAME:-sala_eagle3_forensics}
PROMPTS_JSONL=${PROMPTS_JSONL:-$RUN_DIR/prompts.jsonl}
SPEC_ATTENTION_MODES=${SPEC_ATTENTION_MODES:-"prefill decode"}
INCLUDE_FLASHINFER=${INCLUDE_FLASHINFER:-0}

HEAD_SPECS=${HEAD_SPECS:-old_hot32k=$OLD_EAGLE3_HEAD}
if [[ -n "${FULL_VOCAB_HEAD:-}" ]]; then
  HEAD_SPECS="${HEAD_SPECS},full_vocab=${FULL_VOCAB_HEAD}"
fi
if [[ -n "${HOT32K_800_HEAD:-}" ]]; then
  HEAD_SPECS="${HEAD_SPECS},hot32k_800=${HOT32K_800_HEAD}"
fi
if [[ -n "${HOT32K_1600_HEAD:-}" ]]; then
  HEAD_SPECS="${HEAD_SPECS},hot32k_1600=${HOT32K_1600_HEAD}"
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
        "id": "natural",
        "text": "Human: Please explain speculative decoding in two concise sentences.\n\nAssistant:",
    },
    {
        "id": "repeat",
        "text": "Human: Continue this pattern and output only the next segment: abc abc abc abc abc abc\n\nAssistant:",
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
  python - "$PROMPTS_JSONL" <<'PY'
import json
import sys
print(json.dumps([json.loads(line)["id"] for line in open(sys.argv[1], encoding="utf-8") if line.strip()]))
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
  python - "$port" "$name" "$MAX_NEW_TOKENS" "$PROMPTS_JSONL" "$RUN_DIR" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

port, case, max_new, prompt_path, run_dir = sys.argv[1:6]
run_dir = Path(run_dir)
for line in Path(prompt_path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    payload = {
        "text": row["text"],
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
    out_path = run_dir / f"{case}_{row['id']}.json"
    try:
        out_path.write_text(urllib.request.urlopen(req, timeout=720).read().decode(), encoding="utf-8")
    except Exception as exc:
        out_path.write_text(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

head_vocab_mode() {
  local label="$1"
  if [[ "$label" == *hot* ]]; then
    echo hot
  elif [[ "$label" == *full* ]]; then
    echo full
  else
    echo unknown
  fi
}

append_manifest() {
  CASE="$1" BASELINE_CASE="$2" HEAD_LABEL="$3" HEAD_PATH_VALUE="$4" BACKEND="$5" SPEC_MODE="$6" \
    CASE_KIND="$7" TOPK_VALUE="$8" FORCE_REJECT_VALUE="$9" TRACE_PATH_VALUE="${10}" \
    HEAD_VOCAB_MODE_VALUE="$(head_vocab_mode "$3")" PROMPT_IDS_JSON="$PROMPT_IDS_JSON" \
    FORCE_REJECT_CASE_VALUE="${FORCE_REJECT_CASE:-}" \
    python - "$RUN_DIR/case_manifest.jsonl" <<'PY'
import json
import os
import sys

path = sys.argv[1]
topk = os.environ["TOPK_VALUE"]
row = {
    "case": os.environ["CASE"],
    "baseline_case": os.environ["BASELINE_CASE"],
    "head_label": os.environ["HEAD_LABEL"],
    "head_path": os.environ["HEAD_PATH_VALUE"],
    "head_vocab_mode": os.environ["HEAD_VOCAB_MODE_VALUE"],
    "backend": os.environ["BACKEND"],
    "spec_attention_mode": os.environ["SPEC_MODE"],
    "case_kind": os.environ["CASE_KIND"],
    "topk": None if topk == "" else int(topk),
    "is_topk1": topk == "1",
    "is_force_reject": os.environ["FORCE_REJECT_VALUE"] == "1",
    "force_reject_case": os.environ["FORCE_REJECT_CASE_VALUE"],
    "trace_path": os.environ["TRACE_PATH_VALUE"],
    "prompt_ids": json.loads(os.environ["PROMPT_IDS_JSON"]),
}
with open(path, "a", encoding="utf-8") as fout:
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

run_case() {
  local name="$1" baseline_case="$2" head_label="$3" head_path="$4" backend="$5" spec_mode="$6" case_kind="$7" topk="$8" force_reject="$9"
  shift 9
  local trace_path="$RUN_DIR/${name}_trace.jsonl"
  rm -f "$trace_path"
  echo "===== $name $(date -Iseconds) =====" | tee -a "$RUN_DIR/launch.log"
  printf '%q ' python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${name}_command.txt"
  echo >> "$RUN_DIR/${name}_command.txt"

  append_manifest "$name" "$baseline_case" "$head_label" "$head_path" "$backend" "$spec_mode" "$case_kind" "$topk" "$force_reject" "$trace_path"

  env \
    SGLANG_SALA_SPEC_TRACE=1 \
    SGLANG_SALA_SPEC_TRACE_PATH="$trace_path" \
    SGLANG_FORCE_REJECT_EAGLE_DRAFT="$force_reject" \
    setsid python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${name}_server.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUN_DIR/${name}_server.pid"

  if wait_ready "$PORT" "$name"; then
    generate_prompts "$PORT" "$name" || true
  fi
  cleanup_server "$RUN_DIR/${name}_server.pid"
  echo "$name done $(date -Iseconds)" | tee -a "$RUN_DIR/launch.log"
}

run_baseline() {
  local name="$1" backend="$2"
  run_case "$name" "$name" baseline "" "$backend" none baseline "" 0 \
    "${BASE_COMMON[@]}" --attention-backend "$backend"
}

run_spec_set() {
  local label="$1" head="$2" backend="$3" spec_mode="$4"
  local prefix="${label}_${backend}_${spec_mode}"
  local force_case="${prefix}_force_reject"
  local head_args=(
    --attention-backend "$backend"
    --speculative-algorithm EAGLE3
    --speculative-draft-model-path "$head"
    --speculative-attention-mode "$spec_mode"
  )

  FORCE_REJECT_CASE="$force_case" run_case "$force_case" "baseline_${backend}" "$label" "$head" "$backend" "$spec_mode" force_reject 1 1 \
    "${BASE_COMMON[@]}" "${head_args[@]}" --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
  FORCE_REJECT_CASE="$force_case" run_case "${prefix}_s1_topk1_draft2" "baseline_${backend}" "$label" "$head" "$backend" "$spec_mode" s1_topk1_draft2 1 0 \
    "${BASE_COMMON[@]}" "${head_args[@]}" --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
  FORCE_REJECT_CASE="$force_case" run_case "${prefix}_s3_topk1_draft4" "baseline_${backend}" "$label" "$head" "$backend" "$spec_mode" s3_topk1_draft4 1 0 \
    "${BASE_COMMON[@]}" "${head_args[@]}" --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  FORCE_REJECT_CASE="$force_case" run_case "${prefix}_s5_topk1_draft8" "baseline_${backend}" "$label" "$head" "$backend" "$spec_mode" s5_topk1_draft8 1 0 \
    "${BASE_COMMON[@]}" "${head_args[@]}" --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8
  FORCE_REJECT_CASE="$force_case" run_case "${prefix}_s2_topk2_draft5" "baseline_${backend}" "$label" "$head" "$backend" "$spec_mode" s2_topk2_draft5 2 0 \
    "${BASE_COMMON[@]}" "${head_args[@]}" --speculative-num-steps 2 --speculative-eagle-topk 2 --speculative-num-draft-tokens 5
  FORCE_REJECT_CASE="$force_case" run_case "${prefix}_s2_topk4_draft9" "baseline_${backend}" "$label" "$head" "$backend" "$spec_mode" s2_topk4_draft9 4 0 \
    "${BASE_COMMON[@]}" "${head_args[@]}" --speculative-num-steps 2 --speculative-eagle-topk 4 --speculative-num-draft-tokens 9
}

{
  print_run_header "$SESSION_NAME" "$RUN_DIR/launch.log" "$0 RUN_DIR=$RUN_DIR HEAD_SPECS=$HEAD_SPECS SPEC_ATTENTION_MODES=$SPEC_ATTENTION_MODES"
  echo "prompts=$PROMPTS_JSONL"
  echo "prompt_ids=$PROMPT_IDS_JSON"
  : > "$RUN_DIR/case_manifest.jsonl"

  run_baseline baseline_triton triton

  IFS=',' read -r -a head_specs <<< "$HEAD_SPECS"
  for spec in "${head_specs[@]}"; do
    label="${spec%%=*}"
    head="${spec#*=}"
    if [[ "$label" == "$head" || ! -d "$head" ]]; then
      echo "SKIP head spec '$spec': expected label=/path/to/head and existing directory" | tee -a "$RUN_DIR/launch.log"
      continue
    fi
    for spec_mode in $SPEC_ATTENTION_MODES; do
      run_spec_set "$label" "$head" triton "$spec_mode"
    done
  done

  if [[ "$INCLUDE_FLASHINFER" == 1 ]]; then
    run_baseline baseline_flashinfer flashinfer
    for spec in "${head_specs[@]}"; do
      label="${spec%%=*}"
      head="${spec#*=}"
      if [[ "$label" == "$head" || ! -d "$head" ]]; then
        continue
      fi
      run_spec_set "$label" "$head" flashinfer prefill
    done
  fi

  python "$SCRIPT_DIR/trace_minicpm_sala_spec_mismatch.py" \
    --run-dir "$RUN_DIR" \
    --manifest "$RUN_DIR/case_manifest.jsonl" \
    --output-md "$RUN_DIR/eagle_mismatch_forensics.md" \
    --output-json "$RUN_DIR/mismatch_summary.json"
  cp "$RUN_DIR/eagle_mismatch_forensics.md" "$SALA_EAGLE3_ROOT/reports/eagle_mismatch_forensics.md"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_DIR/final_gpu.csv" || true
  echo "summary=$RUN_DIR/eagle_mismatch_forensics.md"
  date -Is
} 2>&1 | tee "$RUN_DIR/forensics.log"
