#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

RUN_DIR=${RUN_DIR:-$SALA_EAGLE3_ROOT/eval/eagle_logit_margin_probe_$(timestamp)}
PORT=${PORT:-30270}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
SESSION_NAME=${SESSION_NAME:-sala_eagle3_logit_margin_probe}
PROMPTS_JSONL=${PROMPTS_JSONL:-$RUN_DIR/prompts.jsonl}
SPECULATIVE_ATTENTION_MODE=${SPECULATIVE_ATTENTION_MODE:-prefill}
DETERMINISTIC=${DETERMINISTIC:-1}
CONFIG_LABEL=${CONFIG_LABEL:-s2_topk4_draft9}
SPEC_STEPS=${SPEC_STEPS:-2}
SPEC_TOPK=${SPEC_TOPK:-4}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-9}
USE_TREE_EXTEND_KERNEL=${USE_TREE_EXTEND_KERNEL:-1}
USE_TREE_DECODE_KERNEL=${USE_TREE_DECODE_KERNEL:-0}
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
  --attention-backend triton
)

if [[ "$DETERMINISTIC" == 1 ]]; then
  export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
  BASE_COMMON+=(--enable-deterministic-inference)
fi

HEAD_ARGS=(
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path "$HEAD"
  --speculative-attention-mode "$SPECULATIVE_ATTENTION_MODE"
  --speculative-num-steps "$SPEC_STEPS"
  --speculative-eagle-topk "$SPEC_TOPK"
  --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS"
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
        "id": "natural_zh",
        "text": "Human: 用两句话解释什么是投机解码，不要列表。\n\nAssistant:",
    },
    {
        "id": "natural_en",
        "text": "Human: Please explain speculative decoding in two concise sentences.\n\nAssistant:",
    },
]
with path.open("w", encoding="utf-8") as fout:
    for row in prompts:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
fi

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

generate_one() {
  local port="$1" case_name="$2" prompt_id="$3" prompt_text="$4"
  python - "$port" "$case_name" "$prompt_id" "$prompt_text" "$MAX_NEW_TOKENS" "$RUN_DIR" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

port, case_name, prompt_id, text, max_new, run_dir = sys.argv[1:7]
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
out_path = Path(run_dir) / f"{case_name}_{prompt_id}.json"
try:
    out = urllib.request.urlopen(req, timeout=720).read().decode()
except Exception as exc:
    out = json.dumps({"error": str(exc)}, ensure_ascii=False)
out_path.write_text(out, encoding="utf-8")
PY
}

run_server_case() {
  local case_name="$1" prompt_id="$2" prompt_text="$3" is_spec="$4"
  shift 4
  local log_dir="$RUN_DIR/logits/${prompt_id}/${case_name}"
  local trace_path="$RUN_DIR/${case_name}_${prompt_id}_trace.jsonl"
  rm -rf "$log_dir"
  mkdir -p "$log_dir"
  rm -f "$trace_path"

  echo "===== $case_name / $prompt_id $(date -Iseconds) =====" | tee -a "$RUN_DIR/probe.log"
  printf '%q ' python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${case_name}_${prompt_id}_command.txt"
  echo >> "$RUN_DIR/${case_name}_${prompt_id}_command.txt"

  if [[ "$is_spec" == 1 ]]; then
    env \
      SGLANG_SALA_SPEC_TRACE=1 \
      SGLANG_SALA_SPEC_TRACE_PATH="$trace_path" \
      SGLANG_FORCE_REJECT_EAGLE_DRAFT=1 \
      SGLANG_TARGET_VERIFY_TREE_USE_TRITON_EXTEND_KERNEL="$USE_TREE_EXTEND_KERNEL" \
      SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL="$USE_TREE_DECODE_KERNEL" \
      SGLANG_TARGET_VERIFY_ROOT_USE_TRITON_DECODE_KERNEL=0 \
      SGLANG_SALA_DUMP_VERIFY_LOGITS_DIR="$log_dir" \
      setsid python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${case_name}_${prompt_id}_server.log" 2>&1 &
  else
    env \
      SGLANG_SALA_SPEC_TRACE=1 \
      SGLANG_SALA_SPEC_TRACE_PATH="$trace_path" \
      SGLANG_SALA_LOGITS_TRACE_MODES=extend,decode \
      SGLANG_SALA_DUMP_FORWARD_LOGITS_DIR="$log_dir" \
      setsid python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${case_name}_${prompt_id}_server.log" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$RUN_DIR/${case_name}_${prompt_id}_server.pid"

  if wait_ready "$PORT" "${case_name}_${prompt_id}"; then
    generate_one "$PORT" "$case_name" "$prompt_id" "$prompt_text" || true
  fi
  cleanup_server "$RUN_DIR/${case_name}_${prompt_id}_server.pid"
  echo "$case_name / $prompt_id done $(date -Iseconds)" | tee -a "$RUN_DIR/probe.log"
}

{
  print_run_header "$SESSION_NAME" "$RUN_DIR/probe.log" "$0 RUN_DIR=$RUN_DIR HEAD=$HEAD CONFIG=$CONFIG_LABEL DETERMINISTIC=$DETERMINISTIC USE_TREE_EXTEND_KERNEL=$USE_TREE_EXTEND_KERNEL"
  echo "head=$HEAD"
  echo "prompts=$PROMPTS_JSONL"
  echo "max_new_tokens=$MAX_NEW_TOKENS"
  echo "speculative_attention_mode=$SPECULATIVE_ATTENTION_MODE"
  echo "config=$CONFIG_LABEL steps=$SPEC_STEPS topk=$SPEC_TOPK draft_tokens=$SPEC_DRAFT_TOKENS"
  echo "use_tree_extend_kernel=$USE_TREE_EXTEND_KERNEL use_tree_decode_kernel=$USE_TREE_DECODE_KERNEL"

  : > "$RUN_DIR/comparisons.jsonl"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    prompt_id=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$line")
    prompt_text=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["text"])' "$line")

    run_server_case baseline "$prompt_id" "$prompt_text" 0 "${BASE_COMMON[@]}"
    run_server_case "force_reject_${CONFIG_LABEL}" "$prompt_id" "$prompt_text" 1 "${BASE_COMMON[@]}" "${HEAD_ARGS[@]}"

    python "$SCRIPT_DIR/compare_eagle_verify_logits.py" \
      --prompt-id "$prompt_id" \
      --baseline-json "$RUN_DIR/baseline_${prompt_id}.json" \
      --spec-json "$RUN_DIR/force_reject_${CONFIG_LABEL}_${prompt_id}.json" \
      --baseline-logits-dir "$RUN_DIR/logits/${prompt_id}/baseline" \
      --verify-logits-dir "$RUN_DIR/logits/${prompt_id}/force_reject_${CONFIG_LABEL}" \
      --output-json "$RUN_DIR/${prompt_id}_logit_compare.json" \
      --output-md "$RUN_DIR/${prompt_id}_logit_compare.md"
    python - "$RUN_DIR/${prompt_id}_logit_compare.json" "$RUN_DIR/comparisons.jsonl" <<'PY'
import json
import sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
with dst.open("a", encoding="utf-8") as fout:
    fout.write(json.dumps(json.loads(src.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
PY
  done < "$PROMPTS_JSONL"

  python - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = [json.loads(line) for line in (run_dir / "comparisons.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
lines = [
    "# EAGLE force-reject logit margin probe",
    "",
    f"- run_dir: `{run_dir}`",
    "",
    "| prompt | classification | first output mismatch | first argmax mismatch | first nonzero diff | compared |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for row in rows:
    lines.append(
        f"| `{row['prompt_id']}` | `{row['classification']}` | `{row['first_output_mismatch']}` | `{row['first_verify_argmax_mismatch']}` | `{row['first_nonzero_logit_diff']}` | `{row['max_compared_tokens']}` |"
    )
lines.append("")
lines.append("## Interpretation rule")
lines.append("")
lines.append("- `verify_argmax_changed_at_output_mismatch`: exact mismatch is a real greedy-decision change in target verify; kernel/metadata/state still need fixing for lossless speculative decoding.")
lines.append("- `output_mismatch_without_root_argmax_mismatch`: logits top1 was stable at the mismatch point; inspect commit/sampling/output plumbing before kernel work.")
lines.append("- `outputs_exact`: this prompt does not reproduce the mismatch under this probe.")
(run_dir / "logit_margin_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((run_dir / "logit_margin_summary.md").read_text(encoding="utf-8"))
PY
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_DIR/final_gpu.csv" || true
  echo "summary=$RUN_DIR/logit_margin_summary.md"
  date -Is
} 2>&1 | tee "$RUN_DIR/probe.log"
