#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

RUN_DIR=${RUN_DIR:-$SALA_EAGLE3_ROOT/eval_quality/flash_native_quality_smoke_$(timestamp)}
DATA_PATH=${DATA_PATH:-/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_fast_eval.jsonl}
TOOLKIT_DIR=${TOOLKIT_DIR:-/root/autodl-tmp/SOAR-Toolkit}
STREAM_EVAL=${STREAM_EVAL:-/root/autodl-tmp/codex-drafts/stream_eval_model.py}
HEAD=${HEAD:-$SALA_EAGLE3_ROOT/outputs/main2000_hot32k_lr5e5_ttt5/epoch_0_step_800}
PORT_BASE=${PORT_BASE:-30480}
NUM_SAMPLES=${NUM_SAMPLES:-9}
MAX_OUT_LEN=${MAX_OUT_LEN:-1024}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-262144}
CONCURRENCY=${CONCURRENCY:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
SPEC_STEPS=${SPEC_STEPS:-2}
SPEC_TOPK=${SPEC_TOPK:-4}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-9}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_SPEC=${RUN_SPEC:-1}

mkdir -p "$RUN_DIR"
cd "$SGLANG_REPO"

export PYTHONPATH="$SGLANG_SRC:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

BASE_SERVER_ARGS=(
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

wait_ready() {
  local port="$1" case_name="$2" pid="$3"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "$RUN_DIR/${case_name}.models.json" 2> "$RUN_DIR/${case_name}.ready.err"; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${case_name}: server exited before ready" >&2
      tail -200 "$RUN_DIR/${case_name}.server.log" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "${case_name}: server not ready" >&2
  tail -200 "$RUN_DIR/${case_name}.server.log" >&2 || true
  return 1
}

cleanup_server() {
  local pid="${1:-}"
  if [[ -n "$pid" ]]; then
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

run_eval_case() {
  local case_name="$1"
  local port="$2"
  shift 2
  local extra_args=("$@")
  local server_log="$RUN_DIR/${case_name}.server.log"
  local eval_log="$RUN_DIR/${case_name}.eval.log"
  local out_dir="$RUN_DIR/${case_name}"

  mkdir -p "$out_dir"
  echo "case=${case_name}" | tee "$RUN_DIR/${case_name}.meta"
  echo "port=${port}" | tee -a "$RUN_DIR/${case_name}.meta"
  echo "backend=${ATTENTION_BACKEND}" | tee -a "$RUN_DIR/${case_name}.meta"
  echo "server_extra=${extra_args[*]}" | tee -a "$RUN_DIR/${case_name}.meta"
  echo "data=${DATA_PATH}" | tee -a "$RUN_DIR/${case_name}.meta"
  echo "head=${HEAD}" | tee -a "$RUN_DIR/${case_name}.meta"

  (
    set -x
    python -m sglang.launch_server \
      "${BASE_SERVER_ARGS[@]}" \
      --port "$port" \
      "${extra_args[@]}"
  ) > "$server_log" 2>&1 &
  local server_pid=$!
  echo "$server_pid" > "$RUN_DIR/${case_name}.server.pid"

  trap 'cleanup_server "$server_pid"' RETURN
  wait_ready "$port" "$case_name" "$server_pid"

  (
    set -x
    MAX_PROMPT_LEN="$MAX_SEQ_LEN" python "$STREAM_EVAL" \
      --toolkit_dir "$TOOLKIT_DIR" \
      --model_path "$MODEL_PATH" \
      --api_base "http://127.0.0.1:${port}" \
      --data_path "$DATA_PATH" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --max_out_len "$MAX_OUT_LEN" \
      --concurrency "$CONCURRENCY" \
      --num_samples "$NUM_SAMPLES" \
      --output_dir "$out_dir"
  ) > "$eval_log" 2>&1

  cleanup_server "$server_pid"
  trap - RETURN
}

{
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "run_dir=$RUN_DIR"
  echo "model_path=$MODEL_PATH"
  echo "data_path=$DATA_PATH"
  echo "attention_backend=$ATTENTION_BACKEND"
  echo "num_samples=$NUM_SAMPLES"
  echo "max_out_len=$MAX_OUT_LEN"
  echo "concurrency=$CONCURRENCY"
  date -Is
} | tee "$RUN_DIR/run_header.txt"

if [[ "$RUN_BASELINE" == 1 ]]; then
  run_eval_case "original_nospec_${ATTENTION_BACKEND}" "$PORT_BASE"
fi

export SGLANG_MINICPM_SALA_EXACT_MODE=0
export SGLANG_MINICPM_SALA_SPEC_EXACT_MODE=0
export SGLANG_MINICPM_SALA_EAGLE_EXACT_MODE=0
export SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=0
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM=0
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_QKV=0
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MLP=0
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_ATTENTION=0

if [[ "$RUN_SPEC" == 1 ]]; then
  run_eval_case "flash_native_noexact_eagle_s${SPEC_STEPS}_topk${SPEC_TOPK}_draft${SPEC_DRAFT_TOKENS}" "$((PORT_BASE + 1))" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$HEAD" \
    --speculative-attention-mode prefill \
    --speculative-num-steps "$SPEC_STEPS" \
    --speculative-eagle-topk "$SPEC_TOPK" \
    --speculative-num-draft-tokens "$SPEC_DRAFT_TOKENS"
fi

python - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
rows = []
for summary in sorted(run_dir.glob("*/summary.json")):
    case = summary.parent.name
    payload = json.loads(summary.read_text())
    stream_summary = summary.parent / "summary.stream.json"
    if stream_summary.exists():
        payload.update({f"stream_{k}": v for k, v in json.loads(stream_summary.read_text()).items()})
    rows.append((case, payload))

print("case\tori_accuracy\toverall_accuracy\tnum_samples\tout_tokens\ttps")
for case, payload in rows:
    print(
        f"{case}\t{payload.get('ori_accuracy')}\t{payload.get('overall_accuracy')}\t"
        f"{payload.get('num_samples')}\t{payload.get('total_output_tokens')}\t{payload.get('tps')}"
    )

out = run_dir / "quality_smoke_summary.tsv"
with out.open("w", encoding="utf-8") as f:
    f.write("case\tori_accuracy\toverall_accuracy\tnum_samples\tout_tokens\ttps\n")
    for case, payload in rows:
        f.write(
            f"{case}\t{payload.get('ori_accuracy')}\t{payload.get('overall_accuracy')}\t"
            f"{payload.get('num_samples')}\t{payload.get('total_output_tokens')}\t{payload.get('tps')}\n"
        )
PY

echo "DONE run_dir=$RUN_DIR"
