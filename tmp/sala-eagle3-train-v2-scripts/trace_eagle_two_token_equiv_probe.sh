#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

RUN_DIR=${RUN_DIR:-$SALA_EAGLE3_ROOT/eval/eagle_two_token_equiv_probe_$(timestamp)}
PORT=${PORT:-30228}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
SESSION_NAME=${SESSION_NAME:-sala_eagle3_two_token_equiv_probe}
HEAD_PATH=${HEAD_PATH:-$SALA_EAGLE3_ROOT/outputs/main2000_hot32k_lr5e5_ttt5/epoch_0_step_800}
PROMPT_TEXT=${PROMPT_TEXT:-"Human: Continue this pattern and output only the next segment: abc abc abc abc abc abc\n\nAssistant:"}
VERIFY_LOGIT_STEPS=${VERIFY_LOGIT_STEPS:-55,56,57,58,59,60,61,62}
SPECULATIVE_ATTENTION_MODE=${SPECULATIVE_ATTENTION_MODE:-prefill}

mkdir -p "$RUN_DIR"
cd "$SGLANG_REPO"

export SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-False}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

COMMON_ARGS=(
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
  --speculative-algorithm EAGLE3
  --speculative-draft-model-path "$HEAD_PATH"
  --speculative-attention-mode "$SPECULATIVE_ATTENTION_MODE"
  --speculative-num-steps 1
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2
)
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  # Intentionally split on spaces for simple flag lists like
  # EXTRA_SERVER_ARGS="--enable-deterministic-inference".
  # Keep complex quoted values out of this smoke harness.
  read -r -a EXTRA_ARGS <<< "$EXTRA_SERVER_ARGS"
  COMMON_ARGS+=("${EXTRA_ARGS[@]}")
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

run_case() {
  local name="$1"
  shift
  local trace_path="$RUN_DIR/${name}_trace.jsonl"
  local logits_dir="$RUN_DIR/${name}_logits"
  rm -rf "$logits_dir"
  mkdir -p "$logits_dir"
  rm -f "$trace_path"
  echo "===== $name $(date -Iseconds) =====" | tee -a "$RUN_DIR/launch.log"
  printf '%q ' python -m sglang.launch_server "$@" --port "$PORT" > "$RUN_DIR/${name}_command.txt"
  echo >> "$RUN_DIR/${name}_command.txt"

  env \
    SGLANG_SALA_SPEC_TRACE=1 \
    SGLANG_SALA_SPEC_TRACE_PATH="$trace_path" \
    SGLANG_SALA_DUMP_VERIFY_LOGITS_DIR="$logits_dir" \
    SGLANG_SALA_DUMP_VERIFY_LOGITS_STEPS="$VERIFY_LOGIT_STEPS" \
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
import glob
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

run_dir = Path(sys.argv[1])

def load_json(name):
    return json.loads((run_dir / f"{name}_repeat.json").read_text(encoding="utf-8"))

def output_ids(obj):
    return obj.get("output_ids") or obj.get("meta_info", {}).get("output_ids") or []

def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "a": x, "b": y}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "a": None, "b": None}
    return None

def load_logits(case):
    out = {}
    for path in glob.glob(str(run_dir / f"{case}_logits" / "*.pt")):
        payload = torch.load(path, map_location="cpu")
        out[int(payload["step"])] = {"path": path, "payload": payload}
    return out

def top(values, k=10):
    vals, ids = torch.topk(values, k=min(k, values.numel()))
    return {"ids": ids.tolist(), "logits": vals.tolist()}

def trace_stats(case):
    path = run_dir / f"{case}_trace.jsonl"
    stats = {
        "trace_path": str(path),
        "events": {},
        "max_commit_source_diff": None,
        "commit_windows": [],
    }
    if not path.exists():
        return stats
    max_diff = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ev = row.get("event")
        stats["events"][ev] = stats["events"].get(ev, 0) + 1
        if ev == "simple_gla_commit_after":
            diff = row.get("commit_source_diff")
            if diff is not None:
                max_diff = max(max_diff, float(diff["max_abs_diff"]))
        if ev == "eagle_verify_commit" and len(stats["commit_windows"]) < 8:
            stats["commit_windows"].append({
                "step": row.get("req_spec_verify_ct"),
                "draft_token_num": row.get("draft_token_num"),
                "seq_lens_before": row.get("seq_lens_before_commit"),
                "seq_lens_after": row.get("seq_lens_after_commit"),
                "out_cache_loc_after": row.get("out_cache_loc_after_commit"),
                "req_to_token_window_after": row.get("req_to_token_window_after_commit"),
                "req_output_ids_tail": row.get("req_output_ids_tail"),
                "kv_committed_len": row.get("kv_committed_len"),
                "kv_allocated_len": row.get("kv_allocated_len"),
            })
    stats["max_commit_source_diff"] = max_diff
    return stats

root_obj = load_json("root_only_force_reject")
two_obj = load_json("two_token_force_reject")
root_ids = output_ids(root_obj)
two_ids = output_ids(two_obj)
root_logits = load_logits("root_only_force_reject")
two_logits = load_logits("two_token_force_reject")

logit_comparisons = []
for step in sorted(set(root_logits) & set(two_logits)):
    r = root_logits[step]["payload"]["logits"][0].float()
    t = two_logits[step]["payload"]["logits"][0].float()
    diff = (r - t).abs()
    logit_comparisons.append({
        "step": step,
        "root_path": root_logits[step]["path"],
        "two_path": two_logits[step]["path"],
        "root_argmax": int(torch.argmax(r).item()),
        "two_argmax": int(torch.argmax(t).item()),
        "same_argmax": bool(torch.argmax(r).item() == torch.argmax(t).item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "cosine": float(F.cosine_similarity(r, t, dim=0).item()),
        "root_top10": top(r),
        "two_top10": top(t),
    })

summary = {
    "root_output_len": len(root_ids),
    "two_output_len": len(two_ids),
    "root_vs_two_exact": root_ids == two_ids,
    "root_vs_two_first_mismatch": first_mismatch(root_ids, two_ids),
    "logit_steps_compared": [x["step"] for x in logit_comparisons],
    "logit_comparisons": logit_comparisons,
    "root_trace_stats": trace_stats("root_only_force_reject"),
    "two_trace_stats": trace_stats("two_token_force_reject"),
}
(run_dir / "two_token_equiv_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

md = ["# EAGLE two-token equivalence probe", ""]
md.append(f"- run_dir: `{run_dir}`")
md.append(f"- output exact: `{summary['root_vs_two_exact']}`")
md.append(f"- first mismatch: `{summary['root_vs_two_first_mismatch']}`")
md.append("")
md.append("## Logit comparisons")
md.append("")
md.append("| step | same argmax | root argmax | two argmax | max abs diff | mean abs diff | cosine |")
md.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
for row in logit_comparisons:
    md.append(
        f"| {row['step']} | {row['same_argmax']} | {row['root_argmax']} | {row['two_argmax']} | "
        f"{row['max_abs_diff']:.8g} | {row['mean_abs_diff']:.8g} | {row['cosine']:.8g} |"
    )
md.append("")
md.append("## State trace summary")
md.append("")
md.append(f"- root max commit-source diff: `{summary['root_trace_stats']['max_commit_source_diff']}`")
md.append(f"- two-token max commit-source diff: `{summary['two_trace_stats']['max_commit_source_diff']}`")
md.append(f"- root events: `{summary['root_trace_stats']['events']}`")
md.append(f"- two-token events: `{summary['two_trace_stats']['events']}`")
md.append("")
md.append("## Interpretation rule")
md.append("")
md.append("- If any compared `max_abs_diff` is non-zero, the extra rejected token affects the root forward path.")
md.append("- If logits are identical but outputs diverge, inspect commit windows and SimpleGLA/KV state.")
(run_dir / "two_token_equiv_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(json.dumps({
    "run_dir": str(run_dir),
    "root_vs_two_exact": summary["root_vs_two_exact"],
    "first_mismatch": summary["root_vs_two_first_mismatch"],
    "logit_steps_compared": summary["logit_steps_compared"],
}, ensure_ascii=False))
PY
}

{
  print_run_header "$SESSION_NAME" "$RUN_DIR/two_token_equiv_probe.log" "$0 RUN_DIR=$RUN_DIR HEAD_PATH=$HEAD_PATH PORT=$PORT"
  echo "head_path=$HEAD_PATH"
  echo "prompt_text=$PROMPT_TEXT"
  echo "verify_logit_steps=$VERIFY_LOGIT_STEPS"

  CASE_EXTRA_ENV=(
    SGLANG_EAGLE_ROOT_ONLY_VERIFY=1
    SGLANG_FORCE_REJECT_EAGLE_DRAFT=1
  )
  run_case root_only_force_reject "${COMMON_ARGS[@]}"

  CASE_EXTRA_ENV=(
    SGLANG_FORCE_REJECT_EAGLE_DRAFT=1
  )
  run_case two_token_force_reject "${COMMON_ARGS[@]}"

  summarize
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_DIR/final_gpu.csv" || true
  echo "run_dir=$RUN_DIR"
  date -Is
} 2>&1 | tee "$RUN_DIR/two_token_equiv_probe.log"
