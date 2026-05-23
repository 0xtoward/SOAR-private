#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common_env.sh"

ABLATION_ROOT=${ABLATION_ROOT:-$SALA_EAGLE3_ROOT/eval/eagle_batch_invariant_op_ablation_$(timestamp)}
CONFIG_MATRIX=${CONFIG_MATRIX:-"s2_topk4_draft9:2:4:9"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
BASE_PORT=${BASE_PORT:-30640}
PROMPTS_JSONL=${PROMPTS_JSONL:-$ABLATION_ROOT/prompts_mismatch3.jsonl}

mkdir -p "$ABLATION_ROOT"

if [[ ! -s "$PROMPTS_JSONL" ]]; then
  python - "$PROMPTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
prompts = [
    {
        "id": "natural_en",
        "text": "Human: Please explain speculative decoding in two concise sentences.\n\nAssistant:",
    },
    {
        "id": "natural_zh",
        "text": "Human: \u7528\u4e24\u53e5\u8bdd\u89e3\u91ca\u4ec0\u4e48\u662f\u6295\u673a\u89e3\u7801\uff0c\u4e0d\u8981\u5217\u8868\u3002\n\nAssistant:",
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

run_variant() {
  local label="$1"
  local port="$2"
  shift 2

  local run_dir="$ABLATION_ROOT/$label"
  mkdir -p "$run_dir"
  echo "===== $label $(date -Iseconds) =====" | tee -a "$ABLATION_ROOT/ablation.log"

  (
    export RUN_DIR="$run_dir"
    export PORT="$port"
    export PROMPTS_JSONL="$PROMPTS_JSONL"
    export CONFIG_MATRIX="$CONFIG_MATRIX"
    export MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
    export DETERMINISTIC=0
    export RUN_FORCE_REJECT_CASES=1
    export SPECULATIVE_ATTENTION_MODE=prefill
    export SGLANG_TARGET_VERIFY_TREE_USE_TRITON_DECODE_KERNEL=true
    export SGLANG_TARGET_VERIFY_TREE_USE_TRITON_EXTEND_KERNEL=false
    export SGLANG_FORCE_ENABLE_BATCH_INVARIANT_OPS=1
    export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
    export SGLANG_BATCH_INVARIANT_OPS_REGISTER_MM=1
    export SGLANG_BATCH_INVARIANT_OPS_REGISTER_ADDMM=1
    export SGLANG_BATCH_INVARIANT_OPS_REGISTER_BMM=1
    export SGLANG_BATCH_INVARIANT_OPS_REGISTER_LOG_SOFTMAX=1
    export SGLANG_BATCH_INVARIANT_OPS_REGISTER_MEAN=1
    export SESSION_NAME="sala_eagle3_bio_${label}"
    local assignment
    for assignment in "$@"; do
      export "$assignment"
    done
    "$SCRIPT_DIR/trace_eagle_deterministic_tree_sweep.sh"
    python "$SCRIPT_DIR/light_summarize_eagle_run.py" --run-dir "$run_dir"
  ) 2>&1 | tee "$run_dir/run.log"
}

run_variant all_on "$BASE_PORT"
run_variant no_mm "$((BASE_PORT + 1))" SGLANG_BATCH_INVARIANT_OPS_REGISTER_MM=0
run_variant no_addmm "$((BASE_PORT + 2))" SGLANG_BATCH_INVARIANT_OPS_REGISTER_ADDMM=0
run_variant no_bmm "$((BASE_PORT + 3))" SGLANG_BATCH_INVARIANT_OPS_REGISTER_BMM=0
run_variant no_log_softmax "$((BASE_PORT + 4))" SGLANG_BATCH_INVARIANT_OPS_REGISTER_LOG_SOFTMAX=0
run_variant no_mean "$((BASE_PORT + 5))" SGLANG_BATCH_INVARIANT_OPS_REGISTER_MEAN=0

python - "$ABLATION_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/eagle_forcebio_fullmatrix_light_summary.json")):
    label = path.parent.name
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data:
        rows.append(
            {
                "variant": label,
                "case": row["case"],
                "exact": row["exact"],
                "exact_count": row["exact_count"],
                "total": row["total"],
                "mismatch_count": len(row["mismatches"]),
                "avg_accept_len": row.get("avg_accept_len_log"),
                "avg_accept_rate": row.get("avg_accept_rate_log"),
                "avg_tps": row.get("avg_tps_log"),
                "mismatches": row["mismatches"],
            }
        )

(root / "ablation_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)

lines = [
    "# Batch-Invariant Operator Ablation",
    "",
    f"root: `{root}`",
    "",
    "| variant | case | exact | mismatch count | avg accept len | avg accept rate | avg TPS |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for row in rows:
    def fmt(value):
        return "" if value is None else f"{value:.2f}"
    lines.append(
        "| `{variant}` | `{case}` | {exact_count}/{total} | {mismatch_count} | {avg_accept_len} | {avg_accept_rate} | {avg_tps} |".format(
            variant=row["variant"],
            case=row["case"],
            exact_count=row["exact_count"],
            total=row["total"],
            mismatch_count=row["mismatch_count"],
            avg_accept_len=fmt(row["avg_accept_len"]),
            avg_accept_rate=fmt(row["avg_accept_rate"]),
            avg_tps=fmt(row["avg_tps"]),
        )
    )

bad = [row for row in rows if row["mismatches"]]
if bad:
    lines += ["", "## First Mismatches"]
    for row in bad:
        for mismatch in row["mismatches"]:
            lines.append(
                f"- `{row['variant']}` / `{row['case']}` / `{mismatch['prompt']}`: "
                f"`{json.dumps(mismatch['mismatch'], ensure_ascii=False)}`"
            )
else:
    lines += ["", "All ablation rows were exact."]

(root / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(root / "ablation_summary.md")
PY

cp "$ABLATION_ROOT/ablation_summary.md" "$SALA_EAGLE3_ROOT/reports/batch_invariant_op_ablation.md"
echo "summary=$ABLATION_ROOT/ablation_summary.md"
