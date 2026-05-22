#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH=${DATA_PATH:-/root/autodl-tmp/SOAR-Toolkit/test_results/winning99_eagle3_topk2_nograph_smoke_nocap_after_full_20260512_152925/no_cap_smoke_data.jsonl}
ROW_ORDINAL=${ROW_ORDINAL:-5}

PROMPT_TEXT="$(
  DATA_PATH="$DATA_PATH" ROW_ORDINAL="$ROW_ORDINAL" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DATA_PATH"])
row_ordinal = int(os.environ["ROW_ORDINAL"])
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(rows[row_ordinal]["question"])
PY
)"

VERIFY_LOGIT_STEPS="$(
  python3 - <<'PY'
print(",".join(map(str, range(0, 128))))
PY
)"

export SGLANG_SALA_SKIP_TARGET_VERIFY_TEMPORAL_WRITE=1
if [[ "${FORCE_NONDETERMINISTIC:-0}" == "1" ]]; then
  unset SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM || true
  unset SGLANG_ENABLE_DETERMINISTIC_INFERENCE || true
  unset CUBLAS_WORKSPACE_CONFIG || true
fi

cd /root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/scripts
RUN_DIR="$RUN_DIR" \
PORT="${PORT:-30228}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}" \
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}" \
HEAD_PATH="${HEAD_PATH:-/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/outputs/main2000_hot32k_lr5e5_ttt5/epoch_0_step_800}" \
PROMPT_TEXT="$PROMPT_TEXT" \
VERIFY_LOGIT_STEPS="$VERIFY_LOGIT_STEPS" \
SPECULATIVE_ATTENTION_MODE="${SPECULATIVE_ATTENTION_MODE:-decode}" \
bash ./trace_eagle_two_token_equiv_probe.sh
