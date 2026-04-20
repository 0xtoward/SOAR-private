#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-rtx6000-2}"
ENV="${ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-v1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30025}"
LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-/Users/ql/cursor/openbmb/tmp/perf_public_medium_overgen8_eval_v1.jsonl}"
LOCAL_SCRIPT_PATH="${LOCAL_SCRIPT_PATH:-/Users/ql/cursor/openbmb/scripts/overgen_stop_audit.py}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/root/autodl-tmp/codex-drafts/overgen_stop_audit}"
RUN_ID="${RUN_ID:-overgen_stop_audit_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/SOAR-Toolkit/test_results/${RUN_ID}}"

echo "[audit] syncing script+dataset to ${REMOTE_HOST}:${REMOTE_WORKDIR}"
tar -C /Users/ql/cursor/openbmb -cf - \
  "scripts/$(basename "${LOCAL_SCRIPT_PATH}")" \
  "tmp/$(basename "${LOCAL_DATA_PATH}")" | \
  ssh "${REMOTE_HOST}" "rm -rf '${REMOTE_WORKDIR}' && mkdir -p '${REMOTE_WORKDIR}' '${RUN_DIR}' && tar -C '${REMOTE_WORKDIR}' -xf -"

ssh "${REMOTE_HOST}" "bash -lc '
set -euo pipefail
export PATH=\"${ENV}/bin:\${PATH}\"
export LD_LIBRARY_PATH=\"/root/miniconda3/envs/py310/lib:\${LD_LIBRARY_PATH:-}\"

SERVER_LOG=\"${RUN_DIR}/server.log\"
AUDIT_LOG=\"${RUN_DIR}/audit.log\"
AUDIT_JSON=\"${RUN_DIR}/overgen_stop_audit.json\"

pkill -f \"sglang.launch_server --host ${HOST} --port ${PORT}\" >/dev/null 2>&1 || true

\"${ENV}/bin/python\" -m sglang.launch_server \
  --model-path \"${MODEL_PATH}\" \
  --quantization gptq_marlin \
  --dtype float16 \
  --trust-remote-code \
  --host \"${HOST}\" \
  --port \"${PORT}\" \
  --attention-backend flashinfer \
  --force-dense-minicpm \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --skip-server-warmup \
  --enable-request-time-stats-logging \
  > \"${RUN_DIR}/server.log\" 2>&1 &
SERVER_PID=\$!
trap \"kill \${SERVER_PID} >/dev/null 2>&1 || true\" EXIT

READY=0
for _ in \$(seq 1 120); do
  if curl -sf \"http://${HOST}:${PORT}/v1/models\" >/dev/null; then
    READY=1
    break
  fi
  sleep 2
done

if [[ \"\${READY}\" -ne 1 ]]; then
  echo \"[audit] server failed to become ready\" >&2
  tail -n 120 \"${RUN_DIR}/server.log\" >&2
  exit 1
fi

\"${ENV}/bin/python\" \"${REMOTE_WORKDIR}/scripts/$(basename "${LOCAL_SCRIPT_PATH}")\" \
  --api-base \"http://${HOST}:${PORT}\" \
  --model-path \"${MODEL_PATH}\" \
  --data-path \"${REMOTE_WORKDIR}/tmp/$(basename "${LOCAL_DATA_PATH}")\" \
  --output-path \"${RUN_DIR}/overgen_stop_audit.json\" \
  2>&1 | tee \"${RUN_DIR}/audit.log\"

python3 - <<PY
import json
from pathlib import Path
p = Path(\"${RUN_DIR}/overgen_stop_audit.json\")
data = json.loads(p.read_text(encoding=\"utf-8\"))
print(\"[summary] finish_reason_counts=\", data[\"finish_reason_counts\"])
print(\"[summary] total_output_tokens=\", data[\"total_output_tokens\"])
print(\"[summary] avg_output_tokens=\", round(data[\"avg_output_tokens\"], 2))
print(\"[summary] avg_wall_s=\", round(data[\"avg_wall_s\"], 2))
PY
'"

echo "[audit] run_dir=${RUN_DIR}"
