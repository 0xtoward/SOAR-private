#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-root@connect.bjb1.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-14968}"
REMOTE_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts -p "${REMOTE_PORT}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/autodl-tmp/codex-drafts/eagle-spec-minimal}"
REMOTE_STREAM_EVAL="${REMOTE_SCRIPT_DIR}/stream_eval_model.py"

ssh "${REMOTE_OPTS[@]}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_SCRIPT_DIR}'"
scp -P "${REMOTE_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/codex_known_hosts \
  "${SCRIPT_DIR}/stream_eval_model.py" \
  "${REMOTE_HOST}:${REMOTE_STREAM_EVAL}"

REMOTE_ENV_PREFIX=""
for key in \
  TOOLKIT PKG MODEL_PATH KV_SCALE_JSON DATA_PATH DRAFT_PATH PORT MAX_OUT_LEN \
  CONCURRENCY NUM_SAMPLES SPEC_STEPS SPEC_TOPK SPEC_DRAFT_TOKENS \
  SPEC_ATTENTION_MODE ATTENTION_BACKEND WAIT_FOR_SESSION RUN_ID SESSION
do
  if [[ -n "${!key+x}" ]]; then
    printf -v quoted "%q" "${!key}"
    REMOTE_ENV_PREFIX+=" ${key}=${quoted}"
  fi
done
printf -v remote_stream_quoted "%q" "${REMOTE_STREAM_EVAL}"
REMOTE_ENV_PREFIX+=" STREAM_EVAL=${remote_stream_quoted}"

ssh "${REMOTE_OPTS[@]}" "${REMOTE_HOST}" "${REMOTE_ENV_PREFIX} bash -s" <<'REMOTE'
set -euo pipefail

source /etc/network_turbo >/dev/null 2>&1 || true
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SALA_SKIP_TARGET_VERIFY_TEMPORAL_WRITE="${SGLANG_SALA_SKIP_TARGET_VERIFY_TEMPORAL_WRITE:-1}"

TOOLKIT="${TOOLKIT:-/root/autodl-tmp/SOAR-Toolkit}"
PKG="${PKG:-/root/autodl-tmp/codex-drafts/submission-w4a16-marlin-attnqkv-allowlist-gate0fixorder-cgbs1-fp8-kvscale-mambabf16-v1}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-attnqkv-allowlist-v1-gate0-fixorder}"
KV_SCALE_JSON="${KV_SCALE_JSON:-/root/autodl-tmp/SOAR-Toolkit/test_results/kvscale_input_full_20260421_160936/kv_scales.json}"
DATA_PATH="${DATA_PATH:-${TOOLKIT}/eval_dataset/perf_public_set.jsonl}"
DRAFT_PATH="${DRAFT_PATH:-/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2/outputs/main2000_full_vocab_lr5e5_ttt5/epoch_0_step_1600}"
STREAM_EVAL="${STREAM_EVAL:-/root/autodl-tmp/codex-drafts/eagle-spec-minimal/stream_eval_model.py}"
PORT="${PORT:-30058}"
HOST="127.0.0.1"
MAX_OUT_LEN="${MAX_OUT_LEN:-32768}"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-4}"
SPEC_ATTENTION_MODE="${SPEC_ATTENTION_MODE:-decode}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"
RUN_ID="${RUN_ID:-winning99_eagle3_stream_s${SPEC_STEPS}_k${SPEC_TOPK}_d${SPEC_DRAFT_TOKENS}_$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="${TOOLKIT}/test_results/${RUN_ID}"
SESSION="${SESSION:-codex_winning99_eagle3_stream_eval}"
LAUNCHER="${RESULT_DIR}/launcher.sh"
LAUNCHER_LOG="${RESULT_DIR}/launcher.log"

mkdir -p "${RESULT_DIR}"

cat > "${LAUNCHER}" <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

source /etc/network_turbo >/dev/null 2>&1 || true
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_SALA_SKIP_TARGET_VERIFY_TEMPORAL_WRITE="${SGLANG_SALA_SKIP_TARGET_VERIFY_TEMPORAL_WRITE:-1}"

TOOLKIT="__TOOLKIT__"
PKG="__PKG__"
MODEL_PATH="__MODEL_PATH__"
KV_SCALE_JSON="__KV_SCALE_JSON__"
DATA_PATH="__DATA_PATH__"
DRAFT_PATH="__DRAFT_PATH__"
STREAM_EVAL="__STREAM_EVAL__"
HOST="127.0.0.1"
PORT="__PORT__"
MAX_OUT_LEN="__MAX_OUT_LEN__"
CONCURRENCY="__CONCURRENCY__"
NUM_SAMPLES="__NUM_SAMPLES__"
SPEC_STEPS="__SPEC_STEPS__"
SPEC_TOPK="__SPEC_TOPK__"
SPEC_DRAFT_TOKENS="__SPEC_DRAFT_TOKENS__"
SPEC_ATTENTION_MODE="__SPEC_ATTENTION_MODE__"
ATTENTION_BACKEND="__ATTENTION_BACKEND__"
WAIT_FOR_SESSION="__WAIT_FOR_SESSION__"
RESULT_DIR="__RESULT_DIR__"
SERVER_LOG="${RESULT_DIR}/server.log"
EVAL_LOG="${RESULT_DIR}/stream_eval.log"
MONITOR_LOG="${RESULT_DIR}/monitor.log"
CONTEXT_JSON="${RESULT_DIR}/launch_context.json"
MODELS_JSON="${RESULT_DIR}/models.json"
EVAL_OUTPUT_DIR="${RESULT_DIR}/outputs"

mkdir -p "${RESULT_DIR}" "${EVAL_OUTPUT_DIR}"
cd "${TOOLKIT}"

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[launch] waiting for tmux session to finish: ${WAIT_FOR_SESSION}" | tee -a "${EVAL_LOG}"
  while tmux has-session -t "${WAIT_FOR_SESSION}" >/dev/null 2>&1; do
    sleep 30
  done
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" >/dev/null 2>&1 || true
    wait "${MONITOR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

{
  echo "[launch] host=$(hostname)"
  echo "[launch] cwd=${TOOLKIT}"
  echo "[launch] result_dir=${RESULT_DIR}"
  echo "[launch] package=${PKG}"
  echo "[launch] model=${MODEL_PATH}"
  echo "[launch] draft=${DRAFT_PATH}"
  echo "[launch] data=${DATA_PATH}"
  echo "[launch] max_out_len=${MAX_OUT_LEN}"
  echo "[launch] num_samples=${NUM_SAMPLES:-all}"
  echo "[launch] port=${PORT}"
  echo "[launch] concurrency=${CONCURRENCY}"
  echo "[launch] attention_backend=${ATTENTION_BACKEND}"
  echo "[launch] speculative=EAGLE3 steps=${SPEC_STEPS} topk=${SPEC_TOPK} draft_tokens=${SPEC_DRAFT_TOKENS} attention_mode=${SPEC_ATTENTION_MODE}"
  echo "[launch] start=$(date '+%F %T')"
} | tee "${EVAL_LOG}"

bash "${PKG}/run_attnqkv_server.sh" \
  --context-file "${CONTEXT_JSON}" \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size 1 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --force-dense-minicpm \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --cuda-graph-bs 1 \
  --skip-server-warmup \
  --quantization gptq_marlin \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --quantization-param-path "${KV_SCALE_JSON}" \
  --mamba-ssm-dtype bfloat16 \
  --enable-request-time-stats-logging \
  --decode-log-interval 1 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "${DRAFT_PATH}" \
  --speculative-draft-model-quantization unquant \
  --speculative-attention-mode "${SPEC_ATTENTION_MODE}" \
  --speculative-num-steps "${SPEC_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

SERVER_READY=0
for _ in $(seq 1 240); do
  if curl -sf "http://${HOST}:${PORT}/v1/models" > "${MODELS_JSON}"; then
    SERVER_READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[launch] server process exited before ready" | tee -a "${EVAL_LOG}"
    tail -n 200 "${SERVER_LOG}" | tee -a "${EVAL_LOG}" || true
    exit 1
  fi
  sleep 2
done

if [[ "${SERVER_READY}" -ne 1 ]]; then
  echo "[launch] server failed to become ready" | tee -a "${EVAL_LOG}"
  tail -n 240 "${SERVER_LOG}" | tee -a "${EVAL_LOG}" || true
  exit 1
fi

echo "[launch] server ready $(date '+%F %T')" | tee -a "${EVAL_LOG}"

(
  while true; do
    req_done="$(grep -c "Req Time Stats" "${SERVER_LOG}" 2>/dev/null || true)"
    cap_hits="$(grep -c "output len=${MAX_OUT_LEN}" "${SERVER_LOG}" 2>/dev/null || true)"
    latest_tps="$(grep "gen throughput" "${SERVER_LOG}" 2>/dev/null | tail -n 1 || true)"
    latest_accept="$(grep "accept len:" "${SERVER_LOG}" 2>/dev/null | tail -n 1 || true)"
    latest_acc="$(python3 - "${EVAL_OUTPUT_DIR}/summary.stream.json" <<'PY' 2>/dev/null || true
import json, sys
try:
    data=json.load(open(sys.argv[1]))
    print(f"done={data.get('completed')}/{data.get('num_samples')} acc_done={data.get('partial_accuracy_done')} acc_total={data.get('partial_accuracy_total')} tps={data.get('stream_tps')}")
except Exception:
    pass
PY
)"
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    echo "[monitor] $(date '+%F %T') req_done=${req_done} cap_hits=${cap_hits} gpu=${gpu} latest_acc='${latest_acc}' latest_tps='${latest_tps}' latest_accept='${latest_accept}'" | tee -a "${MONITOR_LOG}"
    sleep 30
  done
) &
MONITOR_PID=$!

NUM_ARGS=()
if [[ -n "${NUM_SAMPLES}" ]]; then
  NUM_ARGS=(--num_samples "${NUM_SAMPLES}")
fi

MAX_PROMPT_LEN=128000 \
stdbuf -oL -eL "${PKG}/runtime_envs/eval_py310_env/bin/python" "${STREAM_EVAL}" \
  --toolkit_dir "${TOOLKIT}" \
  --model_path "${MODEL_PATH}" \
  --api_base "http://${HOST}:${PORT}" \
  --data_path "${DATA_PATH}" \
  --max_seq_len 262144 \
  --max_out_len "${MAX_OUT_LEN}" \
  --concurrency "${CONCURRENCY}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  "${NUM_ARGS[@]}" \
  2>&1 | tee -a "${EVAL_LOG}"

python3 - "${SERVER_LOG}" "${RESULT_DIR}/accept_stats.json" <<'PY' || true
import json, re, statistics, sys
server_log, out = sys.argv[1:3]
accept_lens = []
accept_rates = []
tps = []
for line in open(server_log, errors="ignore"):
    m = re.search(r"accept len: ([0-9.]+), accept rate: ([0-9.]+)", line)
    if m:
        accept_lens.append(float(m.group(1)))
        accept_rates.append(float(m.group(2)))
    m = re.search(r"gen throughput \(token/s\): ([0-9.]+)", line)
    if m:
        tps.append(float(m.group(1)))
payload = {
    "accept_len_count": len(accept_lens),
    "accept_len_avg": statistics.mean(accept_lens) if accept_lens else None,
    "accept_len_max": max(accept_lens) if accept_lens else None,
    "accept_rate_avg": statistics.mean(accept_rates) if accept_rates else None,
    "accept_rate_max": max(accept_rates) if accept_rates else None,
    "decode_tps_avg_last_100": statistics.mean(tps[-100:]) if tps else None,
    "decode_tps_last": tps[-1] if tps else None,
}
open(out, "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[launch] done $(date '+%F %T')" | tee -a "${EVAL_LOG}"
SCRIPT

python3 - "${LAUNCHER}" \
  "__TOOLKIT__=${TOOLKIT}" \
  "__PKG__=${PKG}" \
  "__MODEL_PATH__=${MODEL_PATH}" \
  "__KV_SCALE_JSON__=${KV_SCALE_JSON}" \
  "__DATA_PATH__=${DATA_PATH}" \
  "__DRAFT_PATH__=${DRAFT_PATH}" \
  "__STREAM_EVAL__=${STREAM_EVAL}" \
  "__PORT__=${PORT}" \
  "__MAX_OUT_LEN__=${MAX_OUT_LEN}" \
  "__CONCURRENCY__=${CONCURRENCY}" \
  "__NUM_SAMPLES__=${NUM_SAMPLES}" \
  "__SPEC_STEPS__=${SPEC_STEPS}" \
  "__SPEC_TOPK__=${SPEC_TOPK}" \
  "__SPEC_DRAFT_TOKENS__=${SPEC_DRAFT_TOKENS}" \
  "__SPEC_ATTENTION_MODE__=${SPEC_ATTENTION_MODE}" \
  "__ATTENTION_BACKEND__=${ATTENTION_BACKEND}" \
  "__WAIT_FOR_SESSION__=${WAIT_FOR_SESSION}" \
  "__RESULT_DIR__=${RESULT_DIR}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    text = text.replace(key, value)
path.write_text(text, encoding="utf-8")
PY
chmod +x "${LAUNCHER}"

tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
tmux new-session -d -s "${SESSION}" "bash '${LAUNCHER}' > '${LAUNCHER_LOG}' 2>&1"

cat <<EOF
HOST=$(hostname)
CWD=${TOOLKIT}
SESSION=${SESSION}
RESULT_DIR=${RESULT_DIR}
LAUNCHER=${LAUNCHER}
LAUNCHER_LOG=${LAUNCHER_LOG}
EVAL_LOG=${RESULT_DIR}/stream_eval.log
SERVER_LOG=${RESULT_DIR}/server.log
MONITOR_LOG=${RESULT_DIR}/monitor.log
COMMAND=tmux new-session -d -s ${SESSION} "bash '${LAUNCHER}' > '${LAUNCHER_LOG}' 2>&1"
SPEC=EAGLE3 steps=${SPEC_STEPS} topk=${SPEC_TOPK} draft_tokens=${SPEC_DRAFT_TOKENS} attention_mode=${SPEC_ATTENTION_MODE} draft=${DRAFT_PATH}
STREAM_OUTPUT=${RESULT_DIR}/outputs/predictions.stream.jsonl
STREAM_SUMMARY=${RESULT_DIR}/outputs/summary.stream.json
WAIT_FOR_SESSION=${WAIT_FOR_SESSION:-none}
EOF
REMOTE
