#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-rtx6000-2}"
ENV="${ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_eval_env}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-publichybrid-noattn-skipdown151617-v1}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30021}"
LOCAL_PROBE_DIR="${LOCAL_PROBE_DIR:-/Users/ql/cursor/openbmb/tmp/earlystop_prompt_probe}"
REMOTE_PROBE_DIR="${REMOTE_PROBE_DIR:-/root/autodl-tmp/codex-drafts/earlystop_prompt_probe}"
TMUX_SOCKET="${TMUX_SOCKET:-/root/autodl-tmp/tmux/codex-soar.sock}"
SUFFIX_TEXT="${SUFFIX_TEXT:-Important: provide only the final answer. No explanation, no reasoning, and stop immediately after the answer.}"

RUN_ID="promptclose_full_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="/root/autodl-tmp/SOAR-Toolkit/test_results/${RUN_ID}"
SESSION_NAME="${RUN_ID}"

echo "[launch] syncing probe package to ${REMOTE_HOST}:${REMOTE_PROBE_DIR}"
tar -C "${LOCAL_PROBE_DIR}" -cf - . | ssh "${REMOTE_HOST}" "rm -rf '${REMOTE_PROBE_DIR}' && mkdir -p '${REMOTE_PROBE_DIR}' && tar -C '${REMOTE_PROBE_DIR}' -xf -"

ssh "${REMOTE_HOST}" "mkdir -p '${RUN_DIR}' '/root/autodl-tmp/tmux'"

ssh "${REMOTE_HOST}" "cat > '${RUN_DIR}/launch_eval.sh' <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ENV='${ENV}'
MODEL_PATH='${MODEL_PATH}'
DATA_PATH='${DATA_PATH}'
RUN_DIR='${RUN_DIR}'
REMOTE_PROBE_DIR='${REMOTE_PROBE_DIR}'
HOST='${HOST}'
PORT='${PORT}'
export PYTHONUNBUFFERED=1
export PATH=\"\${ENV}/bin:\${PATH}\"
export LD_LIBRARY_PATH=\"/root/miniconda3/envs/py310/lib:\${LD_LIBRARY_PATH:-}\"
export PYTHONPATH=\"\${REMOTE_PROBE_DIR}/sglang/python:\${PYTHONPATH:-}\"
export SGLANG_APPEND_FINAL_USER_SUFFIX='${SUFFIX_TEXT}'

SERVER_LOG=\"\${RUN_DIR}/server.log\"
EVAL_LOG=\"\${RUN_DIR}/eval_model.log\"
LAUNCH_LOG=\"\${RUN_DIR}/launch.txt\"

echo \"[launch] start \$(date '+%F %T')\" | tee \"\${LAUNCH_LOG}\"
echo \"[launch] model=\${MODEL_PATH}\" | tee -a \"\${LAUNCH_LOG}\"
echo \"[launch] probe=\${REMOTE_PROBE_DIR}\" | tee -a \"\${LAUNCH_LOG}\"
echo \"[launch] suffix=\${SGLANG_APPEND_FINAL_USER_SUFFIX}\" | tee -a \"\${LAUNCH_LOG}\"

pkill -f \"sglang.launch_server --host \${HOST} --port \${PORT}\" >/dev/null 2>&1 || true
pkill -f \"eval_model.py --api_base http://\${HOST}:\${PORT}\" >/dev/null 2>&1 || true

stdbuf -oL -eL \"\${ENV}/bin/python\" -m sglang.launch_server \
  --model-path \"\${MODEL_PATH}\" \
  --quantization gptq_marlin \
  --dtype float16 \
  --trust-remote-code \
  --host \"\${HOST}\" \
  --port \"\${PORT}\" \
  --attention-backend flashinfer \
  --force-dense-minicpm \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --skip-server-warmup \
  --enable-request-time-stats-logging \
  > \"\${SERVER_LOG}\" 2>&1 &
SERVER_PID=\$!
echo \$SERVER_PID > \"\${RUN_DIR}/server.pid\"

READY=0
for _ in \$(seq 1 120); do
  if curl -sf \"http://\${HOST}:\${PORT}/v1/models\" >/dev/null; then
    READY=1
    break
  fi
  sleep 2
done

if [[ \"\${READY}\" -ne 1 ]]; then
  echo \"[launch] server failed to become ready\" | tee -a \"\${LAUNCH_LOG}\"
  tail -n 80 \"\${SERVER_LOG}\" | tee -a \"\${LAUNCH_LOG}\"
  exit 1
fi

echo \"[launch] server ready \$(date '+%F %T')\" | tee -a \"\${LAUNCH_LOG}\"

set +e
stdbuf -oL -eL \"\${ENV}/bin/python\" /root/autodl-tmp/SOAR-Toolkit/eval_model.py \
  --model_path \"\${MODEL_PATH}\" \
  --api_base \"http://\${HOST}:\${PORT}\" \
  --data_path \"\${DATA_PATH}\" \
  --max_seq_len 262144 \
  --concurrency 8 \
  --verbose 2>&1 | tee \"\${EVAL_LOG}\"
EVAL_EXIT=\${PIPESTATUS[0]}
set -e

echo \"[launch] eval exit=\${EVAL_EXIT} \$(date '+%F %T')\" | tee -a \"\${LAUNCH_LOG}\"
exit \"\${EVAL_EXIT}\"
SH
chmod +x '${RUN_DIR}/launch_eval.sh'"

ssh "${REMOTE_HOST}" "tmux -S '${TMUX_SOCKET}' kill-session -t '${SESSION_NAME}' >/dev/null 2>&1 || true; tmux -S '${TMUX_SOCKET}' new-session -d -s '${SESSION_NAME}' bash '${RUN_DIR}/launch_eval.sh'; printf '%s\n%s\n' '${RUN_DIR}' '${SESSION_NAME}'"

echo "[launch] run_dir=${RUN_DIR}"
echo "[launch] tmux_session=${SESSION_NAME}"
echo "[launch] monitor: ssh ${REMOTE_HOST} 'tail -f ${RUN_DIR}/eval_model.log'"
