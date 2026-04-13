#!/usr/bin/env bash
set -euo pipefail

ENV="${ENV:-/root/autodl-tmp/sglang/sglang_submitmatch_uv_py310_gptq_env}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models}"
CALIB_PATH="${CALIB_PATH:-/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_pg19_v2.jsonl}"
OUT_PATH="${OUT_PATH:-/root/autodl-tmp/models-gptq-w4a16-py310-quantcheck}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
DTYPE="${DTYPE:-float16}"
SCRIPT_SRC="${SCRIPT_SRC:-/root/autodl-tmp/codex-drafts/quantize_gptq_w4a16.py}"
LOG_PATH="${LOG_PATH:-/root/autodl-tmp/SOAR-Toolkit/test_results/gptq_py310_quantcheck.log}"
DYNAMIC_CONFIG_PATH="${DYNAMIC_CONFIG_PATH:-}"
QUANT_EXTRA_ARGS="${QUANT_EXTRA_ARGS:-}"
OFFLOAD_TO_DISK_PATH="${OFFLOAD_TO_DISK_PATH:-}"

export LD_LIBRARY_PATH="/root/miniconda3/envs/py310/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1

mkdir -p "$(dirname "${LOG_PATH}")"

echo "[quantcheck] start $(date '+%F %T')" | tee "${LOG_PATH}"
echo "[quantcheck] host=$(hostname) shell_pid=$$ pwd=$(pwd)" | tee -a "${LOG_PATH}"
echo "[quantcheck] env=${ENV}" | tee -a "${LOG_PATH}"
echo "[quantcheck] model=${MODEL_PATH}" | tee -a "${LOG_PATH}"
echo "[quantcheck] calib=${CALIB_PATH}" | tee -a "${LOG_PATH}"
echo "[quantcheck] out=${OUT_PATH}" | tee -a "${LOG_PATH}"
echo "[quantcheck] samples=${NUM_SAMPLES} dtype=${DTYPE}" | tee -a "${LOG_PATH}"
echo "[quantcheck] dynamic_config=${DYNAMIC_CONFIG_PATH:-<unset>}" | tee -a "${LOG_PATH}"
echo "[quantcheck] offload_to_disk_path=${OFFLOAD_TO_DISK_PATH:-<unset>}" | tee -a "${LOG_PATH}"
echo "[quantcheck] extra_args=${QUANT_EXTRA_ARGS:-<unset>}" | tee -a "${LOG_PATH}"
echo "[quantcheck] PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER} TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES}" | tee -a "${LOG_PATH}"
ulimit -c unlimited >/dev/null 2>&1 || true
echo "[quantcheck] core_ulimit=$(ulimit -c 2>/dev/null || echo unavailable)" | tee -a "${LOG_PATH}"

rm -rf "${OUT_PATH}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --calibration-path "${CALIB_PATH}"
  --output-path "${OUT_PATH}"
  --num-samples "${NUM_SAMPLES}"
  --dtype "${DTYPE}"
)

if [[ -n "${DYNAMIC_CONFIG_PATH}" ]]; then
  ARGS+=(--dynamic-config-path "${DYNAMIC_CONFIG_PATH}")
fi

if [[ -n "${OFFLOAD_TO_DISK_PATH}" ]]; then
  ARGS+=(--offload-to-disk-path "${OFFLOAD_TO_DISK_PATH}")
fi

if [[ -n "${QUANT_EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS_ARRAY <<< "${QUANT_EXTRA_ARGS}"
  ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

heartbeat() {
  local py_pid="$1"
  while kill -0 "${py_pid}" >/dev/null 2>&1; do
    local gpu_line="unavailable"
    local rss_line="unavailable"
    local cgroup_line=""
    local cgroup_current=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_line="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
      gpu_line="${gpu_line%;}"
    fi
    rss_line="$(ps -p "${py_pid}" -o pid=,etime=,%cpu=,%mem=,rss=,vsz=,stat= 2>/dev/null | xargs || true)"
    if [[ -f /sys/fs/cgroup/memory.events ]]; then
      cgroup_line="$(tr '\n' ';' </sys/fs/cgroup/memory.events | sed 's/;$/ /')"
    fi
    if [[ -f /sys/fs/cgroup/memory.current ]]; then
      cgroup_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || true)"
    fi
    echo "[quantcheck] heartbeat ts=$(date '+%F %T') gpu=[${gpu_line}] proc=[${rss_line}] cgroup=[${cgroup_line}] memory_current_bytes=${cgroup_current}" | tee -a "${LOG_PATH}"
    sleep 60
  done
}

set +e
"${ENV}/bin/python" -X faulthandler "${SCRIPT_SRC}" \
  "${ARGS[@]}" > >(tee -a "${LOG_PATH}") 2> >(tee -a "${LOG_PATH}" >&2) &
PY_PID=$!
echo "[quantcheck] python_pid=${PY_PID}" | tee -a "${LOG_PATH}"

heartbeat "${PY_PID}" &
HEARTBEAT_PID=$!

wait "${PY_PID}"
STATUS=$?

kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
wait "${HEARTBEAT_PID}" 2>/dev/null || true
set -e

echo "[quantcheck] python_exit_code=${STATUS}" | tee -a "${LOG_PATH}"
if (( STATUS != 0 )); then
  if (( STATUS > 128 )); then
    echo "[quantcheck] python_exit_signal=$((STATUS - 128))" | tee -a "${LOG_PATH}"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[quantcheck] final_nvidia_smi=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' | sed 's/;$/ /')" | tee -a "${LOG_PATH}"
  fi
  if [[ -f /sys/fs/cgroup/memory.events ]]; then
    echo "[quantcheck] final_cgroup_memory_events=$(tr '\n' ';' </sys/fs/cgroup/memory.events | sed 's/;$/ /')" | tee -a "${LOG_PATH}"
  fi
  if [[ -f /sys/fs/cgroup/memory.current ]]; then
    echo "[quantcheck] final_cgroup_memory_current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || true)" | tee -a "${LOG_PATH}"
  fi
  exit "${STATUS}"
fi

echo "[quantcheck] done $(date '+%F %T')" | tee -a "${LOG_PATH}"
