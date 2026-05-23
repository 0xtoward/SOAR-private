#!/usr/bin/env bash
set -euo pipefail

# Shared defaults for MiniCPM-SALA EAGLE3 training experiments.
# These scripts are intended to run on the remote GPU host, usually rtx6000-1.

export SALA_EAGLE3_ROOT=${SALA_EAGLE3_ROOT:-/root/autodl-tmp/codex-drafts/sala-eagle3-train-v2}
export MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models}
export SGLANG_REPO=${SGLANG_REPO:-/root/autodl-tmp/codex-drafts/sglang-sala-tree-verify}
export SGLANG_SRC=${SGLANG_SRC:-$SGLANG_REPO/python}
export LEGACY_EAGLE3_ROOT=${LEGACY_EAGLE3_ROOT:-/root/autodl-tmp/codex-drafts/sala-eagle3-smoke}
export SPECFORGE_SRC=${SPECFORGE_SRC:-$SALA_EAGLE3_ROOT/SpecForge}
export OLD_EAGLE3_HEAD=${OLD_EAGLE3_HEAD:-$LEGACY_EAGLE3_ROOT/outputs/eagle3_real_head_20260504_224509/epoch_7_step_400}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}
export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

if [[ -f /root/autodl-tmp/use_soar_uv.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/use_soar_uv.sh
fi

mkdir -p "$SALA_EAGLE3_ROOT"/{logs,reports,data,configs,outputs,cache}

if [[ ! -d "$SPECFORGE_SRC" && -d "$LEGACY_EAGLE3_ROOT/SpecForge" ]]; then
  ln -s "$LEGACY_EAGLE3_ROOT/SpecForge" "$SPECFORGE_SRC"
fi

export PYTHONPATH="$SPECFORGE_SRC:$SGLANG_SRC:${PYTHONPATH:-}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

print_run_header() {
  local session_name="$1"
  local log_path="$2"
  local cmd_text="$3"
  echo "host=$(hostname)"
  echo "cwd=$(pwd)"
  echo "tmux_session=$session_name"
  echo "log_path=$log_path"
  echo "cmd=$cmd_text"
  date -Is
}
