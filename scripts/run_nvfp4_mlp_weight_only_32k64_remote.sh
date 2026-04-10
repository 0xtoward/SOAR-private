#!/usr/bin/env bash
set -euo pipefail

REMOTE_ARGS=(
  -p 50884
  -i /Users/ql/.ssh/id_ed25519
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  root@106.120.183.117
)

ssh "${REMOTE_ARGS[@]}" 'bash -lc "
set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true
export HF_ENDPOINT=https://hf-mirror.com
source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1
mkdir -p /root/autodl-tmp/SOAR-Toolkit/test_results
tmux kill-window -t codex-soar:nvfp4-w32k64 >/dev/null 2>&1 || true
tmux new-window -d -t codex-soar -n nvfp4-w32k64 \"bash -lc \\\"source /etc/network_turbo >/dev/null 2>&1 || true; export HF_ENDPOINT=https://hf-mirror.com; source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1; python3 /root/autodl-tmp/SOAR-Toolkit/submissions/nvfp4-mlp-only-draft/quantize_modelopt_nvfp4_conservative.py --model-path /root/autodl-tmp/models --export-dir /root/autodl-tmp/models-nvfp4-mlp-weight-only-curated32k64-public-v1 --cfg mlp_weight_only --dtype bfloat16 --local-calib-files /root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_32k_public_v1.jsonl --local-calib-samples 64 --max-length 32768 --min-good-batches 48 --overwrite > /root/autodl-tmp/SOAR-Toolkit/test_results/nvfp4_mlp_weight_only_32k64_public_v1.log 2>&1\\\"\"
tmux list-windows -t codex-soar | tail -n 4
"'
