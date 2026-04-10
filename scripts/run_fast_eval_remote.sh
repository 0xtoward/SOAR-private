#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <label> <model_path> <quantization:none|modelopt|gptq|gptq_marlin>" >&2
  exit 2
fi

python3 /Users/ql/cursor/openbmb/scripts/run_remote_candidate_bench.py \
  --label "$1" \
  --model-path "$2" \
  --quantization "$3" \
  --dtype float16 \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --dense-as-sparse \
  --profile fast-eval
