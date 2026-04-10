#!/usr/bin/env bash
set -euo pipefail

INPUT=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
  echo "Usage: bash prepare_model.sh --input <path> --output <path>" >&2
  exit 1
fi

echo "[probe] start $(date '+%F %T')"
echo "[probe] pwd=$(pwd)"
echo "[probe] user=$(whoami)"
echo "[probe] input=${INPUT}"
echo "[probe] output=${OUTPUT}"
echo "[probe] input_exists=$(test -e "${INPUT}" && echo yes || echo no)"
echo "[probe] output_parent=$(dirname "${OUTPUT}")"
echo "[probe] python=$(command -v python || true)"
echo "[probe] uv=$(command -v uv || true)"
echo "[probe] pip=$(command -v pip || true)"
echo "[probe] git=$(command -v git || true)"
echo "[probe] python_version=$(python -V 2>&1 || true)"
echo "[probe] uv_version=$(uv --version 2>&1 || true)"
echo "[probe] git_version=$(git --version 2>&1 || true)"
echo "[probe] network_turbo=$(test -f /etc/network_turbo && echo present || echo absent)"
echo "[probe] nvidia=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)"

python - <<'PY'
import importlib
import os
import site
import sys

mods = [
    "torch",
    "transformers",
    "sglang",
    "flash_attn",
    "gptqmodel",
    "modelopt",
    "accelerate",
    "optimum",
    "sentencepiece",
    "huggingface_hub",
    "tokenizers",
    "triton",
]

print(f"[probe] sys.executable={sys.executable}")
print(f"[probe] sys.version={sys.version.splitlines()[0]}")
print(f"[probe] site_packages={site.getsitepackages()}")
print(f"[probe] cwd={os.getcwd()}")

for name in mods:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "<no __version__>")
        origin = getattr(mod, "__file__", "<built-in>")
        print(f"[probe] {name} version={version} file={origin}")
    except Exception as exc:
        print(f"[probe] {name} import_error={type(exc).__name__}: {exc}")
PY

uv pip list | egrep '^(accelerate|flash-attn|flashinfer|gptqmodel|huggingface-hub|modelopt|optimum|sentencepiece|sglang|tokenizers|torch|transformers|triton) ' || true

echo "[probe] intentionally failing after environment dump"
exit 17
