#!/usr/bin/env bash
set -euo pipefail

ENV="/root/autodl-tmp/sglang/sglang_t550_py312_env"
BASE_ENV="/root/autodl-tmp/sglang/sglang_minicpm_sala_env"
BASE_PY="${BASE_ENV}/bin/python3"
LOG_PREFIX="[evalmatch-env]"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "${LOG_PREFIX} start $(date '+%F %T')"
echo "${LOG_PREFIX} env=${ENV}"
echo "${LOG_PREFIX} base_env=${BASE_ENV}"

if [[ -f /etc/network_turbo ]]; then
  source /etc/network_turbo >/dev/null 2>&1 || true
fi

if [[ ! -x "${BASE_PY}" ]]; then
  echo "${LOG_PREFIX} missing base python: ${BASE_PY}" >&2
  exit 1
fi

if [[ ! -x "${ENV}/bin/python" ]]; then
  echo "${LOG_PREFIX} creating venv with system site packages"
  "${BASE_PY}" -m venv --system-site-packages "${ENV}"
fi

PIP_INDEX_URL="${PIP_INDEX_URL}" \
"${ENV}/bin/pip" install --upgrade pip setuptools wheel packaging

PIP_INDEX_URL="${PIP_INDEX_URL}" \
"${ENV}/bin/pip" install --no-deps \
  "transformers==5.5.0" \
  "accelerate==1.13.0" \
  "optimum==2.1.0" \
  "sentencepiece==0.2.1" \
  "huggingface_hub==1.10.1" \
  "tokenizers==0.22.2" \
  "safetensors==0.7.0" \
  "regex==2026.4.4" \
  "tqdm==4.67.3" \
  "pyyaml==6.0.3"

PIP_INDEX_URL="${PIP_INDEX_URL}" \
"${ENV}/bin/pip" install --no-deps --no-build-isolation \
  "numpy==2.2.6" \
  "threadpoolctl==3.6.0" \
  "device-smi==0.5.3" \
  "protobuf==7.34.1" \
  "pillow==12.2.0" \
  "pypcre==0.2.15" \
  "hf_transfer==0.1.9" \
  "tokenicer==0.0.10" \
  "logbar==0.2.3" \
  "maturin==1.13.1" \
  "datasets==4.8.4" \
  "pyarrow==23.0.1" \
  "dill==0.4.1" \
  "torchao==0.17.0" \
  "kernels==0.12.3" \
  "defuser==0.0.16" \
  "gptqmodel==5.8.0"

"${ENV}/bin/python" - <<'PY'
mods = [
    "torch",
    "transformers",
    "gptqmodel",
    "accelerate",
    "optimum",
    "sentencepiece",
    "huggingface_hub",
    "einops",
    "fla",
    "flash_attn",
]
for m in mods:
    mod = __import__(m)
    print(m, getattr(mod, "__version__", "<no __version__>"))
PY

echo "${LOG_PREFIX} done $(date '+%F %T')"
