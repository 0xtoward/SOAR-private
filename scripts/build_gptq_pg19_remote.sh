#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-rtx6000-2}"
REMOTE_SOAR_ROOT="${REMOTE_SOAR_ROOT:-/root/autodl-tmp/SOAR-Toolkit}"
REMOTE_MODEL_ROOT="${REMOTE_MODEL_ROOT:-/root/autodl-tmp/models}"
LOCAL_ROOT="${LOCAL_ROOT:-/Users/ql/cursor/openbmb}"

PG19_CORPUS_PATH="${PG19_CORPUS_PATH:-${REMOTE_SOAR_ROOT}/calibration_sources/pg19_train_books_v1.jsonl}"
PG19_TARGET_BOOKS="${PG19_TARGET_BOOKS:-8}"
PG19_MIN_BOOK_TOKENS="${PG19_MIN_BOOK_TOKENS:-160000}"
PG19_MAX_ATTEMPTS="${PG19_MAX_ATTEMPTS:-192}"
PUBLIC_QUOTAS="${PUBLIC_QUOTAS:-mcq:12,qa:14,niah:14,fwe:4,cwe:4}"
PG19_QUOTA="${PG19_QUOTA:-16}"
PG19_OUTPUT_PATH="${PG19_OUTPUT_PATH:-${REMOTE_SOAR_ROOT}/calibration/calibration_gptq_w4a16_pg19_v2.jsonl}"
TRUE160K_OUTPUT_PATH="${TRUE160K_OUTPUT_PATH:-${REMOTE_SOAR_ROOT}/calibration/calibration_gptq_w4a16_true160k_v1.jsonl}"

scp \
  "${LOCAL_ROOT}/scripts/build_gptq_calibration.py" \
  "${LOCAL_ROOT}/scripts/materialize_pg19_books.py" \
  "${REMOTE_HOST}:${REMOTE_SOAR_ROOT}/scripts/"

ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
mkdir -p "${REMOTE_SOAR_ROOT}/calibration_sources"
source /etc/network_turbo >/dev/null 2>&1 || true
export HF_ENDPOINT=https://hf-mirror.com
source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1

"\${SOAR_UV}" run --python "\${SOAR_PYTHON}" python "${REMOTE_SOAR_ROOT}/scripts/materialize_pg19_books.py" \
  --model-path "${REMOTE_MODEL_ROOT}" \
  --output-path "${PG19_CORPUS_PATH}" \
  --target-books "${PG19_TARGET_BOOKS}" \
  --min-book-tokens "${PG19_MIN_BOOK_TOKENS}" \
  --max-attempts "${PG19_MAX_ATTEMPTS}"

"\${SOAR_UV}" run --python "\${SOAR_PYTHON}" python "${REMOTE_SOAR_ROOT}/scripts/build_gptq_calibration.py" \
  --model-path "${REMOTE_MODEL_ROOT}" \
  --public-path "${REMOTE_SOAR_ROOT}/eval_dataset/perf_public_set.jsonl" \
  --output-path "${PG19_OUTPUT_PATH}" \
  --public-quotas "${PUBLIC_QUOTAS}" \
  --open-quotas "pg19_local:${PG19_QUOTA}" \
  --pg19-local-path "${PG19_CORPUS_PATH}" \
  --pg19-min-book-tokens "${PG19_MIN_BOOK_TOKENS}" \
  --pg19-slice-strategy contiguous \
  --max-sample-tokens 32768 \
  --head-tokens 4096 \
  --middle-tokens 4096 \
  --tail-tokens 24576

"\${SOAR_UV}" run --python "\${SOAR_PYTHON}" python "${REMOTE_SOAR_ROOT}/scripts/build_gptq_calibration.py" \
  --model-path "${REMOTE_MODEL_ROOT}" \
  --public-path "${REMOTE_SOAR_ROOT}/eval_dataset/perf_public_set.jsonl" \
  --output-path "${TRUE160K_OUTPUT_PATH}" \
  --public-quotas "${PUBLIC_QUOTAS}" \
  --open-quotas "pg19_local:${PG19_QUOTA}" \
  --pg19-local-path "${PG19_CORPUS_PATH}" \
  --pg19-min-book-tokens "${PG19_MIN_BOOK_TOKENS}" \
  --pg19-slice-strategy contiguous \
  --max-sample-tokens 160000 \
  --head-tokens 4096 \
  --middle-tokens 4096 \
  --tail-tokens 151808
EOF

printf '%s\n%s\n' "${PG19_OUTPUT_PATH}" "${TRUE160K_OUTPUT_PATH}"
