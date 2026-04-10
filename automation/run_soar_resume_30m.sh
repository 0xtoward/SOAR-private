#!/bin/zsh
set -euo pipefail

PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

WORKDIR="/Users/ql/cursor/openbmb"
SESSION_ID="019d6caa-e905-75d3-98f2-18d4243ade0e"
PROMPT_FILE="${WORKDIR}/automation/soar_resume_30m_prompt.txt"
LOG_DIR="${WORKDIR}/automation/logs"
LOCK_DIR="${WORKDIR}/automation/.resume_30m.lock"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/run_${RUN_TS}.log"
LAST_MSG="${LOG_DIR}/last_message_${RUN_TS}.txt"
LATEST_MSG="${LOG_DIR}/last_message_latest.txt"
RUN_START_UTC="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

mkdir -p "${LOG_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[$(date '+%F %T')] skip: another automation run is still active" >> "${LOG_DIR}/scheduler.log"
  exit 0
fi

cleanup() {
  rm -rf "${LOCK_DIR}"
}
trap cleanup EXIT

echo "[$(date '+%F %T')] start session=${SESSION_ID}" >> "${LOG_DIR}/scheduler.log"

cd "${WORKDIR}"

set +e
/opt/homebrew/bin/codex exec \
  --full-auto \
  --skip-git-repo-check \
  -C "${WORKDIR}" \
  -o "${LAST_MSG}" \
  resume "${SESSION_ID}" - \
  < "${PROMPT_FILE}" \
  > "${RUN_LOG}" 2>&1
STATUS=$?
set -e

REPLY_STATE="direct"
SESSION_FILE=""
if [[ ! -s "${LAST_MSG}" ]]; then
  SESSION_FILE="$(find "${HOME}/.codex/sessions" -type f -name "*${SESSION_ID}.jsonl" 2>/dev/null | sort | tail -n 1)"
  if [[ -n "${SESSION_FILE}" ]]; then
    python3 - "${SESSION_FILE}" "${RUN_START_UTC}" "${LAST_MSG}" <<'PY'
import json
import sys
from datetime import datetime, timezone

session_file, since_text, output_path = sys.argv[1:4]

def parse_ts(text: str) -> datetime:
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

since = parse_ts(since_text)
last_text = None

with open(session_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = obj.get("timestamp")
        if not ts:
            continue
        try:
            if parse_ts(ts) < since:
                continue
        except Exception:
            continue
        if obj.get("type") != "response_item":
            continue
        payload = obj.get("payload") or {}
        if payload.get("type") != "message" or payload.get("role") != "assistant":
            continue
        parts = []
        for item in payload.get("content") or []:
            if item.get("type") == "output_text" and item.get("text"):
                parts.append(item["text"])
        if parts:
            last_text = "\n".join(parts).strip()

if last_text:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(last_text + "\n")
PY
    if [[ -s "${LAST_MSG}" ]]; then
      REPLY_STATE="recovered"
    fi
  fi
fi

if [[ -s "${LAST_MSG}" ]]; then
  cp "${LAST_MSG}" "${LATEST_MSG}"
else
  REPLY_STATE="missing"
fi
echo "[$(date '+%F %T')] done status=${STATUS} reply_state=${REPLY_STATE} log=${RUN_LOG} reply=${LAST_MSG}" >> "${LOG_DIR}/scheduler.log"
exit "${STATUS}"
