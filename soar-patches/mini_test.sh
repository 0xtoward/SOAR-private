#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:30000}"
TIMEOUT="${TIMEOUT:-0}"
MINI_PER_TASK="${MINI_PER_TASK:-4}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-0}"
EVAL_MIN_TOKENS="${EVAL_MIN_TOKENS:-512}"
EVAL_TASK_MAX_TOKENS="${EVAL_TASK_MAX_TOKENS:-}"
EVAL_DATA="${EVAL_DATA:-/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl}"
EVAL_USE_ALL_ROWS="${EVAL_USE_ALL_ROWS:-0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [[ "${TIMEOUT}" =~ ^[0-9]+$ ]] && (( TIMEOUT > 0 )); then
    TIMEOUT_LABEL="${TIMEOUT}s"
else
    TIMEOUT_LABEL="none"
fi

echo -e "${CYAN}========== MINI EVAL ==========${NC}"
echo "API_BASE=${API_BASE}"
echo "MINI_PER_TASK=${MINI_PER_TASK}  TIMEOUT=${TIMEOUT_LABEL}"
echo "EVAL_USE_ALL_ROWS=${EVAL_USE_ALL_ROWS}"
if [[ "${EVAL_MAX_TOKENS}" == "0" ]]; then
    echo "EVAL_MAX_TOKENS=auto(no-trunc by dataset completion_tokens, floor=${EVAL_MIN_TOKENS})"
else
    echo "EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS}"
fi
if [[ -n "${EVAL_TASK_MAX_TOKENS}" ]]; then
    echo "EVAL_TASK_MAX_TOKENS=${EVAL_TASK_MAX_TOKENS}"
fi
echo -e "${YELLOW}[NOTE] mini 是分层抽样，不是官方 150 题 full eval。适合日常迭代，不适合最终结论。${NC}"

if ! curl -s --max-time 3 "${API_BASE}/v1/models" > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] SGLang 服务未运行在 ${API_BASE}${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 服务可达，开始 mini eval...${NC}"

API_BASE="${API_BASE}" TIMEOUT="${TIMEOUT}" MINI_PER_TASK="${MINI_PER_TASK}" EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS}" EVAL_MIN_TOKENS="${EVAL_MIN_TOKENS}" EVAL_TASK_MAX_TOKENS="${EVAL_TASK_MAX_TOKENS}" EVAL_DATA="${EVAL_DATA}" EVAL_USE_ALL_ROWS="${EVAL_USE_ALL_ROWS}" python3 -u - <<'PY'
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

api_base = os.environ["API_BASE"].rstrip("/")
timeout_raw = os.environ["TIMEOUT"].strip()
mini_per_task = int(os.environ["MINI_PER_TASK"])
eval_max_tokens = int(os.environ["EVAL_MAX_TOKENS"])
eval_min_tokens = int(os.environ["EVAL_MIN_TOKENS"])
task_caps_raw = os.environ.get("EVAL_TASK_MAX_TOKENS", "").strip()
eval_data = os.environ["EVAL_DATA"]
use_all_rows = os.environ.get("EVAL_USE_ALL_ROWS", "0").strip().lower() in {"1", "true", "yes", "on"}

def parse_timeout(raw: str) -> Optional[int]:
    if not raw:
        return None
    if raw.lower() in {"0", "none", "off"}:
        return None
    value = int(raw)
    return value if value > 0 else None

request_timeout = parse_timeout(timeout_raw)
if request_timeout is None:
    socket.setdefaulttimeout(None)

def parse_task_caps(raw: str) -> dict[str, int]:
    caps = {}
    if not raw:
        return caps
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if ":" in piece:
            task, value = piece.split(":", 1)
        elif "=" in piece:
            task, value = piece.split("=", 1)
        else:
            raise ValueError(f"Invalid EVAL_TASK_MAX_TOKENS entry: {piece}")
        caps[task.strip()] = int(value.strip())
    return caps

task_caps = parse_task_caps(task_caps_raw)

def post_json(url: str, payload: dict, timeout_s: Optional[int]) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if timeout_s is None:
        resp = urllib.request.urlopen(req)
    else:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    with resp:
        return json.loads(resp.read().decode("utf-8"))

def get_json(url: str, timeout_s: Optional[int]) -> dict:
    if timeout_s is None:
        resp = urllib.request.urlopen(url)
    else:
        resp = urllib.request.urlopen(url, timeout=timeout_s)
    with resp:
        return json.loads(resp.read().decode("utf-8"))

def extract_final_answer(pred: str) -> str:
    parts = pred.split("</think>")
    return parts[-1].strip() if len(parts) > 1 else pred.strip()

def extract_mcq_answer(pred: str):
    match = re.search(r"(?i)ANSWER\s*:\s*([A-D])", pred)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{\\text\{([A-D])\}\}", pred)
    if match:
        return match.group(1).upper()
    match = re.search(r"\\boxed\{([A-D])\}", pred)
    if match:
        return match.group(1).upper()
    return None

def score_item(task, pred, gold):
    if not pred or gold is None:
        return 0.0
    final = extract_final_answer(pred)
    if task == "mcq":
        extracted = extract_mcq_answer(final)
        return 1.0 if extracted and extracted.upper() == str(gold).upper() else 0.0
    if not isinstance(gold, list):
        gold = [gold]
    if task in ["qa", "niah", "lcx"]:
        return 1.0 if any(str(r).lower() in final.lower() for r in gold) else 0.0
    hits = sum(1.0 if str(r).lower() in final.lower() else 0.0 for r in gold)
    return hits / len(gold) if gold else 0.0

def verdict(score: float) -> str:
    if score >= 0.999:
        return "PASS"
    if score <= 0.001:
        return "FAIL"
    return "PART"

def pick_quantile_samples(items, count):
    items = sorted(items, key=lambda x: x.get("prompt_tokens", 10**9))
    if count >= len(items):
        return items
    if count <= 1:
        return [items[len(items) // 2]]
    selected = []
    used = set()
    n = len(items)
    for i in range(count):
        pos = round(i * (n - 1) / (count - 1))
        while pos in used and pos + 1 < n:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        if pos in used:
            continue
        used.add(pos)
        selected.append(items[pos])
    return selected

def run_one(prompt: str, max_new_tokens: int) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }
    start = time.perf_counter()
    try:
        result = post_json(f"{api_base}/generate", payload, request_timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        return {"ok": False, "error": f"HTTP {exc.code}: {body}", "latency": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "latency": None}
    meta = result.get("meta_info", {})
    text = (result.get("text") or "").strip()
    latency = meta.get("e2e_latency", time.perf_counter() - start)
    return {
        "ok": True,
        "text": text,
        "latency": latency,
        "prompt_tokens": meta.get("prompt_tokens", 0),
        "completion_tokens": meta.get("completion_tokens", 0),
        "finish_reason": meta.get("finish_reason"),
    }

try:
    model_info = get_json(f"{api_base}/v1/models", request_timeout)
    model_id = model_info["data"][0]["id"]
except Exception as exc:
    print(f"[ERROR] 读取 /v1/models 失败: {exc}")
    sys.exit(1)

print(f"[INFO] model={model_id}")
try:
    with open(eval_data, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]
except Exception as exc:
    print(f"[ERROR] 读取评测数据失败: {exc}")
    sys.exit(1)

by_task = {}
for item in dataset:
    task = item.get("task", "unknown")
    by_task.setdefault(task, []).append(item)

selected = []
if use_all_rows:
    selected = [dict(item) for item in dataset]
else:
    for task, items in sorted(by_task.items()):
        picks = pick_quantile_samples(items, mini_per_task)
        for item in picks:
            row = dict(item)
            row["_task_rank"] = len(selected)
            selected.append(row)

selected = sorted(selected, key=lambda x: (x.get("prompt_tokens", 10**9), x.get("task", "")))
print(f"[INFO] selected_samples={len(selected)} from {len(by_task)} tasks")
for task, items in sorted(by_task.items()):
    task_selected = [x for x in selected if x.get("task") == task]
    token_list = ",".join(str(x.get("prompt_tokens", 0)) for x in task_selected)
    print(f"  task={task:>4s} picks={len(task_selected)} prompt_tokens=[{token_list}]")

print("")
scores = []
pass_count = 0
part_count = 0
fail_count = 0
empty_count = 0
for idx, item in enumerate(selected, 1):
    task = item.get("task", "unknown")
    if eval_max_tokens > 0:
        max_new_tokens = eval_max_tokens
    elif task in task_caps:
        max_new_tokens = task_caps[task]
    else:
        max_new_tokens = max(int(item.get("completion_tokens", 0) or 0), eval_min_tokens)
    result = run_one(item["question"], max_new_tokens=max_new_tokens)
    if not result["ok"]:
        print(f"  {idx:02d}/{len(selected)} ERROR {task:>4s} | {result['error']}")
        sys.exit(1)
    score = score_item(task, result["text"], item.get("gold"))
    state = verdict(score)
    scores.append(score)
    if state == "PASS":
        pass_count += 1
    elif state == "PART":
        part_count += 1
    else:
        fail_count += 1
    if not result["text"]:
        empty_count += 1
    print(
        f"  {idx:02d}/{len(selected)} {state:>4s} {task:>4s} | "
        f"in={item.get('prompt_tokens', 0):>6} | req_out={max_new_tokens:>5} | "
        f"lat={result['latency']:.2f}s | out={result['completion_tokens']:>4} | "
        f"finish={str(result['finish_reason']):<10} | score={score:.2f}"
    )

avg_score = (sum(scores) / len(scores) * 100.0) if scores else 0.0
print("")
print(
    f"[SUMMARY] avg_score={avg_score:.2f}%  pass={pass_count}  part={part_count}  "
    f"fail={fail_count}  empty={empty_count}/{len(selected)}"
)
if empty_count == len(selected):
    print("[WARN] mini eval 完成，但样本基本为空输出")
else:
    print("[PASS] mini eval 完成")
PY
