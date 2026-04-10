#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:30000}"
TIMEOUT="${TIMEOUT:-0}"
# Eval defaults to no forced truncation.
# If EVAL_MAX_TOKENS is unset or <= 0, use each sample's completion_tokens
# with a minimum floor to avoid judging quality from a 64-token cutoff.
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-0}"
EVAL_MIN_TOKENS="${EVAL_MIN_TOKENS:-512}"
BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS:-16}"
EVAL_DATA="${EVAL_DATA:-/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl}"
RUN_EVAL=1
RUN_BENCH=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

for arg in "$@"; do
    case "$arg" in
        --eval-only)
            RUN_BENCH=0
            ;;
        --bench-only)
            RUN_EVAL=0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown argument: ${arg}${NC}"
            echo "Usage: bash fast_test.sh [--eval-only|--bench-only]"
            exit 2
            ;;
    esac
done

if [[ "${TIMEOUT}" =~ ^[0-9]+$ ]] && (( TIMEOUT > 0 )); then
    TIMEOUT_LABEL="${TIMEOUT}s"
else
    TIMEOUT_LABEL="none"
fi

echo -e "${CYAN}========== MIN TEST ==========${NC}"
echo "API_BASE=${API_BASE}"
if [[ "${EVAL_MAX_TOKENS}" == "0" ]]; then
    echo "EVAL_MAX_TOKENS=auto(no-trunc by dataset completion_tokens, floor=${EVAL_MIN_TOKENS})  BENCH_MAX_TOKENS=${BENCH_MAX_TOKENS}  TIMEOUT=${TIMEOUT_LABEL}"
else
    echo "EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS}  BENCH_MAX_TOKENS=${BENCH_MAX_TOKENS}  TIMEOUT=${TIMEOUT_LABEL}"
fi
echo -e "${YELLOW}[NOTE] eval 默认不再使用 64-token 强制截断；64-token 结果只能当 smoke，不可直接下质量结论。${NC}"

if ! curl -s --max-time 3 "${API_BASE}/v1/models" > /dev/null 2>&1; then
    echo -e "${RED}[ERROR] SGLang 服务未运行在 ${API_BASE}${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 服务可达，开始最小测试...${NC}"

API_BASE="${API_BASE}" TIMEOUT="${TIMEOUT}" EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS}" EVAL_MIN_TOKENS="${EVAL_MIN_TOKENS}" BENCH_MAX_TOKENS="${BENCH_MAX_TOKENS}" EVAL_DATA="${EVAL_DATA}" RUN_EVAL="${RUN_EVAL}" RUN_BENCH="${RUN_BENCH}" python3 -u - <<'PY'
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
eval_max_tokens = int(os.environ["EVAL_MAX_TOKENS"])
eval_min_tokens = int(os.environ["EVAL_MIN_TOKENS"])
bench_max_tokens = int(os.environ["BENCH_MAX_TOKENS"])
eval_data = os.environ["EVAL_DATA"]
run_eval = os.environ["RUN_EVAL"] == "1"
run_bench = os.environ["RUN_BENCH"] == "1"

BUCKETS = [
    ("0-4K", 0, 4000),
    ("4K-16K", 4000, 16000),
    ("16K-32K", 16000, 32000),
    ("32K-128K", 32000, 128000),
    ("128K-160K", 128000, 160000),
]

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

def run_one(prompt: str) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": current_max_tokens,
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

eval_samples = []
for task, items in sorted(by_task.items()):
    items = sorted(items, key=lambda x: x.get("prompt_tokens", 10**9))
    eval_samples.append(items[0])

bench_samples = []
for label, low, high in BUCKETS:
    candidates = [
        item for item in dataset
        if low <= item.get("prompt_tokens", 0) < high
    ]
    if not candidates:
        bench_samples.append((label, None))
        continue
    candidates.sort(key=lambda x: x.get("prompt_tokens", 10**9))
    bench_samples.append((label, candidates[0]))

empty_eval = 0
empty_bench = 0

if run_eval:
    print("")
    if eval_max_tokens > 0:
        print(f"[mini-eval fixed-cap={eval_max_tokens}]")
    else:
        print(f"[mini-eval no-trunc floor={eval_min_tokens}]")
    eval_scores = []
    eval_pass = 0
    eval_partial = 0
    eval_fail = 0
    for idx, item in enumerate(eval_samples, 1):
        task = item.get("task", "unknown")
        if eval_max_tokens > 0:
            current_max_tokens = eval_max_tokens
        else:
            current_max_tokens = max(int(item.get("completion_tokens", 0) or 0), eval_min_tokens)
        result = run_one(item["question"])
        if not result["ok"]:
            print(f"  {idx:02d}/{len(eval_samples)} ERROR {task:>4s} | {result['error']}")
            sys.exit(1)
        score = score_item(task, result["text"], item.get("gold"))
        eval_scores.append(score)
        state = verdict(score)
        if state == "PASS":
            eval_pass += 1
        elif state == "PART":
            eval_partial += 1
        else:
            eval_fail += 1
        if not result["text"]:
            empty_eval += 1
        print(
            f"  {idx:02d}/{len(eval_samples)} {state:>4s} {task:>4s} | "
            f"in={item.get('prompt_tokens', 0):>6} | req_out={current_max_tokens:>5} | "
            f"lat={result['latency']:.2f}s | out={result['completion_tokens']:>4} | "
            f"finish={str(result['finish_reason']):<10} | score={score:.2f}"
        )
    avg_score = (sum(eval_scores) / len(eval_scores) * 100.0) if eval_scores else 0.0
    print(
        f"  avg_score={avg_score:.2f}%  pass={eval_pass}  part={eval_partial}  "
        f"fail={eval_fail}  empty={empty_eval}/{len(eval_samples)}"
    )

if run_bench:
    print("")
    print("[ultra-bench]")
    current_max_tokens = bench_max_tokens
    bench_latencies = []
    bench_output_tokens = 0
    bench_count = 0
    for label, item in bench_samples:
        if item is None:
            print(f"  {label:>9s} | missing")
            continue
        started = time.perf_counter()
        result = run_one(item["question"])
        wall = time.perf_counter() - started
        if not result["ok"]:
            print(f"  [ERROR] {label}: {result['error']}")
            sys.exit(1)
        bench_count += 1
        bench_latencies.append(wall)
        bench_output_tokens += result["completion_tokens"]
        if not result["text"]:
            empty_bench += 1
        print(
            f"  {bench_count:02d}/{len(bench_samples)} {label:>9s} | "
            f"in={item.get('prompt_tokens', 0):>6} | wall={wall:.2f}s | "
            f"out={result['completion_tokens']:>4} | finish={str(result['finish_reason']):<10}"
        )

    bench_total = sum(bench_latencies)
    bench_avg = bench_total / len(bench_latencies) if bench_latencies else 0.0
    tps = bench_output_tokens / bench_total if bench_total > 0 else 0.0
    print(f"  total={bench_total:.2f}s  avg={bench_avg:.2f}s  out_tokens={bench_output_tokens}  tps={tps:.2f}")
    print(f"  empty={empty_bench}/{len(bench_latencies)}")

print("")
if run_eval and empty_eval == len(eval_samples):
    print("[WARN] 评测完成，但 eval 样本基本为空输出")
elif run_bench and bench_latencies and empty_bench == len(bench_latencies):
    print("[WARN] 测试完成，但 bench 样本基本为空输出")
else:
    print("[PASS] 测试完成")
PY
