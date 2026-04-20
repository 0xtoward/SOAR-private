#!/usr/bin/env python3
"""
Sequential remote benchmark runner for SOAR candidate models on `rtx6000-2`.

This script is intentionally conservative:
- one candidate at a time
- one remote server process at a time
- smoke-quality eval separated from speed benchmarking
- stable proxy datasets for relative A/B comparison
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REMOTE_HOST = os.environ.get("SOAR_REMOTE_HOST", "rtx6000-2")
REMOTE_SOAR_ROOT = os.environ.get("SOAR_REMOTE_SOAR_ROOT", "/root/autodl-tmp/SOAR-Toolkit")
REMOTE_SGLANG_ROOT = os.environ.get("SOAR_REMOTE_SGLANG_ROOT", "/root/autodl-tmp/sglang")
REMOTE_PYTHON = os.environ.get(
    "SOAR_REMOTE_PYTHON",
    f"{REMOTE_SGLANG_ROOT}/sglang_minicpm_sala_env/bin/python3",
)
REMOTE_ENV = os.environ.get(
    "SOAR_REMOTE_ENV",
    "source /etc/network_turbo >/dev/null 2>&1; "
    "export HF_ENDPOINT=https://hf-mirror.com; "
    "source /root/autodl-tmp/use_soar_uv.sh >/dev/null 2>&1",
)
REMOTE_RESULTS_ROOT = f"{REMOTE_SOAR_ROOT}/test_results/candidate_benches"

SMOKE_S1 = f"{REMOTE_SOAR_ROOT}/bench_speed_v2_smoke.jsonl"
SMOKE_S8 = f"{REMOTE_SOAR_ROOT}/bench_speed_v2_smoke.jsonl"
MINI_S1 = f"{REMOTE_SOAR_ROOT}/bench_speed_v2_mini.jsonl"
PROXY_S1 = f"{REMOTE_SOAR_ROOT}/bench_speed_v2_proxy11.jsonl"
PROXY_S8 = f"{REMOTE_SOAR_ROOT}/bench_speed_v2_proxy11.jsonl"
PROXY_SMAX = f"{REMOTE_SOAR_ROOT}/bench_speed_v2_proxy11.jsonl"
LOCAL_PATCHED_MINI_TEST = Path("/Users/ql/cursor/openbmb/soar-patches/mini_test.sh")
REMOTE_PATCHED_MINI_TEST = f"{REMOTE_SOAR_ROOT}/.codex_mini_eval_task_caps.sh"
MEDIUM_EVAL_DATA = f"{REMOTE_SOAR_ROOT}/eval_dataset/perf_public_medium_eval_v1.jsonl"
FAST_EVAL_DATA = f"{REMOTE_SOAR_ROOT}/eval_dataset/perf_public_fast_eval.jsonl"
FAST_EVAL_META = f"{REMOTE_SOAR_ROOT}/eval_dataset/perf_public_fast_eval.meta.json"
FAST_EVAL_PER_BUCKET = 1
FAST_EVAL_COMPLETION_CAPS = {
    "mcq": 64,
    "qa": 128,
    "niah": 128,
    "fwe": 256,
    "cwe": 256,
}
PUBLIC_SPEED_SOURCE = f"{REMOTE_SOAR_ROOT}/eval_dataset/perf_public_set.jsonl"
PUBLIC_SPEED_DATA = f"{REMOTE_SOAR_ROOT}/bench_speed_public_full.jsonl"
PUBLIC_SPEED_META = f"{REMOTE_SOAR_ROOT}/bench_speed_public_full.meta.json"
PUBLIC_SPEED_TASK_ORDER = ("mcq", "qa", "niah", "fwe", "cwe")

PROXY_SELECTION = {
    # Target input-bucket counts for 11 rows are:
    # 0-4K: 3, 4-16K: 1, 16-32K: 2, 32-128K: 4, 128-160K: 1.
    # We exclude current >160K mini rows to stay closer to the public length cap.
    "bench_speed_v2_smoke.jsonl": [1, 3, 4, 5],
    "bench_speed_v2_mini.jsonl": [1, 3, 4, 6, 8, 11, 14],
}

def evenly_pick(items: list[dict], k: int) -> list[dict]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    picked = []
    used = set()
    n = len(items)
    for i in range(k):
        pos = round(i * (n - 1) / (k - 1)) if k > 1 else n // 2
        while pos in used and pos + 1 < n:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        if pos in used:
            continue
        used.add(pos)
        picked.append(items[pos])
    return picked

SSH_BASE_ARGS = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    REMOTE_HOST,
    "bash",
    "-s",
]
SSH_RETRY_CODES = {255}
SSH_RETRY_SUBSTRINGS = (
    "Could not resolve hostname",
    "Name or service not known",
    "nodename nor servname provided",
    "Connection timed out",
    "Connection reset by peer",
    "Operation timed out",
)
SSH_MAX_ATTEMPTS = 4
SSH_RETRY_DELAY_S = 3
PREFLIGHT_MAX_GPU_UTILIZATION_PCT = 5
PREFLIGHT_MAX_GPU_MEMORY_MIB = 256

TRANSPORT_FAILURE_RULES = (
    (
        "dns_resolution",
        True,
        (
            "Could not resolve hostname",
            "Name or service not known",
            "nodename nor servname provided",
        ),
    ),
    (
        "connect_timeout",
        True,
        (
            "Connection timed out",
            "Operation timed out",
        ),
    ),
    (
        "connection_reset",
        True,
        (
            "Connection reset by peer",
            "Broken pipe",
        ),
    ),
    (
        "auth_or_hostkey",
        False,
        (
            "Permission denied",
            "Host key verification failed",
            "REMOTE HOST IDENTIFICATION HAS CHANGED",
        ),
    ),
)


@dataclass
class RunSummary:
    label: str
    run_dir: str
    server_log: str
    preflight: dict | None
    fast_eval_log: str | None
    mini_eval_log: str | None
    official_eval_log: str | None
    bench_smoke_log: str | None
    bench_mini_log: str | None
    fast_avg_score: float | None
    mini_eval_avg_score: float | None
    official_avg_score: float | None
    official_duration_s: float | None
    fast_pass: int | None
    fast_part: int | None
    fast_fail: int | None
    mini_eval_pass: int | None
    mini_eval_part: int | None
    mini_eval_fail: int | None
    smoke_duration: dict | None
    mini_duration: dict | None


def classify_ssh_failure(
    returncode: int,
    stdout: str | None,
    stderr: str | None,
) -> dict:
    combined = "\n".join(part for part in (stdout or "", stderr or "") if part).strip()
    detail = next((line.strip() for line in combined.splitlines() if line.strip()), f"ssh exit {returncode}")
    for category, retryable, markers in TRANSPORT_FAILURE_RULES:
        if any(marker in combined for marker in markers):
            return {
                "category": category,
                "retryable": retryable,
                "detail": detail,
            }
    if returncode == 255:
        return {
            "category": "transport_other",
            "retryable": True,
            "detail": detail,
        }
    return {
        "category": "remote_command_failure",
        "retryable": False,
        "detail": detail,
    }


def ssh(script: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    last_proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, SSH_MAX_ATTEMPTS + 1):
        proc = subprocess.run(
            SSH_BASE_ARGS,
            input=script,
            check=False,
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0:
            return proc

        last_proc = proc
        failure = classify_ssh_failure(proc.returncode, proc.stdout, proc.stderr)
        retryable = proc.returncode in SSH_RETRY_CODES and failure["retryable"] and any(
            marker in f"{proc.stdout}\n{proc.stderr}" for marker in SSH_RETRY_SUBSTRINGS
        )
        if not retryable or attempt == SSH_MAX_ATTEMPTS:
            break
        print(
            "[bench] ssh retry "
            f"{attempt}/{SSH_MAX_ATTEMPTS} "
            f"category={failure['category']} "
            f"detail={failure['detail']}",
            file=sys.stderr,
        )
        time.sleep(SSH_RETRY_DELAY_S * attempt)

    assert last_proc is not None
    if check:
        raise subprocess.CalledProcessError(
            last_proc.returncode,
            SSH_BASE_ARGS,
            output=last_proc.stdout,
            stderr=last_proc.stderr,
        )
    return last_proc


def build_server_args(args: argparse.Namespace) -> list[str]:
    parts = [
        REMOTE_PYTHON,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--attention-backend",
        args.attention_backend,
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--skip-server-warmup",
    ]
    if args.quantization != "none":
        parts.extend(["--quantization", args.quantization])
    if args.disable_radix_cache:
        parts.append("--disable-radix-cache")
    if args.disable_cuda_graph:
        parts.append("--disable-cuda-graph")
    if args.dense_as_sparse:
        parts.append("--dense-as-sparse")
    if args.fuse_topk:
        parts.append("--fuse-topk")
    if args.split_stage1:
        parts.append("--split-stage1")
    if args.max_running_requests is not None:
        parts.extend(["--max-running-requests", str(args.max_running_requests)])
    for extra in args.extra_server_arg:
        parts.append(extra)
    return parts


def remote_quote_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def start_server(args: argparse.Namespace, run_dir: str) -> tuple[str, str]:
    server_log = f"{run_dir}/server.log"
    pid_file = f"{run_dir}/server.pid"
    server_cmd = remote_quote_cmd(build_server_args(args))
    script = f"""
set -e
mkdir -p {shlex.quote(run_dir)}
pkill -f 'sglang.launch_server --host 127.0.0.1 --port {args.port}' >/dev/null 2>&1 || true
rm -f {shlex.quote(pid_file)}
{REMOTE_ENV}
nohup {server_cmd} > {shlex.quote(server_log)} 2>&1 &
echo $! > {shlex.quote(pid_file)}
cat {shlex.quote(pid_file)}
"""
    proc = ssh(script)
    pid = proc.stdout.strip().splitlines()[-1]
    return pid, server_log


def wait_for_server(port: int, timeout_s: int = 240) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        probe = ssh(
            f"curl -sf http://127.0.0.1:{port}/v1/models >/dev/null 2>&1",
            check=False,
        )
        if probe.returncode == 0:
            return True
        time.sleep(2)
    return False


def stop_server(pid: str, port: int) -> None:
    script = f"""
kill {shlex.quote(pid)} >/dev/null 2>&1 || true
sleep 2
pkill -f 'sglang.launch_server --host 127.0.0.1 --port {port}' >/dev/null 2>&1 || true
"""
    ssh(script, check=False)


def run_fast_eval(port: int, run_dir: str) -> tuple[str, str]:
    log_path = f"{run_dir}/fast_eval.log"
    script = f"""
set -e
{REMOTE_ENV}
cd {shlex.quote(REMOTE_SOAR_ROOT)}
API_BASE=http://127.0.0.1:{port} TIMEOUT=0 bash fast_test.sh --eval-only | tee {shlex.quote(log_path)}
"""
    proc = ssh(script)
    return proc.stdout, log_path


def run_mini_eval(
    port: int,
    run_dir: str,
    *,
    eval_data: str | None = None,
    mini_per_task: int = 4,
    log_name: str = "mini_eval.log",
    task_caps: str | None = None,
    use_all_rows: bool = False,
) -> tuple[str, str]:
    log_path = f"{run_dir}/{log_name}"
    eval_data_export = (
        f"EVAL_DATA={shlex.quote(eval_data)} " if eval_data else ""
    )
    task_caps_export = (
        f"EVAL_TASK_MAX_TOKENS={shlex.quote(task_caps)} " if task_caps else ""
    )
    mini_script = "mini_test.sh"
    if task_caps:
        patch_text = LOCAL_PATCHED_MINI_TEST.read_text(encoding="utf-8")
        sync_script = f"""
set -e
cat > {shlex.quote(REMOTE_PATCHED_MINI_TEST)} <<'EOF'
{patch_text}
EOF
chmod +x {shlex.quote(REMOTE_PATCHED_MINI_TEST)}
"""
        ssh(sync_script)
        mini_script = REMOTE_PATCHED_MINI_TEST
    script = f"""
set -e
{REMOTE_ENV}
cd {shlex.quote(REMOTE_SOAR_ROOT)}
API_BASE=http://127.0.0.1:{port} TIMEOUT=0 {eval_data_export}{task_caps_export}MINI_PER_TASK={mini_per_task} EVAL_USE_ALL_ROWS={"1" if use_all_rows else "0"} EVAL_MAX_TOKENS=0 EVAL_MIN_TOKENS=512 bash {shlex.quote(mini_script)} | tee {shlex.quote(log_path)}
"""
    proc = ssh(script)
    return proc.stdout, log_path


def run_official_eval(
    port: int,
    run_dir: str,
    *,
    model_path: str,
    max_seq_len: int,
    data_path: str,
    concurrency: int,
    num_samples: int | None,
) -> tuple[str, str]:
    log_path = f"{run_dir}/official_eval.log"
    num_samples_arg = f"--num_samples {num_samples}" if num_samples is not None else ""
    script = f"""
set -e
{REMOTE_ENV}
cd {shlex.quote(REMOTE_SOAR_ROOT)}
python3 eval_model.py \
  --model_path {shlex.quote(model_path)} \
  --api_base http://127.0.0.1:{port} \
  --model_name {shlex.quote(model_path)} \
  --max_seq_len {max_seq_len} \
  --data_path {shlex.quote(data_path)} \
  --concurrency {concurrency} \
  {num_samples_arg} | tee {shlex.quote(log_path)}
"""
    proc = ssh(script)
    return proc.stdout, log_path


def run_bench(port: int, run_dir: str, *, s1: str | None, s8: str | None, smax: str | None, log_name: str) -> tuple[str, str]:
    log_path = f"{run_dir}/{log_name}"
    exports = [
        f"API_BASE=http://127.0.0.1:{port}",
        f"SPEED_DATA_S1={shlex.quote(s1)}" if s1 else "SPEED_DATA_S1=",
        f"SPEED_DATA_S8={shlex.quote(s8)}" if s8 else "SPEED_DATA_S8=",
        f"SPEED_DATA_SMAX={shlex.quote(smax)}" if smax else "SPEED_DATA_SMAX=",
    ]
    script = f"""
set -e
{REMOTE_ENV}
cd {shlex.quote(REMOTE_SOAR_ROOT)}
{' '.join(exports)} bash bench_serving.sh http://127.0.0.1:{port} | tee {shlex.quote(log_path)}
"""
    proc = ssh(script)
    return proc.stdout, log_path


def run_preflight(port: int) -> dict:
    script = f"""
set -e
python3 - <<'PY'
import json
import os
import re
import subprocess
from datetime import datetime

pattern = re.compile(r"(sglang\\.launch_server|bench_serving|eval_model\\.py|fast_test\\.sh)")

gpu_proc = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
    check=True,
    capture_output=True,
    text=True,
)
gpu_line = next(line.strip() for line in gpu_proc.stdout.splitlines() if line.strip())
name, memory_used, utilization = [part.strip() for part in gpu_line.split(",", 2)]
memory_used_mib = int(memory_used.split()[0])
utilization_pct = int(utilization.split()[0])

ps_proc = subprocess.run(
    ["ps", "-eo", "pid=,args="],
    check=True,
    capture_output=True,
    text=True,
)
active_jobs = []
for line in ps_proc.stdout.splitlines():
    line = line.strip()
    if not line:
        continue
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    pid_str, cmd = parts
    pid = int(pid_str)
    if pid == os.getpid():
        continue
    if pattern.search(cmd):
        active_jobs.append({{"pid": pid, "cmd": cmd}})

remote_idle = (
    memory_used_mib <= {PREFLIGHT_MAX_GPU_MEMORY_MIB}
    and utilization_pct <= {PREFLIGHT_MAX_GPU_UTILIZATION_PCT}
    and not active_jobs
)

print(json.dumps({{
    "transport_ok": True,
    "remote_idle": remote_idle,
    "host": subprocess.check_output(["hostname"], text=True).strip(),
    "remote_time": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    "port": {port},
    "gpu_name": name,
    "gpu_memory_used_mib": memory_used_mib,
    "gpu_utilization_pct": utilization_pct,
    "active_heavy_jobs": active_jobs,
    "idle_thresholds": {{
        "gpu_memory_used_mib_max": {PREFLIGHT_MAX_GPU_MEMORY_MIB},
        "gpu_utilization_pct_max": {PREFLIGHT_MAX_GPU_UTILIZATION_PCT},
    }},
}}, ensure_ascii=False))
PY
"""
    proc = ssh(script)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def ensure_proxy_dataset() -> None:
    selection_json = json.dumps(PROXY_SELECTION, ensure_ascii=False)
    script = f"""
set -e
{REMOTE_ENV}
{REMOTE_PYTHON} - <<'PY'
import json
from pathlib import Path

target = Path({PROXY_S1!r})
selection = json.loads({selection_json!r})
rows = []
for source_name, indices in selection.items():
    source = Path({REMOTE_SOAR_ROOT!r}) / source_name
    with source.open("r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]
    for idx in indices:
        rows.append(all_rows[idx - 1])

target.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows),
    encoding="utf-8",
)
print(target)
print(f"rows={{len(rows)}}")
PY
"""
    ssh(script)


def ensure_fast_eval_dataset() -> None:
    caps_json = json.dumps(FAST_EVAL_COMPLETION_CAPS, ensure_ascii=False)
    script = f"""
set -e
{REMOTE_ENV}
{REMOTE_PYTHON} - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path

source_path = Path({MEDIUM_EVAL_DATA!r})
target_path = Path({FAST_EVAL_DATA!r})
meta_path = Path({FAST_EVAL_META!r})
per_bucket = {FAST_EVAL_PER_BUCKET}
caps = json.loads({caps_json!r})

def bucket(prompt_tokens: int) -> str:
    if prompt_tokens < 4096:
        return "0-4K"
    if prompt_tokens < 16384:
        return "4-16K"
    if prompt_tokens < 32768:
        return "16K-32K"
    if prompt_tokens < 131072:
        return "32K-128K"
    return "128K-160K"

def evenly_pick(items, k):
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    picked = []
    used = set()
    n = len(items)
    for i in range(k):
        pos = round(i * (n - 1) / (k - 1)) if k > 1 else n // 2
        while pos in used and pos + 1 < n:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        if pos in used:
            continue
        used.add(pos)
        picked.append(items[pos])
    return picked

rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
by_task_bucket = defaultdict(list)
for row in rows:
    task = row["task"]
    bkt = bucket(int(row.get("prompt_tokens", 0)))
    by_task_bucket[(task, bkt)].append(row)

selected = []
for (task, bkt), items in sorted(by_task_bucket.items()):
    task_bucket_rows = sorted(
        items,
        key=lambda row: (int(row.get("prompt_tokens", 0)), int(row.get("index", 0))),
    )
    selected.extend(evenly_pick(task_bucket_rows, per_bucket))

selected = sorted(
    selected,
    key=lambda row: (row.get("task", ""), int(row.get("prompt_tokens", 0)), int(row.get("index", 0))),
)

for row in selected:
    task = row.get("task")
    cap = int(caps[task])
    original_completion = int(row.get("completion_tokens", cap))
    row["completion_tokens_original"] = original_completion
    row["completion_tokens"] = min(original_completion, cap)
    row["fast_eval_cap"] = cap

target_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in selected),
    encoding="utf-8",
)

meta = {{
    "source": str(source_path),
    "count": len(selected),
    "per_bucket": per_bucket,
    "completion_token_caps": caps,
    "by_task": dict(sorted(Counter(row["task"] for row in selected).items())),
    "by_bucket": dict(sorted(Counter(bucket(int(row.get("prompt_tokens", 0))) for row in selected).items())),
    "selected_prompt_tokens": {{
        task: [int(row.get("prompt_tokens", 0)) for row in sorted(task_rows, key=lambda row: int(row.get("prompt_tokens", 0)))]
        for task, task_rows in sorted((task, [row for row in selected if row["task"] == task]) for task in Counter(row["task"] for row in selected))
    }},
}}
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(target_path)
print(json.dumps(meta, ensure_ascii=False))
PY
"""
    ssh(script)


def ensure_public_speed_dataset(
    model_path: str,
    *,
    completion_cap: int | None,
    per_task_bucket: int | None,
) -> dict:
    task_order_json = json.dumps(list(PUBLIC_SPEED_TASK_ORDER), ensure_ascii=False)
    completion_cap_literal = "None" if completion_cap is None else str(int(completion_cap))
    per_task_bucket_literal = "None" if per_task_bucket is None else str(int(per_task_bucket))
    script = """
set -e
%s
%s - <<'PY'
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

source_path = Path(%r)
target_path = Path(%r)
meta_path = Path(%r)
model_path = %r
task_order = json.loads(%r)
completion_cap = %s
per_task_bucket = %s

rows = [
    json.loads(line)
    for line in source_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

def bucket(prompt_tokens: int) -> str:
    if prompt_tokens < 4096:
        return "0-4K"
    if prompt_tokens < 16384:
        return "4-16K"
    if prompt_tokens < 32768:
        return "16K-32K"
    if prompt_tokens < 131072:
        return "32K-128K"
    return "128K-160K"

def evenly_pick(items, k):
    if k is None:
        return list(items)
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    picked = []
    used = set()
    n = len(items)
    for i in range(k):
        pos = round(i * (n - 1) / (k - 1)) if k > 1 else n // 2
        while pos in used and pos + 1 < n:
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        if pos in used:
            continue
        used.add(pos)
        picked.append(items[pos])
    return picked

def find_repeat_token_id() -> tuple[int, str]:
    seed_texts = [" a", " hello", " test", " the", " word", " ok", ".", ",", " 1"]
    candidate_ids = []
    for text in seed_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in special_ids:
            candidate_ids.append(ids[0])
    vocab_cap = min(int(getattr(tokenizer, "vocab_size", 8192) or 8192), 8192)
    candidate_ids.extend(tid for tid in range(vocab_cap) if tid not in special_ids)

    seen = set()
    for tid in candidate_ids:
        if tid in seen or tid in special_ids:
            continue
        seen.add(tid)
        text = tokenizer.decode(
            [tid] * 64,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not text:
            continue
        if len(tokenizer.encode(text, add_special_tokens=False)) == 64:
            piece = tokenizer.decode(
                [tid],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return tid, piece
    raise RuntimeError("Failed to find a stable repeat token for synthetic speed outputs")

repeat_token_id, repeat_piece = find_repeat_token_id()

def build_response(target_tokens: int) -> tuple[str, int]:
    if target_tokens <= 0:
        return "", 0
    ids = [repeat_token_id] * target_tokens
    text = ""
    actual = 0
    for _ in range(8):
        text = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        if actual == target_tokens:
            return text, actual
        diff = target_tokens - actual
        if diff > 0:
            ids.extend([repeat_token_id] * diff)
        else:
            keep = max(1, len(ids) + diff)
            ids = ids[:keep]
    return text, actual

rows_by_task = {task: [] for task in task_order}
selected_rows = []
rows_by_task_bucket = {}
for row in rows:
    task = str(row.get("task", ""))
    rows_by_task.setdefault(task, []).append(row)
    bkt = bucket(int(row.get("prompt_tokens", 0)))
    rows_by_task_bucket.setdefault((task, bkt), []).append(row)

if per_task_bucket is not None:
    selected_rows = []
    for key, task_bucket_rows in sorted(rows_by_task_bucket.items()):
        task_bucket_rows = sorted(
            task_bucket_rows,
            key=lambda row: (
                int(row.get("prompt_tokens", 0)),
                int(row.get("completion_tokens", 0)),
                int(row.get("index", 0)),
            ),
        )
        selected_rows.extend(evenly_pick(task_bucket_rows, per_task_bucket))
else:
    selected_rows = list(rows)

rows_by_task = {task: [] for task in task_order}
for row in selected_rows:
    task = str(row.get("task", ""))
    rows_by_task.setdefault(task, []).append(row)

for task_rows in rows_by_task.values():
    task_rows.sort(
        key=lambda row: (
            int(row.get("prompt_tokens", 0)),
            int(row.get("completion_tokens", 0)),
            int(row.get("index", 0)),
        )
    )

ordered_rows = []
max_task_rows = max((len(task_rows) for task_rows in rows_by_task.values()), default=0)
for pos in range(max_task_rows):
    for task in task_order:
        task_rows = rows_by_task.get(task, [])
        if pos < len(task_rows):
            ordered_rows.append(task_rows[pos])

bench_rows = []
actual_completion_tokens = []
target_completion_tokens = []
for row in ordered_rows:
    completion_tokens_source = int(row.get("completion_tokens", 0))
    completion_tokens_target = completion_tokens_source
    if completion_cap is not None:
        completion_tokens_target = min(completion_tokens_target, int(completion_cap))
    model_response, completion_tokens_actual = build_response(completion_tokens_target)
    target_completion_tokens.append(completion_tokens_target)
    actual_completion_tokens.append(completion_tokens_actual)
    bench_rows.append(
        {
            "task": row.get("task"),
            "index": row.get("index"),
            "question": row.get("question", ""),
            "model_response": model_response,
            "prompt_tokens_source": int(row.get("prompt_tokens", 0)),
            "completion_tokens_source": completion_tokens_source,
            "completion_tokens_bench_target": completion_tokens_target,
            "completion_tokens_bench_actual": completion_tokens_actual,
        }
    )

target_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in bench_rows),
    encoding="utf-8",
)

meta = {
    "source": str(source_path),
    "model_path": model_path,
    "count": len(bench_rows),
    "task_order": list(task_order),
    "completion_cap": completion_cap,
    "per_task_bucket": per_task_bucket,
    "repeat_token_id": repeat_token_id,
    "repeat_piece_preview": repeat_piece[:32],
    "by_task": dict(sorted(Counter(str(row.get("task", "")) for row in bench_rows).items())),
    "by_bucket": dict(
        sorted(
            Counter(bucket(int(row.get("prompt_tokens_source", 0))) for row in bench_rows).items()
        )
    ),
    "prompt_tokens_total": sum(int(row.get("prompt_tokens_source", 0)) for row in bench_rows),
    "completion_tokens_source_total": sum(
        int(row.get("completion_tokens_source", 0)) for row in bench_rows
    ),
    "completion_tokens_bench_target_total": sum(target_completion_tokens),
    "completion_tokens_bench_actual_total": sum(actual_completion_tokens),
    "max_completion_delta": max(
        abs(target - actual)
        for target, actual in zip(target_completion_tokens, actual_completion_tokens)
    ) if bench_rows else 0,
}
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False))
PY
""" % (
        REMOTE_ENV,
        REMOTE_PYTHON,
        PUBLIC_SPEED_SOURCE,
        PUBLIC_SPEED_DATA,
        PUBLIC_SPEED_META,
        model_path,
        task_order_json,
        completion_cap_literal,
        per_task_bucket_literal,
    )
    proc = ssh(script)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def parse_fast_eval(output: str) -> tuple[float | None, int | None, int | None, int | None]:
    match = re.search(
        r"avg_score=([0-9.]+)%\s+pass=(\d+)\s+part=(\d+)\s+fail=(\d+)",
        output,
    )
    if not match:
        return None, None, None, None
    return float(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))


def parse_official_eval(output: str) -> tuple[float | None, float | None]:
    score_match = re.search(r"Average Score:\s*([0-9.]+)%", output)
    duration_match = re.search(r"Total Duration:\s*([0-9.]+)\s*s", output)
    avg_score = float(score_match.group(1)) if score_match else None
    duration_s = float(duration_match.group(1)) if duration_match else None
    return avg_score, duration_s


def parse_bench_json(output: str) -> dict | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument(
        "--quantization",
        choices=["none", "modelopt", "gptq", "gptq_marlin"],
        default="none",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--attention-backend", default="minicpm_flashinfer")
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument(
        "--profile",
        choices=[
            "smoke",
            "speed-smoke",
            "proxy",
            "public-speed",
            "mini",
            "mini-eval",
            "fast-eval",
            "medium-eval",
            "eval-only",
            "bench-only",
            "official",
        ],
        default="smoke",
    )
    parser.add_argument("--official-data-path", default="eval_dataset/perf_public_set.jsonl")
    parser.add_argument("--official-max-seq-len", type=int, default=524288)
    parser.add_argument("--official-concurrency", type=int, default=8)
    parser.add_argument("--official-num-samples", type=int, default=None)
    parser.add_argument("--disable-radix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dense-as-sparse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-topk", action="store_true")
    parser.add_argument("--split-stage1", action="store_true")
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument("--public-speed-completion-cap", type=int, default=None)
    parser.add_argument("--public-speed-per-task-bucket", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.profile == "fast-eval"
        and "--enable-request-time-stats-logging" not in args.extra_server_arg
    ):
        args.extra_server_arg.append("--enable-request-time-stats-logging")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.label)
    run_dir = f"{REMOTE_RESULTS_ROOT}/{timestamp}_{safe_label}"

    print(f"[bench] label={args.label}")
    print(f"[bench] run_dir={run_dir}")
    print(f"[bench] model_path={args.model_path}")

    pid = ""
    server_log = ""
    preflight = None
    fast_log = None
    mini_eval_log = None
    official_log = None
    bench_smoke_log = None
    bench_mini_log = None
    fast_avg = fast_pass = fast_part = fast_fail = None
    mini_eval_avg = mini_eval_pass = mini_eval_part = mini_eval_fail = None
    official_avg = None
    official_duration_s = None
    smoke_duration = None
    mini_duration = None

    try:
        preflight = run_preflight(args.port)
        print(
            "[bench] preflight "
            f"transport=ok "
            f"remote_idle={preflight['remote_idle']} "
            f"host={preflight['host']} "
            f"time={preflight['remote_time']} "
            f"gpu_mem_mib={preflight['gpu_memory_used_mib']} "
            f"gpu_util_pct={preflight['gpu_utilization_pct']} "
            f"active_jobs={len(preflight['active_heavy_jobs'])}"
        )
        if not preflight["remote_idle"]:
            print(
                "[bench] preflight blocked: remote is not idle enough for a serialized heavy run",
                file=sys.stderr,
            )
            print(json.dumps(preflight, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        if args.preflight_only:
            print(json.dumps({"label": args.label, "preflight": preflight}, ensure_ascii=False, indent=2))
            return 0

        if args.profile == "proxy":
            ensure_proxy_dataset()
        if args.profile == "fast-eval":
            ensure_fast_eval_dataset()
        if args.profile == "public-speed":
            public_speed_meta = ensure_public_speed_dataset(
                args.model_path,
                completion_cap=args.public_speed_completion_cap,
                per_task_bucket=args.public_speed_per_task_bucket,
            )
            print(
                "[bench] public_speed_meta="
                + json.dumps(public_speed_meta, ensure_ascii=False)
            )

        pid, server_log = start_server(args, run_dir)
        print(f"[bench] server pid={pid}")
        if not wait_for_server(args.port):
            print("[bench] server failed to become ready", file=sys.stderr)
            tail = ssh(f"tail -n 120 {shlex.quote(server_log)}", check=False)
            if tail.stdout:
                print(tail.stdout, file=sys.stderr)
            return 1

        if args.profile in {"smoke", "proxy", "mini", "eval-only"}:
            fast_out, fast_log = run_fast_eval(args.port, run_dir)
            fast_avg, fast_pass, fast_part, fast_fail = parse_fast_eval(fast_out)
            print(f"[bench] fast_eval avg={fast_avg} pass={fast_pass} part={fast_part} fail={fast_fail}")

        if args.profile == "mini-eval":
            mini_eval_out, mini_eval_log = run_mini_eval(args.port, run_dir)
            mini_eval_avg, mini_eval_pass, mini_eval_part, mini_eval_fail = parse_fast_eval(mini_eval_out)
            print(
                "[bench] mini_eval "
                f"avg={mini_eval_avg} pass={mini_eval_pass} "
                f"part={mini_eval_part} fail={mini_eval_fail}"
            )

        if args.profile == "fast-eval":
            task_caps = ",".join(
                f"{task}:{cap}" for task, cap in FAST_EVAL_COMPLETION_CAPS.items()
            )
            official_out, official_log = run_mini_eval(
                args.port,
                run_dir,
                eval_data=FAST_EVAL_DATA,
                mini_per_task=999,
                log_name="fast_eval.log",
                task_caps=task_caps,
                use_all_rows=True,
            )
            official_avg, official_pass, official_part, official_fail = parse_fast_eval(official_out)
            print(
                "[bench] fast_eval "
                f"avg={official_avg} pass={official_pass} "
                f"part={official_part} fail={official_fail}"
            )

        if args.profile in {"official", "medium-eval"}:
            official_data_path = args.official_data_path
            if args.profile == "medium-eval":
                official_data_path = "eval_dataset/perf_public_medium_eval_v1.jsonl"
            official_out, official_log = run_official_eval(
                args.port,
                run_dir,
                model_path=args.model_path,
                max_seq_len=args.official_max_seq_len,
                data_path=official_data_path,
                concurrency=args.official_concurrency,
                num_samples=args.official_num_samples,
            )
            official_avg, official_duration_s = parse_official_eval(official_out)
            print(
                "[bench] official_eval "
                f"avg={official_avg} duration_s={official_duration_s}"
            )

        if args.profile in {"smoke", "speed-smoke", "bench-only"}:
            bench_out, bench_smoke_log = run_bench(
                args.port,
                run_dir,
                s1=SMOKE_S1,
                s8=SMOKE_S8,
                smax=None,
                log_name="bench_smoke.log",
            )
            smoke_duration = parse_bench_json(bench_out)
            print(f"[bench] smoke_duration={smoke_duration}")

        if args.profile == "proxy":
            bench_out, bench_smoke_log = run_bench(
                args.port,
                run_dir,
                s1=PROXY_S1,
                s8=PROXY_S8,
                smax=PROXY_SMAX,
                log_name="bench_proxy.log",
            )
            smoke_duration = parse_bench_json(bench_out)
            print(f"[bench] proxy_duration={smoke_duration}")

        if args.profile == "public-speed":
            bench_out, bench_smoke_log = run_bench(
                args.port,
                run_dir,
                s1=PUBLIC_SPEED_DATA,
                s8=PUBLIC_SPEED_DATA,
                smax=PUBLIC_SPEED_DATA,
                log_name="bench_public_speed.log",
            )
            smoke_duration = parse_bench_json(bench_out)
            print(f"[bench] public_speed_duration={smoke_duration}")

        if args.profile == "mini":
            bench_out, bench_mini_log = run_bench(
                args.port,
                run_dir,
                s1=MINI_S1,
                s8=None,
                smax=None,
                log_name="bench_mini.log",
            )
            mini_duration = parse_bench_json(bench_out)
            print(f"[bench] mini_duration={mini_duration}")

    except subprocess.CalledProcessError as exc:
        failure = classify_ssh_failure(exc.returncode, exc.output, exc.stderr)
        print(
            f"[bench] remote command failed exit={exc.returncode}",
            file=sys.stderr,
        )
        print(
            f"[bench] transport_classification={failure['category']} retryable={failure['retryable']}",
            file=sys.stderr,
        )
        print(f"[bench] transport_detail={failure['detail']}", file=sys.stderr)
        if exc.output:
            print(f"[bench] stdout:\n{exc.output}", file=sys.stderr)
        if exc.stderr:
            print(f"[bench] stderr:\n{exc.stderr}", file=sys.stderr)
        return 1
    finally:
        if pid:
            stop_server(pid, args.port)

    summary = RunSummary(
        label=args.label,
        run_dir=run_dir,
        server_log=server_log,
        preflight=preflight,
        fast_eval_log=fast_log,
        mini_eval_log=mini_eval_log,
        official_eval_log=official_log,
        bench_smoke_log=bench_smoke_log,
        bench_mini_log=bench_mini_log,
        fast_avg_score=fast_avg,
        mini_eval_avg_score=mini_eval_avg,
        official_avg_score=official_avg,
        official_duration_s=official_duration_s,
        fast_pass=fast_pass,
        fast_part=fast_part,
        fast_fail=fast_fail,
        mini_eval_pass=mini_eval_pass,
        mini_eval_part=mini_eval_part,
        mini_eval_fail=mini_eval_fail,
        smoke_duration=smoke_duration,
        mini_duration=mini_duration,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
