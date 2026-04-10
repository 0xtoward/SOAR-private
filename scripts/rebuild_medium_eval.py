#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--per-task", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    meta = Path(args.meta)

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    selected: list[dict] = []
    for task in sorted(by_task):
        task_rows = sorted(
            by_task[task],
            key=lambda row: (int(row.get("prompt_tokens", 0)), int(row.get("index", 0))),
        )
        selected.extend(evenly_pick(task_rows, args.per_task))

    selected = sorted(
        selected,
        key=lambda row: (row.get("task", ""), int(row.get("prompt_tokens", 0)), int(row.get("index", 0))),
    )

    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )

    meta_payload = {
        "count": len(selected),
        "per_task": args.per_task,
        "by_task": dict(sorted(Counter(row["task"] for row in selected).items())),
        "by_bucket": dict(sorted(Counter(bucket(int(row.get("prompt_tokens", 0))) for row in selected).items())),
        "prompt_tokens_min": min(int(row.get("prompt_tokens", 0)) for row in selected) if selected else 0,
        "prompt_tokens_max": max(int(row.get("prompt_tokens", 0)) for row in selected) if selected else 0,
        "indices": [int(row["index"]) for row in sorted(selected, key=lambda row: int(row.get("index", 0)))],
    }
    meta.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta_payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
