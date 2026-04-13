#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SOAR eval predictions.jsonl files.")
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions_path)
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"No rows found in {predictions_path}")

    by_task = defaultdict(lambda: {"count": 0, "score_sum": 0.0, "output_tokens": 0})
    summary_rows = []
    total_score = 0.0
    total_output_tokens = 0
    for row in rows:
        task = row.get("task", "<unknown>")
        score = float(row.get("score", 0.0))
        output_tokens = int(row.get("output_tokens", 0))
        input_tokens = int(row.get("input_tokens", 0))
        total_score += score
        total_output_tokens += output_tokens
        by_task[task]["count"] += 1
        by_task[task]["score_sum"] += score
        by_task[task]["output_tokens"] += output_tokens
        summary_rows.append(
            {
                "index": row.get("index"),
                "task": task,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "score": score,
            }
        )

    task_rows = []
    for task, item in sorted(by_task.items()):
        task_rows.append(
            {
                "task": task,
                "count": item["count"],
                "avg_score": item["score_sum"] / item["count"],
                "output_tokens": item["output_tokens"],
            }
        )

    summary_rows.sort(key=lambda row: (row["task"], row["input_tokens"], row["index"]))
    payload = {
        "predictions_path": str(predictions_path),
        "count": len(rows),
        "avg_score": total_score / len(rows),
        "total_output_tokens": total_output_tokens,
        "tasks": task_rows,
        "rows": summary_rows,
        "by_output_tokens_desc": sorted(summary_rows, key=lambda row: row["output_tokens"], reverse=True),
        "by_score_asc": sorted(summary_rows, key=lambda row: (row["score"], -row["output_tokens"])),
        "task_counts": dict(sorted(Counter(row["task"] for row in summary_rows).items())),
    }
    if args.output_path:
        Path(args.output_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
