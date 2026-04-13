#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


STOP_MARKERS = (
    "<|im_end|>",
    "</s>",
    "<|eot_id|>",
    "<|endoftext|>",
)

SYSTEM_MARKERS = (
    "<|system|>",
    "<|im_start|>system",
    "\nsystem\n",
)

ASSISTANT_MARKERS = (
    "<|assistant|>",
    "<|im_start|>assistant",
    "\nAssistant:",
    "\nassistant:",
    "\nANSWER:",
    "\nAnswer:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a GPTQ calibration JSONL dataset.")
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--jsonl-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--summary-out", required=True)
    return parser.parse_args()


def pctl(values: list[int], frac: float) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, max(0, math.floor((len(values) - 1) * frac)))
    return values[idx]


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def contains_system_prompt_text(text: str) -> bool:
    stripped = text.lstrip()
    return contains_any(text, SYSTEM_MARKERS) or stripped.startswith("You are ")


def classify_template_type(record: dict[str, Any]) -> str:
    source = record.get("source", "")
    task = record.get("task", "")
    if source == "soar_public":
        return f"public_{task}_plain"
    if source == "soar_chat_close":
        return f"chat_close_{task}"
    if source == "open_long_semantic_tail":
        template_idx = record.get("semantic_template_index", 0)
        bucket_target = record.get("bucket_target", "")
        return f"semantic_tail_t{template_idx}_{bucket_target}"
    if source == "pg19_local":
        return f"pg19_{record.get('slice_mode', 'unknown')}"
    if source == "synthetic_ultralong":
        return "synthetic_ultralong"
    return f"{source or 'unknown'}_{task or 'unknown'}"


def classify_bucket(record: dict[str, Any], token_len: int, truncated_token_len: int, text: str) -> str:
    source = record.get("source", "")
    task = record.get("task", "")
    record_bucket = str(record.get("bucket", ""))
    has_stop = contains_any(text, STOP_MARKERS)
    has_assistant = contains_any(text, ASSISTANT_MARKERS)
    if "```" in text or "\ndef " in text or "\nclass " in text or "\nimport " in text:
        return "code"
    if source == "soar_chat_close":
        return "short_close"
    if has_assistant or has_stop:
        return "chat" if truncated_token_len > 2048 else "short_close"
    if task == "qa":
        return "qa"
    if token_len >= 4096 or record_bucket in {"4K-16K", "16K-32K", "32K-128K", "128K-160K"}:
        return "long_context"
    return "other"


def to_last_20_tokens(tokenizer, token_ids: list[int]) -> list[str]:
    tail_ids = token_ids[-20:]
    try:
        return tokenizer.convert_ids_to_tokens(tail_ids)
    except Exception:
        return [str(x) for x in tail_ids]


def build_row(tokenizer, row_idx: int, record: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(record.get("text") or record.get("calib_text") or "")
    calib_text = str(record.get("calib_text") or raw_text)
    raw_token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
    calib_token_ids = tokenizer.encode(calib_text, add_special_tokens=False)
    contains_system_prompt = contains_system_prompt_text(calib_text)
    contains_assistant_prefix = contains_any(calib_text, ASSISTANT_MARKERS)
    contains_eos_or_stop_marker = contains_any(calib_text, STOP_MARKERS)
    bucket = classify_bucket(record, len(raw_token_ids), len(calib_token_ids), calib_text)
    return {
        "index": row_idx,
        "source_id": record.get("source_id"),
        "task": record.get("task"),
        "source": record.get("source"),
        "template_type": classify_template_type(record),
        "slice_mode": record.get("slice_mode"),
        "raw_text_char_len": len(raw_text),
        "token_len": len(raw_token_ids),
        "truncated_token_len": len(calib_token_ids),
        "contains_system_prompt": contains_system_prompt,
        "contains_assistant_prefix": contains_assistant_prefix,
        "contains_eos_or_stop_marker": contains_eos_or_stop_marker,
        "last_50_chars": calib_text[-50:],
        "last_20_tokens": to_last_20_tokens(tokenizer, calib_token_ids),
        "bucket": bucket,
        "original_bucket": record.get("bucket"),
        "bucket_target": record.get("bucket_target"),
        "base_book_id": record.get("book_id"),
        "short_book_title": record.get("short_book_title"),
        "semantic_template_index": record.get("semantic_template_index"),
        "body_token_count": record.get("body_token_count"),
    }


def detect_distribution_shift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(rows))
    bucket_counts = Counter(row["bucket"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    stop_coverage = sum(1 for row in rows if row["contains_eos_or_stop_marker"]) / total
    assistant_coverage = sum(1 for row in rows if row["contains_assistant_prefix"]) / total
    long_share = bucket_counts.get("long_context", 0) / total
    short_close_share = bucket_counts.get("short_close", 0) / total
    chat_share = bucket_counts.get("chat", 0) / total

    flags: list[str] = []
    if stop_coverage < 0.1:
        flags.append("low_stop_marker_coverage")
    if assistant_coverage < 0.1:
        flags.append("low_assistant_prefix_coverage")
    if source_counts.get("open_long_semantic_tail", 0) / total > 0.2:
        flags.append("open_semantic_tail_share_gt_20pct")
    if long_share > 0.7 and short_close_share < 0.05:
        flags.append("heavy_long_context_bias_with_few_short_closing_rows")
    if bucket_counts.get("qa", 0) / total < 0.1:
        flags.append("low_direct_qa_bucket_share")

    return {
        "likely_shift_flags": flags,
        "likely_shift_assessment": (
            "likely_shifted_away_from_answer_conditioned_close_states" if flags else "no_strong_shift_flag"
        ),
        "long_context_share": round(long_share, 4),
        "short_close_share": round(short_close_share, 4),
        "chat_share": round(chat_share, 4),
        "stop_marker_coverage_share": round(stop_coverage, 4),
        "assistant_prefix_coverage_share": round(assistant_coverage, 4),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_lens = sorted(row["token_len"] for row in rows)
    trunc_lens = sorted(row["truncated_token_len"] for row in rows)
    bucket_counts = Counter(row["bucket"] for row in rows)
    source_counts = Counter(row["source"] for row in rows)
    task_counts = Counter(row["task"] for row in rows)
    template_counts = Counter(row["template_type"] for row in rows)
    total = max(1, len(rows))
    summary = {
        "count": len(rows),
        "token_len_distribution": {
            "min": token_lens[0] if token_lens else 0,
            "p50": pctl(token_lens, 0.50),
            "p90": pctl(token_lens, 0.90),
            "p95": pctl(token_lens, 0.95),
            "max": token_lens[-1] if token_lens else 0,
            "mean": round(statistics.mean(token_lens), 2) if token_lens else 0,
        },
        "truncated_token_len_distribution": {
            "min": trunc_lens[0] if trunc_lens else 0,
            "p50": pctl(trunc_lens, 0.50),
            "p90": pctl(trunc_lens, 0.90),
            "p95": pctl(trunc_lens, 0.95),
            "max": trunc_lens[-1] if trunc_lens else 0,
            "mean": round(statistics.mean(trunc_lens), 2) if trunc_lens else 0,
        },
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "bucket_share": {k: round(v / total, 4) for k, v in sorted(bucket_counts.items())},
        "source_counts": dict(sorted(source_counts.items())),
        "source_share": {k: round(v / total, 4) for k, v in sorted(source_counts.items())},
        "task_counts": dict(sorted(task_counts.items())),
        "task_share": {k: round(v / total, 4) for k, v in sorted(task_counts.items())},
        "template_counts": dict(sorted(template_counts.items())),
        "eos_stop_coverage": {
            "count": sum(1 for row in rows if row["contains_eos_or_stop_marker"]),
            "share": round(sum(1 for row in rows if row["contains_eos_or_stop_marker"]) / total, 4),
        },
        "assistant_prefix_coverage": {
            "count": sum(1 for row in rows if row["contains_assistant_prefix"]),
            "share": round(sum(1 for row in rows if row["contains_assistant_prefix"]) / total, 4),
        },
        "system_prompt_coverage": {
            "count": sum(1 for row in rows if row["contains_system_prompt"]),
            "share": round(sum(1 for row in rows if row["contains_system_prompt"]) / total, 4),
        },
        "long_vs_short_close": {
            "long_context_count": bucket_counts.get("long_context", 0),
            "short_close_count": bucket_counts.get("short_close", 0),
            "chat_count": bucket_counts.get("chat", 0),
            "ratio_long_to_short_close": None
            if bucket_counts.get("short_close", 0) == 0
            else round(bucket_counts.get("long_context", 0) / bucket_counts.get("short_close", 1), 4),
        },
    }
    summary.update(detect_distribution_shift(rows))
    return summary


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
    )

    calib_path = Path(args.calibration_path)
    records = []
    with calib_path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            records.append(build_row(tokenizer, row_idx, json.loads(line)))

    jsonl_out = Path(args.jsonl_out)
    csv_out = Path(args.csv_out)
    summary_out = Path(args.summary_out)
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_out.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    fieldnames = [
        "index",
        "source_id",
        "task",
        "source",
        "template_type",
        "slice_mode",
        "raw_text_char_len",
        "token_len",
        "truncated_token_len",
        "contains_system_prompt",
        "contains_assistant_prefix",
        "contains_eos_or_stop_marker",
        "last_50_chars",
        "last_20_tokens",
        "bucket",
        "original_bucket",
        "bucket_target",
        "base_book_id",
        "short_book_title",
        "semantic_template_index",
        "body_token_count",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            dumped = dict(row)
            dumped["last_20_tokens"] = json.dumps(dumped["last_20_tokens"], ensure_ascii=False)
            writer.writerow(dumped)

    summary = build_summary(records)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "count": len(records),
        "jsonl_out": str(jsonl_out),
        "csv_out": str(csv_out),
        "summary_out": str(summary_out),
        "summary": summary,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
