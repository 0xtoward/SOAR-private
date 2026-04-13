#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce long calibration rows with progressively smaller keep ratios."
    )
    parser.add_argument("--input", required=True, help="Input calibration jsonl path.")
    parser.add_argument("--output", required=True, help="Output calibration jsonl path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=int,
        default=32768,
        help="Keep all rows with calib_prompt_tokens <= threshold.",
    )
    parser.add_argument(
        "--band-size",
        type=int,
        default=8192,
        help="Band size above the threshold.",
    )
    parser.add_argument(
        "--base-keep-ratio",
        type=float,
        default=0.5,
        help="Keep ratio for the first band above threshold.",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.5,
        help="Per-band multiplicative decay applied to the keep ratio.",
    )
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.125,
        help="Lower bound for keep ratio on the longest bands.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def length_band(tokens: int, threshold: int, band_size: int) -> str:
    if tokens <= threshold:
        return f"<= {threshold}"
    start = threshold + 1 + ((tokens - threshold - 1) // band_size) * band_size
    end = start + band_size - 1
    return f"{start}-{end}"


def keep_ratio(tokens: int, threshold: int, band_size: int, base_keep_ratio: float, decay: float, min_keep_ratio: float) -> float:
    if tokens <= threshold:
        return 1.0
    band_index = (tokens - threshold - 1) // band_size
    ratio = base_keep_ratio * (decay ** band_index)
    return max(min_keep_ratio, ratio)


def summarize(rows: list[dict], threshold: int, band_size: int) -> dict:
    by_source = Counter(row.get("source", "<unknown>") for row in rows)
    by_task = Counter(row.get("task", "<unknown>") for row in rows)
    by_bucket = Counter(row.get("bucket", "<unknown>") for row in rows)
    by_band = Counter(
        length_band(int(row.get("calib_prompt_tokens", 0)), threshold, band_size)
        for row in rows
    )
    calib_tokens = sorted(int(row.get("calib_prompt_tokens", 0)) for row in rows)
    return {
        "count": len(rows),
        "by_source": dict(sorted(by_source.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_length_band": dict(sorted(by_band.items())),
        "calib_tokens": {
            "p50": calib_tokens[len(calib_tokens) // 2] if calib_tokens else 0,
            "p90": calib_tokens[min(len(calib_tokens) - 1, int(len(calib_tokens) * 0.9))] if calib_tokens else 0,
            "max": calib_tokens[-1] if calib_tokens else 0,
        },
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_rows(input_path)
    if not rows:
        raise SystemExit(f"Input calibration is empty: {input_path}")

    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        tokens = int(row.get("calib_prompt_tokens", 0))
        band = length_band(tokens, args.threshold, args.band_size)
        key = (
            row.get("source", "<unknown>"),
            row.get("task", "<unknown>"),
            row.get("bucket", "<unknown>"),
            band,
        )
        grouped[key].append(row)

    selected: list[dict] = []
    reduction_report: list[dict] = []
    for key, group_rows in sorted(grouped.items()):
        tokens = int(group_rows[0].get("calib_prompt_tokens", 0))
        ratio = keep_ratio(
            tokens=tokens,
            threshold=args.threshold,
            band_size=args.band_size,
            base_keep_ratio=args.base_keep_ratio,
            decay=args.decay,
            min_keep_ratio=args.min_keep_ratio,
        )
        if ratio >= 1.0:
            kept = list(group_rows)
        else:
            keep_n = max(1, math.floor(len(group_rows) * ratio))
            keep_n = min(keep_n, len(group_rows))
            shuffled = list(group_rows)
            rng.shuffle(shuffled)
            kept = shuffled[:keep_n]
        selected.extend(kept)
        reduction_report.append(
            {
                "key": list(key),
                "input_rows": len(group_rows),
                "kept_rows": len(kept),
                "keep_ratio_target": ratio,
            }
        )

    selected.sort(
        key=lambda row: (
            row.get("source", ""),
            row.get("task", ""),
            int(row.get("calib_prompt_tokens", 0)),
            row.get("question_id", row.get("index", "")),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "seed": args.seed,
        "threshold": args.threshold,
        "band_size": args.band_size,
        "base_keep_ratio": args.base_keep_ratio,
        "decay": args.decay,
        "min_keep_ratio": args.min_keep_ratio,
        "input_summary": summarize(rows, args.threshold, args.band_size),
        "output_summary": summarize(selected, args.threshold, args.band_size),
        "reduction_report": reduction_report,
    }
    (output_path.with_suffix(output_path.suffix + ".meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(meta["output_summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
