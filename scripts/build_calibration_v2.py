#!/usr/bin/env python3
"""
Build a more semantics-heavy calibration set for MiniCPM-SALA quantization.

Goals:
- Keep the pool bounded and reproducible.
- Reuse existing local calibration artifacts first.
- Downweight low-semantic tasks like fwe/cwe.
- Add a small number of long, more continuous-text samples from merged raw prompts.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


SEMANTIC_EXCLUDE_PATTERNS = [
    "uuid",
    "coded text",
    "coded word",
    "frequency of each coded word",
    "most frequently appeared coded",
    "numbered list of words",
    "memorize the ones that appear more often",
    "random single-digit",
    "random string",
    "track the frequency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-jsonl",
        default="/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v1.jsonl",
    )
    parser.add_argument(
        "--merged-jsonl",
        default="/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_data_merged.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/Users/ql/cursor/openbmb/submission-drafts/nvfp4-mlp-only-draft/calibration/calibration_curated_16k_v2.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-samples", type=int, default=128)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def char_head_mid_tail(text: str, target_chars: int = 64000) -> str:
    text = text.strip()
    if len(text) <= target_chars:
        return text

    head = int(target_chars * 0.15)
    middle = int(target_chars * 0.20)
    tail = target_chars - head - middle
    center = max(0, (len(text) // 2) - (middle // 2))

    return (
        text[:head].rstrip()
        + "\n\n[...snip...]\n\n"
        + text[center : center + middle].strip()
        + "\n\n[...snip...]\n\n"
        + text[-tail:].lstrip()
    )


def is_semantic_long_text(text: str) -> bool:
    lower = text.lower()
    if len(text) < 12000:
        return False
    return not any(p in lower for p in SEMANTIC_EXCLUDE_PATTERNS)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    base_rows = load_jsonl(Path(args.base_jsonl))
    merged_rows = load_jsonl(Path(args.merged_jsonl))

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in base_rows:
        by_task[row.get("task", "unknown")].append(row)

    # Favor longer representatives inside each task bucket.
    for rows in by_task.values():
        rows.sort(key=lambda x: int(x.get("calib_prompt_tokens") or 0), reverse=True)

    selected: list[dict] = []
    selected.extend(by_task.get("qa", [])[:30])
    selected.extend(by_task.get("niah", [])[:30])
    selected.extend(by_task.get("open_text", [])[:32])
    selected.extend(by_task.get("mcq", [])[:20])

    seen_texts = {row.get("calib_text") or row.get("text") or "" for row in selected}
    extra_needed = max(0, args.total_samples - len(selected))

    semantic_candidates: list[dict] = []
    for idx, row in enumerate(merged_rows):
        text = str(row.get("text") or "").strip()
        if not text or not is_semantic_long_text(text):
            continue
        calib_text = char_head_mid_tail(text)
        if not calib_text or calib_text in seen_texts:
            continue
        semantic_candidates.append(
            {
                "task": "open_text",
                "source": "merged_semantic_v2",
                "source_id": idx,
                "slice_mode": "head_mid_tail_char",
                "text": text,
                "calib_text": calib_text,
                "raw_prompt_tokens": None,
                "calib_prompt_tokens": None,
            }
        )

    rng.shuffle(semantic_candidates)
    selected.extend(semantic_candidates[:extra_needed])

    if len(selected) < args.total_samples:
        raise SystemExit(
            f"only built {len(selected)} samples, expected {args.total_samples}; "
            "not enough semantic candidates"
        )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in selected[: args.total_samples]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for row in selected[: args.total_samples]:
        counts[row.get("task", "unknown")] += 1

    print(f"wrote={out_path}")
    print(f"num_rows={args.total_samples}")
    print(f"task_counts={dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
