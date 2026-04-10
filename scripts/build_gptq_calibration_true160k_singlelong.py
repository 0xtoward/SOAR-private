#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--meta", required=True)
    return p.parse_args()


def percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, int(len(sorted_vals) * q))
    return sorted_vals[idx]


def main() -> int:
    args = parse_args()
    src = Path(args.input)
    out = Path(args.output)
    meta = Path(args.meta)

    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    long_rows = [r for r in rows if r.get("bucket") == "128K-160K"]
    keep_rows = [r for r in rows if r.get("bucket") != "128K-160K"]

    if not long_rows:
        raise SystemExit("no 128K-160K rows found")

    longs_by_tokens = sorted(long_rows, key=lambda r: int(r.get("raw_prompt_tokens", 0)))
    chosen = longs_by_tokens[len(longs_by_tokens) // 2]
    keep_rows.append(chosen)

    # Stable ordering by original row order from source.
    key_to_idx = {id(row): idx for idx, row in enumerate(rows)}
    keep_rows.sort(key=lambda row: key_to_idx[id(row)])

    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep_rows), encoding="utf-8")

    raws = sorted(int(r.get("raw_prompt_tokens", 0)) for r in keep_rows)
    calibs = sorted(int(r.get("calib_prompt_tokens", 0)) for r in keep_rows)
    meta_payload = {
        "count": len(keep_rows),
        "bucket_counts": dict(sorted(Counter(r.get("bucket", "<unknown>") for r in keep_rows).items())),
        "source_counts": dict(sorted(Counter(r.get("source", "<unknown>") for r in keep_rows).items())),
        "task_counts": dict(sorted(Counter(r.get("task", "<unknown>") for r in keep_rows).items())),
        "raw_tokens": {
            "p50": percentile(raws, 0.5),
            "p90": percentile(raws, 0.9),
            "max": raws[-1] if raws else 0,
        },
        "calib_tokens": {
            "p50": percentile(calibs, 0.5),
            "p90": percentile(calibs, 0.9),
            "max": calibs[-1] if calibs else 0,
        },
        "kept_long_row": {
            "source": chosen.get("source"),
            "task": chosen.get("task"),
            "raw_prompt_tokens": chosen.get("raw_prompt_tokens"),
            "calib_prompt_tokens": chosen.get("calib_prompt_tokens"),
        },
    }
    meta.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
