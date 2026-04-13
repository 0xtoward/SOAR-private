#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_SPEC = "mcq:103,167,202,344,589;cwe:31433,31697,63430,127483,127738"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed eval subset by task + prompt_tokens."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument(
        "--spec",
        default=DEFAULT_SPEC,
        help="Semicolon-separated task specs such as mcq:103,167;cwe:31433,31697",
    )
    return parser.parse_args()


def parse_spec(raw: str) -> list[tuple[str, int]]:
    targets: list[tuple[str, int]] = []
    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        task, values = group.split(":", 1)
        for item in values.split(","):
            item = item.strip()
            if item:
                targets.append((task.strip(), int(item)))
    if not targets:
        raise SystemExit("Focus subset spec resolved to zero targets")
    return targets


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    meta = Path(args.meta)
    targets = parse_spec(args.spec)

    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = {(task, prompt_tokens): None for task, prompt_tokens in targets}
    for row in rows:
        key = (row.get("task"), int(row.get("prompt_tokens", -1)))
        if key in wanted and wanted[key] is None:
            wanted[key] = row

    missing = [f"{task}:{prompt_tokens}" for (task, prompt_tokens), row in wanted.items() if row is None]
    if missing:
        raise SystemExit(f"Missing focus subset rows: {', '.join(missing)}")

    selected = [wanted[(task, prompt_tokens)] for task, prompt_tokens in targets]
    output.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    meta_payload = {
        "source": str(source),
        "count": len(selected),
        "spec": args.spec,
        "indices": [int(row["index"]) for row in selected],
        "tasks": [row["task"] for row in selected],
        "prompt_tokens": [int(row["prompt_tokens"]) for row in selected],
        "by_task": dict(sorted(Counter(row["task"] for row in selected).items())),
    }
    meta.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta_payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
