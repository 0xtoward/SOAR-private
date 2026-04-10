#!/usr/bin/env python3
"""
Build a clean, token-aware calibration JSONL for MiniCPM-SALA quantization.

The output file stores one JSON object per sample and keeps both the original
prompt text and the token-sliced `calib_text` that should actually be used for
calibration.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_PUBLIC_QUOTAS = {
    "mcq": 20,
    "qa": 30,
    "niah": 30,
    "fwe": 8,
    "cwe": 8,
}

DEFAULT_OPEN_QUOTAS = {
    "cnn_dailymail": 16,
    "wikitext": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models")
    parser.add_argument(
        "--public-path",
        default="/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_16k_v1.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--public-quotas",
        default="",
        help="Optional comma-separated task quotas such as mcq:20,qa:30,niah:30,fwe:8,cwe:8",
    )
    parser.add_argument(
        "--open-quotas",
        default="",
        help="Optional comma-separated source quotas such as cnn_dailymail:8,wikitext:8,pg19:16",
    )
    parser.add_argument("--max-sample-tokens", type=int, default=16384)
    parser.add_argument("--head-tokens", type=int, default=2048)
    parser.add_argument("--middle-tokens", type=int, default=2048)
    parser.add_argument("--tail-tokens", type=int, default=12288)
    return parser.parse_args()


def parse_quota_map(raw: str, defaults: dict[str, int]) -> dict[str, int]:
    if not raw.strip():
        return dict(defaults)
    parsed: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        parsed[key.strip()] = int(value.strip())
    return parsed


def evenly_pick(items: list[dict], k: int) -> list[dict]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    positions = [int((idx + 0.5) * len(items) / k) for idx in range(k)]
    picked = []
    seen = set()
    for pos in positions:
        pos = min(len(items) - 1, max(0, pos))
        while pos in seen and pos + 1 < len(items):
            pos += 1
        seen.add(pos)
        picked.append(items[pos])
    return picked


def tokenize_length(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def bucket_for_tokens(num_tokens: int) -> str:
    if num_tokens < 4096:
        return "0-4K"
    if num_tokens < 16384:
        return "4K-16K"
    if num_tokens < 32768:
        return "16K-32K"
    if num_tokens < 131072:
        return "32K-128K"
    return "128K-160K"


def slice_token_ids(
    token_ids: list[int],
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
) -> tuple[list[int], str]:
    if len(token_ids) <= max_tokens:
        return token_ids, "full"

    tail = min(tail_tokens, max_tokens)
    remaining = max_tokens - tail
    head = min(head_tokens, max(0, remaining))
    remaining -= head
    middle = min(middle_tokens, max(0, remaining))
    remaining -= middle
    head += remaining

    selected = []
    if head > 0:
        selected.extend(token_ids[:head])
    if middle > 0:
        start = max(head, (len(token_ids) - middle) // 2)
        end = min(len(token_ids) - tail, start + middle)
        selected.extend(token_ids[start:end])
    if tail > 0:
        selected.extend(token_ids[-tail:])
    return selected, "head_middle_tail"


def make_record(
    *,
    tokenizer,
    text: str,
    source: str,
    source_id: str,
    task: str,
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
) -> dict:
    raw_token_ids = tokenizer.encode(text, add_special_tokens=False)
    sliced_ids, slice_mode = slice_token_ids(
        raw_token_ids,
        max_tokens=max_tokens,
        head_tokens=head_tokens,
        middle_tokens=middle_tokens,
        tail_tokens=tail_tokens,
    )
    calib_text = tokenizer.decode(sliced_ids, skip_special_tokens=True)
    raw_prompt_tokens = len(raw_token_ids)
    calib_prompt_tokens = len(sliced_ids)
    return {
        "source": source,
        "source_id": source_id,
        "task": task,
        "text": text,
        "calib_text": calib_text,
        "raw_prompt_tokens": raw_prompt_tokens,
        "calib_prompt_tokens": calib_prompt_tokens,
        "bucket": bucket_for_tokens(raw_prompt_tokens),
        "slice_mode": slice_mode,
    }


def load_public_examples(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            examples.append(item)
    return examples


def collect_public_records(
    *,
    tokenizer,
    public_examples: list[dict],
    quotas: dict[str, int],
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in public_examples:
        by_task[item["task"]].append(item)

    records: list[dict] = []
    for task, quota in quotas.items():
        items = sorted(
            by_task.get(task, []),
            key=lambda item: item.get("prompt_tokens", 0),
        )
        for item in evenly_pick(items, quota):
            records.append(
                make_record(
                    tokenizer=tokenizer,
                    text=item["question"],
                    source="soar_public",
                    source_id=f"public_{item['index']}",
                    task=task,
                    max_tokens=max_tokens,
                    head_tokens=head_tokens,
                    middle_tokens=middle_tokens,
                    tail_tokens=tail_tokens,
                )
            )
    return records


def iter_cnn_dailymail(seed: int) -> Iterable[str]:
    start = seed % 1024
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"train[{start}:{start + 128}]")
    dataset = dataset.shuffle(seed=seed)
    for item in dataset:
        article = (item.get("article") or "").strip()
        if len(article) < 2000:
            continue
        yield (
            "Read the following news article and keep track of the important facts.\n\n"
            f"Article:\n{article}\n\n"
            "Question: What is the central event described above?\nAnswer:"
        )


def iter_wikitext(seed: int) -> Iterable[str]:
    start = 4096 + (seed % 2048)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[{start}:{start + 512}]")
    dataset = dataset.shuffle(seed=seed)
    buffer: list[str] = []
    for item in dataset:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        buffer.append(text)
        joined = "\n".join(buffer)
        if len(joined) < 5000:
            continue
        yield (
            "Read the following reference passage and keep track of the factual details.\n\n"
            f"Passage:\n{joined}\n\n"
            "Question: What is the passage mainly about?\nAnswer:"
        )
        buffer = []


def iter_pg19(seed: int) -> Iterable[str]:
    start = seed % 256
    dataset = load_dataset("pg19", split=f"train[{start}:{start + 32}]")
    dataset = dataset.shuffle(seed=seed)
    for item in dataset:
        text = (item.get("text") or "").strip()
        if len(text) < 30000:
            continue
        yield (
            "Read the following long-form passage and keep track of the important entities, "
            "facts, and narrative state.\n\n"
            f"Passage:\n{text}\n\n"
            "Question: What is the passage mainly about?\nAnswer:"
        )


OPEN_SOURCE_ITERATORS = {
    "cnn_dailymail": iter_cnn_dailymail,
    "wikitext": iter_wikitext,
    "pg19": iter_pg19,
}


def collect_open_records(
    *,
    tokenizer,
    quotas: dict[str, int],
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
    seed: int,
) -> list[dict]:
    records: list[dict] = []
    for source, quota in quotas.items():
        if quota <= 0:
            continue
        iterator = OPEN_SOURCE_ITERATORS[source](seed)
        for idx, text in enumerate(iterator, start=1):
            records.append(
                make_record(
                    tokenizer=tokenizer,
                    text=text,
                    source=source,
                    source_id=f"{source}_{idx}",
                    task="open_text",
                    max_tokens=max_tokens,
                    head_tokens=head_tokens,
                    middle_tokens=middle_tokens,
                    tail_tokens=tail_tokens,
                )
            )
            if idx >= quota:
                break
    return records


def dedupe_records(records: list[dict]) -> list[dict]:
    unique = []
    seen = set()
    for record in records:
        key = record["calib_text"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def summarize(records: list[dict]) -> None:
    by_source: dict[str, int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    token_lengths = []
    for record in records:
        by_source[record["source"]] += 1
        by_task[record["task"]] += 1
        token_lengths.append(record["calib_prompt_tokens"])
    token_lengths.sort()
    p50 = token_lengths[len(token_lengths) // 2]
    p90 = token_lengths[min(len(token_lengths) - 1, int(len(token_lengths) * 0.9))]
    print("records", len(records))
    print("by_source", dict(sorted(by_source.items())))
    print("by_task", dict(sorted(by_task.items())))
    print("tokens", {"p50": p50, "p90": p90, "max": token_lengths[-1]})


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    public_quotas = parse_quota_map(args.public_quotas, DEFAULT_PUBLIC_QUOTAS)
    open_quotas = parse_quota_map(args.open_quotas, DEFAULT_OPEN_QUOTAS)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    public_examples = load_public_examples(Path(args.public_path))
    records = collect_public_records(
        tokenizer=tokenizer,
        public_examples=public_examples,
        quotas=public_quotas,
        max_tokens=args.max_sample_tokens,
        head_tokens=args.head_tokens,
        middle_tokens=args.middle_tokens,
        tail_tokens=args.tail_tokens,
    )
    records.extend(
        collect_open_records(
            tokenizer=tokenizer,
            quotas=open_quotas,
            max_tokens=args.max_sample_tokens,
            head_tokens=args.head_tokens,
            middle_tokens=args.middle_tokens,
            tail_tokens=args.tail_tokens,
            seed=args.seed,
        )
    )
    records = dedupe_records(records)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    summarize(records)
    print("output_path", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
