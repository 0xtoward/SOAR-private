#!/usr/bin/env python3
"""
Build a GPTQ-oriented calibration JSONL for MiniCPM-SALA.

This builder intentionally differs from the older NVFP4-focused calibration:
- keep all public task families represented
- downweight low-semantic-signal fwe/cwe samples
- add semantic open-text rows from offline local sources when available
- support coherent long-book slices instead of synthetic ultra-long stitching

The output keeps both the raw text and the token-sliced calibration text.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
import zlib

from transformers import AutoTokenizer


DEFAULT_PUBLIC_QUOTAS = {
    "mcq": 12,
    "qa": 14,
    "niah": 14,
    "fwe": 4,
    "cwe": 4,
}

DEFAULT_OPEN_QUOTAS = {
    "pg19_local": 16,
}

DEFAULT_CHAT_CLOSE_QUOTAS: dict[str, int] = {}
DEFAULT_SEMANTIC_TAIL_QUOTAS: dict[str, int] = {}
SEMANTIC_TAIL_TEMPLATES = (
    "Answer the question based on the given passage. Only give me the answer and do not output any other words.\n"
    "Question: Based on the passage above, what is happening now, which entities or concepts matter most, and what details are most likely to be queried next?",
    "Answer the question based on the given passage. Only give me the answer and do not output any other words.\n"
    "Question: Summarize the key facts, relationships, and most recent developments in the passage above.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models")
    parser.add_argument(
        "--public-path",
        default="/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_v1.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--public-quotas",
        default="",
        help="Optional comma-separated task quotas such as mcq:10,qa:12,niah:12,fwe:4,cwe:4",
    )
    parser.add_argument(
        "--open-quotas",
        default="",
        help="Optional comma-separated source quotas such as pg19_local:16,synthetic_ultralong:8",
    )
    parser.add_argument(
        "--chat-close-quotas",
        default="",
        help="Optional comma-separated task quotas for chat-templated short-answer closing samples such as mcq:4,qa:4,niah:4,fwe:2,cwe:2",
    )
    parser.add_argument(
        "--semantic-tail-quotas",
        default="",
        help="Optional comma-separated bucket quotas such as 4-16K:18,128K-160K:30 for open-source semantic-tail long-form samples.",
    )
    parser.add_argument("--max-sample-tokens", type=int, default=32768)
    parser.add_argument("--head-tokens", type=int, default=4096)
    parser.add_argument("--middle-tokens", type=int, default=4096)
    parser.add_argument("--tail-tokens", type=int, default=24576)
    parser.add_argument(
        "--pg19-local-path",
        default="/root/autodl-tmp/SOAR-Toolkit/calibration_sources/pg19_train_books_v1.jsonl",
    )
    parser.add_argument("--pg19-min-book-tokens", type=int, default=32768)
    parser.add_argument(
        "--pg19-slice-strategy",
        choices=["contiguous", "head_middle_tail"],
        default="contiguous",
    )
    parser.add_argument("--ultralong-target-min", type=int, default=131072)
    parser.add_argument("--ultralong-target-max", type=int, default=155648)
    parser.add_argument("--chat-close-answer-max-chars", type=int, default=512)
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
    picked: list[dict] = []
    seen = set()
    for pos in positions:
        pos = min(len(items) - 1, max(0, pos))
        while pos in seen and pos + 1 < len(items):
            pos += 1
        seen.add(pos)
        picked.append(items[pos])
    return picked


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


def bucket_bounds(bucket_name: str) -> tuple[int, int]:
    mapping = {
        "0-4K": (0, 4095),
        "4-16K": (4096, 16383),
        "16K-32K": (16384, 32767),
        "32K-128K": (32768, 131071),
        "128K-160K": (131072, 163840),
    }
    if bucket_name not in mapping:
        raise ValueError(f"Unsupported bucket name: {bucket_name}")
    return mapping[bucket_name]


def slice_token_ids(
    token_ids: list[int],
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
    strategy: str = "head_middle_tail",
    slice_salt: str = "",
) -> tuple[list[int], str]:
    if len(token_ids) <= max_tokens:
        return token_ids, "full"

    if strategy == "contiguous":
        window = min(max_tokens, len(token_ids))
        max_start = len(token_ids) - window
        start = 0
        if max_start > 0:
            start = zlib.crc32(slice_salt.encode("utf-8")) % (max_start + 1)
        end = start + window
        return token_ids[start:end], "contiguous"

    tail = min(tail_tokens, max_tokens)
    remaining = max_tokens - tail
    head = min(head_tokens, max(0, remaining))
    remaining -= head
    middle = min(middle_tokens, max(0, remaining))
    remaining -= middle
    head += remaining

    selected: list[int] = []
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
    slice_strategy: str = "head_middle_tail",
    slice_salt: str = "",
    extra_fields: dict | None = None,
    skip_special_tokens: bool = True,
) -> dict:
    raw_token_ids = tokenizer.encode(text, add_special_tokens=False)
    sliced_ids, slice_mode = slice_token_ids(
        raw_token_ids,
        max_tokens=max_tokens,
        head_tokens=head_tokens,
        middle_tokens=middle_tokens,
        tail_tokens=tail_tokens,
        strategy=slice_strategy,
        slice_salt=slice_salt,
    )
    calib_text = tokenizer.decode(sliced_ids, skip_special_tokens=skip_special_tokens)
    raw_prompt_tokens = len(raw_token_ids)
    calib_prompt_tokens = len(sliced_ids)
    record = {
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
    if extra_fields:
        record.update(extra_fields)
    return record


def get_first_gold_answer(item: dict, *, max_chars: int) -> str | None:
    gold = item.get("gold")
    candidates: list[str] = []
    if isinstance(gold, list):
        candidates = [str(x).strip() for x in gold if str(x).strip()]
    elif gold is not None:
        text = str(gold).strip()
        if text:
            candidates = [text]
    if not candidates:
        return None
    answer = min(candidates, key=len)
    if len(answer) > max_chars:
        answer = answer[:max_chars].rstrip()
    return answer or None


def render_chat_close_text(tokenizer, item: dict, *, max_answer_chars: int) -> str | None:
    task = item.get("task", "unknown")
    answer = get_first_gold_answer(item, max_chars=max_answer_chars)
    if not answer:
        return None

    if task == "mcq":
        answer = f"ANSWER: {answer}"
    elif task in {"cwe", "fwe"} and isinstance(item.get("gold"), list):
        short_list = [str(x).strip() for x in item["gold"] if str(x).strip()]
        if short_list:
            answer = ", ".join(short_list[:10])
            if len(answer) > max_answer_chars:
                answer = answer[:max_answer_chars].rstrip()

    messages = [
        {"role": "user", "content": item["question"]},
        {"role": "assistant", "content": answer},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )


def load_public_examples(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            examples.append(json.loads(raw))
    return examples


def load_local_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing local JSONL source: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if isinstance(item, dict):
                rows.append(item)
    return rows


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
        items = sorted(by_task.get(task, []), key=lambda item: item.get("prompt_tokens", 0))
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


def collect_chat_close_records(
    *,
    tokenizer,
    public_examples: list[dict],
    quotas: dict[str, int],
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
    max_answer_chars: int,
) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in public_examples:
        by_task[item["task"]].append(item)

    records: list[dict] = []
    for task, quota in quotas.items():
        if quota <= 0:
            continue
        items = sorted(by_task.get(task, []), key=lambda item: item.get("prompt_tokens", 0))
        for item in evenly_pick(items, quota):
            chat_text = render_chat_close_text(
                tokenizer,
                item,
                max_answer_chars=max_answer_chars,
            )
            if not chat_text:
                continue
            records.append(
                make_record(
                    tokenizer=tokenizer,
                    text=chat_text,
                    source="soar_chat_close",
                    source_id=f"chat_close_{task}_{item['index']}",
                    task=task,
                    max_tokens=max_tokens,
                    head_tokens=head_tokens,
                    middle_tokens=middle_tokens,
                    tail_tokens=tail_tokens,
                    slice_salt=f"chat_close:{task}:{item['index']}",
                    extra_fields={
                        "chat_close": True,
                        "base_index": item["index"],
                    },
                    skip_special_tokens=False,
                )
            )
    return records


def iter_raw_long_paragraphs(
    *,
    public_examples: list[dict],
    seed: int,
) -> Iterable[str]:
    rng = random.Random(seed)
    public_rows = list(public_examples)
    rng.shuffle(public_rows)
    for item in public_rows:
        text = (item.get("question") or "").strip()
        if len(text) >= 800:
            yield text


def build_synthetic_ultralong_records(
    *,
    tokenizer,
    quota: int,
    public_examples: list[dict],
    seed: int,
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
    target_min_tokens: int,
    target_max_tokens: int,
) -> list[dict]:
    if quota <= 0:
        return []

    rng = random.Random(seed)
    separator = "\n\n==========\n\n"
    prefix = (
        "Read the following ultra-long multi-document context and keep track of the entities, "
        "facts, and discourse state across the full context.\n\nContext:\n"
    )
    suffix = "\n\nQuestion: What are the main topics and entities discussed above?\nAnswer:"

    records: list[dict] = []
    chunks: list[str] = []
    current_tokens = len(tokenizer.encode(prefix + suffix, add_special_tokens=False))

    for text in iter_raw_long_paragraphs(
        public_examples=public_examples,
        seed=seed,
    ):
        if len(records) >= quota:
            break
        chunk = text.strip()
        if not chunk:
            continue
        if rng.random() < 0.15:
            continue
        chunk_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
        if chunk_tokens < 128:
            continue

        sep_tokens = 0 if not chunks else len(tokenizer.encode(separator, add_special_tokens=False))
        projected = current_tokens + sep_tokens + chunk_tokens
        if chunks and projected > target_max_tokens:
            full_text = prefix + separator.join(chunks) + suffix
            records.append(
                make_record(
                    tokenizer=tokenizer,
                    text=full_text,
                    source="synthetic_ultralong",
                    source_id=f"synthetic_ultralong_{len(records) + 1}",
                    task="open_text",
                    max_tokens=max_tokens,
                    head_tokens=head_tokens,
                    middle_tokens=middle_tokens,
                    tail_tokens=tail_tokens,
                )
            )
            chunks = [chunk]
            current_tokens = len(tokenizer.encode(prefix + chunk + suffix, add_special_tokens=False))
            continue

        chunks.append(chunk)
        current_tokens = projected
        if current_tokens >= target_min_tokens:
            full_text = prefix + separator.join(chunks) + suffix
            records.append(
                make_record(
                    tokenizer=tokenizer,
                    text=full_text,
                    source="synthetic_ultralong",
                    source_id=f"synthetic_ultralong_{len(records) + 1}",
                    task="open_text",
                    max_tokens=max_tokens,
                    head_tokens=head_tokens,
                    middle_tokens=middle_tokens,
                    tail_tokens=tail_tokens,
                )
            )
            chunks = []
            current_tokens = len(tokenizer.encode(prefix + suffix, add_special_tokens=False))

    return records


def collect_pg19_local_records(
    *,
    tokenizer,
    quota: int,
    pg19_local_path: Path,
    min_book_tokens: int,
    slice_strategy: str,
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
) -> list[dict]:
    if quota <= 0:
        return []

    books = load_local_jsonl(pg19_local_path)
    eligible: list[dict] = []
    for item in books:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        token_count = item.get("token_count")
        if token_count is None:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if int(token_count) < min_book_tokens:
            continue
        eligible.append({**item, "token_count": int(token_count), "text": text})

    if not eligible:
        raise ValueError(
            f"No PG19 local books met min_book_tokens={min_book_tokens} from {pg19_local_path}"
        )

    eligible = sorted(eligible, key=lambda item: item["token_count"])
    picked_books = evenly_pick(eligible, min(len(eligible), quota))
    records: list[dict] = []
    for idx in range(quota):
        item = picked_books[idx % len(picked_books)]
        window_idx = idx // len(picked_books)
        book_id = str(item.get("book_id") or item.get("id") or idx)
        title = str(item.get("short_book_title") or f"PG19 book {book_id}")
        publication_date = item.get("publication_date")
        url = item.get("url")
        text = (
            "Read the following long-form book passage and keep track of the narrative state, "
            "major entities, and factual details.\n\n"
            f"Title: {title}\n\n"
            f"Passage:\n{item['text']}\n\n"
            "Question: What is happening in the passage above?\nAnswer:"
        )
        records.append(
            make_record(
                tokenizer=tokenizer,
                text=text,
                source="pg19_local",
                source_id=f"pg19_{book_id}_w{window_idx + 1}",
                task="open_text",
                max_tokens=max_tokens,
                head_tokens=head_tokens,
                middle_tokens=middle_tokens,
                tail_tokens=tail_tokens,
                slice_strategy=slice_strategy,
                slice_salt=f"pg19:{book_id}:window:{window_idx + 1}",
                extra_fields={
                    "book_id": book_id,
                    "short_book_title": title,
                    "publication_date": publication_date,
                    "url": url,
                    "window_index": window_idx + 1,
                },
            )
        )
    return records


def render_semantic_tail_text(
    *,
    tokenizer,
    title: str,
    body_text: str,
    template_index: int,
) -> str:
    question = SEMANTIC_TAIL_TEMPLATES[template_index % len(SEMANTIC_TAIL_TEMPLATES)]
    return (
        f"Title: {title}\n\n"
        f"Passage:\n{body_text}\n\n"
        f"{question}"
    )


def select_token_window_text(
    *,
    tokenizer,
    token_ids: list[int],
    min_total_tokens: int,
    max_total_tokens: int,
    prompt_overhead_tokens: int,
    slice_salt: str,
) -> tuple[str, int]:
    min_body_tokens = max(256, min_total_tokens - prompt_overhead_tokens)
    max_body_tokens = max(min_body_tokens, max_total_tokens - prompt_overhead_tokens)
    if len(token_ids) < min_body_tokens:
        raise ValueError(
            f"Window source too short for target bucket: need {min_body_tokens}, got {len(token_ids)}"
        )

    target_span = min(max_body_tokens, len(token_ids))
    if target_span > min_body_tokens:
        width = target_span - min_body_tokens
        target_span = min_body_tokens + (zlib.crc32(slice_salt.encode("utf-8")) % (width + 1))

    max_start = max(0, len(token_ids) - target_span)
    start = 0
    if max_start > 0:
        start = zlib.crc32(f"start:{slice_salt}".encode("utf-8")) % (max_start + 1)
    end = start + target_span
    body_ids = token_ids[start:end]
    body_text = tokenizer.decode(body_ids, skip_special_tokens=True)
    return body_text, len(body_ids)


def collect_open_long_semantic_tail_records(
    *,
    tokenizer,
    quota: int,
    bucket_name: str,
    pg19_local_path: Path,
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
) -> list[dict]:
    if quota <= 0:
        return []

    min_total_tokens, max_total_tokens = bucket_bounds(bucket_name)
    books = load_local_jsonl(pg19_local_path)
    eligible: list[dict] = []
    for item in books:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < max(512, min_total_tokens - 512):
            continue
        eligible.append(
            {
                **item,
                "text": text,
                "token_ids": token_ids,
                "token_count": len(token_ids),
            }
        )

    if not eligible:
        raise ValueError(
            f"No long-form local books available for semantic-tail bucket {bucket_name} from {pg19_local_path}"
        )

    eligible = sorted(eligible, key=lambda item: item["token_count"])
    picked_books = evenly_pick(eligible, min(len(eligible), max(1, quota)))

    records: list[dict] = []
    for idx in range(quota):
        item = picked_books[idx % len(picked_books)]
        window_idx = idx // len(picked_books)
        book_id = str(item.get("book_id") or item.get("id") or idx)
        title = str(item.get("short_book_title") or f"PG19 book {book_id}")
        template_index = idx
        prompt_shell = render_semantic_tail_text(
            tokenizer=tokenizer,
            title=title,
            body_text="",
            template_index=template_index,
        )
        prompt_overhead_tokens = len(tokenizer.encode(prompt_shell, add_special_tokens=False))
        body_text, body_token_count = select_token_window_text(
            tokenizer=tokenizer,
            token_ids=item["token_ids"],
            min_total_tokens=min_total_tokens,
            max_total_tokens=max_total_tokens,
            prompt_overhead_tokens=prompt_overhead_tokens,
            slice_salt=f"semantic_tail:{bucket_name}:{book_id}:{window_idx + 1}",
        )
        text = render_semantic_tail_text(
            tokenizer=tokenizer,
            title=title,
            body_text=body_text,
            template_index=template_index,
        )
        records.append(
            make_record(
                tokenizer=tokenizer,
                text=text,
                source="open_long_semantic_tail",
                source_id=f"semantic_tail_{bucket_name}_{book_id}_w{window_idx + 1}",
                task="open_text",
                max_tokens=max_tokens,
                head_tokens=head_tokens,
                middle_tokens=middle_tokens,
                tail_tokens=tail_tokens,
                slice_strategy="head_middle_tail",
                slice_salt=f"semantic_tail:{bucket_name}:{book_id}:{window_idx + 1}",
                extra_fields={
                    "bucket_target": bucket_name,
                    "book_id": book_id,
                    "short_book_title": title,
                    "window_index": window_idx + 1,
                    "semantic_template_index": template_index % len(SEMANTIC_TAIL_TEMPLATES),
                    "body_token_count": body_token_count,
                },
            )
        )
    return records


def collect_open_records(
    *,
    tokenizer,
    quotas: dict[str, int],
    public_examples: list[dict],
    max_tokens: int,
    head_tokens: int,
    middle_tokens: int,
    tail_tokens: int,
    seed: int,
    pg19_local_path: Path,
    pg19_min_book_tokens: int,
    pg19_slice_strategy: str,
    ultralong_target_min: int,
    ultralong_target_max: int,
    semantic_tail_quotas: dict[str, int],
) -> list[dict]:
    records: list[dict] = []
    for source, quota in quotas.items():
        if quota <= 0:
            continue
        if source == "pg19_local":
            records.extend(
                collect_pg19_local_records(
                    tokenizer=tokenizer,
                    quota=quota,
                    pg19_local_path=pg19_local_path,
                    min_book_tokens=pg19_min_book_tokens,
                    slice_strategy=pg19_slice_strategy,
                    max_tokens=max_tokens,
                    head_tokens=head_tokens,
                    middle_tokens=middle_tokens,
                    tail_tokens=tail_tokens,
                )
            )
            continue
        if source == "synthetic_ultralong":
            records.extend(
                build_synthetic_ultralong_records(
                    tokenizer=tokenizer,
                    quota=quota,
                    public_examples=public_examples,
                    seed=seed,
                    max_tokens=max_tokens,
                    head_tokens=head_tokens,
                    middle_tokens=middle_tokens,
                    tail_tokens=tail_tokens,
                    target_min_tokens=ultralong_target_min,
                    target_max_tokens=ultralong_target_max,
                )
            )
            continue

        raise ValueError(f"Unsupported local source: {source}")
    for bucket_name, quota in semantic_tail_quotas.items():
        if quota <= 0:
            continue
        records.extend(
            collect_open_long_semantic_tail_records(
                tokenizer=tokenizer,
                quota=quota,
                bucket_name=bucket_name,
                pg19_local_path=pg19_local_path,
                max_tokens=max_tokens,
                head_tokens=head_tokens,
                middle_tokens=middle_tokens,
                tail_tokens=tail_tokens,
            )
        )
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
    by_bucket: dict[str, int] = defaultdict(int)
    raw_lengths: list[int] = []
    calib_lengths: list[int] = []
    for record in records:
        by_source[record["source"]] += 1
        by_task[record["task"]] += 1
        by_bucket[record["bucket"]] += 1
        raw_lengths.append(record["raw_prompt_tokens"])
        calib_lengths.append(record["calib_prompt_tokens"])
    raw_lengths.sort()
    calib_lengths.sort()
    print("records", len(records))
    print("by_source", dict(sorted(by_source.items())))
    print("by_task", dict(sorted(by_task.items())))
    print("by_bucket", dict(sorted(by_bucket.items())))
    print(
        "raw_tokens",
        {
            "p50": raw_lengths[len(raw_lengths) // 2],
            "p90": raw_lengths[min(len(raw_lengths) - 1, int(len(raw_lengths) * 0.9))],
            "max": raw_lengths[-1],
        },
    )
    print(
        "calib_tokens",
        {
            "p50": calib_lengths[len(calib_lengths) // 2],
            "p90": calib_lengths[min(len(calib_lengths) - 1, int(len(calib_lengths) * 0.9))],
            "max": calib_lengths[-1],
        },
    )


def build_summary(records: list[dict]) -> dict:
    by_source: dict[str, int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    by_bucket: dict[str, int] = defaultdict(int)
    raw_lengths: list[int] = []
    calib_lengths: list[int] = []
    for record in records:
        by_source[record["source"]] += 1
        by_task[record["task"]] += 1
        by_bucket[record["bucket"]] += 1
        raw_lengths.append(record["raw_prompt_tokens"])
        calib_lengths.append(record["calib_prompt_tokens"])
    raw_lengths.sort()
    calib_lengths.sort()
    return {
        "count": len(records),
        "by_source": dict(sorted(by_source.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "raw_tokens": {
            "p50": raw_lengths[len(raw_lengths) // 2],
            "p90": raw_lengths[min(len(raw_lengths) - 1, int(len(raw_lengths) * 0.9))],
            "max": raw_lengths[-1],
        },
        "calib_tokens": {
            "p50": calib_lengths[len(calib_lengths) // 2],
            "p90": calib_lengths[min(len(calib_lengths) - 1, int(len(calib_lengths) * 0.9))],
            "max": calib_lengths[-1],
        },
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    public_quotas = parse_quota_map(args.public_quotas, DEFAULT_PUBLIC_QUOTAS)
    open_quotas = parse_quota_map(args.open_quotas, DEFAULT_OPEN_QUOTAS)
    chat_close_quotas = parse_quota_map(args.chat_close_quotas, DEFAULT_CHAT_CLOSE_QUOTAS)
    semantic_tail_quotas = parse_quota_map(args.semantic_tail_quotas, DEFAULT_SEMANTIC_TAIL_QUOTAS)

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
        collect_chat_close_records(
            tokenizer=tokenizer,
            public_examples=public_examples,
            quotas=chat_close_quotas,
            max_tokens=args.max_sample_tokens,
            head_tokens=args.head_tokens,
            middle_tokens=args.middle_tokens,
            tail_tokens=args.tail_tokens,
            max_answer_chars=args.chat_close_answer_max_chars,
        )
    )
    records.extend(
        collect_open_records(
            tokenizer=tokenizer,
            quotas=open_quotas,
            public_examples=public_examples,
            max_tokens=args.max_sample_tokens,
            head_tokens=args.head_tokens,
            middle_tokens=args.middle_tokens,
            tail_tokens=args.tail_tokens,
            seed=args.seed,
            pg19_local_path=Path(args.pg19_local_path),
            pg19_min_book_tokens=args.pg19_min_book_tokens,
            pg19_slice_strategy=args.pg19_slice_strategy,
            ultralong_target_min=args.ultralong_target_min,
            ultralong_target_max=args.ultralong_target_max,
            semantic_tail_quotas=semantic_tail_quotas,
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
    summary = build_summary(records)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("output_path", output_path)
    print("meta_path", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
