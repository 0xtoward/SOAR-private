#!/usr/bin/env python3
"""
Materialize a local PG19 long-book cache for offline calibration building.

This script avoids `datasets.load_dataset("pg19")`, which now fails because the
dataset depends on a deprecated dataset script. Instead it:

- fetches the tiny HF index files via `huggingface_hub`
- downloads selected raw book texts from DeepMind's public GCS bucket
- tokenizes them with the target model tokenizer
- writes a local JSONL corpus that later calibration builders can consume offline
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from urllib.error import URLError
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


HF_REPO_ID = "pg19"
HF_REPO_TYPE = "dataset"
ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"
METADATA_URL = ASSET_ROOT_URL + "metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models")
    parser.add_argument(
        "--output-path",
        default="/root/autodl-tmp/SOAR-Toolkit/calibration_sources/pg19_train_books_v1.jsonl",
    )
    parser.add_argument(
        "--download-dir",
        default="/root/autodl-tmp/SOAR-Toolkit/calibration_sources/pg19_raw_train",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-books", type=int, default=8)
    parser.add_argument("--min-book-tokens", type=int, default=160000)
    parser.add_argument("--min-book-bytes", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=192)
    return parser.parse_args()


def download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urllib.request.urlretrieve(url, path)


def get_remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            header = response.headers.get("Content-Length")
            return int(header) if header else None
    except (URLError, ValueError):
        return None


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
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


def load_metadata(metadata_path: Path) -> dict[str, dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(
            f,
            fieldnames=["book_id", "short_book_title", "publication_date", "url"],
        )
        return {row["book_id"]: row for row in reader}


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_path)
    download_dir = Path(args.download_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    train_files_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename="data/train_files.txt",
        )
    )
    metadata_path = output_path.parent / "pg19_metadata.csv"
    download_url(METADATA_URL, metadata_path)

    metadata = load_metadata(metadata_path)
    existing_rows = load_existing_rows(output_path)
    selected_ids = {str(row.get("book_id")) for row in existing_rows}
    selected_rows = list(existing_rows)
    min_book_bytes = args.min_book_bytes or (args.min_book_tokens * 4)

    with train_files_path.open("r", encoding="utf-8") as f:
        relpaths = [line.strip() for line in f if line.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(relpaths)

    attempts = 0
    for relpath in relpaths:
        if len(selected_rows) >= args.target_books:
            break
        if attempts >= args.max_attempts:
            break
        attempts += 1

        book_id = Path(relpath).stem
        if book_id in selected_ids:
            continue

        url = ASSET_ROOT_URL + relpath
        local_text_path = download_dir / relpath
        if local_text_path.exists():
            local_size = local_text_path.stat().st_size
        else:
            remote_size = get_remote_content_length(url)
            if remote_size is not None and remote_size < min_book_bytes:
                continue
            download_url(url, local_text_path)
            local_size = local_text_path.stat().st_size

        if local_size < min_book_bytes:
            continue

        text = local_text_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count < args.min_book_tokens:
            continue

        meta = metadata.get(
            book_id,
            {
                "book_id": book_id,
                "short_book_title": "",
                "publication_date": "",
                "url": "",
            },
        )
        row = {
            "book_id": book_id,
            "relpath": relpath,
            "short_book_title": meta.get("short_book_title", ""),
            "publication_date": meta.get("publication_date", ""),
            "url": meta.get("url", ""),
            "token_count": token_count,
            "text": text,
        }
        selected_rows.append(row)
        selected_ids.add(book_id)
        print(
            f"selected book_id={book_id} tokens={token_count} "
            f"title={row['short_book_title']!r}"
        )

    selected_rows = sorted(selected_rows, key=lambda row: int(row.get("token_count", 0)))
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    summary = {
        "output_path": str(output_path),
        "selected_books": len(selected_rows),
        "min_book_tokens": args.min_book_tokens,
        "min_book_bytes": min_book_bytes,
        "max_attempts": args.max_attempts,
        "token_min": min(int(row["token_count"]) for row in selected_rows) if selected_rows else 0,
        "token_max": max(int(row["token_count"]) for row in selected_rows) if selected_rows else 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
