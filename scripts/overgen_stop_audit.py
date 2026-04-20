#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit over-generation on a small dataset using the same chat/completions path as eval_model.py."
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--mcq-suffix", default="")
    parser.add_argument("--infer-mcq", action="store_true")
    parser.add_argument(
        "--mcq-patch-mode",
        choices=["task_or_infer", "task_only", "infer_only"],
        default="task_or_infer",
    )
    parser.add_argument("--task-filter", default="")
    return parser.parse_args()


def extract_mcq_answer(text: str) -> str | None:
    answer_matches = re.findall(r"(?i)ANSWER\s*:\s*([A-D])", text)
    if answer_matches:
        return answer_matches[-1].upper()
    boxed_text_matches = re.findall(r"\\boxed\{\\text\{([A-D])\}\}", text)
    if boxed_text_matches:
        return boxed_text_matches[-1].upper()
    boxed_matches = re.findall(r"\\boxed\{([A-D])\}", text)
    if boxed_matches:
        return boxed_matches[-1].upper()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        line_match = re.search(r"([A-D])", last)
        if line_match:
            return line_match.group(1).upper()
    return None


def looks_like_mcq_prompt(text: str) -> bool:
    lowered = text.lower()
    has_answer_contract = "answer: $letter" in lowered or "one of abcd" in lowered
    has_mcq_phrase = "multiple choice question" in lowered
    has_letter_options = all(token in text for token in ["A)", "B)", "C)", "D)"])
    return (has_answer_contract or has_mcq_phrase) and has_letter_options


def load_stop_words(model_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(model_path)

    eos_token_ids: list[int] = []
    if hasattr(generation_config, "eos_token_id"):
        eos = generation_config.eos_token_id
        eos_token_ids = [eos] if isinstance(eos, int) else list(eos or [])

    stop_words: list[str] = []
    for token_id in eos_token_ids:
        text = tokenizer.decode(token_id)
        if text:
            stop_words.append(text)
    if tokenizer.eos_token:
        stop_words.append(tokenizer.eos_token)

    deduped_stop_words: list[str] = []
    seen = set()
    for item in stop_words:
        if item and item not in seen:
            deduped_stop_words.append(item)
            seen.add(item)

    return {
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", model_path),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "generation_config_eos_token_id": eos_token_ids,
        "stop_words": deduped_stop_words,
    }


def main() -> int:
    args = parse_args()
    api_base = args.api_base.rstrip("/")
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    stop_info = load_stop_words(args.model_path)

    rows = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.task_filter:
        allowed_tasks = {piece.strip() for piece in args.task_filter.split(",") if piece.strip()}
        rows = [row for row in rows if row.get("task") in allowed_tasks]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    if not rows:
        raise SystemExit(f"No rows loaded from {data_path}")

    model_name = args.model_name
    if not model_name:
        response = requests.get(f"{api_base}/v1/models", timeout=min(args.timeout, 60))
        response.raise_for_status()
        model_name = response.json()["data"][0]["id"]

    stop_words = stop_info["stop_words"]
    audited_rows: list[dict[str, Any]] = []
    finish_counter: Counter[str] = Counter()
    task_finish_counter: dict[str, Counter[str]] = defaultdict(Counter)
    patch_counter: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        prompt = row["question"]
        patch_reason = "none"
        allow_task_patch = args.mcq_patch_mode in {"task_or_infer", "task_only"}
        allow_infer_patch = args.mcq_patch_mode in {"task_or_infer", "infer_only"}
        if args.mcq_suffix and allow_task_patch and row.get("task") == "mcq":
            prompt = f"{prompt.rstrip()}\n\n{args.mcq_suffix}"
            patch_reason = "task=mcq"
        elif (
            args.mcq_suffix
            and args.infer_mcq
            and allow_infer_patch
            and looks_like_mcq_prompt(prompt)
        ):
            prompt = f"{prompt.rstrip()}\n\n{args.mcq_suffix}"
            patch_reason = "infer=mcq"
        requested_max_tokens = (
            args.max_output_tokens
            if args.max_output_tokens > 0
            else int(row.get("completion_tokens", 0) or 0)
        )
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": requested_max_tokens,
            "stop": stop_words,
        }

        wall_start = time.perf_counter()
        response = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=args.timeout,
        )
        wall_s = time.perf_counter() - wall_start
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        finish_reason = choice.get("finish_reason") or "unknown"
        content = choice.get("message", {}).get("content", "")
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + output_tokens) or 0)
        mcq_extracted_answer = None
        mcq_correct = None
        if row.get("task") == "mcq":
            mcq_extracted_answer = extract_mcq_answer(content or "")
            if mcq_extracted_answer is not None and row.get("gold") is not None:
                mcq_correct = mcq_extracted_answer.upper() == str(row["gold"]).upper()
            else:
                mcq_correct = False

        finish_counter[finish_reason] += 1
        task_finish_counter[row.get("task", "<unknown>")][finish_reason] += 1
        patch_counter[patch_reason] += 1

        audited_rows.append(
            {
                "index": index,
                "task": row.get("task"),
                "question": row["question"],
                "effective_question": prompt,
                "gold": row.get("gold"),
                "dataset_prompt_tokens": row.get("prompt_tokens"),
                "dataset_completion_tokens": row.get("completion_tokens"),
                "requested_max_tokens": requested_max_tokens,
                "api_prompt_tokens": prompt_tokens,
                "api_completion_tokens": output_tokens,
                "api_total_tokens": total_tokens,
                "finish_reason": finish_reason,
                "wall_s": wall_s,
                "patch_reason": patch_reason,
                "mcq_extracted_answer": mcq_extracted_answer,
                "mcq_correct": mcq_correct,
                "prediction": content,
            }
        )

        print(
            f"{index:02d}/{len(rows)} task={row.get('task','?'):>4} "
            f"req_out={requested_max_tokens:>5} out={output_tokens:>5} "
            f"finish={finish_reason:<8} wall={wall_s:>7.2f}s "
            f"patch={patch_reason:<9} "
            f"mcq={mcq_extracted_answer if mcq_extracted_answer is not None else '-'}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "api_base": api_base,
        "model_path": args.model_path,
        "model_name": model_name,
        "data_path": str(data_path),
        "count": len(audited_rows),
        "stop_info": stop_info,
        "patch_counts": dict(sorted(patch_counter.items())),
        "finish_reason_counts": dict(sorted(finish_counter.items())),
        "avg_output_tokens": (
            sum(row["api_completion_tokens"] for row in audited_rows) / len(audited_rows)
        ),
        "total_output_tokens": sum(row["api_completion_tokens"] for row in audited_rows),
        "avg_wall_s": sum(row["wall_s"] for row in audited_rows) / len(audited_rows),
        "task_finish_reason_counts": {
            task: dict(sorted(counter.items()))
            for task, counter in sorted(task_finish_counter.items())
        },
        "by_output_tokens_desc": sorted(
            audited_rows,
            key=lambda row: (row["api_completion_tokens"], row["wall_s"]),
            reverse=True,
        ),
        "rows": audited_rows,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[overgen-stop-audit] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
