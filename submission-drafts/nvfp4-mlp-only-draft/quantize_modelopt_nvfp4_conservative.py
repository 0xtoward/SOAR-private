#!/usr/bin/env python3
"""
Conservative ModelOpt NVFP4 quantization driver for MiniCPM-SALA.

Goals:
- Keep quantization on GPU.
- Prefer conservative configs over full NVFP4 by default.
- Allow calibration from local JSONL/plain-text files and/or ModelOpt datasets.
- Export a Hugging Face checkpoint that SGLang can load with quantization=modelopt.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

try:
    from modelopt.torch.utils.dataset_utils import get_dataset_dataloader
except Exception:
    get_dataset_dataloader = None


CFG_MAP = {
    "default": mtq.NVFP4_DEFAULT_CFG,
    "mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
    "mlp_weight_only": mtq.NVFP4_MLP_WEIGHT_ONLY_CFG,
    "svdquant": mtq.NVFP4_SVDQUANT_DEFAULT_CFG,
}

DEFAULT_LOCAL_CALIB_FILES = [
    "/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_curated_16k_v2.jsonl",
]

COPY_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "configuration_minicpm_sala.py",
    "modeling_minicpm_sala.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/autodl-tmp/models")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument(
        "--cfg",
        default="mlp_only",
        choices=sorted(CFG_MAP.keys()),
        help="Conservative config to use. Default is mlp_only.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--local-calib-files",
        default=",".join(DEFAULT_LOCAL_CALIB_FILES),
        help="Comma-separated local calibration files. Supports JSONL or plain text.",
    )
    parser.add_argument(
        "--local-calib-samples",
        type=int,
        default=128,
        help="Number of local calibration samples to use.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated ModelOpt dataset names such as cnn_dailymail,wikitext.",
    )
    parser.add_argument(
        "--dataset-samples",
        default="",
        help="Optional comma-separated sample counts aligned with --datasets.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument(
        "--min-good-batches",
        type=int,
        default=64,
        help="Minimum successful calibration batches required after skipping bad batches.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete export-dir before writing.",
    )
    return parser.parse_args()


def read_local_texts(paths: list[str], limit: int, seed: int) -> list[str]:
    texts: list[str] = []
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        json_records = 0
        skipped_non_json = 0
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                if not (raw.startswith("{") and raw.endswith("}")):
                    skipped_non_json += 1
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    skipped_non_json += 1
                    continue
                if not isinstance(item, dict):
                    skipped_non_json += 1
                    continue
                text = item.get("calib_text") or item.get("text") or item.get("question")
                text = str(text).strip() if text is not None else ""
                if text:
                    texts.append(text)
                    json_records += 1

        if json_records == 0 and skipped_non_json > 0:
            print(
                f"[warn] skipped non-JSONL calibration file: {p}. "
                "Use a clean JSONL with one object per sample.",
                file=sys.stderr,
            )
        elif skipped_non_json > 0:
            print(
                f"[warn] ignored {skipped_non_json} non-JSON lines in calibration file: {p}",
                file=sys.stderr,
            )

    if not texts:
        return []

    random.Random(seed).shuffle(texts)
    unique_texts = list(dict.fromkeys(texts))
    return unique_texts[:limit]


def make_local_batches(
    tokenizer,
    texts: Iterable[str],
    max_length: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        batches.append(encoded)
    return batches


def make_modelopt_batches(
    dataset_names: list[str],
    dataset_sizes: list[int],
    tokenizer,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[dict[str, torch.Tensor]]:
    if not dataset_names:
        return []
    if get_dataset_dataloader is None:
        raise RuntimeError("modelopt dataset utils are unavailable in this environment")

    dataloader = get_dataset_dataloader(
        dataset_name=dataset_names,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=dataset_sizes,
        max_sample_length=max_length,
        device=device,
    )
    return list(dataloader)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def get_batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    input_ids = batch.get("input_ids")
    if input_ids is None:
        return -1
    return int(input_ids.shape[-1])


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        print("GPU is required for this workflow.", file=sys.stderr)
        return 1

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    cfg = CFG_MAP[args.cfg]

    export_dir = Path(args.export_dir)
    if export_dir.exists() and args.overwrite:
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MiniCPM-SALA conservative NVFP4 quantization")
    print(f"model_path={args.model_path}")
    print(f"export_dir={export_dir}")
    print(f"cfg={args.cfg}")
    print(f"dtype={args.dtype}")
    print(f"cuda={torch.cuda.get_device_name(0)}")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("[1/4] Loading base model on GPU...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = model.to(device="cuda")
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    local_paths = parse_csv_list(args.local_calib_files)
    local_texts = read_local_texts(local_paths, args.local_calib_samples, args.seed)
    local_batches = make_local_batches(tokenizer, local_texts, args.max_length)

    dataset_names = parse_csv_list(args.datasets)
    dataset_sizes = parse_csv_ints(args.dataset_samples) if args.dataset_samples else []
    if dataset_names and not dataset_sizes:
        dataset_sizes = [args.local_calib_samples for _ in dataset_names]
    if dataset_names and len(dataset_sizes) != len(dataset_names):
        raise ValueError("--dataset-samples must align with --datasets")

    print("[2/4] Preparing calibration batches...")
    print(f"local_batches={len(local_batches)} from {local_paths}")
    print(f"modelopt_datasets={dataset_names or 'none'}")

    modelopt_batches = make_modelopt_batches(
        dataset_names=dataset_names,
        dataset_sizes=dataset_sizes,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device="cpu",
    )
    print(f"modelopt_batches={len(modelopt_batches)}")

    all_batches = local_batches + modelopt_batches
    if not all_batches:
        raise RuntimeError("No calibration data available.")

    def forward_loop(model) -> None:
        good_batches = 0
        skipped_batches = 0
        with torch.no_grad():
            for idx, batch in enumerate(all_batches, start=1):
                batch = move_batch(batch, get_model_device(model))
                batch_tokens = get_batch_tokens(batch)
                try:
                    outputs = model(**batch)
                    logits = getattr(outputs, "logits", None)
                    if logits is not None and not torch.isfinite(logits).all():
                        skipped_batches += 1
                        print(
                            f"  skipped {idx}/{len(all_batches)} tokens={batch_tokens} "
                            "reason=non_finite_logits"
                        )
                        continue
                    good_batches += 1
                except Exception as exc:
                    skipped_batches += 1
                    print(
                        f"  skipped {idx}/{len(all_batches)} tokens={batch_tokens} "
                        f"reason={type(exc).__name__}: {exc}"
                    )
                    torch.cuda.empty_cache()
                    continue
                if idx % 16 == 0 or idx == len(all_batches):
                    print(
                        f"  calibrated {idx}/{len(all_batches)} "
                        f"good={good_batches} skipped={skipped_batches}"
                    )
        if good_batches < args.min_good_batches:
            raise RuntimeError(
                f"Only {good_batches} calibration batches succeeded; "
                f"need at least {args.min_good_batches}."
            )
        print(
            f"calibration_complete good_batches={good_batches} "
            f"skipped_batches={skipped_batches}"
        )
        forward_loop.good_batches = good_batches
        forward_loop.skipped_batches = skipped_batches

    print("[3/4] Quantizing with ModelOpt...")
    t1 = time.time()
    mtq.quantize(model, cfg, forward_loop=forward_loop)
    print(f"Quantized in {time.time() - t1:.1f}s")
    mtq.print_quant_summary(model)

    print("[4/4] Exporting checkpoint...")
    t2 = time.time()
    export_hf_checkpoint(model, export_dir=str(export_dir), dtype=dtype)

    for name in COPY_FILES:
        src = Path(args.model_path) / name
        dst = export_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    manifest = {
        "cfg": args.cfg,
        "dtype": args.dtype,
        "local_calib_files": local_paths,
        "local_calib_samples": len(local_batches),
        "datasets": dataset_names,
        "dataset_samples": dataset_sizes,
        "max_length": args.max_length,
        "min_good_batches": args.min_good_batches,
        "good_batches": getattr(forward_loop, "good_batches", None),
        "skipped_batches": getattr(forward_loop, "skipped_batches", None),
        "seed": args.seed,
    }
    with (export_dir / "codex_quant_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total_size = sum(p.stat().st_size for p in export_dir.rglob("*") if p.is_file())
    print(f"Exported in {time.time() - t2:.1f}s")
    print(f"total_size_gb={total_size / 1e9:.2f}")
    print(f"manifest={export_dir / 'codex_quant_manifest.json'}")
    print()
    print("Suggested serve command:")
    print(
        "python -m sglang.launch_server "
        f"--model-path {export_dir} "
        "--trust-remote-code "
        "--quantization modelopt "
        "--disable-cuda-graph "
        "--disable-radix-cache "
        "--attention-backend minicpm_flashinfer "
        "--chunked-prefill-size 8192 "
        "--skip-server-warmup "
        "--dense-as-sparse"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
