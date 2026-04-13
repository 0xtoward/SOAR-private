#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


TARGET_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

QUANTILES = (0.5, 0.9, 0.99, 0.999)
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze base-weight distribution for Marlin/GPTQ target layers."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=262144)
    parser.add_argument("--row-sample-size", type=int, default=256)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def suffix_of(name: str) -> str:
    for suffix in TARGET_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    raise ValueError(f"unexpected tensor: {name}")


def layer_of(name: str) -> int:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else -1


def evenly_sample_1d(x: torch.Tensor, limit: int) -> torch.Tensor:
    if x.numel() <= limit:
        return x
    step = max(1, x.numel() // limit)
    return x[::step][:limit]


def evenly_sample_rows(x: torch.Tensor, limit: int) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] <= limit:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, steps=limit, dtype=torch.long)
    return x.index_select(0, idx)


def quantile_map(sample_abs: torch.Tensor) -> dict[str, float]:
    q = torch.tensor(QUANTILES, dtype=torch.float32)
    values = torch.quantile(sample_abs, q)
    return {
        "p50_abs": float(values[0].item()),
        "p90_abs": float(values[1].item()),
        "p99_abs": float(values[2].item()),
        "p999_abs": float(values[3].item()),
    }


def summarize_group_spikes(
    matrix_abs: torch.Tensor, group_size: int, row_sample_size: int
) -> dict[str, float] | None:
    if matrix_abs.ndim != 2 or matrix_abs.shape[1] < group_size:
        return None

    sampled = evenly_sample_rows(matrix_abs, row_sample_size)
    usable = (sampled.shape[1] // group_size) * group_size
    if usable == 0:
        return None

    groups = sampled[:, :usable].reshape(-1, group_size)
    group_max = groups.max(dim=1).values
    group_mean = groups.mean(dim=1)
    ratios = group_max / group_mean.clamp_min(1e-12)
    q = torch.quantile(ratios, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32))
    return {
        "group_ratio_p50": float(q[0].item()),
        "group_ratio_p90": float(q[1].item()),
        "group_ratio_p99": float(q[2].item()),
        "group_ratio_max": float(ratios.max().item()),
        "group_ratio_gt8_frac": float((ratios > 8.0).float().mean().item()),
        "group_ratio_gt12_frac": float((ratios > 12.0).float().mean().item()),
        "sampled_groups": int(ratios.numel()),
    }


def summarize_tensor(
    name: str,
    tensor: torch.Tensor,
    group_size: int,
    sample_size: int,
    row_sample_size: int,
) -> dict[str, object]:
    x = tensor.detach().to(dtype=torch.float32, device="cpu")
    flat = x.reshape(-1)
    abs_flat = flat.abs()
    abs_sample = evenly_sample_1d(abs_flat, sample_size)

    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    min_v = float(flat.min().item())
    max_v = float(flat.max().item())
    abs_max = float(abs_flat.max().item())
    qmap = quantile_map(abs_sample)

    centered = flat - mean
    if std > 0:
        m4 = float((centered.pow(4).mean() / (std ** 4)).item())
        frac_gt_6sigma = float((abs_flat > (6.0 * std)).float().mean().item())
        frac_gt_8sigma = float((abs_flat > (8.0 * std)).float().mean().item())
    else:
        m4 = 0.0
        frac_gt_6sigma = 0.0
        frac_gt_8sigma = 0.0

    group_stats = summarize_group_spikes(x.abs(), group_size, row_sample_size)
    p999 = max(qmap["p999_abs"], 1e-12)
    p99 = max(qmap["p99_abs"], 1e-12)
    p90 = max(qmap["p90_abs"], 1e-12)

    result: dict[str, object] = {
        "name": name,
        "layer": layer_of(name),
        "suffix": suffix_of(name),
        "shape": list(x.shape),
        "numel": int(flat.numel()),
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "abs_max": abs_max,
        "absmax_over_p90": abs_max / p90,
        "absmax_over_p99": abs_max / p99,
        "absmax_over_p999": abs_max / p999,
        "kurtosis": m4,
        "frac_gt_6sigma": frac_gt_6sigma,
        "frac_gt_8sigma": frac_gt_8sigma,
        "sample_size": int(abs_sample.numel()),
    }
    result.update(qmap)
    if group_stats is not None:
        result.update(group_stats)
    return result


def aggregate(records: list[dict[str, object]], field: str) -> dict[str, float]:
    values = sorted(float(record[field]) for record in records if field in record)
    if not values:
        return {}
    mid = len(values) // 2
    return {
        "min": values[0],
        "median": values[mid],
        "p90": values[min(len(values) - 1, math.floor(len(values) * 0.9))],
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    target_names = sorted(
        name for name in weight_map if any(name.endswith(suffix) for suffix in TARGET_SUFFIXES)
    )
    shard_to_names: dict[str, list[str]] = defaultdict(list)
    for name in target_names:
        shard_to_names[weight_map[name]].append(name)

    per_tensor: list[dict[str, object]] = []
    for shard_name in sorted(shard_to_names):
        shard_path = model_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for name in sorted(shard_to_names[shard_name]):
                tensor = f.get_tensor(name)
                per_tensor.append(
                    summarize_tensor(
                        name=name,
                        tensor=tensor,
                        group_size=args.group_size,
                        sample_size=args.sample_size,
                        row_sample_size=args.row_sample_size,
                    )
                )

    by_suffix: dict[str, dict[str, object]] = {}
    for suffix in TARGET_SUFFIXES:
        records = [record for record in per_tensor if record["suffix"] == suffix]
        by_suffix[suffix] = {
            "count": len(records),
            "absmax_over_p999": aggregate(records, "absmax_over_p999"),
            "kurtosis": aggregate(records, "kurtosis"),
            "frac_gt_6sigma": aggregate(records, "frac_gt_6sigma"),
            "group_ratio_p99": aggregate(records, "group_ratio_p99"),
            "group_ratio_max": aggregate(records, "group_ratio_max"),
        }

    overall = {
        "tensor_count": len(per_tensor),
        "target_suffixes": list(TARGET_SUFFIXES),
        "absmax_over_p999": aggregate(per_tensor, "absmax_over_p999"),
        "kurtosis": aggregate(per_tensor, "kurtosis"),
        "frac_gt_6sigma": aggregate(per_tensor, "frac_gt_6sigma"),
        "group_ratio_p99": aggregate(per_tensor, "group_ratio_p99"),
        "group_ratio_max": aggregate(per_tensor, "group_ratio_max"),
    }

    top_absmax = sorted(per_tensor, key=lambda r: float(r["absmax_over_p999"]), reverse=True)[:20]
    top_group = sorted(per_tensor, key=lambda r: float(r.get("group_ratio_max", 0.0)), reverse=True)[:20]
    top_kurtosis = sorted(per_tensor, key=lambda r: float(r["kurtosis"]), reverse=True)[:20]

    output = {
        "model_dir": str(model_dir),
        "group_size": args.group_size,
        "sample_size": args.sample_size,
        "row_sample_size": args.row_sample_size,
        "overall": overall,
        "by_suffix": by_suffix,
        "top_absmax_over_p999": top_absmax,
        "top_group_ratio_max": top_group,
        "top_kurtosis": top_kurtosis,
        "per_tensor": per_tensor,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
