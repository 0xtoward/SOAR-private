#!/usr/bin/env python3
"""Compare baseline decode logits with EAGLE force-reject verify logits.

This is a narrow correctness probe for MiniCPM-SALA speculative decoding.  It
answers one question: at the first output mismatch, did the target-verify root
row already change greedy argmax, or are logits only numerically different while
top-1 remains stable?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_ids(path: Path) -> List[int]:
    obj = load_json(path)
    ids = obj.get("output_ids")
    if not isinstance(ids, list):
        raise ValueError(f"{path} does not contain output_ids")
    return [int(x) for x in ids]


def first_mismatch(lhs: List[int], rhs: List[int]) -> Optional[int]:
    for idx, (a, b) in enumerate(zip(lhs, rhs)):
        if a != b:
            return idx
    if len(lhs) != len(rhs):
        return min(len(lhs), len(rhs))
    return None


def _time_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"_(\d+)\.pt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (path.stat().st_mtime_ns, path.name)


def _verify_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"verify_step_(\d+|unknown)_", path.name)
    if match and match.group(1) != "unknown":
        return (int(match.group(1)), path.name)
    return _time_key(path)


def _row0_logits(payload: Dict[str, Any]) -> torch.Tensor:
    logits = payload["logits"]
    if logits.ndim == 1:
        return logits.float()
    return logits[0].float()


def load_forward_logits(path: Path) -> List[Dict[str, Any]]:
    files = sorted(path.glob("forward_*.pt"), key=_time_key)
    rows = []
    for file in files:
        payload = torch_load(file)
        logits = _row0_logits(payload)
        top2_vals, top2_ids = torch.topk(logits, k=2)
        argmax_id = int(torch.argmax(logits).item())
        rows.append(
            {
                "path": str(file),
                "payload": payload,
                "logits": logits,
                "argmax": argmax_id,
                "top2": [int(top2_ids[0].item()), int(top2_ids[1].item())],
                "top2_logits": [float(top2_vals[0].item()), float(top2_vals[1].item())],
                "margin": float((top2_vals[0] - top2_vals[1]).item()),
                "mode": payload.get("mode"),
                "positions": payload.get("positions"),
            }
        )
    return rows


def load_verify_logits(path: Path) -> List[Dict[str, Any]]:
    files = sorted(path.glob("verify_step_*_draft_*.pt"), key=_verify_key)
    rows = []
    for file in files:
        payload = torch_load(file)
        logits = _row0_logits(payload)
        top2_vals, top2_ids = torch.topk(logits, k=2)
        argmax_id = int(torch.argmax(logits).item())
        rows.append(
            {
                "path": str(file),
                "payload": payload,
                "logits": logits,
                "argmax": argmax_id,
                "top2": [int(top2_ids[0].item()), int(top2_ids[1].item())],
                "top2_logits": [float(top2_vals[0].item()), float(top2_vals[1].item())],
                "margin": float((top2_vals[0] - top2_vals[1]).item()),
                "step": payload.get("step"),
                "fields": payload.get("fields", {}),
            }
        )
    return rows


def best_alignment(
    logit_argmax: List[int],
    generated: List[int],
    max_logit_start: int = 4,
    max_generated_start: int = 4,
) -> Dict[str, Any]:
    best = {
        "logit_start": 0,
        "generated_start": 0,
        "prefix_matches": -1,
        "total_matches": -1,
    }
    for logit_start in range(max_logit_start + 1):
        for generated_start in range(max_generated_start + 1):
            prefix = 0
            total = 0
            for k, token in enumerate(generated[generated_start:]):
                j = logit_start + k
                if j >= len(logit_argmax):
                    break
                if logit_argmax[j] == token:
                    total += 1
                    if prefix == k:
                        prefix += 1
                elif prefix == k:
                    # Prefix match ends at first mismatch.
                    pass
            cand = {
                "logit_start": logit_start,
                "generated_start": generated_start,
                "prefix_matches": prefix,
                "total_matches": total,
            }
            if (cand["prefix_matches"], cand["total_matches"]) > (
                best["prefix_matches"],
                best["total_matches"],
            ):
                best = cand
    return best


def vector_metrics(base: torch.Tensor, verify: torch.Tensor) -> Dict[str, float]:
    diff = base - verify
    base_norm = torch.linalg.vector_norm(base.float())
    verify_norm = torch.linalg.vector_norm(verify.float())
    denom = float((base_norm * verify_norm).item())
    cosine = float(torch.dot(base.float(), verify.float()).item() / denom) if denom else math.nan
    return {
        "max_abs_diff": float(torch.max(torch.abs(diff)).item()),
        "mean_abs_diff": float(torch.mean(torch.abs(diff)).item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "cosine": cosine,
    }


def token_window(ids: List[int], idx: Optional[int], radius: int = 8) -> List[int]:
    if idx is None:
        return []
    start = max(0, idx - radius)
    end = min(len(ids), idx + radius + 1)
    return ids[start:end]


def compare(args: argparse.Namespace) -> Dict[str, Any]:
    baseline_ids = output_ids(Path(args.baseline_json))
    spec_ids = output_ids(Path(args.spec_json))
    baseline_rows = load_forward_logits(Path(args.baseline_logits_dir))
    verify_rows = load_verify_logits(Path(args.verify_logits_dir))
    if not baseline_rows:
        raise ValueError(f"No baseline forward logits in {args.baseline_logits_dir}")
    if not verify_rows:
        raise ValueError(f"No verify logits in {args.verify_logits_dir}")

    base_align = best_alignment([r["argmax"] for r in baseline_rows], baseline_ids)
    verify_align = best_alignment([r["argmax"] for r in verify_rows], spec_ids)
    out_mismatch = first_mismatch(baseline_ids, spec_ids)

    compare_start = max(base_align["generated_start"], verify_align["generated_start"])
    compare_end = min(
        len(baseline_ids),
        len(spec_ids),
        base_align["generated_start"] + len(baseline_rows) - base_align["logit_start"],
        verify_align["generated_start"] + len(verify_rows) - verify_align["logit_start"],
    )
    rows = []
    first_argmax_mismatch = None
    first_nonzero_diff = None
    for token_idx in range(compare_start, compare_end):
        base_row = baseline_rows[
            base_align["logit_start"] + token_idx - base_align["generated_start"]
        ]
        verify_row = verify_rows[
            verify_align["logit_start"] + token_idx - verify_align["generated_start"]
        ]
        metrics = vector_metrics(base_row["logits"], verify_row["logits"])
        if first_nonzero_diff is None and metrics["max_abs_diff"] > args.diff_eps:
            first_nonzero_diff = token_idx
        if first_argmax_mismatch is None and base_row["argmax"] != verify_row["argmax"]:
            first_argmax_mismatch = token_idx
        base_top = base_row["argmax"]
        verify_top = verify_row["argmax"]
        row = {
            "token_index": token_idx,
            "baseline_output_id": baseline_ids[token_idx],
            "spec_output_id": spec_ids[token_idx],
            "output_mismatch": baseline_ids[token_idx] != spec_ids[token_idx],
            "baseline_argmax": base_top,
            "verify_argmax": verify_top,
            "argmax_mismatch": base_top != verify_top,
            "baseline_margin": base_row["margin"],
            "verify_margin": verify_row["margin"],
            "baseline_top2": base_row["top2"],
            "verify_top2": verify_row["top2"],
            "baseline_top2_logits": base_row["top2_logits"],
            "verify_top2_logits": verify_row["top2_logits"],
            "baseline_top_logit_in_verify": float(verify_row["logits"][base_top].item()),
            "verify_top_logit_in_baseline": float(base_row["logits"][verify_top].item()),
            "baseline_file": base_row["path"],
            "verify_file": verify_row["path"],
            "verify_step": verify_row.get("step"),
            **metrics,
        }
        if (
            token_idx <= 3
            or token_idx == out_mismatch
            or token_idx == first_argmax_mismatch
            or (out_mismatch is not None and abs(token_idx - out_mismatch) <= args.context)
        ):
            rows.append(row)

    focus_indices = sorted(
        {
            idx
            for idx in [
                0,
                first_nonzero_diff,
                first_argmax_mismatch,
                out_mismatch,
            ]
            if idx is not None and compare_start <= idx < compare_end
        }
    )
    focus = []
    for idx in focus_indices:
        base_row = baseline_rows[
            base_align["logit_start"] + idx - base_align["generated_start"]
        ]
        verify_row = verify_rows[
            verify_align["logit_start"] + idx - verify_align["generated_start"]
        ]
        focus.append(
            {
                "token_index": idx,
                "baseline_output_id": baseline_ids[idx],
                "spec_output_id": spec_ids[idx],
                "baseline_argmax": base_row["argmax"],
                "verify_argmax": verify_row["argmax"],
                "baseline_margin": base_row["margin"],
                "verify_margin": verify_row["margin"],
                **vector_metrics(base_row["logits"], verify_row["logits"]),
            }
        )

    classification = "unknown"
    if out_mismatch is None:
        classification = "outputs_exact"
    elif first_argmax_mismatch == out_mismatch:
        classification = "verify_argmax_changed_at_output_mismatch"
    elif first_argmax_mismatch is not None and first_argmax_mismatch < out_mismatch:
        classification = "verify_argmax_changed_before_output_mismatch"
    elif first_argmax_mismatch is None or first_argmax_mismatch > out_mismatch:
        classification = "output_mismatch_without_root_argmax_mismatch"

    return {
        "prompt_id": args.prompt_id,
        "classification": classification,
        "baseline_json": args.baseline_json,
        "spec_json": args.spec_json,
        "baseline_len": len(baseline_ids),
        "spec_len": len(spec_ids),
        "first_output_mismatch": out_mismatch,
        "first_verify_argmax_mismatch": first_argmax_mismatch,
        "first_nonzero_logit_diff": first_nonzero_diff,
        "baseline_alignment": base_align,
        "verify_alignment": verify_align,
        "baseline_output_window": token_window(baseline_ids, out_mismatch),
        "spec_output_window": token_window(spec_ids, out_mismatch),
        "num_baseline_logit_rows": len(baseline_rows),
        "num_verify_logit_rows": len(verify_rows),
        "compare_start": compare_start,
        "compare_end": compare_end,
        "max_compared_tokens": max(0, compare_end - compare_start),
        "focus": focus,
        "rows": rows,
    }


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        f"# EAGLE verify logit comparison: {summary['prompt_id']}",
        "",
        f"- classification: `{summary['classification']}`",
        f"- first_output_mismatch: `{summary['first_output_mismatch']}`",
        f"- first_verify_argmax_mismatch: `{summary['first_verify_argmax_mismatch']}`",
        f"- first_nonzero_logit_diff: `{summary['first_nonzero_logit_diff']}`",
        f"- baseline_alignment: `{summary['baseline_alignment']}`",
        f"- verify_alignment: `{summary['verify_alignment']}`",
        f"- compared_tokens: `{summary['max_compared_tokens']}`",
        "",
        "## Focus Rows",
        "",
        "| token_idx | base_out | spec_out | base_argmax | verify_argmax | base_margin | verify_margin | max_abs | mean_abs | cosine |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["focus"]:
        lines.append(
            "| {token_index} | {baseline_output_id} | {spec_output_id} | {baseline_argmax} | {verify_argmax} | {baseline_margin:.6g} | {verify_margin:.6g} | {max_abs_diff:.6g} | {mean_abs_diff:.6g} | {cosine:.8f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Output Window",
            "",
            f"- baseline: `{summary['baseline_output_window']}`",
            f"- spec: `{summary['spec_output_window']}`",
            "",
            "## Nearby Rows",
            "",
            "| token_idx | out_mismatch | argmax_mismatch | base_out | spec_out | base_argmax | verify_argmax | base_top2 | verify_top2 | max_abs | mean_abs | cosine |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| {token_index} | {output_mismatch} | {argmax_mismatch} | {baseline_output_id} | {spec_output_id} | {baseline_argmax} | {verify_argmax} | `{baseline_top2}` | `{verify_top2}` | {max_abs_diff:.6g} | {mean_abs_diff:.6g} | {cosine:.8f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--baseline-logits-dir", required=True)
    parser.add_argument("--verify-logits-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--diff-eps", type=float, default=0.0)
    parser.add_argument("--context", type=int, default=3)
    args = parser.parse_args()

    summary = compare(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, Path(args.output_md))


if __name__ == "__main__":
    main()
