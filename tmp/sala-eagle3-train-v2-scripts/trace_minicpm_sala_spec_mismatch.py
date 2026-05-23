#!/usr/bin/env python3
"""Summarize MiniCPM-SALA speculative mismatch traces.

The companion shell script launches SGLang cases and writes one response JSON
per prompt plus one opt-in trace JSONL per speculative case.  This summarizer
keeps the root-cause decision table in one place so every run produces the same
forensic report shape.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"event": "json_decode_error", "raw": line[:500]})
    return rows


def first_mismatch(base_ids: Optional[List[int]], spec_ids: Optional[List[int]]) -> Optional[Dict[str, Any]]:
    if base_ids is None or spec_ids is None:
        return {"missing": True}
    for idx, (base, spec) in enumerate(zip(base_ids, spec_ids)):
        if base != spec:
            start = max(0, idx - 10)
            end = idx + 10
            return {
                "index": idx,
                "baseline_token": base,
                "spec_token": spec,
                "baseline_window": base_ids[start:end],
                "spec_window": spec_ids[start:end],
            }
    if len(base_ids) != len(spec_ids):
        return {
            "index": min(len(base_ids), len(spec_ids)),
            "baseline_len": len(base_ids),
            "spec_len": len(spec_ids),
        }
    return None


def output_ids(obj: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if obj is None:
        return None
    ids = obj.get("output_ids")
    return ids if isinstance(ids, list) else None


def meta(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return obj.get("meta_info", {}) if isinstance(obj, dict) else {}


def prompt_ids_from_manifest(manifest: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for row in manifest:
        for prompt_id in row.get("prompt_ids", []):
            if prompt_id not in ids:
                ids.append(prompt_id)
    return ids


def classify(rows: List[Dict[str, Any]]) -> List[str]:
    by_case_prompt = {(r["case"], r["prompt_id"]): r for r in rows}
    findings = []

    for row in rows:
        if row.get("is_force_reject") and row.get("mismatch"):
            findings.append(
                f"{row['case']}:{row['prompt_id']} force-reject mismatched -> target verify / backend / position ids / cache loc suspect"
            )

    for row in rows:
        if row.get("is_force_reject") or not row.get("mismatch"):
            continue
        force_case = row.get("force_reject_case")
        force_row = by_case_prompt.get((force_case, row["prompt_id"])) if force_case else None
        if force_row and not force_row.get("mismatch"):
            findings.append(
                f"{row['case']}:{row['prompt_id']} force-reject exact but accept path mismatched -> accepted KV/SimpleGLA commit or draft-cache update suspect"
            )

    exact_topk1 = {
        (r.get("head_label"), r.get("spec_attention_mode"), r["prompt_id"])
        for r in rows
        if r.get("is_topk1") and not r.get("mismatch")
    }
    for row in rows:
        key = (row.get("head_label"), row.get("spec_attention_mode"), row["prompt_id"])
        topk = row.get("topk") or 0
        if topk > 1 and row.get("mismatch") and key in exact_topk1:
            findings.append(
                f"{row['case']}:{row['prompt_id']} topk1 exact but tree mismatched -> tree parent/sibling SimpleGLA state suspect"
            )

    full_exact = {
        (r.get("case_kind"), r.get("spec_attention_mode"), r["prompt_id"])
        for r in rows
        if r.get("head_vocab_mode") == "full" and not r.get("mismatch")
    }
    for row in rows:
        key = (row.get("case_kind"), row.get("spec_attention_mode"), row["prompt_id"])
        if row.get("head_vocab_mode") == "hot" and row.get("mismatch") and key in full_exact:
            findings.append(
                f"{row['case']}:{row['prompt_id']} full-vocab exact but hot-vocab mismatched -> d2t/t2d/OOV remap suspect"
            )

    return findings or [
        "No automatic classification. Inspect trace JSONL for missing fields or add a narrower reproduction."
    ]


def summarize_trace(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    events = list(events)
    counts = Counter(str(e.get("event", "unknown")) for e in events)
    samples = defaultdict(list)
    accept_lengths: List[int] = []
    for event in events:
        name = str(event.get("event", "unknown"))
        if len(samples[name]) < 2:
            samples[name].append(event)
        if name == "eagle_verify_accept":
            snapshot = event.get("accept_length")
            if isinstance(snapshot, dict):
                for value in snapshot.get("values", []):
                    try:
                        accept_lengths.append(int(value))
                    except (TypeError, ValueError):
                        pass
    return {
        "event_count": len(events),
        "event_counts": dict(counts),
        "samples": dict(samples),
        "accept_length_histogram": dict(sorted(Counter(accept_lengths).items())),
        "max_accept_length": max(accept_lengths) if accept_lengths else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = Path(args.manifest) if args.manifest else run_dir / "case_manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    prompt_ids = prompt_ids_from_manifest(manifest)

    rows: List[Dict[str, Any]] = []
    for case in manifest:
        case_name = case["case"]
        baseline_case = case.get("baseline_case") or case_name
        trace_summary = summarize_trace(load_jsonl(run_dir / f"{case_name}_trace.jsonl"))
        for prompt_id in prompt_ids:
            obj = load_json(run_dir / f"{case_name}_{prompt_id}.json")
            base = load_json(run_dir / f"{baseline_case}_{prompt_id}.json")
            mismatch = None if case_name == baseline_case else first_mismatch(output_ids(base), output_ids(obj))
            case_meta = meta(obj)
            output_len = len(output_ids(obj) or [])
            e2e_latency = case_meta.get("e2e_latency")
            rows.append(
                {
                    **case,
                    "prompt_id": prompt_id,
                    "mismatch": mismatch,
                    "exact": mismatch is None,
                    "output_len": output_len,
                    "baseline_len": len(output_ids(base) or []),
                    "e2e_latency": e2e_latency,
                    "tokens_per_second": (
                        output_len / e2e_latency
                        if isinstance(e2e_latency, (int, float)) and e2e_latency > 0
                        else None
                    ),
                    "spec_accept_length": case_meta.get("spec_accept_length"),
                    "spec_accept_token_num": case_meta.get("spec_accept_token_num"),
                    "spec_verify_ct": case_meta.get("spec_verify_ct"),
                    "trace_accept_histogram": trace_summary["accept_length_histogram"],
                    "trace_max_accept_length": trace_summary["max_accept_length"],
                    "trace_summary": trace_summary,
                }
            )

    findings = classify(rows)
    summary = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "rows": rows,
        "findings": findings,
    }

    output_json = Path(args.output_json) if args.output_json else run_dir / "mismatch_summary.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# EAGLE mismatch forensics",
        "",
        f"- run_dir: `{run_dir}`",
        f"- manifest: `{manifest_path}`",
        f"- summary_json: `{output_json}`",
        "",
        "## Case Matrix",
        "",
        "| case | prompt | exact | mismatch | avg_gen_per_verify | max_accept | tok/s | trace events |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        mismatch = row["mismatch"]
        mismatch_text = "none" if mismatch is None else json.dumps(mismatch, ensure_ascii=False)
        tok_s = row.get("tokens_per_second")
        tok_s_text = "" if tok_s is None else f"{tok_s:.2f}"
        lines.append(
            f"| `{row['case']}` | `{row['prompt_id']}` | {row['exact']} | `{mismatch_text}` | "
            f"{row.get('spec_accept_length')} | {row.get('trace_max_accept_length')} | "
            f"{tok_s_text} | {row['trace_summary']['event_count']} |"
        )

    lines.extend(["", "## Acceptance Histograms", ""])
    for row in rows:
        if row.get("trace_accept_histogram"):
            lines.append(
                f"- `{row['case']}` / `{row['prompt_id']}`: "
                f"`{json.dumps(row['trace_accept_histogram'], ensure_ascii=False)}`"
            )

    lines.extend(["", "## Automatic Classification", ""])
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Required Manual Follow-up",
            "",
            "- If force-reject mismatched, inspect `eagle_verify_accept`, `eagle_verify_commit`, and `simple_gla_forward_temporal_write` events first.",
            "- If tree-only mismatched, inspect `simple_gla_parent_tokens` and per-step `parent_step` checksums.",
            "- If hot-vocab-only mismatched, run `audit_hot_vocab_token_map.py` on the same checkpoint.",
        ]
    )

    output_md = Path(args.output_md) if args.output_md else run_dir / "eagle_mismatch_forensics.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_json={output_json}")
    print(f"wrote_md={output_md}")


if __name__ == "__main__":
    main()
