#!/usr/bin/env python3
"""Lightweight output-id summary for EAGLE forensic runs.

This avoids reading large trace JSONL files when the full trace summary is too
large to aggregate cheaply.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


ACCEPT_RE = re.compile(r"accept len: ([0-9.]+), accept rate: ([0-9.]+)")
TPS_RE = re.compile(r"gen throughput \(token/s\): ([0-9.]+)")


def load_ids(run_dir: pathlib.Path, case: str, prompt_id: str):
    path = run_dir / f"{case}_{prompt_id}.json"
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, {"error": str(exc), "path": str(path)}
    return row.get("output_ids"), row


def first_mismatch(baseline_ids, spec_ids):
    if baseline_ids is None or spec_ids is None:
        return {"index": None, "error": "missing"}

    for index, (baseline_token, spec_token) in enumerate(zip(baseline_ids, spec_ids)):
        if baseline_token != spec_token:
            return {
                "index": index,
                "baseline_token": baseline_token,
                "spec_token": spec_token,
                "baseline_window": baseline_ids[max(0, index - 10) : index + 10],
                "spec_window": spec_ids[max(0, index - 10) : index + 10],
            }

    if len(baseline_ids) != len(spec_ids):
        return {
            "index": min(len(baseline_ids), len(spec_ids)),
            "baseline_len": len(baseline_ids),
            "spec_len": len(spec_ids),
        }

    return None


def parse_metrics(run_dir: pathlib.Path, case: str):
    accept_lens = []
    accept_rates = []
    tps_values = []
    path = run_dir / f"{case}_server.log"
    if not path.exists():
        return {}

    for line in path.read_text(errors="ignore").splitlines():
        accept_match = ACCEPT_RE.search(line)
        if accept_match:
            accept_lens.append(float(accept_match.group(1)))
            accept_rates.append(float(accept_match.group(2)))
        tps_match = TPS_RE.search(line)
        if tps_match:
            tps_values.append(float(tps_match.group(1)))

    metrics = {}
    if accept_lens:
        metrics["avg_accept_len_log"] = sum(accept_lens) / len(accept_lens)
        metrics["max_accept_len_log"] = max(accept_lens)
        metrics["avg_accept_rate_log"] = sum(accept_rates) / len(accept_rates)
    if tps_values:
        metrics["avg_tps_log"] = sum(tps_values) / len(tps_values)
        metrics["min_tps_log"] = min(tps_values)
        metrics["max_tps_log"] = max(tps_values)
    return metrics


def summarize(run_dir: pathlib.Path):
    manifest_path = run_dir / "case_manifest.jsonl"
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    for item in manifest:
        case = item["case"]
        baseline_case = item["baseline_case"]
        exact_count = 0
        mismatches = []

        for prompt_id in item["prompt_ids"]:
            baseline_ids, _ = load_ids(run_dir, baseline_case, prompt_id)
            spec_ids, _ = load_ids(run_dir, case, prompt_id)
            mismatch = first_mismatch(baseline_ids, spec_ids)
            if mismatch is None:
                exact_count += 1
            else:
                mismatches.append({"prompt": prompt_id, "mismatch": mismatch})

        row = {
            "case": case,
            "exact_count": exact_count,
            "total": len(item["prompt_ids"]),
            "exact": exact_count == len(item["prompt_ids"]),
            "mismatches": mismatches[:5],
        }
        row.update(parse_metrics(run_dir, case))
        rows.append(row)

    return rows


def format_num(row, key: str, digits: int = 2):
    if key not in row:
        return ""
    return f"{row[key]:.{digits}f}"


def write_outputs(run_dir: pathlib.Path, rows):
    json_path = run_dir / "eagle_forcebio_fullmatrix_light_summary.json"
    md_path = run_dir / "eagle_forcebio_fullmatrix_light_summary.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# ForceBIO full matrix light summary",
        "",
        f"run_dir: `{run_dir}`",
        "",
        "| case | exact | mismatch count | avg accept len | max accept len | avg accept rate | avg TPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{case}` | {exact_count}/{total} | {mismatch_count} | {avg_accept_len} | {max_accept_len} | {avg_accept_rate} | {avg_tps} |".format(
                case=row["case"],
                exact_count=row["exact_count"],
                total=row["total"],
                mismatch_count=len(row["mismatches"]),
                avg_accept_len=format_num(row, "avg_accept_len_log"),
                max_accept_len=format_num(row, "max_accept_len_log", 0),
                avg_accept_rate=format_num(row, "avg_accept_rate_log"),
                avg_tps=format_num(row, "avg_tps_log"),
            )
        )

    if any(row["mismatches"] for row in rows):
        lines += ["", "## First mismatches"]
        for row in rows:
            for mismatch in row["mismatches"]:
                lines.append(
                    "- `{case}` / `{prompt}`: `{payload}`".format(
                        case=row["case"],
                        prompt=mismatch["prompt"],
                        payload=json.dumps(mismatch["mismatch"], ensure_ascii=False),
                    )
                )
    else:
        lines += ["", "All cases exact against `baseline_triton` output_ids."]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    rows = summarize(args.run_dir)
    json_path, md_path = write_outputs(args.run_dir, rows)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
