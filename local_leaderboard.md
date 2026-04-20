# Local Leaderboard

Last updated: 2026-04-15

This file is a lightweight comparison board for SOAR local/public evals.
Only compare rows within the same dataset bucket.

Columns:
- `dataset`: exact eval slice / harness
- `model`: checkpoint or route label
- `avg_score`: accuracy-like score on that dataset
- `out_tokens`: total generated output tokens
- `duration_s`: end-to-end wall clock
- `tps_out`: output tokens per second
- `status`: `verified` if recovered from logs, `pending` if we only have partial metadata
- `notes`: short interpretation

## full150 (`perf_public_set.jsonl`)

| dataset | model | avg_score | out_tokens | duration_s | tps_out | status | notes |
|---|---|---:|---:|---:|---:|---|---|
| full150 | stock base (official original) | ? | 1359286 | ? | ? | pending | the known `official_stock_base_20260409_034318` run did not leave a valid final summary; this token baseline is a later-referenced number, not a recovered official summary line |
| full150 | attnqkv allowlist `gate0/fixorder` | 81.11% | 1193331 | 3767.77 | 316.72 | verified | better than later expanded attnW route; fewer output tokens |
| full150 | attnqkv allowlist `v2/gate1` | 76.82% | 1683315 | 4937.46 | 340.93 | verified | per-token throughput improved, but over-generation got much worse |

## medium25 (`perf_public_medium_eval_v1.jsonl`, active 25-row half-medium)

| dataset | model | avg_score | out_tokens | duration_s | tps_out | status | notes |
|---|---|---:|---:|---:|---:|---|---|
| medium25 | stock base | 50.40% | 430343 | 2405.74 | 178.88 | verified | valid original-weight medium control |
| medium25 | GPTQ py310 dense-fallback baseline | 72.00% | 330508 | 1256.88 | 262.96 | verified | first promising quantized medium control |
| medium25 | `stopaligned64k_v1` | 84.40% | 207140 | 1206.01 | 171.76 | verified | strongest validated medium accuracy anchor in local logs |
| medium25 | `stopaligned64k_v1` rerun | 76.40% | 208484 | 1222.97 | 170.47 | verified | later rerun regressed from the stronger 84.40 anchor |

## fast9 (`perf_public_fast_eval.jsonl`, bounded)

| dataset | model | avg_score | out_tokens | duration_s | tps_out | status | notes |
|---|---|---:|---:|---:|---:|---|---|
| fast9 | stock base | 32.59% | - | - | - | verified | bounded stock fast reference |
| fast9 | GPTQ-Marlin bounded fast | 35.19% | - | - | - | verified | small but real fast-gate lift over stock |
| fast9 | `stopaligned64k_v1` fast retry | 58.89% | - | - | - | verified | strongest fast gate among the logged GPTQ routes |

## focus10 (`perf_public_selective_focus10_v1.jsonl`)

| dataset | model | avg_score | out_tokens | duration_s | tps_out | status | notes |
|---|---|---:|---:|---:|---:|---|---|
| focus10 | stop-aligned stronger baseline | 0.91 | 201146 | - | - | verified | recovered from earlier medium predictions |
| focus10 | stop-aligned latest rerun baseline | 0.71 | 202859 | - | - | verified | more realistic later baseline for selective routing checks |
| focus10 | Candidate B `skip-o-down64` | 0.66 | 285992 | - | - | verified | rejected; violated healthy short-tail rule |
| focus10 | Candidate C later read | 0.52 | - | - | - | verified | selective expansion continued to regress |

## Practical Use

- New weight / runtime route:
  1. run `focus10` first if the change is selective or layer-specific
  2. then run `fast9`
  3. then run `medium25`
  4. only spend `full150` on candidates that survive the cheaper gates
- For speed regressions, always inspect `out_tokens` before blaming raw kernel throughput.
