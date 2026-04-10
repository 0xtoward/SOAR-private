# Calibration Handoff

This note is the shortest safe handoff for a second local Codex conversation that should improve the calibration set for the `W4A16 + GPTQ` route.

## Read These First

- `/Users/ql/cursor/openbmb/clone-server-handoff.md`
- `/Users/ql/cursor/openbmb/quantization-log.md`
- `/Users/ql/cursor/openbmb/worklog.md`
- `/Users/ql/cursor/openbmb/scripts/build_gptq_calibration.py`

## Hard Rules

- Do not use:
  - `calibration_data_merged.jsonl`
- Do not make calibration depend on live HF dataset downloads.
- `source /etc/network_turbo` can help network speed, but it does not fix:
  - old `datasets` compatibility
  - dataset-script deprecations
  - mirror-side `500` errors

## Current Active Calibration

- Remote file:
  - `/root/autodl-tmp/SOAR-Toolkit/calibration/calibration_gptq_w4a16_v1.jsonl`
- Builder:
  - `/Users/ql/cursor/openbmb/scripts/build_gptq_calibration.py`
- Current mix:
  - `48` public-task-aligned rows
  - `16` synthetic ultra-long rows
- Current live route update:
  - the earlier MiniCPM-SALA mapping blocker has largely been fixed
  - the new blocker is numerical:
    - repeated `Hessian remained non positive-definite after diagonal floor attempts`
    - then:
      - `rtn failsafe`
  - this means calibration quality is now directly relevant again

## Distribution Facts

- Public set file:
  - `/root/autodl-tmp/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`
- Public set actual length coverage:
  - no true `4K-16K`
  - no true `128K-160K`
  - longest real public prompt is about `127.7K`
- Therefore the current `128K-160K` coverage in calibration is synthetic.

## What To Improve

- Keep task alignment with the public set.
- Keep long-context coverage.
- Improve semantic quality of the long rows.
- Improve Hessian conditioning for GPTQ, not just superficial length coverage.
- Prefer:
  - public-set-derived multi-document synthesis
  - long coherent text from local sources already present in repo or on server
- Prefer more rows with diverse semantics over a small number of extremely long synthetic rows.
- Avoid:
  - random/noisy text
  - over-weighting `fwe/cwe`
  - letting synthetic `128K-160K` rows dominate the calibration distribution

## Recommended Next Move

- Raise the real GPTQ calibration row count materially:
  - target `128` at minimum
  - `256+` is better for the serious run
- For the first serious GPTQ pass, it is acceptable to trade some synthetic `128K-160K` coverage for much better Hessian stability.
- Focus on coherent long prose / book-like / article-like rows instead of stitched random long rows.

## Current Quantization Blocker

- The active blocker is no longer calibration loading.
- The active blocker is no longer primarily the MiniCPM-SALA model mapping.
- The current blocker is GPTQ numerical stability under the active calibration mix.
- Quantization work is happening in:
  - `/Users/ql/cursor/openbmb/scripts/quantize_gptq_w4a16.py`

## Ask For The Other Conversation

Improve only the calibration data and its builder.
Do not touch runtime benchmarks or NVFP4.
